"""Read-only completion audit for the fixed checkpoint/native diagnostic."""
import argparse
import datetime
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_checkpoint_native import summarize, STEPS, SOURCE_SHA


def audit(root):
    root=Path(root); summary=json.loads((root/'summary.json').read_text())
    if summarize(root)!=summary:raise ValueError('summary recomputation mismatch')
    if summary['contexts']!=464 or summary['queries']!=1856:raise ValueError('incomplete matrix')
    d=json.loads((root/'data/design.json').read_text()); source=Path(d['source_root'])
    if file_digest(source/'data/scenes.jsonl')!=SOURCE_SHA:raise ValueError('source changed')
    old={r['id']:r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()) if r['segment_tokens']==2048}
    checked_records=0
    for r in map(json.loads,(root/'data/scenes.jsonl').read_text().splitlines()):
        s=old[r['id']]
        if r['chunks']!=[s['chunks'][i]+s['chunks'][i+1] for i in (0,2,4)] or r['queries']!=s['queries'] or r['choice_ids']!=s['choice_ids']:
            raise ValueError('reblocking changed tokens or queries')
        for x,y in zip(r['records'],s['records']):
            old_absolute=(y['chunk']-1)*2048+y['offset'];new_absolute=(x['chunk']-1)*4096+x['offset']
            if old_absolute!=new_absolute or x['entries']!=y['entries']:raise ValueError('record moved/rewritten')
            checked_records+=len(x['entries'])
    if checked_records!=768:raise ValueError('record count')
    files={}
    for items in summary['loading_audits'].values():
        for a in items:
            if a['config'].get('dtype',a['config'].get('torch_dtype'))!='bfloat16':raise ValueError('wrong dtype')
            for name,sha in {**a['source_files_sha256'],'config.json':a['config_sha256']}.items():
                p=Path(a['path'])/name
                if p not in files:files[p]=file_digest(p)
                if files[p]!=sha:raise ValueError('checkpoint changed')
    exports={}
    for step in STEPS[:-1]:
        p=root/f'exports/step{step}/export_audit.json';a=json.loads(p.read_text());src=Path(a['source'])
        inv={x.name:{'size':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns} for x in sorted(src.iterdir()) if x.is_file()}
        if inv!=a['source_inventory'] or file_digest(src/'.metadata')!=a['metadata_sha256'] or not a['source_inventory_unchanged']:
            raise ValueError('DCP source changed')
        for arm in (f'step{step}_plain',f'step{step}_native'):
            if any(x['export_audit_sha256']!=file_digest(p) for x in summary['loading_audits'][arm]):raise ValueError('export audit changed')
        exports[str(step)]={'audit_sha256':file_digest(p),'tensors_equal_to_dcp':a['tensors_equal_to_dcp'],'source_inventory_unchanged':True}
    if (root/'failed_exit_code.txt').exists():raise ValueError('run failed')
    if (root/'terminal_status.txt').read_text().strip()!=summary['terminal_status']:raise ValueError('terminal mismatch')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    return {'audit_passed':True,'summary_recomputed':True,'checkpoint_files_rehashed':len(files),
            'record_absolute_positions_checked':checked_records,'exports':exports,'contexts':464,'queries':1856,
            'confirm_scored':False,'code_commit':(root/'code_commit.txt').read_text().strip(),
            'started_at':start,'completed_at':end,
            'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
            'data_sha256':d['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
            'summary_sha256':file_digest(root/'summary.json'),'audit_script_sha256':file_digest(Path(__file__)),
            'rows_sha256':summary['rows_sha256'],'bridges':summary['bridges'],'terminal_status':summary['terminal_status']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.root);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
