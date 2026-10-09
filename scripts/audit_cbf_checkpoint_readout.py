"""Independently validate completed checkpoint comparison and recompute its summary."""
import argparse
import datetime
import json
from pathlib import Path
from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_checkpoint_readout import summarize, ARMS, DATA_SHA


def audit(root,source):
    root=Path(root);source=Path(source)
    actual=json.loads((root/'summary.json').read_text());expected=summarize(root,source)
    if actual!=expected:raise ValueError('recomputed summary differs')
    old=json.loads((source/'execution_audit.json').read_text())
    if not old['audit_passed'] or old['data_sha256']!=DATA_SHA:raise ValueError('source construction audit')
    for name in ('summary','design'):
        path=source/('data/design.json' if name=='design' else 'summary.json')
        if file_digest(path)!=old[name+'_sha256']:raise ValueError('source archive changed')
    checked_files={}
    for arm,items in actual['loading_audits'].items():
        for a in items:
            model=Path(a['path'])
            if file_digest(model/'config.json')!=a['config_sha256']:raise ValueError('model config changed')
            for name,sha in a['source_files_sha256'].items():
                path=model/name
                if path not in checked_files:checked_files[path]=file_digest(path)
                if checked_files[path]!=sha:raise ValueError('checkpoint changed')
            if a['attention']!='sdpa' or a['config']['ttt_mode']:raise ValueError('inference mode')
            if a['config'].get('dtype')!='bfloat16' and a['config'].get('torch_dtype')!='bfloat16':raise ValueError('dtype')
    if (root/'failed_exit_code.txt').exists() or not (root/'completed_at.txt').exists():raise ValueError('run incomplete')
    status=(root/'terminal_status.txt').read_text().strip()
    if status!=actual['terminal_status']:raise ValueError('terminal mismatch')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    paths=[root/f'{arm}_{i}.jsonl' for arm in ARMS for i in (0,1)]
    if list(root.glob('confirm*')):raise ValueError('unexpected confirmation output')
    return {'audit_passed':True,'summary_recomputed':True,'source_construction_audit_verified':True,
        'source_checkpoint_hashes_rechecked':True,'contexts':176,'queries':608,'confirm_scored':False,
        'terminal_status':status,'code_commit':(root/'code_commit.txt').read_text().strip(),
        'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'summary_sha256':file_digest(root/'summary.json'),'audit_script_sha256':file_digest(Path(__file__)),
        'data_sha256':DATA_SHA,'source_audit_sha256':file_digest(source/'execution_audit.json'),
        'rows_sha256':{p.name:file_digest(p) for p in paths},'bridges':actual['bridges']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.root,args.source)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
