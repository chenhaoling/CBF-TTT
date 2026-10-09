"""Recheck actual retained suffixes and completed length-diagnostic artifacts."""
import argparse
import datetime
import json
from pathlib import Path
from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_length_readout import summarize,DATA_SHA


def audit(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    if summarize(root)!=summary:raise ValueError('summary recomputation differs')
    d=json.loads((root/'data/design.json').read_text());source=Path(d['source_root'])
    if file_digest(source/'data/scenes.jsonl')!=DATA_SHA:raise ValueError('source changed')
    sources={r['id']:r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()) if not r['is_bridge']}
    rows=list(map(json.loads,(root/'data/scenes.jsonl').read_text().splitlines()))
    records=0;counts={};removed={}
    for r in rows:
        old=sources[r['id']];n=r['segment_tokens'];counts[str(n)]=counts.get(str(n),0)+1
        if r['queries']!=old['queries'] or r['choice_ids']!=old['choice_ids'] or r['sources']!=old['sources']:raise ValueError('truth/query/source changed')
        if len(r['chunks'])!=6 or len(r['records'])!=6:raise ValueError('segment count')
        for i,(before,after) in enumerate(zip(old['chunks'],r['chunks'])):
            rec=r['records'][i];oldrec=old['records'][i]
            if len(after)!=n or after!=before[len(before)-n:]:raise ValueError('not exact suffix')
            if rec['entries']!=oldrec['entries'] or rec['chunk']!=oldrec['chunk'] or rec['offset']!=oldrec['offset']-(4096-n):raise ValueError('record changed')
            if rec['offset']<0 or after[rec['offset']:]!=before[oldrec['offset']:]:raise ValueError('record/header truncated')
            records+=len(rec['entries']);removed[str(n)]=removed.get(str(n),0)+len(before)-len(after)
    if counts!={'1024':64,'2048':64,'4096':16} or records!=1728:raise ValueError('construction counts')
    checked={}
    for items in summary['loading_audits'].values():
        for a in items:
            model=Path(a['path'])
            for name,sha in {**a['source_files_sha256'],'config.json':a['config_sha256']}.items():
                p=model/name
                if p not in checked:checked[p]=file_digest(p)
                if checked[p]!=sha:raise ValueError('checkpoint file changed')
            if a['attention']!='sdpa' or a['config']['ttt_mode'] or a['config'].get('dtype',a['config'].get('torch_dtype'))!='bfloat16':raise ValueError('inference configuration')
    if (root/'failed_exit_code.txt').exists():raise ValueError('failed run')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    if (root/'terminal_status.txt').read_text().strip()!=summary['terminal_status']:raise ValueError('terminal status')
    if list(root.glob('confirm*')):raise ValueError('unexpected confirm output')
    return {'audit_passed':True,'summary_recomputed':True,'independent_suffix_record_check':True,
        'checkpoint_files_rehashed':True,'records_checked_once_per_input':records,'contexts_by_segment_tokens':counts,
        'removed_tokens_by_segment_tokens':removed,'new_contexts':288,'new_queries':1152,
        'analysis_conditions':384,'analysis_queries':1536,'confirm_scored':False,'M':0,
        'code_commit':(root/'code_commit.txt').read_text().strip(),'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'audit_script_sha256':file_digest(Path(__file__)),'summary_sha256':file_digest(root/'summary.json'),
        'data_sha256':file_digest(root/'data/scenes.jsonl'),'design_sha256':file_digest(root/'data/design.json'),
        'reference_summary_sha256':summary['reference_summary_sha256'],'rows_sha256':summary['rows_sha256'],
        'bridges':summary['bridges'],'terminal_status':summary['terminal_status']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.root);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
