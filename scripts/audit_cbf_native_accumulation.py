"""Independent completion audit for native_accumulation_v1."""
import argparse
import glob
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_native_accumulation import ARMS, SOURCE_SUMMARY_SHA, load_data, summarize


def _load(pattern):
    rows=[]
    for path in sorted(glob.glob(str(pattern))):
        rows += list(map(json.loads,Path(path).read_text().splitlines()))
    return {(r['id'],r['cell']):r for r in rows}


def _bridge(left,right):
    keys=set(left)&set(right); errors=[]; answer=[]; predictions=[]; greedy=[]
    if not keys: raise ValueError('empty bridge')
    for key in keys:
        for query in ('target','anchor'):
            for fmt in ('binding','qa'):
                a,b=left[key]['queries'][query][fmt],right[key]['queries'][query][fmt]
                errors.extend(abs(x-y) for x,y in zip(a['choice_nll'],b['choice_nll']))
                answer.append(abs(a['nll']-b['nll']))
                predictions.append(a['prediction']==b['prediction']);greedy.append(a['greedy_id']==b['greedy_id'])
    return {'contexts':len(keys),'queries':len(predictions),'max_choice_nll_error':max(errors),
            'max_answer_nll_error':max(answer),'prediction_agreement':sum(predictions)/len(predictions),
            'greedy_agreement':sum(greedy)/len(greedy)}


def audit(root):
    root=Path(root); rows,design=load_data(root)
    stored=json.loads((root/'summary.json').read_text()); recomputed=summarize(root)
    if stored != recomputed: raise ValueError('stored summary differs from recomputation')
    source=Path(design['source_root'])
    source_audit=json.loads((source/'execution_audit.json').read_text())
    if not source_audit['audit_passed'] or source_audit['summary_sha256'] != SOURCE_SUMMARY_SHA:
        raise ValueError('source audit changed')
    if file_digest(source/'summary.json') != SOURCE_SUMMARY_SHA:
        raise ValueError('source summary changed')
    expected={(r['id'],r['cell']) for r in rows}
    arm_hashes={}
    for arm in ARMS:
        seen=set()
        for shard in (0,1):
            path=root/f'{arm}_{shard}.jsonl'; audit_path=Path(str(path)+'.audit.json')
            part=list(map(json.loads,path.read_text().splitlines())); meta=json.loads(audit_path.read_text())
            if meta['rows_sha256'] != file_digest(path) or meta['rows'] != len(part): raise ValueError('row hash/count')
            seen.update((r['id'],r['cell']) for r in part)
            arm_hashes[path.name]=file_digest(path)
        if seen != expected: raise ValueError('incomplete arm contexts')
    current={arm:_load(root/f'{arm}_?.jsonl') for arm in ARMS}
    bridges={
        'cumulative_vs_previous_native':_bridge(current['cumulative'],_load(source/'step81381_native_?.jsonl')),
        'zero_vs_previous_plain':_bridge(current['zero'],_load(source/'step81381_plain_?.jsonl')),
        'last_only_vs_reset_each':_bridge(current['last_only'],current['reset_each']),
    }
    if any(v['max_choice_nll_error']!=0 or v['max_answer_nll_error']!=0 or v['prediction_agreement']!=1 or v['greedy_agreement']!=1 for v in bridges.values()):
        raise ValueError('exact bridge failed')
    diagnostic_error=max(abs(current['last_only'][key]['queries'][q][f][name]-current['reset_each'][key]['queries'][q][f][name])
        for key in current['last_only'] for q in ('target','anchor') for f in ('binding','qa')
        for name in ('answer_nll','eos_nll','answer_eos_margin'))
    if diagnostic_error!=0: raise ValueError('last/reset diagnostic mismatch')
    bridges['last_only_vs_reset_each']['max_eos_margin_error']=diagnostic_error
    return {'protocol':'native_accumulation_v1','audit_passed':True,
            'data_sha256':design['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
            'summary_sha256':file_digest(root/'summary.json'),'audit_script_sha256':file_digest(Path(__file__)),
            'source_summary_sha256':SOURCE_SUMMARY_SHA,'rows_sha256':arm_hashes,
            'bridges':bridges,'terminal_status':stored['terminal_status'],'decision':stored['decision']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); result=audit(args.root);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
