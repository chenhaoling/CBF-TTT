"""Independently reconstruct matched-position inputs and recompute completed scores."""
import argparse
import copy
from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_readability import backgrounds
from tasks.cbf_record_interference import slots
from tasks.cbf_matched_position import validate,summarize,aggregate,compare,BRIDGES,CELLS


def binding_diagnostic(scenes, outputs):
    """Exploratory supplement; never change the preregistered sample or contrasts."""
    indexed={(r['id'],r['cell']):r for r in outputs}
    strata={}
    for cell in CELLS[1:]:
        strata[cell]={}
        for name in ('target','anchor'):
            strata[cell][name]={}
            for overlap in (False,True):
                selected=[r for r in scenes if r['cell']==cell and
                    (r['queries'][name]['label'] in r['active_colors'])==overlap]
                scored=[indexed[r['id'],cell]['queries'][name] for r in selected]
                strata[cell][name]['overlap' if overlap else 'disjoint']={
                    'count':len(scored),'source_groups':len({r['group'] for r in selected}),
                    'means':{k:sum(q[k] for q in scored)/len(scored) for k in ('correct','nll','greedy_correct')} if scored else None}
    ids={r['id'] for r in scenes}
    common={i for i in ids if all(r['queries']['target']['label'] not in r['active_colors']
        for r in scenes if r['id']==i and r['cell']!='none')}
    selected=[r for r in outputs if r['id'] in common]
    if not selected:raise ValueError('no common disjoint targets')
    means=aggregate(selected)
    return {'scope':'Exploratory input-metadata supplement added after launch, before inspecting new scores; main design unchanged.',
        'strata':strata,'common_disjoint_target_sessions':len(common),
        'common_disjoint_target_groups':len({r['group'] for r in selected}),
        'common_disjoint_target_means':{c:v['target'] for c,v in means.items()},
        'common_disjoint_target_contrasts':{f'd{d}_late_minus_{p}':compare(means,f'd{d}_p6',f'd{d}_p{p}')['target']
            for d in (3,6) for p in (1,3)}}


def audit(root,tokenizer):
    from transformers import AutoTokenizer
    root=Path(root);d=json.loads((root/'data/design.json').read_text());data=root/'data/scenes.jsonl'
    reference,source=Path(d['reference_root']),Path(d['source_root'])
    for directory,tag in ((reference,'reference'),(source,'source')):
        for name in ('data','design'):
            assert file_digest(directory/'data'/('scenes.jsonl' if name=='data' else 'design.json'))==d[f'{tag}_{name}_sha256']
    assert file_digest(data)==d['data_sha256']
    rows=list(map(json.loads,data.read_text().splitlines()));validate(rows)
    original={r['id']:r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines())}
    old={(r['id'],r['cell']):r for r in map(json.loads,(reference/'data/scenes.jsonl').read_text().splitlines())}
    tok=AutoTokenizer.from_pretrained(tokenizer)
    bg=backgrounds(json.loads((source/'data/design.json').read_text()),tok)
    for r in rows:
        orig=original[r['id']];assert orig['split']=='dev' and orig['regime']=='stable'
        assert all(r[k]==orig[k] for k in ('queries','choice_ids','group','domain','variant','sources','chunk_size'))
        ss=slots(orig,tok);assert r['slots']==ss
        expected=copy.deepcopy(orig['chunks'])
        # Start with the original natural background in every irrelevant record body.
        for s in ss:
            c,a,b=s['chunk'],s['start'],s['end'];expected[c-1][a:b]=bg[r['group']][c-1][a:b]
        if r['donor_chunk']:
            donor=[s for s in ss if s['chunk']==r['donor_chunk']]
            destination=[s for s in ss if s['chunk']==r['position']]
            for a,b in zip(donor,destination):
                expected[b['chunk']-1][b['start']:b['end']]=orig['chunks'][a['chunk']-1][a['start']:a['end']]
                assert orig['chunks'][a['chunk']-1][a['end']]==orig['chunks'][b['chunk']-1][b['end']]
        assert r['chunks']==expected
        if r['cell'] in BRIDGES:
            previous=old[r['id'],BRIDGES[r['cell']]]
            assert r['chunks']==previous['chunks'] and r['queries']==previous['queries']
    assert d['context_hashes']==[{k:r[k] for k in ('id','cell','context_sha256','sources','donor_entries')} for r in rows]
    assert d['contexts']==112 and d['queries']==224 and d['groups']==list(range(8)) and d['cells']==list(CELLS)
    assert d['bridges']==BRIDGES and d['filler']=='natural' and not d['confirm_scored']
    paths=[root/f'rows_{i}.jsonl' for i in (0,1)]
    summary=json.loads((root/'summary.json').read_text())
    with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
        output=Path(tmp)/'summary.json'
        summarize(SimpleNamespace(data=str(data),inputs=[str(p) for p in paths],output=str(output),
            reference=[str(reference/f'rows_{i}.jsonl') for i in (0,1)]))
        assert json.loads(output.read_text())==summary
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    assert not (root/'failed_exit_code.txt').exists()
    assert all(not list(p.glob('confirm_*.jsonl')) for p in (root,reference,source))
    outputs=[r for p in paths for r in map(json.loads,p.read_text().splitlines())]
    diagnostic=binding_diagnostic(rows,outputs)
    return {'audit_passed':True,'construction_rechecked_from_raw_sources':True,'summary_recomputed':True,
        'code_commit':(root/'code_commit.txt').read_text().strip(),'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'audit_script_sha256':file_digest(Path(__file__)),'design_sha256':file_digest(root/'data/design.json'),
        'data_sha256':file_digest(data),'summary_sha256':file_digest(root/'summary.json'),
        'rows_sha256':{p.name:file_digest(p) for p in paths},'weights_sha256':summary['weights_sha256'],
        'contexts':112,'queries':224,'source_groups':8,'confirm_scored':False,
        'bridge_checks':summary['bridge_checks'],'bridge_max_error':summary['bridge_max_error'],
        'consistent_late_target_interference':summary['consistent_late_target_interference'],'profile':summary['profile'],
        'exploratory_label_overlap':diagnostic}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='/home/ctj/models/Qwen3-4B');args=p.parse_args()
    result=audit(args.root,args.tokenizer);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
