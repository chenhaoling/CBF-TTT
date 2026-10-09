"""Post-run provenance, exact construction and raw-score audit for record interference."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_readability import backgrounds
from tasks.cbf_record_interference import summarize,validate,slots,CELLS


def audit(root,tokenizer):
    from transformers import AutoTokenizer
    root = Path(root);data = root/'data/scenes.jsonl'
    design = json.loads((root/'data/design.json').read_text())
    source,cal = Path(design['source_root']),Path(design['readability_root'])
    for directory,tag in ((source,'source'),(cal,'readability')):
        for name in ('data','design'):
            p = directory/'data'/('scenes.jsonl' if name=='data' else 'design.json')
            assert file_digest(p)==design[f'{tag}_{name}_sha256']
    assert file_digest(data)==design['data_sha256']
    scenes = list(map(json.loads,data.read_text().splitlines()));validate(scenes)
    original = {r['id']:r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines())}
    rebuilt = {r['id']:r for r in map(json.loads,(cal/'data/scenes.jsonl').read_text().splitlines()) if r['cell']=='long_far/dual/target_last'}
    tok = AutoTokenizer.from_pretrained(tokenizer)
    bg = backgrounds(json.loads((source/'data/design.json').read_text()),tok)
    newline = tok.encode('\n',add_special_tokens=False);assert len(newline)==1
    for r in scenes:
        old = original[r['id']]
        assert old['split']=='dev' and old['regime']=='stable'
        assert all(r[k]==old[k] for k in ('group','domain','variant','sources','choice_ids','queries','chunk_size'))
        assert r['slots']==slots(old,tok)
        if r['cell']=='rebuilt_bridge':
            assert r['chunks']==rebuilt[r['id']]['chunks']
            assert r['queries']=={k:v['qa'] for k,v in rebuilt[r['id']]['queries'].items()}
            continue
        if r['cell']=='original10':assert r['chunks']==old['chunks']
        for s in r['slots']:
            c,a,b = s['chunk'],s['start'],s['end']
            if c in r['active_chunks']:expected = old['chunks'][c-1][a:b]
            elif r['filler']=='natural':expected = bg[r['group']][c-1][a:b]
            else:expected = newline*(b-a)
            assert r['chunks'][c-1][a:b]==expected
        assert r['chunks'][1]==old['chunks'][1]
    assert design['context_hashes']==[{k:r[k] for k in ('id','cell','context_sha256','sources')} for r in scenes]
    assert design['groups']==list(range(8)) and design['contexts']==160 and design['queries']==320
    assert design['cells']==json.loads(json.dumps(CELLS))
    paths = [root/f'rows_{i}.jsonl' for i in (0,1)]
    summary = json.loads((root/'summary.json').read_text())
    with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
        output = Path(tmp)/'summary.json'
        summarize(SimpleNamespace(data=str(data),inputs=[str(p) for p in paths],output=str(output),
            reference=[str(source/f'q_{i}.jsonl') for i in (0,1)],
            readability_reference=[str(cal/f'rows_{i}.jsonl') for i in (0,1)]))
        assert json.loads(output.read_text())==summary
    start = (root/'started_at.txt').read_text().strip();end = (root/'completed_at.txt').read_text().strip()
    assert not (root/'failed_exit_code.txt').exists()
    assert all(not list(p.glob('confirm_*.jsonl')) for p in (root,source,cal))
    return {'audit_passed':True,'summary_recomputed':True,'construction_rechecked':True,
        'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'code_commit':(root/'code_commit.txt').read_text().strip(),'audit_script_sha256':file_digest(Path(__file__)),
        'design_sha256':file_digest(root/'data/design.json'),'data_sha256':file_digest(data),
        'summary_sha256':file_digest(root/'summary.json'),'rows_sha256':{p.name:file_digest(p) for p in paths},
        'contexts':160,'queries':320,'groups':list(range(8)),'confirm_scored':False,
        'cells':dict(Counter(r['cell'] for r in scenes)),
        'weights_sha256':summary['weights_sha256'],'bridge_checks':summary['bridge_checks'],
        'bridge_max_error':summary['bridge_max_error'],'profile':summary['profile'],
        'natural0_readability_passed':summary['natural0_readability_passed']}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='/home/ctj/models/Qwen3-4B');args=p.parse_args()
    result = audit(args.root,args.tokenizer)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
