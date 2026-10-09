"""Reconstruct disjoint-color records and query formats, then recompute all dev scores."""
import argparse
from contextlib import redirect_stdout
import datetime
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from tasks.build_cbf_memory_value import COLORS,file_digest
from tasks.cbf_record_interference import HEADER
from tasks.cbf_disjoint_readout import validate,summarize,BINDING_TEMPLATE,FORMATS,BRIDGE_GROUPS


def audit(root,tokenizer):
    from transformers import AutoTokenizer
    root=Path(root);data=root/'data/scenes.jsonl';d=json.loads((root/'data/design.json').read_text())
    source=Path(d['source_root'])
    assert file_digest(data)==d['data_sha256']
    assert file_digest(source/'data/scenes.jsonl')==d['source_data_sha256']
    assert file_digest(source/'data/design.json')==d['source_design_sha256']
    rows=list(map(json.loads,data.read_text().splitlines()));validate(rows)
    original={r['id']:r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines())}
    tok=AutoTokenizer.from_pretrained(tokenizer);distractors=0;protected_records=0;changed_colors=0
    for r in rows:
        old=original[r['id']];twin=original[old['twin_id']]
        assert old['split']=='dev' and old['group']<8
        assert all(r[k]==old[k] for k in ('group','domain','regime','variant','sources','choice_ids','chunk_size','old_label','twin_id'))
        for name,fs in r['queries'].items():
            assert fs['qa']==old['queries'][name]
            if not r['is_bridge']:
                assert fs['binding']=={**old['queries'][name],'ids':tok.encode(BINDING_TEMPLATE.format(key=old['queries'][name]['key']),add_special_tokens=False)}
        if r['is_bridge']:
            assert r['chunks']==old['chunks'] and r['records']==old['records'];continue
        protected={old['queries']['target']['label'],twin['queries']['target']['label'],old['queries']['anchor']['label'],old['old_label']}
        palette=sorted(set(range(8))-protected);keys={q['key'] for q in old['queries'].values()}
        assert r['protected_labels']==sorted(protected) and r['distractor_palette']==palette
        for i,(before,after) in enumerate(zip(old['records'],r['records'])):
            assert before['chunk']==after['chunk']==i+1 and before['offset']==after['offset']
            entries=[];changes=0
            for slot,(key,label) in enumerate(before['entries']):
                if key in keys:value=label;protected_records+=1
                else:
                    value=palette[(2*i+slot+old['group'])%4];distractors+=1
                    assert value not in protected
                entries.append([key,value]);changes+=int(value!=label)
            assert after['entries']==entries
            suffix=HEADER+''.join(f'Record: The access color for {key} is {COLORS[label]}.\n' for key,label in entries)
            offset=before['offset'];expected=old['chunks'][i][:offset]+tok.encode(suffix,add_special_tokens=False)
            assert r['chunks'][i]==expected and len(expected)==old['chunk_size']
            positions=[j for j,(a,b) in enumerate(zip(old['chunks'][i],expected)) if a!=b]
            assert positions==r['changed_positions'][i] and len(positions)==changes
            changed_colors+=changes
    assert distractors==624 and protected_records==144
    assert d['contexts']==80 and d['queries']==288 and d['groups']==list(range(8))
    assert d['formats']==list(FORMATS) and d['binding_template']==BINDING_TEMPLATE and d['bridge_groups']==list(BRIDGE_GROUPS)
    assert d['context_hashes']==[{k:r[k] for k in ('id','cell','context_sha256','sources')} for r in rows]
    paths=[root/f'rows_{i}.jsonl' for i in (0,1)];summary=json.loads((root/'summary.json').read_text())
    with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
        output=Path(tmp)/'summary.json'
        summarize(SimpleNamespace(data=str(data),inputs=[str(p) for p in paths],output=str(output),reference=[str(source/f'q_{i}.jsonl') for i in (0,1)]))
        assert json.loads(output.read_text())==summary
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    status=(root/'terminal_status.txt').read_text().strip();assert status==summary['terminal_status']
    assert not (root/'failed_exit_code.txt').exists()
    assert all(not list(p.glob('confirm_*.jsonl')) for p in (root,source))
    return {'audit_passed':True,'construction_rechecked':True,'queries_rechecked':True,'summary_recomputed':True,
        'code_commit':(root/'code_commit.txt').read_text().strip(),'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'audit_script_sha256':file_digest(Path(__file__)),'data_sha256':file_digest(data),
        'design_sha256':file_digest(root/'data/design.json'),'summary_sha256':file_digest(root/'summary.json'),
        'rows_sha256':{p.name:file_digest(p) for p in paths},'weights_sha256':summary['weights_sha256'],
        'contexts':80,'queries':288,'source_groups':8,'confirm_scored':False,'terminal_status':status,
        'distractor_records_checked':distractors,'protected_records_unchanged':protected_records,
        'changed_color_tokens':changed_colors,'artificial_distractor_overlap':0,
        'bridge_checks':summary['bridge_checks'],'bridge_max_error':summary['bridge_max_error'],
        'selection':summary['selection'],'profile':summary['profile']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='/home/ctj/models/Qwen3-4B');args=p.parse_args()
    result=audit(args.root,args.tokenizer);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
