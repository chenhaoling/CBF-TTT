"""Audit completed paired-content training using saved artifacts, without new scoring."""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_content import load,select_rows,training_order,summarize,ARMS,STEPS
from tasks.train_cbf_event_writer import data as original_data,write_json


def audit(root,model_path):
    import torch
    from transformers import AutoTokenizer
    root=Path(root);rows,d=load(root);tok=AutoTokenizer.from_pretrained(model_path)
    original,source_manifest=original_data(d['source'])
    if source_manifest['data_sha256']!=d['source_data_sha256'] or sha(Path(d['source'])/'data/manifest.json')!=d['source_manifest_sha256']:
        raise ValueError('source changed')
    rebuilt,groups=select_rows(original,tok)
    if rebuilt!=rows or groups!=d['groups']:raise ValueError('packing/mask/twin reconstruction')
    stored=json.loads((root/'summary.json').read_text())
    if summarize(root,save=False)!=stored:raise ValueError('summary differs')
    old=json.loads((Path(d['source'])/'summary.json').read_text())
    byid={r['id']:r for r in rows};total=0;checkpoints={};initial_states=[]
    for arm in ARMS:
        p=root/arm;steps=list(map(json.loads,(p/'steps.jsonl').read_text().splitlines()))
        expected=training_order(rows);complete=stored['arms'][arm]['complete']
        if len(steps)!=400 or sha(p/'steps.jsonl')!=complete['steps_sha256']:raise ValueError('step count/hash')
        for step_index,(rec,rid) in enumerate(zip(steps,expected),1):
            row=byid[rid]
            if rec['step']!=step_index or rec['ids']!=[rid,row['twin_row_id']] or rec['anchor_pair']!=row['anchor_query']:raise ValueError('training order/labels')
            ce=sum(v['ce'] for v in rec['own'])/2
            balanced=sum(.75*v['digits']+.125*v['prefix']+.125*v['eos'] for v in rec['own'])/2
            def softplus(x):return max(x,0)+math.log1p(math.exp(-abs(x)))
            pair=sum(softplus(.2+a['digits']-b['digits']) for a,b in zip(rec['own'],rec['cross']))/2*float(row['anchor_query'])
            expected_loss=ce if arm=='ce' else balanced+.5*pair
            if abs(rec['loss']-expected_loss)>2e-5 or abs(pair-rec['pair_loss'])>2e-5:raise ValueError('objective decomposition')
            if not 0<=rec['max_relative_delta']<=1:raise ValueError('delta tripwire')
            for k in ('loss','seconds','peak_allocated_gib','peak_reserved_gib','unclipped_grad_norm'):
                if not math.isfinite(rec[k]) or rec[k]<=0:raise ValueError('training numerics/resource profile')
            if any(v is None or not math.isfinite(v) for v in rec['gradient_norms'].values()):raise ValueError('gradient profile')
        if Counter(rid for rec in steps for rid in rec['ids'])!=Counter({r['id']:50 for r in rows if r['split']=='train'}):raise ValueError('exposure counts')
        initial=torch.load(p/'writer_0.pt',map_location='cpu',weights_only=True)
        final=torch.load(p/'writer_400.pt',map_location='cpu',weights_only=True)
        initial_states.append(initial['writer'])
        for n,v in final['writer'].items():
            if v.dtype!=torch.float32 or not torch.isfinite(v).all():raise ValueError('writer saved dtype/numerics')
            delta=float((v-initial['writer'][n]).norm())
            if delta<=0 or abs(delta-complete['writer_update_norms'][n])>1e-7:raise ValueError('actual writer update')
        optimizer=torch.load(p/'optimizer_400.pt',map_location='cpu',weights_only=True)
        if any(v.dtype!=torch.float32 for st in optimizer['state'].values() for v in st.values() if isinstance(v,torch.Tensor)):
            raise ValueError('optimizer FP32')
        if optimizer['param_groups'][0]['lr']!=1e-7 or optimizer['param_groups'][0]['weight_decay']!=0:raise ValueError('optimizer config')
        checkpoints[arm]={str(s):sha(p/f'writer_{s}.pt') for s in STEPS}
        for step in STEPS:
            for rec in map(json.loads,(p/f'evaluation_{step}.jsonl').read_text().splitlines()):
                row=byid[rec['id']]
                expected_policies={'correct','empty','wrong','full_kv'}|({'twin'} if row['anchor_query'] else set())
                if set(rec['policies'])!=expected_policies or rec['split']!=row['split'] or rec['group_id']!=row['group_id']:
                    raise ValueError('evaluation policies/identity')
                for v in rec['policies'].values():
                    total+=1;text=tok.decode(v['generated_ids'],skip_special_tokens=False).strip()
                    code=re.search(r'\bcode_[0-9]+\b',text)
                    if text!=v['generated'] or len(v['generated_ids'])>16:raise ValueError('generation decoding')
                    if v['exact_match']!=int(text==row['answer']) or v['first_line_exact']!=int(text.split('\n')[0].strip()==row['answer']):raise ValueError('EM mismatch')
                    if v['code_correct']!=int(code is not None and code.group()==row['answer']):raise ValueError('code extraction')
                    for k in ('answer_nll','digit_nll','terminal_nll','seconds','peak_allocated_gib','peak_reserved_gib'):
                        if not math.isfinite(v[k]) or v[k]<0:raise ValueError('evaluation numerics')
            # Exact bridge on pre-existing dev controls; new digit metric has no old counterpart.
            now=stored['arms'][arm]['checkpoints'][str(step)]['dev']['means']
            for policy in ('empty','full_kv'):
                for k,v in old['checkpoints']['0']['means'][policy].items():
                    if now[policy][k]!=v:raise ValueError('old dev readout bridge')
        for name,h in final['manifest']['source_files_sha256'].items():
            if sha(Path(model_path)/name)!=h:raise ValueError('original model file changed')
    if any(not torch.equal(initial_states[0][n],initial_states[1][n]) for n in initial_states[0]):raise ValueError('arm initial weights differ')
    if total!=816 or (root/'failed_exit_code.txt').exists():raise ValueError('completion/score count')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    return {'audit_passed':True,'rows_reconstructed':32,'total_optimizer_steps':800,'policy_queries':total,
        'train_exposures_per_row_per_arm':50,'test_scored':False,'old_dev_control_bridge_exact':True,
        'initial_writer_states_equal':True,'saved_writer_optimizer_fp32':True,'original_source_files_rehashed':True,
        'started_at':start,'completed_at':end,'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'code_commit':(root/'code_commit.txt').read_text().strip(),'summary_sha256':sha(root/'summary.json'),
        'design_sha256':sha(root/'design.json'),'data_sha256':d['data_sha256'],'checkpoint_sha256':checkpoints,
        'script_sha256':sha(Path(__file__)),'content_gates':{a:stored['arms'][a]['content_gates'] for a in ARMS}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--model',default='/home/ctj/models/Qwen3-4B');a=p.parse_args()
    v=audit(a.root,a.model);write_json(Path(a.root)/'execution_audit.json',v);print(json.dumps(v,indent=2))
