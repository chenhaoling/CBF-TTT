"""Independent saved-artifact audit of matched schedules and reversible interventions."""
import argparse
from collections import Counter
import datetime
import json
import math
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.cbf_fact_interference import load,schedules,summarize,ARMS,EXPOSURES
from tasks.train_cbf_event_content import load as load_source
from tasks.train_cbf_event_writer import write_json


def audit(root,model_path):
    import torch
    from transformers import AutoTokenizer
    root=Path(root);rows,d=load(root);tok=AutoTokenizer.from_pretrained(model_path)
    original,source=load_source(d['source'])
    if original!=rows or source['data_sha256']!=d['data_sha256'] or sha(Path(d['source'])/'design.json')!=d['source_design_sha256']:
        raise ValueError('source data changed')
    stored=json.loads((root/'summary.json').read_text())
    if stored!=summarize(root,save=False):raise ValueError('summary recomputation')
    old=json.loads((Path(d['source'])/'summary.json').read_text())
    byid={r['id']:r for r in rows};query_count=0;trial_count=0;all_steps=0;hashes={};initials=[]
    for arm in ARMS:
        p=root/arm;complete=stored['arms'][arm]['complete']
        records=list(map(json.loads,(p/'steps.jsonl').read_text().splitlines()));expected=schedules(rows)[arm]
        if len(records)!=len(expected) or sha(p/'steps.jsonl')!=complete['steps_sha256']:raise ValueError('training count/hash')
        exposure=0
        for index,(rec,batch) in enumerate(zip(records,expected),1):
            exposure+=len(batch['ids']);all_steps+=1
            if rec['step']!=index or rec['round']!=batch['round'] or rec['ids']!=batch['ids'] or rec['exposure']!=exposure:raise ValueError('training schedule')
            if len(rec['regions'])!=len(batch['ids']) or abs(rec['loss']-sum(r['ce'] for r in rec['regions'])/len(batch['ids']))>1e-5:raise ValueError('joint loss averaging')
            for rid,value in zip(batch['ids'],rec['regions']):
                r=byid[rid];nd=len(r['digit_positions']);np=len(r['prefix_positions'])
                if abs(value['ce']*(nd+np+1)-value['digits']*nd-value['prefix']*np-value['eos'])>1e-4:raise ValueError('CE region accounting')
            if not 0<=rec['max_relative_delta']<=1:raise ValueError('delta tripwire')
            for k in ('loss','seconds','peak_allocated_gib','peak_reserved_gib','unclipped_grad_norm'):
                if not math.isfinite(rec[k]) or rec[k]<=0:raise ValueError('training numerics')
            if any(v is None or not math.isfinite(v) for v in rec['gradient_norms'].values()):raise ValueError('gradient audit')
        if Counter(rid for r in records for rid in r['ids'])!=Counter({r['id']:50 for r in rows if r['split']=='train'}):raise ValueError('exposure budget')
        initial=torch.load(p/'writer_0.pt',map_location='cpu',weights_only=True)
        final=torch.load(p/'writer_800.pt',map_location='cpu',weights_only=True);initials.append(initial['writer'])
        if final['optimizer_steps']!=len(records) or final['exposure']!=800:raise ValueError('checkpoint step/exposure')
        for name,w in final['writer'].items():
            change=float((w-initial['writer'][name]).norm())
            if w.dtype!=torch.float32 or not torch.isfinite(w).all() or change<=0 or abs(change-complete['writer_update_norms'][name])>1e-7:
                raise ValueError('saved writer update')
        for exposure in EXPOSURES:
            if exposure:
                optimizer=torch.load(p/f'optimizer_{exposure}.pt',map_location='cpu',weights_only=True)
                if any(v.dtype!=torch.float32 for st in optimizer['state'].values() for v in st.values() if isinstance(v,torch.Tensor)):
                    raise ValueError('optimizer FP32')
                expected_step=exposure if arm=='sequential' else exposure//4
                if any(float(st['step'])!=expected_step for st in optimizer['state'].values()):raise ValueError('probe changed optimizer step')
                if optimizer['param_groups'][0]['lr']!=1e-7 or optimizer['param_groups'][0]['weight_decay']!=0:raise ValueError('optimizer config')
            evaluations=list(map(json.loads,(p/f'evaluation_{exposure}.jsonl').read_text().splitlines()))
            for r in evaluations:
                row=byid[r['id']];policies={'correct','empty','wrong','full_kv'}|({'twin'} if row['anchor_query'] else set())
                if set(r['policies'])!=policies or r['split']!=row['split'] or r['group_id']!=row['group_id']:raise ValueError('score identities')
                for v in r['policies'].values():
                    query_count+=1;text=tok.decode(v['generated_ids'],skip_special_tokens=False).strip();code=re.search(r'\bcode_[0-9]+\b',text)
                    if len(v['generated_ids'])>16 or text!=v['generated'] or v['exact_match']!=int(text==row['answer']):raise ValueError('generation/EM')
                    if v['code_correct']!=int(code is not None and code.group()==row['answer']) or v['first_line_exact']!=int(text.split('\n')[0].strip()==row['answer']):raise ValueError('code/line metric')
                    for k in ('answer_nll','digit_nll','terminal_nll','seconds','peak_allocated_gib','peak_reserved_gib'):
                        if not math.isfinite(v[k]) or v[k]<0:raise ValueError('score/resource numerics')
            for split in ('train','dev'):
                for policy in ('empty','full_kv'):
                    now=stored['arms'][arm]['checkpoints'][str(exposure)][split]['means'][policy]
                    if now!=old['arms']['ce']['checkpoints']['0'][split]['means'][policy]:raise ValueError('previous controls bridge')
            if not exposure:continue
            probes=list(map(json.loads,(p/f'probes_{exposure}.jsonl').read_text().splitlines()))
            if {r['context_id'] for r in probes}!={r['context_id'] for r in rows if r['split']=='train'}:raise ValueError('probe contexts')
            event_count=0;harmed_count=0
            for r in probes:
                ids=sorted(x['id'] for x in rows if x['context_id']==r['context_id'])
                if r['ids']!=ids or len(r['before'])!=4 or len(r['single_trials'])!=4:raise ValueError('probe question matrix')
                if not r['restored_parameters_optimizer'] or r['restored_baseline_max_abs']>1e-6:raise ValueError('probe restoration')
                cos=r['gradient_cosines']
                if len(cos)!=4 or any(len(x)!=4 for x in cos):raise ValueError('gradient matrix shape')
                for i in range(4):
                    for j in range(4):
                        if not math.isfinite(cos[i][j]) or not -1<=cos[i][j]<=1 or abs(cos[i][j]-cos[j][i])>1e-6:raise ValueError('gradient cosine symmetry/bounds')
                    if abs(cos[i][i]-1)>1e-5:raise ValueError('gradient normalization')
                # Recompute the predefined interference events directly from raw before/after losses.
                for i,t in enumerate(r['single_trials']+[r['joint_trial']]):
                    trial_count+=1;targets=[ids[i]] if i<4 else ids
                    if t['targets']!=targets or len(t['after'])!=4:raise ValueError('intervention target matrix')
                    if not math.isfinite(t['loss']) or not 0<=t['max_relative_delta']<=1:raise ValueError('intervention loss/update')
                    if any(not math.isfinite(v) for x in t['after']+r['before'] for v in x.values()):raise ValueError('intervention readings')
                    if any(v is None or not math.isfinite(v) for v in t['gradient_norms'].values()):raise ValueError('intervention gradient')
                    for k in ('seconds','peak_allocated_gib','peak_reserved_gib'):
                        if not math.isfinite(t[k]) or t[k]<=0:raise ValueError('probe resource profile')
                    if i<4:
                        changes=[x['digits']-b['digits'] for x,b in zip(t['after'],r['before'])]
                        others=sum(changes[j]>=.01 for j in range(4) if j!=i)
                        harmed_count+=others;event_count+=int(changes[i]<=-.01 and others>0)
            ps=stored['arms'][arm]['probes'][str(exposure)]
            if ps['interference_events']!=event_count or ps['harmed_other_facts']!=harmed_count:raise ValueError('independent event count')
        hashes[arm]={'checkpoints':{str(e):sha(p/f'writer_{e}.pt') for e in EXPOSURES},
            'steps':sha(p/'steps.jsonl'),'probes':{str(e):sha(p/f'probes_{e}.jsonl') for e in (400,800)}}
        for name,h in final['manifest']['source_files_sha256'].items():
            if sha(Path(model_path)/name)!=h:raise ValueError('source model changed')
    if any(not torch.equal(w,other[n]) for other in initials[1:] for n,w in initials[0].items()):raise ValueError('initial writer difference')
    if (query_count,trial_count,all_steps)!=(1224,120,1200) or (root/'failed_exit_code.txt').exists():raise ValueError('complete counts')
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    return {'audit_passed':True,'optimizer_steps':all_steps,'policy_queries':query_count,'intervention_trials':trial_count,
        'test_scored':False,'example_budgets_equal':True,'initial_writer_states_equal':True,'previous_controls_bridge_exact':True,
        'saved_parameters_optimizer_verified':True,'restoration_flags_and_readout_drift_checked':True,
        'gradient_cosine_shape_symmetry_checked':True,'new_model_calls':0,
        'started_at':start,'completed_at':end,'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'code_commit':(root/'code_commit.txt').read_text().strip(),'data_sha256':d['data_sha256'],
        'design_sha256':sha(root/'design.json'),'summary_sha256':sha(root/'summary.json'),'script_sha256':sha(Path(__file__)),
        'artifacts_sha256':hashes,'content_gates':{a:stored['arms'][a]['content_gates'] for a in ARMS}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--model',default='/home/ctj/models/Qwen3-4B');a=p.parse_args()
    result=audit(a.root,a.model);write_json(Path(a.root)/'execution_audit.json',result);print(json.dumps(result,indent=2))
