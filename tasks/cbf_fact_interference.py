"""Budget-matched question/group training and reversible one-step interference probes."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import setup,write_json
from tasks.train_cbf_event_content import load as load_source,regions,evaluate,aggregate,gates

ARMS=('sequential','joint_context','mixed_context')
EXPOSURES=(0,400,800)


def schedules(rows):
    contexts={}
    for r in rows:
        if r['split']=='train':contexts.setdefault(r['context_id'],[]).append(r['id'])
    if len(contexts)!=4 or any(len(v)!=4 for v in contexts.values()):raise ValueError('expected four contexts/four questions')
    rng=random.Random(403);result={a:[] for a in ARMS}
    for round_id in range(1,51):
        keys=sorted(contexts);rng.shuffle(keys);blocks=[]
        for k in keys:
            ids=sorted(contexts[k]);rng.shuffle(ids);blocks.append(ids)
        for block in blocks:
            result['joint_context'].append({'round':round_id,'ids':block})
            result['sequential'].extend({'round':round_id,'ids':[rid]} for rid in block)
        result['mixed_context'].extend({'round':round_id,'ids':list(ids)} for ids in zip(*blocks))
    return result


def prepare(args):
    source=Path(args.source);rows,old=load_source(source);root=Path(args.root)
    with (root/'rows.jsonl').open('x') as f:f.write((source/'rows.jsonl').read_text())
    design={'protocol':'fact_interference_v1','source':args.source,'source_design_sha256':sha(source/'design.json'),
        'data_sha256':sha(root/'rows.jsonl'),'eos_token_id':old['eos_token_id'],'groups':old['groups'],
        'arms':list(ARMS),'exposures':list(EXPOSURES),'rounds':50,'seed':301,'order_seed':403,
        'lr':1e-7,'weight_decay':0.,'clip':1.,'loss':'mean answer-plus-EOS CE',
        'interference_threshold':.01,'negative_cosine_threshold':-.05,'test_scored':False,
        'optimizer_steps':{a:len(v) for a,v in schedules(rows).items()},
        'training_write_blocks':{'sequential':800,'joint_context':200,'mixed_context':800}}
    write_json(root/'design.json',design)


def load(root):
    root=Path(root);d=json.loads((root/'design.json').read_text())
    if sha(root/'rows.jsonl')!=d['data_sha256'] or d['test_scored']:raise ValueError('source data identity')
    rows=list(map(json.loads,(root/'rows.jsonl').read_text().splitlines()))
    if len(rows)!=32 or any(r['split'] not in ('train','dev') or len(r['context_ids'])!=4096 for r in rows):raise ValueError('row matrix')
    return rows,d


def batch_loss(model,rows,eos):
    from cbf_ttt.event_writer import write_memory
    memories={};values=[];relative=[]
    for r in rows:
        key=r['context_id']
        if key not in memories:
            memory=write_memory(model,r['context_ids']);memories[key]=memory
            relative.extend(float(v.detach().float().norm()/model.model.layers[i].mlp.down_proj.weight.float().norm()) for i,v in memory.items())
        values.append(regions(model,memories[key],r,eos))
    if max(relative)>1:raise RuntimeError('delta/base tripwire')
    means={k:sum(v[k] for v in values)/len(values) for k in values[0]}
    return means['ce'],values,max(relative)


def cpu_copy(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def equal_state(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def restore(params,optimizer,weights,state):
    import torch
    with torch.no_grad():
        for n,p in params.items():p.copy_(weights[n])
    # AdamW's CPU step counters must never alias the saved baseline state.
    optimizer.load_state_dict(copy.deepcopy(state));optimizer.zero_grad(set_to_none=True)
    if any(not torch.equal(p,weights[n]) for n,p in params.items()) or not equal_state(optimizer.state_dict(),state):
        raise RuntimeError('probe failed to restore parameters/optimizer')


def cosine_matrix(vectors):
    import torch
    norms=[v.norm() for v in vectors]
    if any(not torch.isfinite(n) or n<=0 for n in norms):raise RuntimeError('undefined gradient cosine')
    return [[float((torch.dot(a,b)/(na*nb)).clamp(-1,1)) for b,nb in zip(vectors,norms)] for a,na in zip(vectors,norms)]


def scalars(values):return [{k:float(v.detach()) for k,v in row.items()} for row in values]


def checked_backward(loss,params,allow_zero_proj=False):
    import torch
    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
    loss.backward()
    norms={n:float(p.grad.norm()) if p.grad is not None else None for n,p in params.items()}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):raise RuntimeError('invalid writer gradient')
    if any(v==0 for n,v in norms.items() if not (allow_zero_proj and '.ttt_proj.' in n)):raise RuntimeError('zero writer gradient')
    return norms


def probe_context(model,params,optimizer,rows,eos,autocast,measure_fn=None):
    import torch
    from tasks.cbf_selective import measure
    measure_fn=measure if measure_fn is None else measure_fn
    weights={n:p.detach().clone() for n,p in params.items()};state=cpu_copy(optimizer.state_dict())
    def readings():
        with torch.no_grad(),autocast():return scalars(batch_loss(model,rows,eos)[1])
    before=readings();vectors=[];trials=[];joint=None
    try:
        for selected in [[r] for r in rows]+[rows]:
            restore(params,optimizer,weights,state)
            def intervention():
                with autocast():loss,values,relative=batch_loss(model,selected,eos)
                grads=checked_backward(loss,params)
                if len(selected)==1:
                    vectors.append(torch.cat([p.grad.detach().reshape(-1).clone() for p in params.values()]))
                norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True));optimizer.step()
                return {'targets':[r['id'] for r in selected],'loss':float(loss.detach()),'gradient_norms':grads,
                    'unclipped_grad_norm':norm,'max_relative_delta':relative,'after':readings()}
            rec,profile=measure_fn(intervention);rec.update(profile)
            if len(selected)==1:trials.append(rec)
            else:joint=rec
        cos=cosine_matrix(vectors)
    finally:
        restore(params,optimizer,weights,state)
    after_restore=readings()
    drift=max(abs(v[k]-w[k]) for v,w in zip(before,after_restore) for k in v)
    if drift>1e-6:raise RuntimeError('baseline readings drift after reversible probes')
    return {'context_id':rows[0]['context_id'],'ids':[r['id'] for r in rows],'before':before,
        'single_trials':trials,'joint_trial':joint,'gradient_cosines':cos,
        'restored_parameters_optimizer':True,'restored_baseline_max_abs':drift}


def describe_probes(records,threshold=.01):
    matrices=[];events=0;harmed=0;cos_negative=0;joint_harm=0;joint_changes=[];single_changes=[]
    for r in records:
        matrix=[[a['digits']-b['digits'] for a,b in zip(t['after'],r['before'])] for t in r['single_trials']]
        jc=[a['digits']-b['digits'] for a,b in zip(r['joint_trial']['after'],r['before'])]
        flags=[matrix[i][i]<=-threshold and any(matrix[i][j]>=threshold for j in range(4) if j!=i) for i in range(4)]
        events+=sum(flags);harmed+=sum(matrix[i][j]>=threshold for i in range(4) for j in range(4) if i!=j)
        cos_negative+=sum(r['gradient_cosines'][i][j]<-.05 for i in range(4) for j in range(i+1,4))
        joint_harm+=sum(x>=threshold for x in jc);joint_changes.extend(jc)
        single_changes.extend(x for row in matrix for x in row)
        matrices.append({'context_id':r['context_id'],'ids':r['ids'],'single_digit_changes':matrix,
            'gradient_cosines':r['gradient_cosines'],'interference_events':flags,'joint_digit_changes':jc})
    return {'contexts':matrices,'interference_events':events,'single_trials':4*len(records),
        'harmed_other_facts':harmed,'other_fact_pairs':12*len(records),'negative_gradient_pairs':cos_negative,
        'gradient_pairs':6*len(records),'joint_harmed_facts':joint_harm,'joint_fact_targets':4*len(records),
        'joint_mean_digit_change':statistics.fmean(joint_changes),'single_mean_all_fact_digit_change':statistics.fmean(single_changes)}


def probe(model,params,opt,rows,d,root,exposure):
    import torch
    from tasks.cbf_selective import measure
    records=[]
    with (root/f'probes_{exposure}.jsonl').open('x') as sink:
        for key in sorted({r['context_id'] for r in rows if r['split']=='train'}):
            subset=sorted([r for r in rows if r['context_id']==key],key=lambda r:r['id'])
            rec,profile=measure(lambda:probe_context(model,params,opt,subset,d['eos_token_id'],lambda:torch.autocast('cuda',dtype=torch.bfloat16)))
            for k in ('peak_allocated_gib','peak_reserved_gib'):
                profile[k]=max(profile[k],*(v[k] for v in rec['single_trials']+[rec['joint_trial']]))
            rec['profile']=profile;sink.write(json.dumps(rec)+'\n');sink.flush();records.append(rec)
    result=describe_probes(records);result['rows_sha256']=sha(root/f'probes_{exposure}.jsonl')
    result['checkpoint_sha256']=sha(root/f'writer_{exposure}.pt')
    write_json(root/f'probes_{exposure}.summary.json',result)


def run(args):
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import backbone_digest
    from tasks.cbf_selective import measure
    rows,d=load(args.root);root=Path(args.root)/args.arm;root.mkdir()
    model,params,meta=setup(args);tok=AutoTokenizer.from_pretrained(args.model)
    meta.update(arm=args.arm,design_sha256=sha(Path(args.root)/'design.json'),data_sha256=d['data_sha256'])
    write_json(root/'training_manifest.json',meta)
    opt=torch.optim.AdamW(list(params.values()),lr=d['lr'],weight_decay=0.)
    initial=cpu_copy(params);versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    byid={r['id']:r for r in rows};records=[];exposure=0
    def save():
        path=root/f'writer_{exposure}.pt';tmp=Path(str(path)+'.tmp')
        torch.save({'exposure':exposure,'optimizer_steps':len(records),'writer':cpu_copy(params),'manifest':meta},tmp);tmp.replace(path)
        if exposure:torch.save(opt.state_dict(),root/f'optimizer_{exposure}.pt')
    save();evaluate(model,tok,rows,d,root,0)
    with (root/'steps.jsonl').open('x') as sink:
        for step,batch in enumerate(schedules(rows)[args.arm],1):
            selected=[byid[rid] for rid in batch['ids']]
            def one_step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,values,relative=batch_loss(model,selected,d['eos_token_id'])
                grads=checked_backward(loss,params,allow_zero_proj=step==1)
                norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True));opt.step()
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('backbone gradient')
                return {'step':step,'round':batch['round'],'ids':batch['ids'],'loss':float(loss.detach()),'regions':scalars(values),
                    'gradient_norms':grads,'unclipped_grad_norm':norm,'max_relative_delta':relative}
            rec,profile=measure(one_step);exposure+=len(selected);rec.update(profile,exposure=exposure);records.append(rec)
            sink.write(json.dumps(rec)+'\n');sink.flush()
            if exposure%40==0:print(json.dumps({k:rec[k] for k in ('step','exposure','loss','seconds','peak_allocated_gib')}),flush=True)
            if exposure in EXPOSURES:
                save();evaluate(model,tok,rows,d,root,exposure);probe(model,params,opt,rows,d,root,exposure)
    if backbone_digest(model)!=meta['backbone_sha256'] or versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad}:raise RuntimeError('backbone changed')
    changes={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in params.items()}
    if any(v<=0 for v in changes.values()):raise RuntimeError('writer did not change')
    fp32=all(v.dtype==torch.float32 for st in opt.state.values() for v in st.values() if isinstance(v,torch.Tensor))
    if not fp32:raise RuntimeError('optimizer dtype')
    # Probe rollback must also match the checkpoint saved before intervention.
    checkpoint=torch.load(root/'writer_800.pt',map_location='cpu',weights_only=True)
    optimizer=torch.load(root/'optimizer_800.pt',map_location='cpu',weights_only=True)
    if not equal_state(cpu_copy(params),checkpoint['writer']) or not equal_state(opt.state_dict(),optimizer):raise RuntimeError('final state modified by probes')
    write_json(root/'complete.json',{'arm':args.arm,'optimizer_steps':len(records),'exposures':exposure,
        'backbone_unchanged':True,'optimizer_state_fp32':True,'probe_final_state_unchanged':True,'test_scored':False,
        'writer_update_norms':changes,'steps_sha256':sha(root/'steps.jsonl'),
        'mean_step_seconds':statistics.fmean(r['seconds'] for r in records),'training_seconds':sum(r['seconds'] for r in records),
        'peak_allocated_gib':max(r['peak_allocated_gib'] for r in records),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in records)})


def summarize(root,save=True):
    root=Path(root);rows,d=load(root);arms={}
    for arm in ARMS:
        p=root/arm;c=json.loads((p/'complete.json').read_text());checkpoints={};probes={}
        if c['exposures']!=800 or c['optimizer_steps']!=d['optimizer_steps'][arm] or not c['probe_final_state_unchanged'] or not c['backbone_unchanged'] or c['test_scored']:
            raise ValueError('incomplete run')
        for exposure in EXPOSURES:
            records=list(map(json.loads,(p/f'evaluation_{exposure}.jsonl').read_text().splitlines()))
            s=json.loads((p/f'evaluation_{exposure}.summary.json').read_text());agg=aggregate(records)
            if len(records)!=32 or {r['id'] for r in records}!={r['id'] for r in rows} or any(s[k]!=v for k,v in agg.items()):raise ValueError('evaluation identities/means')
            if s['rows_sha256']!=sha(p/f'evaluation_{exposure}.jsonl') or s['checkpoint_sha256']!=sha(p/f'writer_{exposure}.pt'):raise ValueError('evaluation hash')
            checkpoints[str(exposure)]=s
            if exposure:
                recs=list(map(json.loads,(p/f'probes_{exposure}.jsonl').read_text().splitlines()))
                ps=json.loads((p/f'probes_{exposure}.summary.json').read_text());desc=describe_probes(recs)
                if len(recs)!=4 or any(ps[k]!=v for k,v in desc.items()) or ps['rows_sha256']!=sha(p/f'probes_{exposure}.jsonl') or ps['checkpoint_sha256']!=s['checkpoint_sha256']:
                    raise ValueError('probe summary/hash')
                probes[str(exposure)]=ps
        for split in ('train','dev'):
            for exposure in EXPOSURES:
                for policy in ('empty','full_kv'):
                    if checkpoints[str(exposure)][split]['means'][policy]!=checkpoints['0'][split]['means'][policy]:raise ValueError('frozen control drift')
        arms[arm]={'complete':c,'checkpoints':checkpoints,'probes':probes,'content_gates':gates(checkpoints['800'])}
    for arm in ARMS[1:]:
        for split in ('train','dev'):
            if arms[arm]['checkpoints']['0'][split]!=arms['sequential']['checkpoints']['0'][split]:raise ValueError('initial arm difference')
    result={'protocol':d['protocol'],'arms':arms,'data_sha256':d['data_sha256'],'design_sha256':sha(root/'design.json'),
        'test_scored':False,'terminal_status':'completed_fact_interference'}
    if save:write_json(root/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','summarize'))
    p.add_argument('--root',required=True);p.add_argument('--source',default='/home/ctj/cbf_ttt_event_content_v1')
    p.add_argument('--model',default='/home/ctj/models/Qwen3-4B');p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.command=='prepare':prepare(a)
    elif a.command=='run':
        if not a.arm:p.error('--arm required for run')
        run(a)
    else:print(json.dumps(summarize(a.root),indent=2))
