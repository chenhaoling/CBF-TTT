"""Fixed paired-content versus CE experiment on two training worlds."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import re
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import data as original_data, setup, score, write_json

STEPS=(0,100,400)
ARMS=('ce','content_pair')
METRICS=('answer_nll','digit_nll','terminal_nll','exact_match','first_line_exact',
         'code_correct','first_token_correct','im_end_probability')


def split_answer(tokenizer,row):
    pieces=[tokenizer.decode([i]) for i in row['answer_ids']]
    digits=[i for i,p in enumerate(pieces) if p and all('0'<=c<='9' for c in p)]
    prefix=[i for i in range(len(pieces)) if i not in digits]
    if not re.fullmatch(r'code_[0-9]{5}',row['answer']):raise ValueError('not a five-digit code')
    if ''.join(pieces[i] for i in digits)!=row['answer'][5:] or ''.join(pieces[i] for i in prefix).strip()!='code_':
        raise ValueError('digit/prefix tokenizer partition mismatch')
    if not digits or not prefix:raise ValueError('empty answer region')
    return digits,prefix


def select_rows(rows,tokenizer):
    selected={s:sorted({r['group_id'] for r in rows if r['split']==s})[:2] for s in ('train','dev')}
    if any(len(v)!=2 for v in selected.values()):raise ValueError('two worlds per split required')
    out=[]
    for old in rows:
        if old['split'] not in selected or old['group_id'] not in selected[old['split']]:continue
        r=copy.deepcopy(old);r['digit_positions'],r['prefix_positions']=split_answer(tokenizer,r)
        groups=selected[r['split']];other=groups[1-groups.index(r['group_id'])]
        r['wrong_context_id']=other+'.'+r['context_id'].split('.')[1];out.append(r)
    if len(out)!=32:raise ValueError('expected 16 train and 16 dev rows')
    byid={r['id']:r for r in out}
    for r in out:
        twin_id=r['twin_context_id']+'.'+r['id'].split('.',2)[2]
        t=byid[twin_id];r['twin_row_id']=twin_id
        if t['query_ids']!=r['query_ids'] or (r['answer']!=t['answer'])!=r['anchor_query']:
            raise ValueError('invalid single-fact twin supervision')
        if r['answer'] in {x['answer'] for x in out if x['context_id']==r['wrong_context_id']}:
            raise ValueError('wrong world accidentally contains the target answer')
    return out,selected


def prepare(args):
    from transformers import AutoTokenizer
    rows,m=original_data(args.source)
    out,groups=select_rows(rows,AutoTokenizer.from_pretrained(args.model))
    root=Path(args.root);p=root/'rows.jsonl'
    with p.open('x') as f:f.write(''.join(json.dumps(r)+'\n' for r in out))
    write_json(root/'design.json',{'protocol':'event_content_v1','groups':groups,'rows':{'train':16,'dev':16},
        'source':args.source,'source_data_sha256':m['data_sha256'],'source_manifest_sha256':sha(Path(args.source)/'data/manifest.json'),
        'data_sha256':sha(p),'eos_token_id':m['eos_token_id'],'test_scored':False,'steps':list(STEPS),
        'arms':list(ARMS),'seed':301,'order_seed':302,'lr':1e-7,'weight_decay':0.,'clip':1.,
        'content_weights':{'digits':.75,'prefix':.125,'eos':.125,'pair':.5,'margin':.2},
        'train_gate':'code>=.75, gain>=.25 vs empty/wrong, each group code>=.5, anchor digit gain vs twin>0',
        'dev_gate':'code>=.25, gain>=.125 vs empty/wrong, each group wrong-correct digit NLL>=.05, anchor digit gain>0'})


def load(root):
    root=Path(root);d=json.loads((root/'design.json').read_text())
    if sha(root/'rows.jsonl')!=d['data_sha256']:raise ValueError('data hash')
    rows=list(map(json.loads,(root/'rows.jsonl').read_text().splitlines()))
    if len(rows)!=32 or any(r['split'] not in ('train','dev') or len(r['context_ids'])!=4096 for r in rows):
        raise ValueError('evaluation matrix')
    return rows,d


def training_order(rows):
    pairs=sorted(r['id'] for r in rows if r['split']=='train' and r['context_id'].endswith('.0'))
    if len(pairs)!=8:raise ValueError('expected eight query pairs')
    rng=random.Random(302);order=[]
    for _ in range(50):
        block=pairs.copy();rng.shuffle(block);order.extend(block)
    return order


def regions(model,memory,row,eos):
    import torch
    import torch.nn.functional as F
    from cbf_ttt.event_writer import answer_logits
    target=row['answer_ids']+[eos]
    logits=answer_logits(model,memory,row['query_ids'],target)
    nll=F.cross_entropy(logits,torch.tensor(target,device=logits.device),reduction='none')
    return {'ce':nll.mean(),'answer':nll[:-1].mean(),'digits':nll[row['digit_positions']].mean(),
            'prefix':nll[row['prefix_positions']].mean(),'eos':nll[-1]}


def objective(own,cross,anchor,arm):
    import torch.nn.functional as F
    ce=sum(v['ce'] for v in own)/2
    balanced=sum(.75*v['digits']+.125*v['prefix']+.125*v['eos'] for v in own)/2
    # Only changed anchor values form negative twins; unchanged facts never get false negatives.
    pair=sum(F.softplus(.2+a['digits']-b['digits']) for a,b in zip(own,cross))/2
    pair=pair*float(anchor)
    if arm=='ce':return ce+0.*pair,pair
    if arm=='content_pair':return balanced+.5*pair,pair
    raise ValueError(arm)


def extra_score(model,tokenizer,row,memory,eos,full_kv=False):
    from cbf_ttt.event_writer import cache_for,forward,answer_logits
    import torch
    import torch.nn.functional as F
    v=score(model,tokenizer,row,memory,eos,full_kv)
    cache=cache_for(model,memory)
    if full_kv:forward(model,row['context_ids'],cache)
    logits=answer_logits(model,memory,row['query_ids'],row['answer_ids'],cache)
    nll=F.cross_entropy(logits,torch.tensor(row['answer_ids'],device=logits.device),reduction='none')
    code=re.search(r'\bcode_[0-9]+\b',v['generated'])
    v.update(digit_nll=float(nll[row['digit_positions']].mean()),
             first_line_exact=int(v['generated'].split('\n')[0].strip()==row['answer']),
             code_correct=int(code is not None and code.group()==row['answer']))
    return v


def aggregate(records):
    result={}
    for split in ('train','dev'):
        subset=[r for r in records if r['split']==split]
        means={p:{k:statistics.fmean(r['policies'][p][k] for r in subset) for k in METRICS}
               for p in ('correct','empty','wrong','full_kv')}
        groups={g:{p:{k:statistics.fmean(r['policies'][p][k] for r in subset if r['group_id']==g) for k in METRICS}
                   for p in means} for g in sorted({r['group_id'] for r in subset})}
        anchor_gain=statistics.fmean(r['policies']['twin']['digit_nll']-r['policies']['correct']['digit_nll'] for r in subset if r['anchor_query'])
        result[split]={'means':means,'groups':groups,'anchor_twin_digit_gain':anchor_gain}
    return result


def evaluate(model,tokenizer,rows,d,root,step):
    import torch
    from cbf_ttt.event_writer import write_memory
    from tasks.cbf_selective import measure
    records=[];profiles={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        with (root/f'evaluation_{step}.jsonl').open('x') as sink:
            for split in ('train','dev'):
                selected=[r for r in rows if r['split']==split]
                contexts={r['context_id']:r['context_ids'] for r in selected};memories={}
                for key,ids in sorted(contexts.items()):
                    memories[key],profiles[key]=measure(lambda:write_memory(model,ids))
                for row in selected:
                    policies={}
                    for p in ('correct','empty','wrong','full_kv')+(('twin',) if row['anchor_query'] else ()):
                        key=row['context_id'] if p=='correct' else row['wrong_context_id'] if p=='wrong' else row['twin_context_id'] if p=='twin' else None
                        memory=memories[key] if key else {}
                        v,profile=measure(lambda:extra_score(model,tokenizer,row,memory,d['eos_token_id'],p=='full_kv'))
                        policies[p]={**v,**profile}
                    rec={k:row[k] for k in ('id','split','group_id','anchor_query')};rec['policies']=policies
                    sink.write(json.dumps(rec)+'\n');sink.flush();records.append(rec)
                del memories,memory
    result=aggregate(records);result.update(step=step,write_profiles=profiles,
        rows_sha256=sha(root/f'evaluation_{step}.jsonl'),checkpoint_sha256=sha(root/f'writer_{step}.pt'))
    write_json(root/f'evaluation_{step}.summary.json',result)
    print(json.dumps({'step':step,'train':result['train']['means'],'dev':result['dev']['means']}),flush=True)


def run(args):
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import write_memory,backbone_digest
    from tasks.cbf_selective import measure
    rows,d=load(args.root);root=Path(args.root)/args.arm;root.mkdir()
    model,params,meta=setup(args);tok=AutoTokenizer.from_pretrained(args.model)
    meta.update(arm=args.arm,design_sha256=sha(Path(args.root)/'design.json'),data_sha256=d['data_sha256'])
    write_json(root/'training_manifest.json',meta)
    opt=torch.optim.AdamW(list(params.values()),lr=d['lr'],weight_decay=0.)
    initial={n:p.detach().cpu().clone() for n,p in params.items()}
    versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    byid={r['id']:r for r in rows};records=[]
    def save(step):
        path=root/f'writer_{step}.pt';tmp=Path(str(path)+'.tmp')
        torch.save({'step':step,'writer':{n:p.detach().cpu() for n,p in params.items()},'manifest':meta},tmp)
        tmp.replace(path)
    save(0);evaluate(model,tok,rows,d,root,0)
    with (root/'steps.jsonl').open('x') as sink:
        for step,rid in enumerate(training_order(rows),1):
            a=byid[rid];b=byid[a['twin_row_id']]
            def one_step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    memories=[write_memory(model,r['context_ids']) for r in (a,b)]
                    own=[regions(model,m,r,d['eos_token_id']) for m,r in zip(memories,(a,b))]
                    cross=[regions(model,m,r,d['eos_token_id']) for m,r in zip(reversed(memories),(a,b))]
                    loss,pair=objective(own,cross,a['anchor_query'],args.arm)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                relative=[max(float(v.float().norm()/model.model.layers[i].mlp.down_proj.weight.float().norm()) for i,v in m.items()) for m in memories]
                if max(relative)>1:raise RuntimeError('delta relative norm exceeds 1')
                loss.backward()
                grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in params.items()}
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):raise RuntimeError('disconnected or nonfinite gradient')
                if any(v==0 for n,v in grads.items() if '.ttt_conv.' in n or step>1):raise RuntimeError('zero writer gradient')
                gradnorm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True));opt.step()
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('backbone gradient')
                return {'step':step,'ids':[a['id'],b['id']],'anchor_pair':a['anchor_query'],'loss':float(loss.detach()),
                    'own':[{k:float(v.detach()) for k,v in r.items()} for r in own],
                    'cross':[{k:float(v.detach()) for k,v in r.items()} for r in cross],
                    'pair_loss':float(pair.detach()),'gradient_norms':grads,'unclipped_grad_norm':gradnorm,'max_relative_delta':max(relative)}
            rec,profile=measure(one_step);rec.update(profile);records.append(rec)
            sink.write(json.dumps(rec)+'\n');sink.flush()
            if step%10==0:print(json.dumps({k:rec[k] for k in ('step','loss','seconds','peak_allocated_gib')}),flush=True)
            if step in STEPS:save(step);evaluate(model,tok,rows,d,root,step)
    if backbone_digest(model)!=meta['backbone_sha256'] or versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad}:
        raise RuntimeError('frozen backbone changed')
    changes={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in params.items()}
    if any(v<=0 for v in changes.values()):raise RuntimeError('writer did not change')
    fp32=all(v.dtype==torch.float32 for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))
    if not fp32:raise RuntimeError('optimizer state dtype')
    torch.save(opt.state_dict(),root/'optimizer_400.pt')
    write_json(root/'complete.json',{'steps':400,'arm':args.arm,'backbone_unchanged':True,'optimizer_state_fp32':True,
        'writer_update_norms':changes,'mean_step_seconds':statistics.fmean(r['seconds'] for r in records),
        'peak_allocated_gib':max(r['peak_allocated_gib'] for r in records),'peak_reserved_gib':max(r['peak_reserved_gib'] for r in records),
        'steps_sha256':sha(root/'steps.jsonl'),'test_scored':False})


def gates(s):
    result={}
    for split,threshold,gain in (('train',.75,.25),('dev',.25,.125)):
        v=s[split];m=v['means'];c=m['correct']['code_correct']
        passed=c>=threshold and all(c-m[p]['code_correct']>=gain for p in ('empty','wrong')) and v['anchor_twin_digit_gain']>0
        if split=='train':passed=passed and all(g['correct']['code_correct']>=.5 for g in v['groups'].values())
        else:passed=passed and all(g['wrong']['digit_nll']-g['correct']['digit_nll']>=.05 for g in v['groups'].values())
        result[split]=passed
    return result


def summarize(root,save=True):
    root=Path(root);rows,d=load(root);arms={}
    for arm in ARMS:
        p=root/arm;complete=json.loads((p/'complete.json').read_text());checkpoints={}
        if complete['steps']!=400 or not complete['backbone_unchanged'] or complete['test_scored']:raise ValueError('run incomplete')
        for step in STEPS:
            records=list(map(json.loads,(p/f'evaluation_{step}.jsonl').read_text().splitlines()))
            s=json.loads((p/f'evaluation_{step}.summary.json').read_text())
            if len(records)!=32 or {r['id'] for r in records}!={r['id'] for r in rows}:raise ValueError('evaluation identities')
            recomputed=aggregate(records)
            if any(s[k]!=v for k,v in recomputed.items()):raise ValueError('summary mismatch')
            if s['rows_sha256']!=sha(p/f'evaluation_{step}.jsonl') or s['checkpoint_sha256']!=sha(p/f'writer_{step}.pt'):raise ValueError('score/checkpoint hash')
            checkpoints[str(step)]=s
        for split in ('train','dev'):
            for step in STEPS:
                for policy in ('empty','full_kv'):
                    if checkpoints[str(step)][split]['means'][policy]!=checkpoints['0'][split]['means'][policy]:raise ValueError('frozen control drift')
            if any(checkpoints['0'][split]['means'][p]!=checkpoints['0'][split]['means']['correct'] for p in ('empty','wrong')):raise ValueError('zero-init parity')
        arms[arm]={'complete':complete,'checkpoints':checkpoints,'content_gates':gates(checkpoints['400'])}
    for split in ('train','dev'):
        if arms['ce']['checkpoints']['0'][split]!=arms['content_pair']['checkpoints']['0'][split]:raise ValueError('arm initialization drift')
    result={'protocol':d['protocol'],'arms':arms,'test_scored':False,'data_sha256':d['data_sha256'],
        'design_sha256':sha(root/'design.json'),'terminal_status':'completed_content_training',
        'content_pair_minus_ce':{s:{k:arms['content_pair']['checkpoints']['400'][s]['means']['correct'][k]-arms['ce']['checkpoints']['400'][s]['means']['correct'][k]
            for k in METRICS} for s in ('train','dev')}}
    if save:write_json(root/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run','summarize'))
    p.add_argument('--root',required=True);p.add_argument('--source',default='/home/ctj/cbf_ttt_event_writer_100step_v1')
    p.add_argument('--model',default='/home/ctj/models/Qwen3-4B');p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.command=='prepare':prepare(a)
    elif a.command=='run':
        if not a.arm:p.error('--arm required for run')
        run(a)
    else:print(json.dumps(summarize(a.root),indent=2))
