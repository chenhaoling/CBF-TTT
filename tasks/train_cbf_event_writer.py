"""100-step native-writer warmup on frozen original Qwen3; asynchronous GPU evaluation."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.pack_cbf_event_warmup import sha

STEPS=(0,25,50,100)


def data(root):
    root=Path(root);m=json.loads((root/'data/manifest.json').read_text());p=root/'data/rows.jsonl'
    if sha(p)!=m['data_sha256'] or m['chunk']!=4096 or m['test_tokenized']:raise ValueError('data manifest')
    rows=list(map(json.loads,p.read_text().splitlines()))
    if len(rows)!=80 or any(len(r['context_ids'])!=4096 or not r['answer_ids'] for r in rows):raise ValueError('packed matrix')
    return rows,m


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def setup(args):
    import torch
    from cbf_ttt.event_writer import load_original,backbone_digest
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,params,info=load_original(args.model)
    digest=backbone_digest(model)
    # Verified in the previous original-checkpoint/plain/native audit.
    if digest!='e75c7303535af6a2711b970f7765d33951074f10e19155ba826d9a3dad967aaf':
        raise ValueError('original backbone differs from audited Qwen3-4B')
    return model,params,{'base_model':args.model,'backbone_sha256':digest,'loading_info':info,
                        'torch':str(torch.__version__),'transformers':__import__('transformers').__version__,
                        'source_files_sha256':{p.name:sha(p) for p in Path(args.model).glob('*.safetensors')},
                        'config_sha256':sha(Path(args.model)/'config.json')}


def train(args):
    import torch
    from cbf_ttt.event_writer import write_memory,answer_loss,backbone_digest
    from tasks.cbf_selective import measure
    root=Path(args.root);rows,m=data(root);training=[r for r in rows if r['split']=='train']
    model,params,meta=setup(args);versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW(list(params.values()),lr=1e-7,weight_decay=0.)
    initial={n:p.detach().cpu().clone() for n,p in params.items()}
    meta.update(lr=1e-7,weight_decay=0.,clip=1.,seed=301,steps=100,train_rows=64,train_groups=8,
                trainable_parameters=sum(p.numel() for p in params.values()),writer_dtype='float32',compute='bfloat16',
                data_sha256=m['data_sha256'],loss='answer plus one terminal EOS; full vocabulary',
                initialization='zero conv, normal projection std=config.initializer_range',g=1)
    write_json(root/'training_manifest.json',meta)
    def save(step):
        path=root/f'writer_{step}.pt';tmp=Path(str(path)+'.tmp')
        torch.save({'step':step,'writer':{n:p.detach().cpu() for n,p in params.items()},'manifest':meta},tmp)
        tmp.replace(path)
        write_json(root/f'writer_{step}.ready.json',{'step':step,'checkpoint_sha256':sha(path),'data_sha256':m['data_sha256']})
    save(0);rng=random.Random(301);order=[];measurements=[]
    with (root/'steps.jsonl').open('x') as sink:
        for step in range(1,101):
            if not order:order=list(range(len(training)));rng.shuffle(order)
            row=training[order.pop()]
            def one_step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    memory=write_memory(model,row['context_ids'])
                    loss,answer_nll,eos_nll=answer_loss(model,memory,row,m['eos_token_id'])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward()
                grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in params.items()}
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):raise RuntimeError('disconnected/nonfinite writer')
                if any(grads[n]==0 for n in grads if '.ttt_conv.' in n):raise RuntimeError('zero conv gradient')
                if any(grads[n]==0 for n in grads if '.ttt_proj.' in n) and step>1:raise RuntimeError('zero projection gradient after first update')
                relative={str(i):float(v.float().norm()/model.model.layers[i].mlp.down_proj.weight.float().norm()) for i,v in memory.items()}
                if max(relative.values())>1.:raise RuntimeError('delta exceeds frozen safety tripwire ||delta||/||W||>1')
                norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True))
                opt.step()
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise RuntimeError('base gradient')
                return {'step':step,'id':row['id'],'group_id':row['group_id'],'loss':float(loss.detach()),
                        'answer_nll':float(answer_nll.detach()),'eos_nll':float(eos_nll.detach()),
                        'unclipped_grad_norm':norm,'gradient_norms':grads,'delta_relative_norm':relative}
            record,profile=measure(one_step);record.update(profile);measurements.append(record)
            sink.write(json.dumps(record)+'\n');sink.flush();print(json.dumps({k:record[k] for k in ('step','loss','answer_nll','seconds','peak_allocated_gib')}),flush=True)
            if step in STEPS:save(step)
    if versions!={n:p._version for n,p in model.named_parameters() if not p.requires_grad} or backbone_digest(model)!=meta['backbone_sha256']:
        raise RuntimeError('backbone changed')
    state_fp32=all(v.dtype==torch.float32 for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    if not state_fp32:raise RuntimeError('optimizer state dtype')
    changes={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in params.items()}
    if any(v<=0 for v in changes.values()):raise RuntimeError('writer not updated')
    torch.save(opt.state_dict(),root/'optimizer_100.pt')
    write_json(root/'training_complete.json',{'steps':100,'backbone_unchanged':True,'optimizer_state_fp32':True,
        'writer_update_norms':changes,'steps_sha256':sha(root/'steps.jsonl'),
        'first10_answer_nll':statistics.fmean(r['answer_nll'] for r in measurements[:10]),
        'last10_answer_nll':statistics.fmean(r['answer_nll'] for r in measurements[-10:]),
        'mean_step_seconds':statistics.fmean(r['seconds'] for r in measurements),
        'peak_allocated_gib':max(r['peak_allocated_gib'] for r in measurements),
        'peak_reserved_gib':max(r['peak_reserved_gib'] for r in measurements)})


def score(model,tokenizer,row,memory,eos,full_kv=False):
    import torch
    from cbf_ttt.event_writer import cache_for,forward,answer_logits
    target=row['answer_ids']+[eos]
    def prefix():
        cache=cache_for(model,memory)
        if full_kv:forward(model,row['context_ids'],cache)
        return cache
    cache=prefix()
    logits=answer_logits(model,memory,row['query_ids'],target,cache)
    logp=logits.log_softmax(-1);labels=torch.tensor(target,device=logits.device)
    nll=-logp[torch.arange(len(target),device=logits.device),labels]
    result={'answer_nll':float(nll[:-1].mean()),'terminal_nll':float(nll[-1]),
            'first_token_correct':int(logits[0].argmax()==target[0]),
            'im_end_probability':float(logp[0,151645].exp())}
    del logits,logp,cache
    cache=prefix();ids=row['query_ids'];generated=[]
    for _ in range(16):
        hidden=forward(model,ids,cache)
        token=int(model.lm_head(hidden[:,-1]).argmax())
        if token in {eos,151643,151645}:break
        generated.append(token);ids=[token]
    text=tokenizer.decode(generated,skip_special_tokens=False).strip()
    result.update(exact_match=int(text==row['answer']),generated=text,generated_ids=generated)
    if not all(math.isfinite(result[k]) for k in ('answer_nll','terminal_nll','im_end_probability')):raise RuntimeError('nonfinite scores')
    return result


def evaluate(args):
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.event_writer import write_memory,backbone_digest
    from tasks.cbf_selective import measure
    root=Path(args.root);rows,m=data(root);dev=[r for r in rows if r['split']=='dev']
    model,params,meta=setup(args);model.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(args.model);contexts={r['context_id']:r['context_ids'] for r in rows}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for step in STEPS:
            ready=root/f'writer_{step}.ready.json'
            while not ready.exists():
                if (root/'train_failed.txt').exists():raise RuntimeError('training failed')
                time.sleep(5)
            info=json.loads(ready.read_text());path=root/f'writer_{step}.pt'
            if info['checkpoint_sha256']!=sha(path):raise ValueError('checkpoint incomplete')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if state['manifest']['data_sha256']!=m['data_sha256'] or state['manifest']['backbone_sha256']!=meta['backbone_sha256']:
                raise ValueError('checkpoint provenance')
            if set(state['writer'])!=set(params):raise ValueError('writer keys mismatch')
            for n,p in params.items():p.copy_(state['writer'][n].to(p.device))
            # Evaluation-only memories: four contexts; no test context is loaded or scored.
            memories={};write_profiles={}
            for key in sorted({r['context_id'] for r in dev}):
                memories[key],write_profiles[key]=measure(lambda:write_memory(model,contexts[key]))
            records=[]
            with (root/f'evaluation_{step}.jsonl').open('x') as sink:
                for row in dev:
                    policies={}
                    for name in ('correct','empty','wrong','full_kv')+ (('twin',) if row['anchor_query'] else ()):
                        memory=memories[row['context_id']] if name=='correct' else memories[row['wrong_context_id']] if name=='wrong' else memories[row['twin_context_id']] if name=='twin' else {}
                        v,profile=measure(lambda:score(model,tokenizer,row,memory,m['eos_token_id'],name=='full_kv'))
                        policies[name]={**v,**profile}
                    rec={'id':row['id'],'group_id':row['group_id'],'anchor_query':row['anchor_query'],'policies':policies}
                    sink.write(json.dumps(rec)+'\n');sink.flush();records.append(rec)
                    print(json.dumps({'checkpoint':step,'dev_completed':len(records),'total':len(dev)}),flush=True)
            del memories
            metrics=('answer_nll','exact_match','first_token_correct','im_end_probability')
            means={p:{k:statistics.fmean(r['policies'][p][k] for r in records) for k in metrics} for p in ('correct','empty','wrong','full_kv')}
            groups={g:{p:{k:statistics.fmean(r['policies'][p][k] for r in records if r['group_id']==g) for k in metrics} for p in means} for g in sorted({r['group_id'] for r in records})}
            anchor=[r for r in records if r['anchor_query']]
            write_json(root/f'evaluation_{step}.summary.json',{'step':step,'rows':16,'groups':groups,'means':means,
                'anchor_correct_minus_twin_nll':statistics.fmean(r['policies']['correct']['answer_nll']-r['policies']['twin']['answer_nll'] for r in anchor),
                'write_profiles':write_profiles,'rows_sha256':sha(root/f'evaluation_{step}.jsonl'),
                'checkpoint_sha256':sha(path),'data_sha256':m['data_sha256'],'test_scored':False})
            if step==100:
                # Fixed training-probe subset: both twins/all facts of two preselected train worlds.
                train_groups=sorted({r['group_id'] for r in rows if r['split']=='train'})[:2]
                train_rows=[r for r in rows if r['group_id'] in train_groups]
                with (root/'final_train_probe.jsonl').open('x') as sink:
                    for row in train_rows:
                        memory=write_memory(model,row['context_ids'])
                        v,profile=measure(lambda:score(model,tokenizer,row,memory,m['eos_token_id']))
                        sink.write(json.dumps({'id':row['id'],'group_id':row['group_id'],**v,**profile})+'\n');sink.flush()
            del state
    if backbone_digest(model)!=meta['backbone_sha256']:raise RuntimeError('evaluation base changed')
    write_json(root/'evaluation_complete.json',{'steps':list(STEPS),'backbone_unchanged':True,'test_scored':False})


def summarize(root):
    root=Path(root);rows,m=data(root)
    train=json.loads((root/'training_complete.json').read_text());ev=json.loads((root/'evaluation_complete.json').read_text())
    if train['steps']!=100 or not train['backbone_unchanged'] or not ev['backbone_unchanged'] or ev['test_scored']:raise ValueError('incomplete run')
    checkpoints={}
    for step in STEPS:
        p=root/f'evaluation_{step}.summary.json';s=json.loads(p.read_text())
        raw=list(map(json.loads,(root/f'evaluation_{step}.jsonl').read_text().splitlines()))
        if len(raw)!=16 or s['rows_sha256']!=sha(root/f'evaluation_{step}.jsonl') or s['checkpoint_sha256']!=sha(root/f'writer_{step}.pt'):raise ValueError('evaluation integrity')
        if {r['id'] for r in raw}!={r['id'] for r in rows if r['split']=='dev'}:raise ValueError('dev identity')
        for policy,metrics in s['means'].items():
            for k,v in metrics.items():
                if v!=statistics.fmean(r['policies'][policy][k] for r in raw):raise ValueError('aggregation mismatch')
        checkpoints[str(step)]=s
    baseline=checkpoints['0']
    for step in STEPS[1:]:
        for policy in ('empty','full_kv'):
            if checkpoints[str(step)]['means'][policy]!=baseline['means'][policy]:raise ValueError('frozen readout control changed')
    for p in ('empty','wrong'):
        if baseline['means'][p]!=baseline['means']['correct']:raise ValueError('zero-init parity')
    # Fixed endpoint, no cherry-picking best intermediate checkpoint.
    final=checkpoints['100'];means=final['means'];passed=all(
        means[p]['answer_nll']-means['correct']['answer_nll']>=.05 and
        means['correct']['exact_match']-means[p]['exact_match']>=.125 for p in ('empty','wrong'))
    passed=passed and all(all(g[p]['answer_nll']>g['correct']['answer_nll'] for p in ('empty','wrong')) for g in final['groups'].values())
    probe=list(map(json.loads,(root/'final_train_probe.jsonl').read_text().splitlines()))
    if len(probe)!=16:raise ValueError('train probe incomplete')
    result={'protocol':'event_writer_warmup_v1','train':train,'checkpoints':checkpoints,
            'final_train_probe':{k:statistics.fmean(r[k] for r in probe) for k in ('answer_nll','exact_match','im_end_probability')},
            'passed_memory_gate':passed,'test_scored':False,'data_sha256':m['data_sha256'],
            'terminal_status':'completed_writer_warmup','rule':'fixed step100, NLL gain >=.05 and EM gain >=.125 against empty/wrong; both dev groups positive NLL gain',
            'files_sha256':{p.name:sha(p) for p in root.glob('*.jsonl')}}
    write_json(root/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('train','evaluate','summarize'))
    p.add_argument('--root',required=True);p.add_argument('--model',default='/home/ctj/models/Qwen3-4B')
    a=p.parse_args()
    if a.command=='summarize':print(json.dumps(summarize(a.root),indent=2))
    else:{'train':train,'evaluate':evaluate}[a.command](a)


if __name__=='__main__':main()
