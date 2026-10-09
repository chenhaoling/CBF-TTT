"""Frozen memory-value experiment: readable facts, causal M controls, held-out gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import validate, REGIMES, digest, file_digest

POLICIES = ('none','retain','half','clear')
MODES = ('full_kv','memory_only')


def mean(items): return statistics.fmean(items)


def tensor_values(cache):
    for layer in cache.layers:
        for key in ('keys','values'):
            value=getattr(layer,key,None)
            if value is not None: yield value
    for value in cache.cbf_memory.values(): yield value
    for value in cache.cbf_candidates.values(): yield value
    for state in cache.ttt_states:
        for value in state:
            if value is not None: yield value


def move_cache(cache,device):
    for layer in cache.layers:
        for key in ('keys','values'):
            value=getattr(layer,key,None)
            if value is not None: setattr(layer,key,value.to(device))
    for name in ('cbf_memory','cbf_candidates'):
        setattr(cache,name,{k:v.to(device) for k,v in getattr(cache,name).items()})
    cache.ttt_states=[tuple(v.to(device) if v is not None else None for v in state) for state in cache.ttt_states]


def state_digest(cache):
    h=hashlib.sha256()
    h.update(str(cache.get_seq_length()).encode())
    for value in tensor_values(cache):
        h.update(str((tuple(value.shape),str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def score_query(session,query,choices,mode,memory):
    import torch
    # The branch owns its KV and M. Query tokens never produce candidates.
    branch=session.clone() if mode=='full_kv' else session.clone_memory_only()
    branch.cache.cbf_memory={k:v.to(session.device).clone() for k,v in memory.items()}
    with torch.inference_mode():
        hidden=branch._forward(query['ids'],collect=False)
        logits=branch.model.lm_head(hidden[:,-1:]).float().reshape(-1)
        logp=logits.log_softmax(-1)
        scores=(-logp[choices]).tolist()
        chosen=min(range(len(choices)),key=lambda i:scores[i])
        result={'nll':float(-logp[query['answer_id']]),'choice_nll':scores,
                'prediction':chosen,'correct':int(chosen==query['label']),
                'greedy_id':int(logits.argmax()),'greedy_correct':int(logits.argmax().item()==query['answer_id'])}
    if not math.isfinite(result['nll']) or not all(math.isfinite(x) for x in scores):
        raise RuntimeError('nonfinite query logits')
    return result


def build_session(model,scene,policy):
    from cbf_ttt.runtime import CBFSession
    session=CBFSession(model)
    for chunk in scene['chunks']:
        session._forward(chunk,collect=policy!='none')
        if policy!='none': session.commit_both({'retain':1.,'half':.5,'clear':0.}[policy],1.)
    if session.cache.cbf_candidates: raise RuntimeError('uncommitted candidate')
    return session


def load_data(args):
    raw=Path(args.data).read_bytes(); rows=[json.loads(x) for x in raw.splitlines()]
    validate(rows)
    design=json.loads(Path(args.design).read_text())
    if digest(raw)!=design['data_sha256']: raise ValueError('data manifest mismatch')
    return rows,design


def collect(args):
    import torch
    from tasks.cbf_ttt import _load_model
    from tasks.cbf_selective import measure
    all_rows,design=load_data(args)
    modes=list(MODES)
    if args.stage=='confirm':
        if not args.selection: raise ValueError('confirm requires locked selection')
        selection=json.loads(Path(args.selection).read_text())
        if selection['data_sha256']!=design['data_sha256'] or selection['model']!=args.model:
            raise ValueError('selection provenance mismatch')
        modes=[m for m in MODES if selection['V'][m]['passed']]
        if not modes: raise ValueError('dev failed: confirm must remain unscored')
    rows=[r for r in all_rows if r['split']==('confirm' if args.stage=='confirm' else 'dev')]
    if args.stage in ('smoke','repeat'): rows=[r for r in rows if r['group'] in (0,2)]
    # Alternating slots distribute the two smoke domains onto different GPUs.
    rows=[r for r in rows if (r['group']//2+r['group']%2)%2==args.shard]
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(208); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=_load_model(args.model,'cuda','bfloat16')
    if model.config.ttt_chunk!=4096: raise ValueError('checkpoint chunk mismatch')
    versions=[p._version for p in model.parameters()]
    weights_before=weight_digest(model)
    started=time.perf_counter(); seen=set(); count=0
    policies=('none',) if args.stage=='q' else POLICIES
    if args.stage=='q': modes=['full_kv']
    with output.open('x') as sink:
        for a in rows:
            if a['id'] in seen: continue
            b=next(r for r in rows if r['id']==a['twin_id'])
            pair=(a,b); seen.update(r['id'] for r in pair)
            for policy in policies:
                sessions=[]; profiles=[]; hashes=[]
                for r in pair:
                    torch.cuda.empty_cache()
                    session,profile=measure(lambda:build_session(model,r,policy))
                    move_cache(session.cache,'cpu')
                    sessions.append(session);profiles.append(profile);hashes.append(state_digest(session.cache))
                for i,(r,session) in enumerate(zip(pair,sessions)):
                    move_cache(session.cache,model.device)
                    results={}
                    for mode in modes:
                        results[mode]={}
                        for intervention in (('self',) if policy=='none' else ('self','zero','twin')):
                            memory=({} if intervention=='zero' else
                                    sessions[1-i].cache.cbf_memory if intervention=='twin' else session.cache.cbf_memory)
                            results[mode][intervention]={}
                            for name,query in r['queries'].items():
                                value,profile=measure(lambda:score_query(session,query,r['choice_ids'],mode,memory))
                                results[mode][intervention][name]={**value,**profile}
                    move_cache(session.cache,'cpu')
                    if state_digest(session.cache)!=hashes[i]: raise RuntimeError('query mutated parent cache')
                    row={k:r[k] for k in ('id','twin_id','group','domain','split','regime','variant','old_label')}
                    row.update({'policy':policy,'stage':args.stage,'model':args.model,
                        'data_sha256':design['data_sha256'],'queries':results,'rollout':profiles[i],
                        'parent_unchanged':True,'parent_sha256':hashes[i],'weights_sha256':weights_before})
                    sink.write(json.dumps(row)+'\n');sink.flush();count+=1
                    print(json.dumps({'scene':r['id'],'policy':policy,'stage':args.stage,'count':count}),flush=True)
                del sessions,session
    if versions!=[p._version for p in model.parameters()] or weights_before!=weight_digest(model):
        raise RuntimeError('backbone changed')
    Path(str(output)+'.audit.json').write_text(json.dumps({'rows':count,'backbone_unchanged':True,
        'weights_sha256':weights_before,'seconds':time.perf_counter()-started,
        'data_sha256':design['data_sha256'],'rows_sha256':file_digest(output)})+'\n')


def weight_digest(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_results(paths,scenes,stage):
    rows=[]
    for path in paths:
        data=Path(path);audit=json.loads(Path(str(path)+'.audit.json').read_text())
        if not audit['backbone_unchanged'] or audit['rows_sha256']!=file_digest(data): raise ValueError('audit/hash failure')
        part=[json.loads(x) for x in data.read_text().splitlines()]
        if len(part)!=audit['rows']: raise ValueError('audit count mismatch')
        rows.extend(part)
    subset=[s for s in scenes if s['split']==('confirm' if stage=='confirm' else 'dev')]
    if stage in ('smoke','repeat'): subset=[s for s in subset if s['group'] in (0,2)]
    expected={(s['id'],p) for s in subset for p in (('none',) if stage=='q' else POLICIES)}
    keys=[(r['id'],r['policy']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected): raise ValueError('incomplete or duplicate results')
    if len({r['model'] for r in rows})!=1 or len({r['data_sha256'] for r in rows})!=1 or len({r['weights_sha256'] for r in rows})!=1:
        raise ValueError('mixed model/data provenance')
    by_id={s['id']:s for s in subset}
    for r in rows:
        s=by_id[r['id']]
        if r['stage']!=stage or not r['parent_unchanged'] or any(r[k]!=s[k] for k in ('group','split','domain','regime','variant','twin_id','old_label')):
            raise ValueError('result identity mismatch')
        for mode,interventions in r['queries'].items():
            if mode not in MODES or set(interventions)!=({'self'} if r['policy']=='none' else {'self','zero','twin'}):
                raise ValueError('missing intervention')
            for qs in interventions.values():
                if set(qs)!={'target','anchor'}: raise ValueError('missing query')
                for name,q in qs.items():
                    if len(q['choice_nll'])!=8 or not all(math.isfinite(v) for v in [q['nll']]+q['choice_nll']):
                        raise ValueError('invalid score')
                    label=s['queries'][name]['label']
                    predicted=min(range(8),key=lambda i:q['choice_nll'][i])
                    if q['prediction']!=predicted or q['correct']!=int(predicted==label) or q['greedy_correct']!=int(q['greedy_id']==s['queries'][name]['answer_id']) or abs(q['nll']-q['choice_nll'][label])>1e-6:
                        raise ValueError('incorrect scoring labels')
    return rows


def q_summary(rows):
    means={regime:{name:mean(r['queries']['full_kv']['self'][name]['correct'] for r in rows if r['regime']==regime)
                   for name in ('target','anchor')} for regime in REGIMES}
    return {'passed':all(v>=.75 for x in means.values() for v in x.values()),'accuracy':means}


def value_metrics(rows,mode,policy):
    stable=[r for r in rows if r['regime']=='stable']
    refs={r['id']:r for r in stable if r['policy']=='none'}
    items=[]
    for r in stable:
        if r['policy']!=policy: continue
        q=r['queries'][mode]; values={n:q[n]['target'] for n in ('self','zero','twin')}
        values['none']=refs[r['id']]['queries'][mode]['self']['target']
        item={'id':r['id'],'group':r['group'],'domain':r['domain']}
        item.update({f'B_{n}':values[n]['nll']-values['self']['nll'] for n in ('none','zero','twin')})
        for n,v in values.items():
            for metric in ('nll','correct','greedy_correct'): item[n+'_'+metric]=v[metric]
        items.append(item)
    keys=[k for k in items[0] if k not in ('id','group','domain')]
    groups={str(g):{k:mean(i[k] for i in items if i['group']==g) for k in keys} for g in sorted({i['group'] for i in items})}
    avg={k:mean(i[k] for i in items) for k in keys}
    positives=[g for g,v in groups.items() if all(v['B_'+n]>0 for n in ('none','zero','twin'))]
    twins=mean(int(all(i['self_correct'] for i in items if i['group']==int(g))) for g in groups)
    threshold=6 if len(groups)==8 else 12
    passed=all(avg['B_'+n]>.05 for n in ('none','zero','twin')) and len(positives)>=threshold
    if mode=='memory_only':
        passed &= avg['self_correct']>=.25 and twins>=.25 and all(avg['self_correct']-avg[n+'_correct']>=.125 for n in ('none','zero','twin'))
    else:
        passed &= all(avg['self_'+metric]>=avg[n+'_'+metric] for n in ('none','zero','twin') for metric in ('correct','greedy_correct'))
    domains={d:sum(str(g) in positives for g in {i['group'] for i in items if i['domain']==d}) for d in ('fineweb','longcrawl')}
    if len(groups)==16: passed &= all(v>=6 for v in domains.values())
    return {'policy':policy,'means':avg,'group_means':groups,'positive_groups':positives,
            'positive_groups_by_domain':domains,'twins_both_correct':twins,'passed':bool(passed)}


def forgetting(rows,mode,locked=None):
    def nll(p,regime): return mean(r['queries'][mode]['self']['target']['nll'] for r in rows if r['policy']==p and r['regime']==regime)
    keep=locked['keep'] if locked else min(('retain','half'),key=lambda p:nll(p,'stable'))
    forget=locked['forget'] if locked else min(('half','clear'),key=lambda p:nll(p,'correction'))
    result={'keep':keep,'forget':forget,'passed':False}
    if locked is None: return result
    groups=sorted({r['group'] for r in rows});deltas={}
    for regime in ('stable','correction'):
        better,worse=(keep,forget) if regime=='stable' else (forget,keep)
        index={(r['id'],r['policy']):r['queries'][mode]['self'] for r in rows if r['regime']==regime}
        records=[]
        for r in rows:
            if r['regime']!=regime or r['policy']!=better: continue
            a,b=index[r['id'],better],index[r['id'],worse]
            records.append({'group':r['group'],'nll_gain':b['target']['nll']-a['target']['nll'],
                'accuracy_gain':a['target']['correct']-b['target']['correct'],
                'anchor_accuracy_gain':a['anchor']['correct']-b['anchor']['correct'],
                'old_error_reduction':int(b['target']['prediction']==r['old_label'])-int(a['target']['prediction']==r['old_label'])})
        deltas[regime]={k:mean(x[k] for x in records) for k in ('nll_gain','accuracy_gain','anchor_accuracy_gain','old_error_reduction')}
        deltas[regime]['positive_groups']=sum(mean(x['nll_gain'] for x in records if x['group']==g)>0 for g in groups)
    result['metrics']=deltas
    result['passed']=keep!=forget and all(v['nll_gain']>.05 and v['positive_groups']>=12 for v in deltas.values()) and deltas['stable']['accuracy_gain']>=0 and deltas['correction']['accuracy_gain']>=.125 and deltas['correction']['old_error_reduction']>0
    return result


def summarize(args):
    scenes,design=load_data(args)
    rows=read_results(args.inputs,scenes,args.stage)
    if rows[0]['data_sha256']!=design['data_sha256']: raise ValueError('wrong design data')
    result={'stage':args.stage,'rows':len(rows),'data_sha256':design['data_sha256'],'model':rows[0]['model'],'weights_sha256':rows[0]['weights_sha256']}
    if args.stage=='q': result['Q']=q_summary(rows)
    elif args.stage in ('smoke','repeat'):
        audits=[json.loads(Path(p+'.audit.json').read_text()) for p in args.inputs]
        profiles=[r['rollout'] for r in rows]+[v for r in rows for m in r['queries'].values() for q in m.values() for v in q.values()]
        peak=max(x['peak_allocated_gib'] for x in profiles)
        estimate=max(a['seconds'] for a in audits)*12*1.25
        result.update({'peak_allocated_gib':peak,'peak_reserved_gib':max(x['peak_reserved_gib'] for x in profiles),
            'estimated_remaining_seconds':estimate,'resource_passed':peak<30 and estimate<=7200})
        if args.reference:
            prior=read_results(args.reference,scenes,'smoke');idx={(r['id'],r['policy']):r for r in prior}
            err=max(abs(v['nll']-idx[r['id'],r['policy']]['queries'][mode][intervention][name]['nll'])
                    for r in rows for mode,qs in r['queries'].items() for intervention,queries in qs.items() for name,v in queries.items())
            result['repeat_max_error']=err;result['repeat_passed']=err<=.005
    else:
        if not args.q_summary: raise ValueError('Q audit required')
        q=json.loads(Path(args.q_summary).read_text())
        if q['data_sha256']!=design['data_sha256'] or q['model']!=result['model'] or q['weights_sha256']!=result['weights_sha256'] or not q['Q']['passed']:
            raise ValueError('Q failed')
        locked=json.loads(Path(args.selection).read_text()) if args.stage=='confirm' else None
        if locked and (locked['weights_sha256']!=result['weights_sha256'] or locked['data_sha256']!=design['data_sha256']): raise ValueError('locked provenance mismatch')
        modes=[m for m in MODES if locked is None or locked['V'][m]['passed']]
        result['V']={};result['F']={}
        for mode in modes:
            for r in rows:
                if mode not in r['queries']: raise ValueError('missing KV condition')
            candidates=[value_metrics(rows,mode,p) for p in ('retain','half','clear')]
            best=(next(v for v in candidates if v['policy']==locked['V'][mode]['policy']) if locked else
                  max(candidates,key=lambda v:min(v['means']['B_'+n] for n in ('none','zero','twin'))))
            result['V'][mode]=best;result['F'][mode]=forgetting(rows,mode,locked['F'][mode] if locked else None)
            result.setdefault('all_candidates',{})[mode]=candidates
        if 'memory_only' in modes:
            none={r['id']:r for r in rows if r['policy']=='none'}
            err=max(abs(v['nll']-none[r['id']]['queries']['memory_only']['self'][name]['nll']) for r in rows if r['policy']!='none'
                    for name,v in r['queries']['memory_only']['zero'].items())
            result['zero_none_max_error']=err
            if err>1e-5: raise ValueError('zero memory differs from no write')
        # All scenes, questions and domains are retained, not only the selected stable target.
        result['strata']={}
        for mode in modes:
            result['strata'][mode]={}
            for policy in POLICIES:
                for regime in REGIMES:
                    for dataset in ('fineweb','longcrawl'):
                        subset=[r for r in rows if r['policy']==policy and r['regime']==regime and r['domain']==dataset]
                        for intervention in (('self',) if policy=='none' else ('self','zero','twin')):
                            result['strata'][mode]['/'.join((policy,regime,dataset,intervention))]={name:{metric:mean(r['queries'][mode][intervention][name][metric] for r in subset) for metric in ('nll','correct','greedy_correct')} for name in ('target','anchor')}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('strata','all_candidates')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('collect','summarize'):
        a=sub.add_parser(command)
        for key in ('data','design','output'): a.add_argument('--'+key,required=True)
        a.add_argument('--stage',choices=('q','smoke','repeat','dev','confirm'),required=True)
        a.add_argument('--selection')
        if command=='collect':
            a.add_argument('--model',required=True);a.add_argument('--shard',type=int,choices=(0,1),required=True)
        else:
            a.add_argument('--inputs',nargs='+',required=True);a.add_argument('--q-summary');a.add_argument('--reference',nargs='+')
    args=p.parse_args();{'collect':collect,'summarize':summarize}[args.command](args)


if __name__=='__main__': main()
