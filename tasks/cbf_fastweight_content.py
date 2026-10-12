"""Test whether one native TTT update carries twin-specific factual content."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_cbf_memory_value import digest, file_digest
from tasks.cbf_checkpoint_native import NativeSession, load_model, native_digest, native_stats
from tasks.cbf_checkpoint_readout import PlainSession, check_loading, parameter_digest
from tasks.cbf_readability import rebuild
from tasks.cbf_selective import measure

SOURCE_DATA_SHA = "c32faa173682cdb9193e9c185de411256a050f206914b925694e7149f408d934"
SOURCE_DESIGN_SHA = "577915a2a9cf74e21252d6270a3537fd34fd7d50dbc7436ec4c32017d68c857b"
Q_GROUPS = tuple(range(8,16))
V_GROUPS = tuple(range(16,24))
LAYERS = (0,6,12,18,24,30,35)
INTERVENTIONS = ("correct","wrong","zero","random0","random1","random2")
RANDOM_COUNT = 3
IDENTITY = ("id","twin_id","group","split","domain","variant","sources","context_sha256")


def shard_for(group):
    return (group//2 + group%2) % 2


def source_backgrounds(design, tokenizer, groups):
    import pyarrow.parquet as pq
    wanted={s for g in groups for s in design['groups'][g]['sources']}
    docs={p['sha256']:p for p in design['candidate_sources'] if p['sha256'] in wanted}
    requests={}
    for sha,item in docs.items(): requests.setdefault(item['parquet'],{}).setdefault(item['row_index'],sha)
    tokens={}
    for path,wanted_rows in requests.items():
        parquet=pq.ParquetFile(path); offset=0
        for rg in range(parquet.num_row_groups):
            size=parquet.metadata.row_group(rg).num_rows
            need={i:sha for i,sha in wanted_rows.items() if offset<=i<offset+size}
            if need:
                for batch in parquet.iter_batches(batch_size=16,row_groups=[rg],columns=['text']):
                    for raw in batch.column(0).to_pylist():
                        if offset in need:
                            sha=need[offset]
                            if digest(raw.encode())!=sha: raise ValueError('original source changed')
                            n=4096 if docs[sha]['domain']=='fineweb' else 6*4096
                            ids=tokenizer.encode(raw,add_special_tokens=False)[:n]
                            if len(ids)!=n: raise ValueError('source too short')
                            tokens[sha]=[ids[i:i+4096] for i in range(0,n,4096)]
                        offset+=1
            else: offset+=size
    if set(tokens)!=wanted: raise ValueError('source rows missing')
    return {g:[chunk for sha in design['groups'][g]['sources'] for chunk in tokens[sha]] for g in groups}


def validate_scenes(rows):
    if len(rows)!=32 or {r['group'] for r in rows}!=set(Q_GROUPS+V_GROUPS): raise ValueError('scene matrix')
    index={r['id']:r for r in rows}
    if len(index)!=32: raise ValueError('duplicate scenes')
    source_owner={}
    for r in rows:
        expected='q' if r['group'] in Q_GROUPS else 'v'
        if r['split']!=expected or r['domain'] not in ('fineweb','longcrawl') or len(r['chunks'])!=2 or any(len(c)!=4096 for c in r['chunks']):
            raise ValueError('scene identity/shape')
        if set(r['queries'])!={'target'} or set(r['queries']['target'])!={'qa'}: raise ValueError('query matrix')
        if r['context_sha256']!=digest(json.dumps(r['chunks']).encode()): raise ValueError('context hash')
        twin=index[r['twin_id']]
        if twin['twin_id']!=r['id'] or any(twin[k]!=r[k] for k in ('group','split','domain','sources','choice_ids')):
            raise ValueError('twin identity')
        if r['queries']['target']['qa']['ids']!=twin['queries']['target']['qa']['ids'] or r['queries']['target']['qa']['label']==twin['queries']['target']['qa']['label']:
            raise ValueError('twin query/answer')
        if sum(a!=b for x,y in zip(r['chunks'],twin['chunks']) for a,b in zip(x,y))!=1: raise ValueError('twin token difference')
        for source in r['sources']:
            if source in source_owner and source_owner[source]!=r['group']: raise ValueError('source reused')
            source_owner[source]=r['group']
    for split in ('q','v'):
        part=[r for r in rows if r['split']==split]
        if len(part)!=16 or {r['domain'] for r in part}!={'fineweb','longcrawl'}: raise ValueError('split balance')
        for domain in ('fineweb','longcrawl'):
            labels=Counter(r['queries']['target']['qa']['label'] for r in part if r['domain']==domain)
            if set(labels)!=set(range(8)) or set(labels.values())!={1}: raise ValueError('label balance')
    if {s for r in rows if r['split']=='q' for s in r['sources']} & {s for r in rows if r['split']=='v' for s in r['sources']}:
        raise ValueError('Q/V source overlap')


def build(source,tokenizer_path,root):
    from transformers import AutoTokenizer
    source,root=Path(source),Path(root)
    if root.exists(): raise FileExistsError(root)
    design_path=source/'data/design.json'; scenes_path=source/'data/scenes.jsonl'
    if file_digest(design_path)!=SOURCE_DESIGN_SHA: raise ValueError('source design changed')
    design=json.loads(design_path.read_text())
    if file_digest(scenes_path)!=SOURCE_DATA_SHA or design['data_sha256']!=SOURCE_DATA_SHA: raise ValueError('source data changed')
    original=list(map(json.loads,scenes_path.read_text().splitlines()))
    selected=[r for r in original if r['group'] in Q_GROUPS+V_GROUPS and r['regime']=='stable']
    if len(selected)!=32 or any(r['split']!='confirm' for r in selected): raise ValueError('expected unscored confirm stable rows')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    backgrounds=source_backgrounds(design,tokenizer,Q_GROUPS+V_GROUPS); rows=[]
    for old in selected:
        row=rebuild(old,backgrounds[old['group']],tokenizer,'short','single')
        row['queries']={'target':{'qa':row['queries']['target']['qa']}}
        row.update(twin_id=old['twin_id'],split='q' if old['group'] in Q_GROUPS else 'v')
        rows.append(row)
    validate_scenes(rows)
    root.mkdir(); data=root/'data'; data.mkdir(); out=data/'scenes.jsonl'
    out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'protocol':'fastweight_content_v1','source_root':str(source),'source_data_sha256':SOURCE_DATA_SHA,
        'source_design_sha256':SOURCE_DESIGN_SHA,'data_sha256':file_digest(out),'q_groups':list(Q_GROUPS),'v_groups':list(V_GROUPS),
        'contexts':32,'q_contexts':16,'v_contexts':16,'interventions':list(INTERVENTIONS),'random_count':RANDOM_COUNT,
        'primary_format':'qa','layout':'short/single','context_tokens':8192,'write_schedule':[0.,.3],
        'q_gate':{'domain_choice_accuracy':.75,'overall_choice_accuracy':.75,'overall_greedy_accuracy':.5,'twin_both_groups':6},
        'content_thresholds':{'nll_gain':.02,'specificity_gain':.05},'confirm_scored_only_after_q':True,
        'context_hashes':[{k:r[k] for k in IDENTITY} for r in rows]}
    (data/'design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def load_data(root):
    root=Path(root); design=json.loads((root/'data/design.json').read_text()); path=root/'data/scenes.jsonl'
    rows=list(map(json.loads,path.read_text().splitlines())); validate_scenes(rows)
    if file_digest(path)!=design['data_sha256'] or design['q_groups']!=list(Q_GROUPS) or design['v_groups']!=list(V_GROUPS) or design['interventions']!=list(INTERVENTIONS):
        raise ValueError('data manifest changed')
    if design['context_hashes']!=[{k:r[k] for k in IDENTITY} for r in rows]: raise ValueError('context manifest')
    source=Path(design['source_root'])
    if file_digest(source/'data/design.json')!=SOURCE_DESIGN_SHA or file_digest(source/'data/scenes.jsonl')!=SOURCE_DATA_SHA: raise ValueError('source changed')
    return rows,design


def plain_score(session,row):
    import torch
    q=row['queries']['target']['qa']; branch=session.clone()
    with torch.inference_mode():
        hidden=branch._forward(q['ids']); logits=branch.model.lm_head(hidden[:,-1:]).float().reshape(-1); logp=logits.log_softmax(-1)
        scores=(-logp[row['choice_ids']]).tolist(); pred=min(range(8),key=lambda i:scores[i]); greedy=int(logits.argmax())
    return {'nll':float(-logp[q['answer_id']]),'choice_nll':scores,'prediction':pred,'correct':int(pred==q['label']),
            'greedy_id':greedy,'greedy_correct':int(greedy==q['answer_id'])}


def collect_q(args):
    import torch
    rows,design=load_data(args.root); rows=[r for r in rows if r['split']=='q' and shard_for(r['group'])==args.shard]
    rows.sort(key=lambda r:(r['group'],r['id'])); out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    torch.manual_seed(312);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,meta=load_model(args.model,'step81381_plain'); before=parameter_digest(model); versions=[p._version for p in model.parameters()];start=time.perf_counter()
    with out.open('x') as sink:
        for i,row in enumerate(rows):
            torch.cuda.empty_cache()
            def prefix():
                session=PlainSession(model)
                for chunk in row['chunks']: session._forward(chunk)
                return session
            session,rollout_profile=measure(prefix); parent=native_digest(session.cache)
            score,query_profile=measure(lambda:plain_score(session,row))
            if parent!=native_digest(session.cache): raise RuntimeError('Q query changed parent')
            item={k:row[k] for k in IDENTITY};item.update(model=args.model,data_sha256=design['data_sha256'],weights_sha256=before,
                parent_unchanged=True,score=score,rollout=rollout_profile,query_profile=query_profile)
            sink.write(json.dumps(item)+'\n');sink.flush();print(json.dumps({'stage':'q','completed':i+1,'total':len(rows)}),flush=True)
    if before!=parameter_digest(model) or versions!=[p._version for p in model.parameters()]: raise RuntimeError('Q weights changed')
    meta.update(stage='q',shard=args.shard,rows=len(rows),rows_sha256=file_digest(out),weights_sha256=before,backbone_unchanged=True,
        seconds=time.perf_counter()-start,torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta,indent=2)+'\n')


def _validate_score(value,row):
    q=row['queries']['target']['qa']; scores=value['choice_nll']
    if len(scores)!=8 or not all(math.isfinite(x) for x in scores+[value['nll']]): raise ValueError('invalid scores')
    pred=min(range(8),key=lambda i:scores[i])
    if value['prediction']!=pred or value['correct']!=int(pred==q['label']) or abs(value['nll']-scores[q['label']])>1e-6 or value['greedy_correct']!=int(value['greedy_id']==q['answer_id']):
        raise ValueError('score mismatch')


def summarize_q(root):
    root=Path(root); scenes,design=load_data(root); expected={r['id']:r for r in scenes if r['split']=='q'};rows=[];audits=[]
    for shard in (0,1):
        path=root/f'q_{shard}.jsonl';part=list(map(json.loads,path.read_text().splitlines()));audit=json.loads(Path(str(path)+'.audit.json').read_text())
        if audit['rows']!=len(part) or audit['rows_sha256']!=file_digest(path) or not audit['backbone_unchanged'] or audit['stage']!='q': raise ValueError('Q audit')
        if any(shard_for(r['group'])!=shard for r in part):raise ValueError('Q shard assignment')
        check_loading(audit['loading_info'],audit['excluded_adapters']);rows+=part;audits.append(audit)
    if len(rows)!=16 or {r['id'] for r in rows}!=set(expected): raise ValueError('Q matrix')
    if len({r['weights_sha256'] for r in rows})!=1 or len({r['model'] for r in rows})!=1 or any(len({json.dumps(a[k],sort_keys=True) for a in audits})!=1 for k in ('weights_sha256','source_files_sha256','config','path')):
        raise ValueError('mixed Q model/shards')
    for r in rows:
        scene=expected[r['id']]
        if any(r[k]!=scene[k] for k in IDENTITY) or r['data_sha256']!=design['data_sha256'] or not r['parent_unchanged']: raise ValueError('Q provenance')
        _validate_score(r['score'],scene)
    def avg(part,key): return statistics.fmean(r['score'][key] for r in part)
    overall={k:avg(rows,k) for k in ('nll','correct','greedy_correct')}
    domains={d:{k:avg([r for r in rows if r['domain']==d],k) for k in ('nll','correct','greedy_correct')} for d in ('fineweb','longcrawl')}
    index={r['id']:r for r in rows};both=0
    for group in Q_GROUPS:
        pair=[r for r in rows if r['group']==group]
        both+=int(len(pair)==2 and all(r['score']['correct'] for r in pair))
    gate=(overall['correct']>=.75 and overall['greedy_correct']>=.5 and all(v['correct']>=.75 for v in domains.values()) and both>=6)
    profiles=[p for r in rows for p in (r['rollout'],r['query_profile'])]
    return {'protocol':'fastweight_content_v1','stage':'q','data_sha256':design['data_sha256'],'contexts':16,'queries':16,
        'overall':overall,'domains':domains,'twin_both_correct_groups':both,'passed':gate,'v_scored':False,
        'weights_sha256':rows[0]['weights_sha256'],'model':rows[0]['model'],
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),'shard_seconds':[a['seconds'] for a in audits]},
        'terminal_status':'q_passed_v_locked' if gate else 'stopped_by_content_Q'}


def _set_lr(model,value):
    for i in model.config.ttt_layers: model.model.layers[i].mlp.ttt_lr=value


def rollout_one_write(model,row):
    session=NativeSession(model);trace=[]
    for chunk,lr in zip(row['chunks'],(0.,.3)):
        _set_lr(model,lr);session._forward(chunk);trace.append(native_stats(session))
    _set_lr(model,.3)
    return session,trace


def move_native_cache(cache,device):
    for layer in cache.layers:
        for key in ('keys','values'):
            value=getattr(layer,key,None)
            if value is not None:setattr(layer,key,value.to(device))
    cache.ttt_states=[tuple(v.to(device) if v is not None else None for v in state) for state in cache.ttt_states]


def _weights(session): return {i:session.cache.ttt_states[i][2] for i in session.model.config.ttt_layers}


def intervention_weights(session,name,wrong_weights=None,seed=None):
    import torch
    result={};ratios={}
    for i in session.model.config.ttt_layers:
        base=session.model.model.layers[i].mlp.down_proj.weight.detach(); own=session.cache.ttt_states[i][2];delta=own-base
        if name=='correct': value=own
        elif name=='wrong': value=wrong_weights[i].to(base.device)
        elif name=='zero': value=base
        elif name.startswith('random'):
            gen=torch.Generator(device=base.device);gen.manual_seed(seed*101+i)
            sign=torch.empty_like(delta).bernoulli_(.5,generator=gen).mul_(2).sub_(1)
            random_delta=delta*sign; value=base+random_delta
            denom=float(delta.float().norm());ratios[str(i)]=float(random_delta.float().norm())/max(denom,1e-30)
            del sign,random_delta
        else: raise ValueError(name)
        result[i]=value.detach().clone()
    return result,ratios


def score_intervention(session,row,name,wrong_weights,eos_id,seed):
    import torch
    q=row['queries']['target']['qa'];weights,ratios=intervention_weights(session,name,wrong_weights,seed);branch=session.clone()
    for i,w in weights.items():
        h,t,_=branch.cache.ttt_states[i]
        if h is not None or t is not None: raise RuntimeError('intervention inside partial chunk')
        branch.cache.TTT_update((None,None,w),i)
    with torch.inference_mode():
        hidden=branch._forward(q['ids']);logits=branch.model.lm_head(hidden[:,-1:]).float().reshape(-1);logp=logits.log_softmax(-1)
        scores=(-logp[row['choice_ids']]).tolist();pred=min(range(8),key=lambda i:scores[i]);greedy=int(logits.argmax())
    twin_label=row['twin_label'];own_nll=scores[q['label']];twin_nll=scores[twin_label]
    value={'nll':own_nll,'own_nll':own_nll,'twin_nll':twin_nll,'own_vs_twin_margin':twin_nll-own_nll,
        'eos_nll':float(-logp[eos_id]),'choice_nll':scores,'prediction':pred,'correct':int(pred==q['label']),
        'greedy_id':greedy,'greedy_correct':int(greedy==q['answer_id']),'eos_greedy':int(greedy==eos_id),
        'random_norm_ratios':ratios}
    if not all(math.isfinite(x) for x in scores+[value['eos_nll'],value['own_vs_twin_margin']]): raise RuntimeError('nonfinite intervention')
    del branch,weights
    return value


def collect_v(args):
    import torch
    rows,design=load_data(args.root);selection=json.loads(Path(args.selection).read_text())
    if not selection['passed'] or selection['data_sha256']!=design['data_sha256'] or selection['v_scored']: raise ValueError('V remains locked')
    selected=[r for r in rows if r['split']=='v' and shard_for(r['group'])==args.shard];selected.sort(key=lambda r:(r['group'],r['id']))
    by_id={r['id']:r for r in selected};out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    torch.manual_seed(313);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,meta=load_model(args.model,'step81381_native');before=parameter_digest(model);versions=[p._version for p in model.parameters()];start=time.perf_counter();seen=set();count=0
    if model.config.eos_token_id!=151645:raise ValueError('unexpected EOS')
    with out.open('x') as sink:
        for first in selected:
            if first['id'] in seen:continue
            second=by_id[first['twin_id']];pair=(first,second);seen.update(r['id'] for r in pair);sessions=[];profiles=[];traces=[];hashes=[]
            for row in pair:
                torch.cuda.empty_cache();(session,trace),profile=measure(lambda r=row:rollout_one_write(model,r));move_native_cache(session.cache,'cpu')
                sessions.append(session);profiles.append(profile);traces.append(trace);hashes.append(native_digest(session.cache))
            cpu_weights=[_weights(s) for s in sessions]
            for index,(row,session) in enumerate(zip(pair,sessions)):
                row=copy.deepcopy(row);row['twin_label']=pair[1-index]['queries']['target']['qa']['label'];move_native_cache(session.cache,model.device);results={}
                for intervention in INTERVENTIONS:
                    seed=3100+row['group']*20+row['variant']*3+(int(intervention[-1]) if intervention.startswith('random') else 0)
                    value,profile=measure(lambda n=intervention,s=seed:score_intervention(session,row,n,cpu_weights[1-index],151645,s))
                    results[intervention]={**value,**profile,'seed':seed if intervention.startswith('random') else None}
                move_native_cache(session.cache,'cpu')
                if native_digest(session.cache)!=hashes[index]:raise RuntimeError('V query changed parent')
                item={k:row[k] for k in IDENTITY};item.update(twin_label=row['twin_label'],model=args.model,data_sha256=design['data_sha256'],weights_sha256=before,
                    parent_sha256=hashes[index],parent_unchanged=True,results=results,rollout=profiles[index],native_update_trace=traces[index])
                sink.write(json.dumps(item)+'\n');sink.flush();count+=1;print(json.dumps({'stage':'v','completed':count,'total':len(selected)}),flush=True)
            del sessions,cpu_weights
    _set_lr(model,.3)
    if before!=parameter_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('V weights changed')
    meta.update(stage='v',shard=args.shard,rows=count,rows_sha256=file_digest(out),weights_sha256=before,backbone_unchanged=True,seconds=time.perf_counter()-start,
        torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta,indent=2)+'\n')


def _policy_value(row,policy,metric):
    if policy!='random_mean':return row['results'][policy][metric]
    return statistics.fmean(row['results'][f'random{i}'][metric] for i in range(RANDOM_COUNT))


def aggregate_v(rows,policy):
    return {metric:statistics.fmean(_policy_value(r,policy,metric) for r in rows) for metric in
        ('own_nll','twin_nll','own_vs_twin_margin','correct','greedy_correct','eos_greedy')}


def paired(rows,left,right,metric,direction=1.,samples=20000,seed=1212):
    values=[]
    for group in V_GROUPS:
        part=[r for r in rows if r['group']==group]
        values.append(statistics.fmean(direction*(_policy_value(r,left,metric)-_policy_value(r,right,metric)) for r in part))
    rng=random.Random(seed);boot=[]
    for _ in range(samples):boot.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    boot.sort();return {'mean':statistics.fmean(values),'ci95':[boot[int(.025*samples)],boot[int(.975*samples)-1]],'group_differences':values,'unit':'source_group','samples':samples,'seed':seed}


def summarize(root):
    root=Path(root);scenes,design=load_data(root);q=json.loads((root/'q_summary.json').read_text())
    if not q['passed'] or q['data_sha256']!=design['data_sha256']:raise ValueError('Q gate not passed')
    expected={r['id']:r for r in scenes if r['split']=='v'};rows=[];audits=[];hashes={}
    for shard in (0,1):
        path=root/f'v_{shard}.jsonl';part=list(map(json.loads,path.read_text().splitlines()));audit=json.loads(Path(str(path)+'.audit.json').read_text())
        if audit['rows']!=len(part) or audit['rows_sha256']!=file_digest(path) or not audit['backbone_unchanged'] or audit['stage']!='v':raise ValueError('V audit')
        if any(shard_for(r['group'])!=shard for r in part):raise ValueError('V shard assignment')
        check_loading(audit['loading_info'],audit['excluded_adapters']);rows+=part;audits.append(audit);hashes[path.name]=file_digest(path)
    if len(rows)!=16 or {r['id'] for r in rows}!=set(expected):raise ValueError('V matrix')
    if len({r['weights_sha256'] for r in rows})!=1 or len({r['model'] for r in rows})!=1 or any(len({json.dumps(a[k],sort_keys=True) for a in audits})!=1 for k in ('weights_sha256','source_files_sha256','config','path')):raise ValueError('mixed V model/shards')
    for r in rows:
        scene=expected[r['id']];twin=expected[scene['twin_id']]
        if any(r[k]!=scene[k] for k in IDENTITY) or r['twin_label']!=twin['queries']['target']['qa']['label'] or r['data_sha256']!=design['data_sha256'] or not r['parent_unchanged']:raise ValueError('V provenance')
        if set(r['results'])!=set(INTERVENTIONS) or len(r['native_update_trace'])!=2:raise ValueError('V intervention matrix')
        if any(v!=0 for v in r['native_update_trace'][0].values()) or not any(v>0 for v in r['native_update_trace'][1].values()):raise ValueError('single write schedule')
        for name,value in r['results'].items():
            _validate_score(value,scene)
            if abs(value['own_nll']-value['choice_nll'][scene['queries']['target']['qa']['label']])>1e-6 or abs(value['twin_nll']-value['choice_nll'][r['twin_label']])>1e-6 or abs(value['own_vs_twin_margin']-(value['twin_nll']-value['own_nll']))>1e-6:raise ValueError('content metric')
            if name.startswith('random') and (set(value['random_norm_ratios'])!=set(map(str,LAYERS)) or any(abs(x-1)>1e-6 for x in value['random_norm_ratios'].values())):raise ValueError('random norm mismatch')
    policies=('correct','wrong','zero','random_mean');tables={p:aggregate_v(rows,p) for p in policies}
    comparisons={
        'correct_vs_wrong_nll_gain':paired(rows,'wrong','correct','own_nll'),
        'correct_vs_random_nll_gain':paired(rows,'random_mean','correct','own_nll'),
        'correct_vs_zero_nll_gain':paired(rows,'zero','correct','own_nll'),
        'correct_vs_wrong_specificity_gain':paired(rows,'correct','wrong','own_vs_twin_margin'),
        'correct_vs_zero_choice_accuracy':paired(rows,'correct','zero','correct'),
    }
    binding=(comparisons['correct_vs_wrong_nll_gain']['mean']>=.02 and comparisons['correct_vs_wrong_nll_gain']['ci95'][0]>0 and
        comparisons['correct_vs_random_nll_gain']['mean']>=.02 and comparisons['correct_vs_random_nll_gain']['ci95'][0]>0 and
        comparisons['correct_vs_wrong_specificity_gain']['mean']>=.05 and comparisons['correct_vs_wrong_specificity_gain']['ci95'][0]>0)
    utility=(comparisons['correct_vs_zero_nll_gain']['mean']>=.02 and comparisons['correct_vs_zero_nll_gain']['ci95'][0]>0 and
        tables['correct']['correct']>=tables['zero']['correct'])
    profiles=[p for r in rows for p in [r['rollout']]+list(r['results'].values())]
    return {'protocol':'fastweight_content_v1','data_sha256':design['data_sha256'],'source_data_sha256':SOURCE_DATA_SHA,
        'q':q,'v_contexts':16,'v_queries':16*len(INTERVENTIONS),'policies':tables,'comparisons':comparisons,
        'decision':{'content_binding_supported':binding,'content_utility_supported':utility,
            'interpretation':'binding_and_utility' if binding and utility else 'binding_without_utility' if binding else 'no_detectable_content_binding'},
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),
            'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in rows),'query_mean_seconds':statistics.fmean(v['seconds'] for r in rows for v in r['results'].values()),'shard_seconds':[a['seconds'] for a in audits]},
        'rows_sha256':hashes,'loading_audits':audits,'terminal_status':'completed_fastweight_content_diagnostic',
        'scope':'Previously unscored source-disjoint confirm groups; constructed short single-key facts, not natural benchmark.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--tokenizer',required=True);b.add_argument('--root',required=True)
    q=sub.add_parser('collect-q');q.add_argument('--root',required=True);q.add_argument('--model',required=True);q.add_argument('--shard',type=int,choices=(0,1),required=True);q.add_argument('--output',required=True)
    sq=sub.add_parser('summarize-q');sq.add_argument('--root',required=True);sq.add_argument('--output',required=True)
    v=sub.add_parser('collect-v');v.add_argument('--root',required=True);v.add_argument('--model',required=True);v.add_argument('--selection',required=True);v.add_argument('--shard',type=int,choices=(0,1),required=True);v.add_argument('--output',required=True)
    s=sub.add_parser('summarize');s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='build':build(a.source,a.tokenizer,a.root)
    elif a.command=='collect-q':collect_q(a)
    elif a.command=='summarize-q':Path(a.output).write_text(json.dumps(summarize_q(a.root),indent=2)+'\n')
    elif a.command=='collect-v':collect_v(a)
    else:Path(a.output).write_text(json.dumps(summarize(a.root),indent=2)+'\n')


if __name__=='__main__':main()
