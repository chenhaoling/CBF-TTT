"""Readout-calibrated v2 of the single-update fast-weight content experiment."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import digest,file_digest
from tasks.cbf_checkpoint_native import load_model,native_digest,parameter_digest
from tasks.cbf_checkpoint_readout import PlainSession,check_loading
from tasks.cbf_fastweight_content import (SOURCE_DATA_SHA,SOURCE_DESIGN_SHA,Q_GROUPS,V_GROUPS,LAYERS,INTERVENTIONS,RANDOM_COUNT,
    source_backgrounds,shard_for,plain_score,_validate_score,rollout_one_write,move_native_cache,_weights,score_intervention,
    aggregate_v,paired)
from tasks.cbf_readability import rebuild
from tasks.cbf_selective import measure

DEV_GROUPS=tuple(range(8))
FORMATS=('lowercase','options','completion')
IDENTITY=('id','twin_id','group','split','domain','variant','sources','context_sha256')


def query_text(key,fmt):
    if fmt=='lowercase':
        return f'Question: What is the latest access color recorded for {key} in this session?\nRespond with exactly one lowercase color word and nothing else.\nAnswer:'
    if fmt=='options':
        return f'Question: What is the latest access color recorded for {key} in this session?\nChoose exactly one lowercase word from: red, blue, green, yellow, black, white, orange, purple. Output only that word.\nAnswer:'
    if fmt=='completion':
        return f'According to the session record, complete this statement with one lowercase color word:\nThe latest access color for {key} is'
    raise ValueError(fmt)


def validate(rows):
    groups=DEV_GROUPS+Q_GROUPS+V_GROUPS
    if len(rows)!=48 or {r['group'] for r in rows}!=set(groups):raise ValueError('v2 matrix')
    index={r['id']:r for r in rows};owners={}
    if len(index)!=48:raise ValueError('duplicate id')
    for r in rows:
        split='dev' if r['group'] in DEV_GROUPS else 'q' if r['group'] in Q_GROUPS else 'v'
        if r['split']!=split or len(r['chunks'])!=2 or any(len(c)!=4096 for c in r['chunks']) or set(r['queries'])!={'target'} or set(r['queries']['target'])!=set(FORMATS):raise ValueError('v2 identity/shape')
        if r['context_sha256']!=digest(json.dumps(r['chunks']).encode()):raise ValueError('context hash')
        twin=index[r['twin_id']]
        if twin['twin_id']!=r['id'] or any(twin[k]!=r[k] for k in ('group','split','domain','sources','choice_ids')):raise ValueError('twin identity')
        if sum(a!=b for x,y in zip(r['chunks'],twin['chunks']) for a,b in zip(x,y))!=1:raise ValueError('twin token')
        for fmt,q in r['queries']['target'].items():
            tq=twin['queries']['target'][fmt]
            if q['ids']!=tq['ids'] or q['label']==tq['label'] or q['answer_id']!=r['choice_ids'][q['label']]:raise ValueError('query/answer')
        for source in r['sources']:
            if source in owners and owners[source]!=r['group']:raise ValueError('source reuse')
            owners[source]=r['group']
    for split in ('dev','q','v'):
        part=[r for r in rows if r['split']==split]
        if len(part)!=16:raise ValueError('split size')
        for domain in ('fineweb','longcrawl'):
            labels=Counter(r['queries']['target']['lowercase']['label'] for r in part if r['domain']==domain)
            if set(labels)!=set(range(8)) or set(labels.values())!={1}:raise ValueError('label balance')
    source_sets=[{s for r in rows if r['split']==split for s in r['sources']} for split in ('dev','q','v')]
    if any(source_sets[i]&source_sets[j] for i in range(3) for j in range(i)):raise ValueError('split source overlap')


def build(source,tokenizer_path,root):
    from transformers import AutoTokenizer
    source,root=Path(source),Path(root)
    if root.exists():raise FileExistsError(root)
    sd=source/'data/design.json';sp=source/'data/scenes.jsonl'
    if file_digest(sd)!=SOURCE_DESIGN_SHA or file_digest(sp)!=SOURCE_DATA_SHA:raise ValueError('source changed')
    design=json.loads(sd.read_text());original=list(map(json.loads,sp.read_text().splitlines()));groups=DEV_GROUPS+Q_GROUPS+V_GROUPS
    selected=[r for r in original if r['group'] in groups and r['regime']=='stable'];tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
    backgrounds=source_backgrounds(design,tokenizer,groups);rows=[]
    for old in selected:
        row=rebuild(old,backgrounds[old['group']],tokenizer,'short','single');base=old['queries']['target'];queries={}
        for fmt in FORMATS:queries[fmt]={**base,'ids':tokenizer.encode(query_text(base['key'],fmt),add_special_tokens=False)}
        row['queries']={'target':queries};row.update(twin_id=old['twin_id'],split='dev' if old['group'] in DEV_GROUPS else 'q' if old['group'] in Q_GROUPS else 'v');rows.append(row)
    validate(rows);root.mkdir();data=root/'data';data.mkdir();out=data/'scenes.jsonl';out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'protocol':'fastweight_content_v2','source_root':str(source),'source_data_sha256':SOURCE_DATA_SHA,'source_design_sha256':SOURCE_DESIGN_SHA,
        'data_sha256':file_digest(out),'dev_groups':list(DEV_GROUPS),'q_groups':list(Q_GROUPS),'v_groups':list(V_GROUPS),'formats':list(FORMATS),
        'interventions':list(INTERVENTIONS),'random_count':RANDOM_COUNT,'contexts':48,'write_schedule':[0.,.3],
        'dev_gate':{'minimum_domain_greedy':.625,'overall_choice':.75},'q_gate':{'domain_choice':.75,'overall_greedy':.5,'both_twins':6},
        'content_thresholds':{'nll_gain':.02,'specificity_gain':.05},'context_hashes':[{k:r[k] for k in IDENTITY} for r in rows]}
    (data/'design.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def load_data(root):
    root=Path(root);design=json.loads((root/'data/design.json').read_text());path=root/'data/scenes.jsonl';rows=list(map(json.loads,path.read_text().splitlines()));validate(rows)
    if file_digest(path)!=design['data_sha256'] or design['formats']!=list(FORMATS) or design['interventions']!=list(INTERVENTIONS):raise ValueError('manifest changed')
    if design['context_hashes']!=[{k:r[k] for k in IDENTITY} for r in rows]:raise ValueError('context manifest')
    return rows,design


def collect_readout(args):
    import torch
    rows,design=load_data(args.root);formats=FORMATS
    if args.stage=='q':
        selection=json.loads(Path(args.selection).read_text())
        if not selection['passed'] or selection['data_sha256']!=design['data_sha256']:raise ValueError('dev gate failed')
        formats=(selection['selected_format'],)
    rows=[r for r in rows if r['split']==args.stage and shard_for(r['group'])==args.shard];rows.sort(key=lambda r:(r['group'],r['id']));out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(314);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,meta=load_model(args.model,'step81381_plain');before=parameter_digest(model);versions=[p._version for p in model.parameters()];start=time.perf_counter()
    with out.open('x') as sink:
        for i,row in enumerate(rows):
            torch.cuda.empty_cache()
            def prefix():
                session=PlainSession(model)
                for chunk in row['chunks']:session._forward(chunk)
                return session
            session,rollout=measure(prefix);parent=native_digest(session.cache);scores={}
            for fmt in formats:
                value,profile=measure(lambda f=fmt:plain_score(session,row,f));scores[fmt]={**value,**profile}
            if parent!=native_digest(session.cache):raise RuntimeError('readout query changed parent')
            item={k:row[k] for k in IDENTITY};item.update(stage=args.stage,model=args.model,data_sha256=design['data_sha256'],weights_sha256=before,parent_unchanged=True,scores=scores,rollout=rollout)
            sink.write(json.dumps(item)+'\n');sink.flush();print(json.dumps({'stage':args.stage,'completed':i+1,'total':len(rows)}),flush=True)
    if before!=parameter_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('readout weights changed')
    meta.update(stage=args.stage,shard=args.shard,rows=len(rows),rows_sha256=file_digest(out),weights_sha256=before,backbone_unchanged=True,seconds=time.perf_counter()-start,
        torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta,indent=2)+'\n')


def read_stage(root,stage):
    root=Path(root);scenes,design=load_data(root);expected={r['id']:r for r in scenes if r['split']==stage};rows=[];audits=[]
    for shard in (0,1):
        path=root/f'{stage}_{shard}.jsonl';part=list(map(json.loads,path.read_text().splitlines()));audit=json.loads(Path(str(path)+'.audit.json').read_text())
        if audit['rows']!=len(part) or audit['rows_sha256']!=file_digest(path) or not audit['backbone_unchanged'] or audit['stage']!=stage or any(shard_for(r['group'])!=shard for r in part):raise ValueError('stage audit/shard')
        check_loading(audit['loading_info'],audit['excluded_adapters']);rows+=part;audits.append(audit)
    if len(rows)!=16 or {r['id'] for r in rows}!=set(expected) or len({r['weights_sha256'] for r in rows})!=1:raise ValueError('stage matrix/model')
    for r in rows:
        scene=expected[r['id']]
        if any(r[k]!=scene[k] for k in IDENTITY) or r['data_sha256']!=design['data_sha256'] or not r['parent_unchanged']:raise ValueError('stage provenance')
        for fmt,value in r['scores'].items():_validate_score(value,scene,fmt)
    return rows,audits,design,expected


def metric(rows,fmt,key,domain=None):
    part=[r for r in rows if domain is None or r['domain']==domain]
    return statistics.fmean(r['scores'][fmt][key] for r in part)


def summarize_dev(root):
    rows,audits,design,_=read_stage(root,'dev');tables={}
    for fmt in FORMATS:
        overall={k:metric(rows,fmt,k) for k in ('nll','correct','greedy_correct')};domains={d:{k:metric(rows,fmt,k,d) for k in ('nll','correct','greedy_correct')} for d in ('fineweb','longcrawl')}
        tables[fmt]={'overall':overall,'domains':domains,'minimum_domain_greedy':min(x['greedy_correct'] for x in domains.values())}
    selected=max(FORMATS,key=lambda f:(tables[f]['minimum_domain_greedy'],tables[f]['overall']['greedy_correct'],-tables[f]['overall']['nll'],-FORMATS.index(f)))
    passed=tables[selected]['minimum_domain_greedy']>=.625 and tables[selected]['overall']['correct']>=.75
    return {'protocol':'fastweight_content_v2','stage':'dev','data_sha256':design['data_sha256'],'formats':tables,'selected_format':selected,'passed':passed,
        'q_scored':False,'terminal_status':'dev_format_selected' if passed else 'stopped_by_v2_dev_readout'}


def summarize_q(root):
    root=Path(root);selection=json.loads((root/'dev_selection.json').read_text());rows,audits,design,_=read_stage(root,'q');fmt=selection['selected_format']
    if any(set(r['scores'])!={fmt} for r in rows):raise ValueError('Q format')
    overall={k:metric(rows,fmt,k) for k in ('nll','correct','greedy_correct')};domains={d:{k:metric(rows,fmt,k,d) for k in ('nll','correct','greedy_correct')} for d in ('fineweb','longcrawl')}
    both=sum(int(all(r['scores'][fmt]['correct'] for r in rows if r['group']==g)) for g in Q_GROUPS)
    passed=overall['correct']>=.75 and overall['greedy_correct']>=.5 and all(x['correct']>=.75 for x in domains.values()) and both>=6
    return {'protocol':'fastweight_content_v2','stage':'q','data_sha256':design['data_sha256'],'selected_format':fmt,'overall':overall,'domains':domains,
        'twin_both_correct_groups':both,'passed':passed,'v_scored':False,'terminal_status':'q_passed_v_locked' if passed else 'stopped_by_v2_Q'}


def collect_v(args):
    import torch
    rows,design=load_data(args.root);selection=json.loads(Path(args.selection).read_text());fmt=selection['selected_format']
    if not selection['passed'] or selection['data_sha256']!=design['data_sha256'] or selection['v_scored']:raise ValueError('V locked')
    selected=[r for r in rows if r['split']=='v' and shard_for(r['group'])==args.shard];selected.sort(key=lambda r:(r['group'],r['id']));by_id={r['id']:r for r in selected};out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(315);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,meta=load_model(args.model,'step81381_native');before=parameter_digest(model);versions=[p._version for p in model.parameters()];start=time.perf_counter();seen=set();count=0
    with out.open('x') as sink:
        for first in selected:
            if first['id'] in seen:continue
            pair=(first,by_id[first['twin_id']]);seen.update(r['id'] for r in pair);sessions=[];profiles=[];traces=[];hashes=[]
            for row in pair:
                torch.cuda.empty_cache();(session,trace),profile=measure(lambda r=row:rollout_one_write(model,r));move_native_cache(session.cache,'cpu');sessions.append(session);profiles.append(profile);traces.append(trace);hashes.append(native_digest(session.cache))
            cpu_weights=[_weights(s) for s in sessions]
            for index,(row,session) in enumerate(zip(pair,sessions)):
                row=copy.deepcopy(row);row['twin_label']=pair[1-index]['queries']['target'][fmt]['label'];move_native_cache(session.cache,model.device);results={}
                for name in INTERVENTIONS:
                    seed=4100+row['group']*20+row['variant']*3+(int(name[-1]) if name.startswith('random') else 0)
                    value,profile=measure(lambda n=name,s=seed:score_intervention(session,row,n,cpu_weights[1-index],151645,s,fmt));results[name]={**value,**profile,'seed':seed if name.startswith('random') else None}
                move_native_cache(session.cache,'cpu')
                if native_digest(session.cache)!=hashes[index]:raise RuntimeError('V parent changed')
                item={k:row[k] for k in IDENTITY};item.update(selected_format=fmt,twin_label=row['twin_label'],model=args.model,data_sha256=design['data_sha256'],weights_sha256=before,parent_sha256=hashes[index],parent_unchanged=True,results=results,rollout=profiles[index],native_update_trace=traces[index])
                sink.write(json.dumps(item)+'\n');sink.flush();count+=1;print(json.dumps({'stage':'v','completed':count,'total':len(selected)}),flush=True)
            del sessions,cpu_weights
    if before!=parameter_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('V weights changed')
    meta.update(stage='v',shard=args.shard,rows=count,rows_sha256=file_digest(out),weights_sha256=before,backbone_unchanged=True,seconds=time.perf_counter()-start,
        torch=torch.__version__,transformers=__import__('transformers').__version__,gpu=torch.cuda.get_device_name())
    Path(str(out)+'.audit.json').write_text(json.dumps(meta,indent=2)+'\n')


def summarize(root):
    root=Path(root);scenes,design=load_data(root);dev=json.loads((root/'dev_selection.json').read_text());q=json.loads((root/'q_summary.json').read_text());fmt=dev['selected_format']
    if not dev['passed'] or not q['passed'] or q['selected_format']!=fmt:raise ValueError('gates not passed')
    expected={r['id']:r for r in scenes if r['split']=='v'};rows=[];audits=[];hashes={}
    for shard in (0,1):
        path=root/f'v_{shard}.jsonl';part=list(map(json.loads,path.read_text().splitlines()));audit=json.loads(Path(str(path)+'.audit.json').read_text())
        if audit['rows']!=len(part) or audit['rows_sha256']!=file_digest(path) or not audit['backbone_unchanged'] or audit['stage']!='v' or any(shard_for(r['group'])!=shard for r in part):raise ValueError('V audit')
        check_loading(audit['loading_info'],audit['excluded_adapters']);rows+=part;audits.append(audit);hashes[path.name]=file_digest(path)
    if len(rows)!=16 or {r['id'] for r in rows}!=set(expected) or len({r['weights_sha256'] for r in rows})!=1:raise ValueError('V matrix')
    for r in rows:
        scene=expected[r['id']];twin=expected[scene['twin_id']]
        if any(r[k]!=scene[k] for k in IDENTITY) or r['selected_format']!=fmt or r['twin_label']!=twin['queries']['target'][fmt]['label'] or not r['parent_unchanged'] or set(r['results'])!=set(INTERVENTIONS):raise ValueError('V provenance')
        if any(v!=0 for v in r['native_update_trace'][0].values()) or not any(v>0 for v in r['native_update_trace'][1].values()):raise ValueError('write schedule')
        for name,value in r['results'].items():
            _validate_score(value,scene,fmt)
            if abs(value['own_nll']-value['choice_nll'][scene['queries']['target'][fmt]['label']])>1e-6 or abs(value['twin_nll']-value['choice_nll'][r['twin_label']])>1e-6:raise ValueError('content metrics')
            if name.startswith('random') and (set(value['random_norm_ratios'])!=set(map(str,LAYERS)) or any(abs(x-1)>1e-6 for x in value['random_norm_ratios'].values())):raise ValueError('random norms')
    tables={p:aggregate_v(rows,p) for p in ('correct','wrong','zero','random_mean')};comparisons={
        'correct_vs_wrong_nll_gain':paired(rows,'wrong','correct','own_nll',seed=1213),
        'correct_vs_random_nll_gain':paired(rows,'random_mean','correct','own_nll',seed=1213),
        'correct_vs_zero_nll_gain':paired(rows,'zero','correct','own_nll',seed=1213),
        'correct_vs_wrong_specificity_gain':paired(rows,'correct','wrong','own_vs_twin_margin',seed=1213),
        'correct_vs_zero_choice_accuracy':paired(rows,'correct','zero','correct',seed=1213)}
    binding=(comparisons['correct_vs_wrong_nll_gain']['mean']>=.02 and comparisons['correct_vs_wrong_nll_gain']['ci95'][0]>0 and comparisons['correct_vs_random_nll_gain']['mean']>=.02 and comparisons['correct_vs_random_nll_gain']['ci95'][0]>0 and comparisons['correct_vs_wrong_specificity_gain']['mean']>=.05 and comparisons['correct_vs_wrong_specificity_gain']['ci95'][0]>0)
    utility=(comparisons['correct_vs_zero_nll_gain']['mean']>=.02 and comparisons['correct_vs_zero_nll_gain']['ci95'][0]>0 and tables['correct']['correct']>=tables['zero']['correct'])
    profiles=[p for r in rows for p in [r['rollout']]+list(r['results'].values())]
    return {'protocol':'fastweight_content_v2','data_sha256':design['data_sha256'],'selected_format':fmt,'dev':dev,'q':q,'v_contexts':16,'v_queries':96,
        'policies':tables,'comparisons':comparisons,'decision':{'content_binding_supported':binding,'content_utility_supported':utility,'interpretation':'binding_and_utility' if binding and utility else 'binding_without_utility' if binding else 'no_detectable_content_binding'},
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in rows),'query_mean_seconds':statistics.fmean(v['seconds'] for r in rows for v in r['results'].values()),'shard_seconds':[a['seconds'] for a in audits]},
        'rows_sha256':hashes,'terminal_status':'completed_fastweight_content_v2','scope':'Readout selected on observed dev, confirmed on separate Q, interventions on previously unscored V.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--tokenizer',required=True);b.add_argument('--root',required=True)
    for command in ('collect-dev','collect-q'):
        c=sub.add_parser(command);c.add_argument('--root',required=True);c.add_argument('--model',required=True);c.add_argument('--shard',type=int,choices=(0,1),required=True);c.add_argument('--output',required=True);c.add_argument('--selection')
    for command in ('summarize-dev','summarize-q','summarize'):
        s=sub.add_parser(command);s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    v=sub.add_parser('collect-v');v.add_argument('--root',required=True);v.add_argument('--model',required=True);v.add_argument('--selection',required=True);v.add_argument('--shard',type=int,choices=(0,1),required=True);v.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='build':build(a.source,a.tokenizer,a.root)
    elif a.command in ('collect-dev','collect-q'):a.stage=a.command.split('-')[1];collect_readout(a)
    elif a.command=='summarize-dev':Path(a.output).write_text(json.dumps(summarize_dev(a.root),indent=2)+'\n')
    elif a.command=='summarize-q':Path(a.output).write_text(json.dumps(summarize_q(a.root),indent=2)+'\n')
    elif a.command=='collect-v':collect_v(a)
    else:Path(a.output).write_text(json.dumps(summarize(a.root),indent=2)+'\n')


if __name__=='__main__':main()
