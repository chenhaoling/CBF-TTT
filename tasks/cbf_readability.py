"""Dev-only factorial calibration of fact position, key load and query format."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import COLORS, digest, file_digest

LAYOUTS=('short','long_far','long_near')
LOADS=('single','dual')
FORMATS=('qa','cloze')


def rebuild(scene,backgrounds,tokenizer,layout,load,order='target_last'):
    length=2 if layout=='short' else 6
    fact_chunk=6 if layout=='long_near' else 2
    target=scene['queries']['target'];anchor=scene['queries']['anchor']
    entries=[(target['key'],target['label'])]
    if load=='dual': entries.insert(0,(anchor['key'],anchor['label']))
    if order=='target_first':entries.reverse()
    suffix='\n\nSession records. For a repeated device, the latest record replaces the earlier record.\n'
    suffix+=''.join(f'Record: The access color for {key} is {COLORS[label]}.\n' for key,label in entries)
    ids=tokenizer.encode(suffix,add_special_tokens=False)
    chunks=[list(x) for x in backgrounds[:length]]
    if len(ids)>=4096:raise ValueError('record too long')
    chunks[fact_chunk-1]=chunks[fact_chunk-1][:4096-len(ids)]+ids
    queries={}
    for name in (('target',) if load=='single' else ('target','anchor')):
        original=scene['queries'][name]
        cloze=f"Record: The access color for {original['key']} is"
        queries[name]={fmt:{**original,'ids':original['ids'] if fmt=='qa' else tokenizer.encode(cloze,add_special_tokens=False)} for fmt in FORMATS}
    row={k:scene[k] for k in ('id','group','domain','variant','sources','choice_ids')}
    row.update({'cell':f'{layout}/{load}/{order}','layout':layout,'load':load,'order':order,
        'chunks':chunks,'queries':queries,'fact_chunk':fact_chunk,'suffix_start':4096-len(ids),
        'context_sha256':digest(json.dumps(chunks).encode())})
    return row


def validate(rows):
    if len(rows)!=128:raise ValueError('expected 128 contexts')
    index={(r['id'],r['cell']):r for r in rows}
    if len(index)!=128 or {r['group'] for r in rows}!=set(range(8)):
        raise ValueError('duplicate or non-dev group')
    for r in rows:
        if any(len(c)!=4096 for c in r['chunks']):raise ValueError('chunk length')
        if r['context_sha256']!=digest(json.dumps(r['chunks']).encode()):raise ValueError('context hash')
        if len(r['chunks'])!=(2 if r['layout']=='short' else 6):raise ValueError('layout length')
        if r['cell']=='bridge':continue
        expected_queries={'target'} if r['load']=='single' else {'target','anchor'}
        if set(r['queries'])!=expected_queries or any(set(q)!=set(FORMATS) for q in r['queries'].values()):
            raise ValueError('query matrix incomplete')
        twin=index[(r['id'][:-1]+str(1-r['variant']),r['cell'])]
        if sum(x!=y for a,b in zip(r['chunks'],twin['chunks']) for x,y in zip(a,b))!=1:
            raise ValueError('twins must differ in one token')
        for name,qs in r['queries'].items():
            for fmt,q in qs.items():
                if q['ids']!=twin['queries'][name][fmt]['ids'] or q['answer_id']!=r['choice_ids'][q['label']]:
                    raise ValueError('query/label mismatch')
        if r['layout']=='short':
            if r['chunks']!=index[(r['id'],f"long_far/{r['load']}/target_last")]['chunks'][:2]:
                raise ValueError('short/long prefix mismatch')
    for cell in {r['cell'] for r in rows}:
        counts=Counter(r['queries']['target']['qa']['label'] for r in rows if r['cell']==cell)
        if set(counts)!=set(range(8)) or len(set(counts.values()))!=1:raise ValueError('imbalanced colors')


def backgrounds(design,tokenizer):
    import pyarrow.parquet as pq
    wanted={s for g in design['groups'][:8] for s in g['sources']}
    docs={p['sha256']:p for p in design['candidate_sources'] if p['sha256'] in wanted}
    requests={}
    for sha,d in docs.items():requests.setdefault(d['parquet'],{}).setdefault(d['row_index'],sha)
    tokens={}
    for path,wanted_rows in requests.items():
        parquet=pq.ParquetFile(path);offset=0
        for rg in range(parquet.num_row_groups):
            size=parquet.metadata.row_group(rg).num_rows
            need={i:sha for i,sha in wanted_rows.items() if offset<=i<offset+size}
            if need:
                for batch in parquet.iter_batches(batch_size=16,row_groups=[rg],columns=['text']):
                    for raw in batch.column(0).to_pylist():
                        if offset in need:
                            sha=need[offset]
                            if digest(raw.encode())!=sha:raise ValueError('original source changed')
                            n=4096 if docs[sha]['domain']=='fineweb' else 6*4096
                            encoded=tokenizer.encode(raw,add_special_tokens=False)[:n]
                            if len(encoded)!=n:raise ValueError('source too short')
                            tokens[sha]=[encoded[i:i+4096] for i in range(0,n,4096)]
                        offset+=1
            else:offset+=size
    if set(tokens)!=wanted:raise ValueError('source row missing')
    return {g:[chunk for sha in item['sources'] for chunk in tokens[sha]] for g,item in enumerate(design['groups'][:8])}


def build(args):
    from transformers import AutoTokenizer
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    source=Path(args.source);design=json.loads((source/'data/design.json').read_text())
    if file_digest(source/'data/scenes.jsonl')!=design['data_sha256']:raise ValueError('source hash')
    scenes=[json.loads(x) for x in (source/'data/scenes.jsonl').read_text().splitlines()]
    scenes=[r for r in scenes if r['split']=='dev' and r['regime']=='stable']
    if len(scenes)!=16:raise ValueError('expected 16 stable dev twins')
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    bg=backgrounds(design,tokenizer);rows=[]
    for scene in scenes:
        for layout in LAYOUTS:
            for load in LOADS:rows.append(rebuild(scene,bg[scene['group']],tokenizer,layout,load))
        rows.append(rebuild(scene,bg[scene['group']],tokenizer,'long_far','dual','target_first'))
        row={k:scene[k] for k in ('id','group','domain','variant','sources','choice_ids','chunks')}
        row.update({'cell':'bridge','layout':'bridge','load':'dual','order':'original',
            'context_sha256':digest(json.dumps(scene['chunks']).encode()),
            'queries':{name:{'qa':q} for name,q in scene['queries'].items()}})
        rows.append(row)
    validate(rows)
    out.mkdir(parents=True);data=out/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'protocol':'readability_v1','source_design_sha256':file_digest(source/'data/design.json'),
        'source_data_sha256':design['data_sha256'],'data_sha256':file_digest(data),
        'source_root':str(source),'groups':list(range(8)),'confirm_scored':False,
        'contexts':128,'queries':sum(len(fs) for r in rows for fs in r['queries'].values()),
        'cells':sorted({r['cell'] for r in rows}),'context_hashes':[{k:r[k] for k in ('id','cell','context_sha256','sources')} for r in rows]}
    (out/'design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def collect(args):
    import torch
    from tasks.cbf_ttt import _load_model
    from tasks.cbf_memory_value import build_session,score_query,state_digest,weight_digest
    from tasks.cbf_selective import measure
    data=Path(args.data);rows=[json.loads(x) for x in data.read_text().splitlines()];validate(rows)
    # Balance dataset mix and keep all paired conditions for a source together.
    rows=[r for r in rows if (r['group']//2+r['group']%2)%2==args.shard]
    rows.sort(key=lambda r:(r['cell']!='bridge',r['group'],r['id'],r['cell']))
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(210);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_load_model(args.model,'cuda','bfloat16');before=weight_digest(model)
    versions=[p._version for p in model.parameters()];started=time.perf_counter()
    with out.open('x') as sink:
        for i,r in enumerate(rows):
            torch.cuda.empty_cache()
            session,profile=measure(lambda:build_session(model,r,'none'))
            parent=state_digest(session.cache);queries={}
            for name,formats in r['queries'].items():
                queries[name]={}
                for fmt,q in formats.items():
                    score,p=measure(lambda:score_query(session,q,r['choice_ids'],'full_kv',{}))
                    queries[name][fmt]={**score,**p}
            if state_digest(session.cache)!=parent or session.cache.cbf_memory:
                raise RuntimeError('parent mutated or M not empty')
            row={k:r[k] for k in ('id','cell','group','domain','variant','layout','load','order','context_sha256')}
            row.update({'queries':queries,'rollout':profile,'parent_unchanged':True,'model':args.model,
                        'weights_sha256':before,'data_sha256':file_digest(data)})
            sink.write(json.dumps(row)+'\n');sink.flush()
            print(json.dumps({'completed':i+1,'total':len(rows),'id':r['id'],'cell':r['cell']}),flush=True)
            del session
    if before!=weight_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('weights changed')
    Path(str(out)+'.audit.json').write_text(json.dumps({'rows':len(rows),'rows_sha256':file_digest(out),'weights_sha256':before,
        'backbone_unchanged':True,'seconds':time.perf_counter()-started})+'\n')


def aggregate(rows,domain=None):
    if domain:rows=[r for r in rows if r['domain']==domain]
    result={}
    for cell in sorted({r['cell'] for r in rows}):
        selected=[r for r in rows if r['cell']==cell];result[cell]={}
        for name in selected[0]['queries']:
            result[cell][name]={}
            for fmt in selected[0]['queries'][name]:
                result[cell][name][fmt]={metric:statistics.fmean(r['queries'][name][fmt][metric] for r in selected)
                                        for metric in ('nll','correct','greedy_correct')}
    return result


def summarize(args):
    data=Path(args.data);scenes=[json.loads(x) for x in data.read_text().splitlines()];validate(scenes)
    expected={(r['id'],r['cell']):r for r in scenes};rows=[];audits=[]
    for name in args.inputs:
        p=Path(name);a=json.loads(Path(str(p)+'.audit.json').read_text());part=[json.loads(x) for x in p.read_text().splitlines()]
        if a['rows_sha256']!=file_digest(p) or a['rows']!=len(part) or not a['backbone_unchanged']:raise ValueError('invalid shard audit')
        rows+=part;audits.append(a)
    if len(rows)!=128 or {(r['id'],r['cell']) for r in rows}!=set(expected):raise ValueError('incomplete/duplicate cells')
    if len({r['weights_sha256'] for r in rows})!=1 or len({r['model'] for r in rows})!=1:raise ValueError('mixed model')
    profiles=[]
    for r in rows:
        scene=expected[r['id'],r['cell']]
        if r['data_sha256']!=file_digest(data) or not r['parent_unchanged'] or any(r[k]!=scene[k] for k in ('group','domain','variant','context_sha256','layout','load','order')):raise ValueError('identity/cache mismatch')
        if set(r['queries'])!=set(scene['queries']):raise ValueError('query missing')
        profiles.append(r['rollout'])
        for name,fs in r['queries'].items():
            if set(fs)!=set(scene['queries'][name]):raise ValueError('format missing')
            for fmt,v in fs.items():
                q=scene['queries'][name][fmt];scores=v['choice_nll']
                if len(scores)!=8 or any(not math.isfinite(x) for x in scores):raise ValueError('invalid scores')
                prediction=min(range(8),key=lambda i:scores[i])
                if abs(v['nll']-scores[q['label']])>1e-6 or v['correct']!=int(prediction==q['label']) or v['prediction']!=prediction or v['greedy_correct']!=int(v['greedy_id']==q['answer_id']):raise ValueError('score mismatch')
                profiles.append(v)
    if any(not math.isfinite(p[k]) or p[k]<=0 for p in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):raise ValueError('invalid profile')
    reference={r['id']:r for p in args.reference for r in map(json.loads,Path(p).read_text().splitlines()) if r['regime']=='stable'}
    errors=[]
    for r in rows:
        if r['cell']!='bridge':continue
        old=reference[r['id']]
        if old['weights_sha256']!=r['weights_sha256'] or old['model']!=r['model']:raise ValueError('bridge model differs')
        for name in r['queries']:
            a=r['queries'][name]['qa'];b=old['queries']['full_kv']['self'][name]
            errors.append(abs(a['nll']-b['nll']))
            if any(a[k]!=b[k] for k in ('prediction','greedy_id')):raise ValueError('bridge predictions differ')
    if len(errors)!=32 or max(errors)>1e-5:raise ValueError('bridge reproduction failed')
    means=aggregate(rows);domains={d:aggregate(rows,d) for d in ('fineweb','longcrawl')}
    eligible=[];cell='long_far/dual/target_last'
    for fmt in FORMATS:
        worst=min(table[cell][name][fmt]['correct'] for table in (means,*domains.values()) for name in ('target','anchor'))
        if worst>=.75:eligible.append((fmt,worst))
    selected=max(eligible,key=lambda x:x[1])[0] if eligible else None
    # Source-group paired effects, accuracy higher and NLL lower are favorable.
    groups={str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)}
    contrasts={}
    pairs=[('length','long_far/dual/target_last','short/dual/target_last','qa','qa'),
           ('recency','long_near/dual/target_last','long_far/dual/target_last','qa','qa'),
           ('key_load','long_far/dual/target_last','long_far/single/target_last','qa','qa'),
           ('format',cell,cell,'cloze','qa'),
           ('slot','long_far/dual/target_first',cell,'qa','qa')]
    for label,a,b,fa,fb in pairs:
        contrasts[label]={'definition':'first condition minus second; accuracy positive, NLL negative favorable',
            'first':[a,fa],'second':[b,fb],
            'group_differences':{g:{k:v[a]['target'][fa][k]-v[b]['target'][fb][k] for k in ('correct','nll','greedy_correct')} for g,v in groups.items()}}
    report={'protocol':'readability_v1','contexts':len(rows),'queries':len(profiles)-len(rows),'data_sha256':file_digest(data),
        'weights_sha256':rows[0]['weights_sha256'],'model':rows[0]['model'],'bridge_checks':len(errors),'bridge_max_error':max(errors),
        'means':means,'domain_means':domains,'group_means':groups,'contrasts':contrasts,
        'selected_format':selected,'eligible_formats':eligible,'confirm_scored':False,'backbone_unchanged':True,
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),
                   'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),'shard_seconds':[a['seconds'] for a in audits]},
        'scope':'Observed dev stable facts only; no memory writes or independent confirmation.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('group_means','contrasts','domain_means')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--tokenizer',required=True);b.add_argument('--output',required=True)
    for name in ('collect','summarize'):
        a=sub.add_parser(name);a.add_argument('--data',required=True);a.add_argument('--output',required=True)
        if name=='collect':a.add_argument('--model',required=True);a.add_argument('--shard',type=int,choices=(0,1),required=True)
        else:a.add_argument('--inputs',nargs='+',required=True);a.add_argument('--reference',nargs='+',required=True)
    args=p.parse_args();{'build':build,'collect':collect,'summarize':summarize}[args.command](args)


if __name__=='__main__':main()
