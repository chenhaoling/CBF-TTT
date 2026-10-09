"""Four-regime dev readability with distractors disjoint from both twins and anchor."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics

from tasks.build_cbf_memory_value import COLORS,REGIMES,domain,file_digest
from tasks.cbf_record_interference import HEADER,row_hash
from tasks.cbf_readability import collect as collect_formats

FORMATS=('qa','binding')
BRIDGE_GROUPS=(0,2)
BRIDGES=tuple('bridge/'+r for r in REGIMES)
IDENTITY=('id','cell','group','domain','variant','regime','is_bridge','context_sha256')
BINDING_TEMPLATE=('Read only the session records for {key}. Ignore records for other devices. '
    'If this device has multiple records, use its last record.\n'
    'Question: What is its access color? Answer with exactly one color word.\nAnswer:')


def clean_scene(scene,twin,tokenizer):
    row=copy.deepcopy(scene)
    protected={scene['queries']['target']['label'],twin['queries']['target']['label'],
               scene['queries']['anchor']['label'],scene['old_label']}
    if len(protected)!=4:raise ValueError('expected four distinct protected colors')
    palette=sorted(set(range(8))-protected)
    keys={q['key'] for q in scene['queries'].values()}
    changed=[]
    for original,record in zip(scene['records'],row['records']):
        c=record['chunk'];count=0;entries=[]
        for slot,(key,label) in enumerate(original['entries']):
            value=label if key in keys else palette[(2*(c-1)+slot+scene['group'])%4]
            entries.append([key,value]);count+=int(value!=label)
        record['entries']=entries
        suffix=HEADER+''.join(f'Record: The access color for {key} is {COLORS[value]}.\n' for key,value in entries)
        ids=tokenizer.encode(suffix,add_special_tokens=False);offset=original['offset']
        if len(ids)!=scene['chunk_size']-offset:raise ValueError('color replacement changed token length')
        new=scene['chunks'][c-1][:offset]+ids
        positions=[i for i,(a,b) in enumerate(zip(scene['chunks'][c-1],new)) if a!=b]
        if len(positions)!=count:raise ValueError('changes are not one token per changed color')
        row['chunks'][c-1]=new;changed.append(positions)
    row.update(protected_labels=sorted(protected),distractor_palette=palette,changed_positions=changed)
    return row


def make_rows(originals,tokenizer):
    index={r['id']:r for r in originals};rows=[]
    if len(originals)!=64 or {r['group'] for r in originals}!=set(range(8)):raise ValueError('expected 64 dev scenes')
    for scene in originals:
        if scene['split']!='dev':raise ValueError('heldout source')
        clean=clean_scene(scene,index[scene['twin_id']],tokenizer)
        for is_bridge,base in ((False,clean),(True,scene)):
            if is_bridge and scene['group'] not in BRIDGE_GROUPS:continue
            row=copy.deepcopy(base);row['queries']={}
            for name,q in scene['queries'].items():
                row['queries'][name]={'qa':copy.deepcopy(q)}
                if not is_bridge:
                    row['queries'][name]['binding']={**q,'ids':tokenizer.encode(BINDING_TEMPLATE.format(key=q['key']),add_special_tokens=False)}
            row.update(cell=('bridge/' if is_bridge else 'clean/')+scene['regime'],is_bridge=is_bridge,
                context_sha256=row_hash(base['chunks']))
            rows.append(row)
    validate(rows)
    return rows


def validate(rows):
    index={(r['id'],r['cell']):r for r in rows}
    if len(rows)!=80 or len(index)!=80:raise ValueError('expected complete 80-context matrix')
    clean=[r for r in rows if not r['is_bridge']];bridge=[r for r in rows if r['is_bridge']]
    expected={(g,reg,v) for g in range(8) for reg in REGIMES for v in (0,1)}
    if {(r['group'],r['regime'],r['variant']) for r in clean}!=expected or len(clean)!=64:raise ValueError('clean dev matrix')
    if {(r['group'],r['regime'],r['variant']) for r in bridge}!={(g,reg,v) for g in BRIDGE_GROUPS for reg in REGIMES for v in (0,1)} or len(bridge)!=16:raise ValueError('bridge matrix')
    for r in rows:
        if r['split']!='dev' or r['domain']!=domain(r['group']) or r['cell']!=('bridge/' if r['is_bridge'] else 'clean/')+r['regime']:
            raise ValueError('identity mismatch')
        if r['id']!=f"g{r['group']:02d}_{r['regime']}_{r['variant']}":raise ValueError('scene ID mismatch')
        if len(r['chunks'])!=6 or any(len(c)!=r['chunk_size'] for c in r['chunks']) or row_hash(r['chunks'])!=r['context_sha256']:raise ValueError('context mismatch')
        twin=index[r['twin_id'],r['cell']]
        if sum(a!=b for x,y in zip(r['chunks'],twin['chunks']) for a,b in zip(x,y))!=1:raise ValueError('twins differ outside single fact')
        if r['sources']!=twin['sources']:raise ValueError('twin source mismatch')
        formats={'qa'} if r['is_bridge'] else set(FORMATS)
        if set(r['queries'])!={'target','anchor'} or any(set(fs)!=formats for fs in r['queries'].values()):raise ValueError('query format matrix')
        for name,fs in r['queries'].items():
            base=fs['qa']
            for fmt,q in fs.items():
                if any(q[k]!=base[k] for k in ('key','label','answer_id')) or q['ids']!=twin['queries'][name][fmt]['ids'] or q['answer_id']!=r['choice_ids'][q['label']]:raise ValueError('query/label mismatch')
            history=[(rec['chunk'],label) for rec in r['records'] for key,label in rec['entries'] if key==base['key']]
            place=2 if name=='anchor' else dict(stable=2,correction=6,recent1_distractor=4,recent2_distractor=5)[r['regime']]
            if not history or history[-1]!=(place,base['label']):raise ValueError('latest-record truth mismatch')
            if name=='target' and r['regime']=='correction' and history!=[(2,r['old_label']),(6,base['label'])]:raise ValueError('correction truth')
        if r['is_bridge']:continue
        protected={r['queries']['target']['qa']['label'],twin['queries']['target']['qa']['label'],r['queries']['anchor']['qa']['label'],r['old_label']}
        if len(protected)!=4 or r['protected_labels']!=sorted(protected) or r['distractor_palette']!=sorted(set(range(8))-protected):raise ValueError('palette mismatch')
        if r['protected_labels']!=twin['protected_labels'] or r['distractor_palette']!=twin['distractor_palette']:raise ValueError('twin palette mismatch')
        keys={fs['qa']['key'] for fs in r['queries'].values()}
        for rec in r['records']:
            if len(rec['entries'])!=2:raise ValueError('record count changed')
            for slot,(key,value) in enumerate(rec['entries']):
                if key not in keys and value!=r['distractor_palette'][(2*(rec['chunk']-1)+slot+r['group'])%4]:raise ValueError('distractor color overlap or assignment mismatch')
    for regime in REGIMES:
        for dataset in ('fineweb','longcrawl'):
            counts=Counter(r['queries']['target']['qa']['label'] for r in clean if r['regime']==regime and r['domain']==dataset)
            if counts!=Counter({k:1 for k in range(8)}):raise ValueError('target label imbalance')


def build(args):
    from transformers import AutoTokenizer
    source=Path(args.source);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    d=json.loads((source/'data/design.json').read_text())
    if file_digest(source/'data/scenes.jsonl')!=d['data_sha256']:raise ValueError('source data hash')
    originals=[r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()) if r['split']=='dev']
    rows=make_rows(originals,AutoTokenizer.from_pretrained(args.tokenizer))
    out.mkdir(parents=True);data=out/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'protocol':'disjoint_readout_v1','source_root':str(source),'source_data_sha256':d['data_sha256'],
        'source_design_sha256':file_digest(source/'data/design.json'),'data_sha256':file_digest(data),
        'contexts':80,'queries':288,'groups':list(range(8)),'bridge_groups':BRIDGE_GROUPS,'formats':FORMATS,
        'binding_template':BINDING_TEMPLATE,'confirm_scored':False,
        'context_hashes':[{k:r[k] for k in ('id','cell','context_sha256','sources')} for r in rows]}
    (out/'design.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def aggregate(rows):
    result={}
    for regime in REGIMES:
        part=[r for r in rows if r['regime']==regime]
        result[regime]={name:{fmt:{**{k:statistics.fmean(r['queries'][name][fmt][k] for r in part)
            for k in ('correct','nll','greedy_correct')},'count':len(part)} for fmt in part[0]['queries'][name]}
            for name in ('target','anchor')}
    return result


def choose_format(domains):
    status={}
    for fmt in FORMATS:
        cells=[(domain,reg,name,v[reg][name][fmt]['correct']) for domain,v in domains.items() for reg in REGIMES for name in ('target','anchor')]
        status[fmt]={'worst_accuracy':min(x[3] for x in cells),'passed':all(x[3]>=.75 for x in cells),
            'failed_cells':[{'domain':d,'regime':r,'query':n,'accuracy':acc} for d,r,n,acc in cells if acc<.75]}
    eligible=[fmt for fmt in FORMATS if status[fmt]['passed']]
    selected=max(eligible,key=lambda fmt:status[fmt]['worst_accuracy']) if eligible else None
    return {'selected_format':selected,'formats':status,'passed':selected is not None,
        'rule':'Every regime/domain/query cell >= 75%; maximize worst cell; ties prefer qa.'}


def summarize(args):
    data=Path(args.data);scenes=list(map(json.loads,data.read_text().splitlines()));validate(scenes)
    expected={(r['id'],r['cell']):r for r in scenes};rows=[];audits=[]
    for shard,path in enumerate(args.inputs):
        p=Path(path);a=json.loads(Path(str(p)+'.audit.json').read_text());part=list(map(json.loads,p.read_text().splitlines()))
        if a['rows_sha256']!=file_digest(p) or a['rows']!=40 or len(part)!=40 or not a['backbone_unchanged']:raise ValueError('shard audit failed')
        if any((r['group']//2+r['group']%2)%2!=shard or r['weights_sha256']!=a['weights_sha256'] for r in part):raise ValueError('shard identity')
        rows+=part;audits.append(a)
    if len(rows)!=80 or {(r['id'],r['cell']) for r in rows}!=set(expected):raise ValueError('incomplete/duplicate results')
    if len({r['model'] for r in rows})!=1 or len({r['weights_sha256'] for r in rows})!=1:raise ValueError('mixed model')
    profiles=[]
    for r in rows:
        scene=expected[r['id'],r['cell']]
        if r['data_sha256']!=file_digest(data) or not r['parent_unchanged'] or any(r[k]!=scene[k] for k in IDENTITY):raise ValueError('row/cache identity')
        if set(r['queries'])!=set(scene['queries']):raise ValueError('queries missing')
        profiles.append(r['rollout'])
        for name,fs in r['queries'].items():
            if set(fs)!=set(scene['queries'][name]):raise ValueError('formats missing')
            for fmt,v in fs.items():
                q=scene['queries'][name][fmt];scores=v['choice_nll']
                if len(scores)!=8 or not all(math.isfinite(x) for x in scores+[v['nll']]):raise ValueError('invalid logits')
                pred=min(range(8),key=lambda i:scores[i])
                if abs(v['nll']-scores[q['label']])>1e-6 or v['prediction']!=pred or v['correct']!=int(pred==q['label']) or v['greedy_correct']!=int(v['greedy_id']==q['answer_id']):raise ValueError('score mismatch')
                profiles.append(v)
    if len(profiles)!=368 or any(not math.isfinite(p[k]) or p[k]<=0 for p in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):raise ValueError('query count/profile mismatch')
    refs={r['id']:r for p in args.reference for r in map(json.loads,Path(p).read_text().splitlines())}
    errors=[]
    for r in rows:
        if not r['is_bridge']:continue
        old=refs[r['id']]
        if any(r[k]!=old[k] for k in ('model','weights_sha256')):raise ValueError('bridge model differs')
        for name,fs in r['queries'].items():
            a=fs['qa'];b=old['queries']['full_kv']['self'][name];errors.append(abs(a['nll']-b['nll']))
            if any(a[k]!=b[k] for k in ('prediction','greedy_id')):raise ValueError('bridge prediction differs')
    if len(errors)!=32 or max(errors)>1e-5:raise ValueError('bridge failed')
    clean=[r for r in rows if not r['is_bridge']];means=aggregate(clean)
    domains={d:aggregate([r for r in clean if r['domain']==d]) for d in ('fineweb','longcrawl')}
    groups={str(g):aggregate([r for r in clean if r['group']==g]) for g in range(8)}
    contrasts={g:{reg:{name:{k:v[reg][name]['binding'][k]-v[reg][name]['qa'][k] for k in ('correct','nll','greedy_correct')}
        for name in ('target','anchor')} for reg in REGIMES} for g,v in groups.items()}
    gate=choose_format(domains)
    report={'protocol':'disjoint_readout_v1','contexts':80,'queries':288,'data_sha256':file_digest(data),
        'model':rows[0]['model'],'weights_sha256':rows[0]['weights_sha256'],'bridge_checks':32,'bridge_max_error':max(errors),
        'means':means,'domain_means':domains,'group_means':groups,'binding_minus_qa_group_differences':contrasts,
        'bridge_means':aggregate([r for r in rows if r['is_bridge']]),'selection':gate,
        'confirm_scored':False,'backbone_unchanged':True,
        'terminal_status':'dev_format_selected' if gate['passed'] else 'stopped_by_disjoint_Q',
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),
            'shard_seconds':[a['seconds'] for a in audits],'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in rows),
            'query_mean_seconds':statistics.fmean(v['seconds'] for r in rows for fs in r['queries'].values() for v in fs.values())},
        'scope':'Observed dev four-regime format calibration, distractors disjoint from four protected colors. Full KV, M=0; no independent confirmation or V/F.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('group_means','binding_minus_qa_group_differences')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--tokenizer',required=True);b.add_argument('--output',required=True)
    for name in ('collect','summarize'):
        a=sub.add_parser(name);a.add_argument('--data',required=True);a.add_argument('--output',required=True)
        if name=='collect':a.add_argument('--model',required=True);a.add_argument('--shard',type=int,choices=(0,1),required=True)
        else:a.add_argument('--inputs',nargs=2,required=True);a.add_argument('--reference',nargs=2,required=True)
    args=p.parse_args()
    if args.command=='collect':collect_formats(args,validator=validate,bridge_cells=BRIDGES,identity_fields=IDENTITY)
    else:{'build':build,'summarize':summarize}[args.command](args)


if __name__=='__main__':main()
