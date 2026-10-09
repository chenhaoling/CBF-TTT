"""Identity-matched two-record position interventions on observed development sources."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_record_interference import row_hash, collect as collect_readonly, validate as validate_reference

DONORS = (3,6)
POSITIONS = (1,3,6)
CELLS = ('none',) + tuple(f'd{d}_p{p}' for d in DONORS for p in POSITIONS)
BRIDGES = {'none':'natural0','d3_p3':'natural_early2','d6_p6':'natural_late2'}


def chunk_slots(row, chunk):
    ss = [s for s in row['slots'] if s['chunk']==chunk]
    if len(ss)!=2:raise ValueError('expected two record slots')
    return ss


def make_rows(previous, originals):
    old = {(r['id'],r['cell']):r for r in previous}
    rows = []
    for scene in originals:
        base = old[scene['id'],'natural0']
        if scene['split']!='dev' or scene['regime']!='stable':raise ValueError('dev stable only')
        if any(base[k]!=scene[k] for k in ('queries','sources','choice_ids','group','domain','variant')):raise ValueError('base provenance mismatch')
        for cell in CELLS:
            donor,pos = (0,0) if cell=='none' else tuple(int(x[1:]) for x in cell.split('_'))
            row = {k:copy.deepcopy(base[k]) for k in ('id','group','domain','variant','sources','choice_ids','queries','chunk_size','slots','chunks')}
            entries = []
            if donor:
                for src,dst in zip(chunk_slots(base,donor),chunk_slots(base,pos)):
                    a,b = src['start'],src['end'];x,y = dst['start'],dst['end']
                    if b-a!=y-x or scene['chunks'][donor-1][b]!=base['chunks'][pos-1][y]:
                        raise ValueError('donor/position token boundaries differ')
                    body = scene['chunks'][donor-1][a:b]
                    row['chunks'][pos-1][x:y] = body
                    entries.append({'key':src['key'],'label':src['label'],'body_sha256':row_hash(body)})
            row.update(cell=cell,donor_chunk=donor,position=pos,donor_entries=entries,
                       active_colors=sorted({e['label'] for e in entries}),context_sha256=row_hash(row['chunks']))
            if cell in BRIDGES and row['chunks']!=old[row['id'],BRIDGES[cell]]['chunks']:
                raise ValueError('bridge input differs')
            rows.append(row)
    validate(rows)
    return rows


def validate(rows):
    index = {(r['id'],r['cell']):r for r in rows}
    if len(rows)!=112 or len(index)!=112 or {r['group'] for r in rows}!=set(range(8)):
        raise ValueError('expected complete dev-only matrix')
    for r in rows:
        if r['cell'] not in CELLS or not r['id'].startswith(f"g{r['group']:02d}_stable_"):
            raise ValueError('condition/identity mismatch')
        if len(r['chunks'])!=6 or any(len(c)!=r['chunk_size'] for c in r['chunks']):raise ValueError('chunk shape')
        if row_hash(r['chunks'])!=r['context_sha256']:raise ValueError('context hash')
        base = index[r['id'],'none'];twin = index[r['id'][:-1]+str(1-r['variant']),r['cell']]
        if any(r[k]!=base[k] for k in ('sources','queries','choice_ids','slots','group','domain','variant')):raise ValueError('base identity changed')
        if sum(a!=b for x,y in zip(r['chunks'],twin['chunks']) for a,b in zip(x,y))!=1:raise ValueError('twins must differ in one target token')
        if set(r['queries'])!={'target','anchor'}:raise ValueError('query missing')
        for name,q in r['queries'].items():
            if q['ids']!=twin['queries'][name]['ids'] or q['answer_id']!=r['choice_ids'][q['label']]:raise ValueError('query/label mismatch')
        donor,pos = (0,0) if r['cell']=='none' else tuple(int(x[1:]) for x in r['cell'].split('_'))
        if (r['donor_chunk'],r['position'])!=(donor,pos):raise ValueError('position metadata')
        if r['active_colors']!=sorted({e['label'] for e in r['donor_entries']}):raise ValueError('active colors')
        allowed = set()
        if donor:
            home = index[r['id'],f'd{donor}_p{donor}']
            if len(r['donor_entries'])!=2 or r['donor_entries']!=home['donor_entries']:raise ValueError('donor identity changed')
            for src,dst,e in zip(chunk_slots(home,donor),chunk_slots(r,pos),r['donor_entries']):
                a,b = src['start'],src['end'];x,y = dst['start'],dst['end']
                body = r['chunks'][pos-1][x:y]
                if body!=home['chunks'][donor-1][a:b] or row_hash(body)!=e['body_sha256']:
                    raise ValueError('donor tokens changed with position')
                if (e['key'],e['label'])!=(src['key'],src['label']):raise ValueError('donor labels changed')
                allowed.update((pos-1,i) for i in range(x,y))
        elif r['donor_entries']:raise ValueError('baseline has donor')
        for c,(a,b) in enumerate(zip(r['chunks'],base['chunks'])):
            if any(x!=y and (c,i) not in allowed for i,(x,y) in enumerate(zip(a,b))):raise ValueError('changed outside destination slots')
    for cell in CELLS:
        for domain in ('fineweb','longcrawl'):
            counts = Counter(r['queries']['target']['label'] for r in rows if r['cell']==cell and r['domain']==domain)
            if counts!=Counter({k:1 for k in range(8)}):raise ValueError('label imbalance')


def build(args):
    out = Path(args.output)
    if out.exists():raise FileExistsError(out)
    reference = Path(args.reference);d = json.loads((reference/'data/design.json').read_text())
    if file_digest(reference/'data/scenes.jsonl')!=d['data_sha256']:raise ValueError('reference data hash')
    source = Path(d['source_root'])
    if file_digest(source/'data/scenes.jsonl')!=d['source_data_sha256']:raise ValueError('source data hash')
    if file_digest(source/'data/design.json')!=d['source_design_sha256']:raise ValueError('source design hash')
    old = list(map(json.loads,(reference/'data/scenes.jsonl').read_text().splitlines()));validate_reference(old)
    originals = [r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()) if r['split']=='dev' and r['regime']=='stable']
    rows = make_rows(old,originals)
    out.mkdir(parents=True);data = out/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = {'protocol':'matched_position_v1','reference_root':str(reference),'source_root':str(source),
        'reference_data_sha256':d['data_sha256'],'reference_design_sha256':file_digest(reference/'data/design.json'),
        'source_data_sha256':d['source_data_sha256'],'source_design_sha256':d['source_design_sha256'],
        'data_sha256':file_digest(data),'groups':list(range(8)),'contexts':112,'queries':224,'cells':CELLS,'bridges':BRIDGES,
        'donor_chunks':DONORS,'positions':POSITIONS,'filler':'natural','confirm_scored':False,
        'context_hashes':[{k:r[k] for k in ('id','cell','context_sha256','sources','donor_entries')} for r in rows]}
    (out/'design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def aggregate(rows):
    return {c:{name:{**{k:statistics.fmean(r['queries'][name][k] for r in rows if r['cell']==c)
        for k in ('correct','nll','greedy_correct')},'count':sum(r['cell']==c for r in rows)}
        for name in ('target','anchor')} for c in CELLS}


def compare(table,a,b):
    return {name:{k:table[a][name][k]-table[b][name][k] for k in ('correct','nll','greedy_correct')}
            for name in ('target','anchor')}


def summarize(args):
    data = Path(args.data);scenes = list(map(json.loads,data.read_text().splitlines()));validate(scenes)
    expected = {(r['id'],r['cell']):r for r in scenes};rows = [];audits = []
    for shard,path in enumerate(args.inputs):
        p = Path(path);a = json.loads(Path(str(p)+'.audit.json').read_text());part = list(map(json.loads,p.read_text().splitlines()))
        if a['rows_sha256']!=file_digest(p) or a['rows']!=56 or len(part)!=56 or not a['backbone_unchanged']:raise ValueError('shard audit failed')
        if any((r['group']//2+r['group']%2)%2!=shard or r['weights_sha256']!=a['weights_sha256'] for r in part):raise ValueError('shard identity')
        rows+=part;audits.append(a)
    if len(rows)!=112 or {(r['id'],r['cell']) for r in rows}!=set(expected):raise ValueError('incomplete/duplicate results')
    if len({r['weights_sha256'] for r in rows})!=1 or len({r['model'] for r in rows})!=1:raise ValueError('mixed model')
    profiles = []
    for r in rows:
        scene = expected[r['id'],r['cell']]
        if r['data_sha256']!=file_digest(data) or not r['parent_unchanged'] or any(r[k]!=scene[k] for k in ('group','domain','variant','active_colors','context_sha256')):raise ValueError('identity/cache mismatch')
        if set(r['queries'])!={'target','anchor'}:raise ValueError('query missing')
        profiles.append(r['rollout'])
        for name,v in r['queries'].items():
            q = scene['queries'][name];scores = v['choice_nll']
            if len(scores)!=8 or not all(math.isfinite(x) for x in scores) or not math.isfinite(v['nll']):raise ValueError('invalid score')
            pred = min(range(8),key=lambda j:scores[j])
            if abs(v['nll']-scores[q['label']])>1e-6 or v['prediction']!=pred or v['correct']!=int(pred==q['label']) or v['greedy_correct']!=int(v['greedy_id']==q['answer_id']):raise ValueError('score mismatch')
            profiles.append(v)
    if any(not math.isfinite(p[k]) or p[k]<=0 for p in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):raise ValueError('profile mismatch')
    previous = {(r['id'],r['cell']):r for p in args.reference for r in map(json.loads,Path(p).read_text().splitlines())}
    errors = []
    for r in rows:
        if r['cell'] not in BRIDGES:continue
        old = previous[r['id'],BRIDGES[r['cell']]]
        if any(r[k]!=old[k] for k in ('model','weights_sha256','context_sha256')):raise ValueError('bridge identity differs')
        for name,q in r['queries'].items():
            ref = old['queries'][name];errors.append(abs(q['nll']-ref['nll']))
            if any(q[k]!=ref[k] for k in ('prediction','greedy_id')):raise ValueError('bridge prediction differs')
    if len(errors)!=96 or max(errors)>1e-5:raise ValueError('bridge reproduction failed')
    means = aggregate(rows);domains = {d:aggregate([r for r in rows if r['domain']==d]) for d in ('fineweb','longcrawl')}
    groups = {str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)}
    pairs = [(f'd{d}_late_minus_{p}',f'd{d}_p6',f'd{d}_p{p}') for d in DONORS for p in (1,3)]
    pairs += [(f'identity_at_{p}',f'd6_p{p}',f'd3_p{p}') for p in POSITIONS]
    pairs += [(f'{c}_minus_none',c,'none') for c in CELLS if c!='none']
    contrasts = {key:{'first':a,'second':b,'overall':compare(means,a,b),
        'domains':{d:compare(v,a,b) for d,v in domains.items()},
        'group_differences':{g:compare(v,a,b) for g,v in groups.items()}} for key,a,b in pairs}
    main = {}
    for p in (1,3):
        values = [contrasts[f'd{d}_late_minus_{p}']['group_differences'] for d in DONORS]
        main[f'late_minus_{p}'] = {g:{n:{k:statistics.fmean(v[g][n][k] for v in values) for k in ('correct','nll','greedy_correct')}
            for n in ('target','anchor')} for g in groups}
    interaction = {str(p):{g:{n:{k:contrasts[f'd6_late_minus_{p}']['group_differences'][g][n][k]-contrasts[f'd3_late_minus_{p}']['group_differences'][g][n][k]
        for k in ('correct','nll','greedy_correct')} for n in ('target','anchor')} for g in groups} for p in (1,3)}
    consistent = all(contrasts[f'd{d}_late_minus_{p}']['domains'][domain]['target']['correct']<0
        for d in DONORS for p in (1,3) for domain in domains)
    report = {'protocol':'matched_position_v1','contexts':112,'queries':224,'data_sha256':file_digest(data),
        'weights_sha256':rows[0]['weights_sha256'],'model':rows[0]['model'],'bridge_checks':96,'bridge_max_error':max(errors),
        'means':means,'domain_means':domains,'group_means':groups,'contrasts':contrasts,
        'main_donor_averaged_group_differences':main,'donor_position_interactions':interaction,
        'consistent_late_target_interference':consistent,'confirm_scored':False,'backbone_unchanged':True,
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),
            'shard_seconds':[a['seconds'] for a in audits],'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in rows),
            'query_mean_seconds':statistics.fmean(v['seconds'] for r in rows for v in r['queries'].values())},
        'scope':'Observed dev stable, natural filler, fixed identities and two records. Full KV and M=0; no V/F or confirmation.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('group_means','contrasts','main_donor_averaged_group_differences','donor_position_interactions')}))


def main():
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='command',required=True)
    b = sub.add_parser('build');b.add_argument('--reference',required=True);b.add_argument('--output',required=True)
    for name in ('collect','summarize'):
        a = sub.add_parser(name);a.add_argument('--data',required=True);a.add_argument('--output',required=True)
        if name=='collect':a.add_argument('--model',required=True);a.add_argument('--shard',type=int,choices=(0,1),required=True)
        else:a.add_argument('--inputs',nargs=2,required=True);a.add_argument('--reference',nargs=2,required=True)
    args = p.parse_args()
    if args.command=='collect':collect_readonly(args,validator=validate,bridge_cells=BRIDGES)
    else:{'build':build,'summarize':summarize}[args.command](args)


if __name__=='__main__':main()
