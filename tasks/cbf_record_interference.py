"""Fixed-slot dev experiment for distracting records; no fast-memory writes."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_cbf_memory_value import COLORS, digest, file_digest
from tasks.cbf_readability import backgrounds

HEADER = '\n\nSession records. For a repeated device, the latest record replaces the earlier record.\n'
CELLS = {
    'original10': ((1,3,4,5,6), 'natural'),
    'natural0': ((), 'natural'),
    'natural_before2': ((1,), 'natural'),
    'natural_early2': ((3,), 'natural'),
    'natural_late2': ((6,), 'natural'),
    'natural_late4': ((5,6), 'natural'),
    'natural_after8': ((3,4,5,6), 'natural'),
    'newline0': ((), 'newline'),
    'newline_late2': ((6,), 'newline'),
    'rebuilt_bridge': ((), 'bridge'),
}


def row_hash(chunks): return digest(json.dumps(chunks).encode())


def slots(scene, tokenizer):
    result = []
    for record in scene['records']:
        c = record['chunk']
        lines = [f'Record: The access color for {key} is {COLORS[value]}.\n' for key,value in record['entries']]
        suffix = HEADER + ''.join(lines)
        encoded = tokenizer.encode(suffix, add_special_tokens=False)
        offset = record['offset']
        if scene['chunks'][c-1][offset:] != encoded:
            raise ValueError('original suffix mismatch')
        prefix = HEADER
        for line, (key,label) in zip(lines, record['entries']):
            a = tokenizer.encode(prefix, add_special_tokens=False)
            prefix += line
            b = tokenizer.encode(prefix, add_special_tokens=False)
            if a != encoded[:len(a)] or b != encoded[:len(b)] or len(b)-len(a)<2:
                raise ValueError('unstable line token boundary')
            # Retain the final token, which may merge punctuation and newline.
            if c != 2:
                result.append({'chunk': c, 'start': offset+len(a), 'end': offset+len(b)-1,
                               'key': key, 'label': label})
    if len(result) != 10: raise ValueError('expected ten distractor slots')
    return result


def make_rows(originals, bg, rebuilt, tokenizer):
    newline = tokenizer.encode('\n', add_special_tokens=False)
    if len(newline) != 1: raise ValueError('newline must be one token')
    rows = []
    for scene in originals:
        ss = slots(scene, tokenizer)
        for cell, (active, filler) in CELLS.items():
            chunks = copy.deepcopy(scene['chunks'])
            if cell == 'rebuilt_bridge': chunks = copy.deepcopy(rebuilt[scene['id']]['chunks'])
            else:
                for slot in ss:
                    c,a,b = slot['chunk'],slot['start'],slot['end']
                    if c not in active:
                        chunks[c-1][a:b] = bg[scene['group']][c-1][a:b] if filler=='natural' else newline*(b-a)
            row = {k:scene[k] for k in ('id','group','domain','variant','sources','choice_ids','queries','chunk_size')}
            row.update(cell=cell,chunks=chunks,context_sha256=row_hash(chunks),slots=ss,
                       active_chunks=list(active),filler=filler,
                       active_colors=sorted({s['label'] for s in ss if s['chunk'] in active}))
            rows.append(row)
    validate(rows)
    return rows


def validate(rows):
    index = {(r['id'],r['cell']):r for r in rows}
    if len(rows)!=160 or len(index)!=160 or {r['group'] for r in rows}!=set(range(8)):
        raise ValueError('expected complete dev-only matrix')
    for r in rows:
        if r['cell'] not in CELLS or not r['id'].startswith(f"g{r['group']:02d}_stable_"):
            raise ValueError('non-stable or unknown condition')
        if len(r['chunks'])!=6 or any(len(c)!=r['chunk_size'] for c in r['chunks']):raise ValueError('chunk shape')
        if row_hash(r['chunks'])!=r['context_sha256']:raise ValueError('context hash')
        original = index[r['id'],'original10']
        twin = index[r['id'][:-1]+str(1-r['variant']),r['cell']]
        if any(r[k]!=original[k] for k in ('group','domain','sources','choice_ids','queries','slots')):
            raise ValueError('identity changed')
        if sum(x!=y for a,b in zip(r['chunks'],twin['chunks']) for x,y in zip(a,b))!=1:
            raise ValueError('twins differ outside target token')
        if set(r['queries'])!={'target','anchor'}:raise ValueError('missing query')
        for name,q in r['queries'].items():
            if q['answer_id']!=r['choice_ids'][q['label']] or q['ids']!=twin['queries'][name]['ids']:
                raise ValueError('query/label mismatch')
        active,filler = CELLS[r['cell']]
        if r['active_chunks']!=list(active) or r['filler']!=filler:raise ValueError('condition mismatch')
        if r['active_colors']!=sorted({s['label'] for s in r['slots'] if s['chunk'] in active}):raise ValueError('color audit')
        if r['cell']=='rebuilt_bridge':continue
        allowed = {(s['chunk']-1,i) for s in r['slots'] if s['chunk'] not in active for i in range(s['start'],s['end'])}
        for c,(a,b) in enumerate(zip(r['chunks'],original['chunks'])):
            if any(x!=y and (c,i) not in allowed for i,(x,y) in enumerate(zip(a,b))):
                raise ValueError('token changed outside intervention slots')
    for cell in CELLS:
        for domain in ('fineweb','longcrawl'):
            counts = Counter(r['queries']['target']['label'] for r in rows if r['cell']==cell and r['domain']==domain)
            if counts!=Counter({k:1 for k in range(8)}):raise ValueError('domain label imbalance')


def build(args):
    from transformers import AutoTokenizer
    out = Path(args.output)
    if out.exists():raise FileExistsError(out)
    source,cal = Path(args.source),Path(args.readability)
    design = json.loads((source/'data/design.json').read_text())
    rd = json.loads((cal/'data/design.json').read_text())
    if file_digest(source/'data/scenes.jsonl')!=design['data_sha256']:raise ValueError('source hash')
    if file_digest(cal/'data/scenes.jsonl')!=rd['data_sha256']:raise ValueError('readability hash')
    if rd['source_data_sha256']!=design['data_sha256']:raise ValueError('source mismatch')
    originals = [r for r in map(json.loads,(source/'data/scenes.jsonl').read_text().splitlines()) if r['split']=='dev' and r['regime']=='stable']
    rebuilt = {r['id']:r for r in map(json.loads,(cal/'data/scenes.jsonl').read_text().splitlines()) if r['cell']=='long_far/dual/target_last'}
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bg = backgrounds(design,tok)
    rows = make_rows(originals,bg,rebuilt,tok)
    for row in rows:
        if row['cell']=='rebuilt_bridge':
            if row['queries']!={k:v['qa'] for k,v in rebuilt[row['id']]['queries'].items()}:raise ValueError('bridge queries differ')
    out.mkdir(parents=True)
    data = out/'scenes.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest = {'protocol':'record_interference_v1','source_root':str(source),'readability_root':str(cal),
        'source_data_sha256':design['data_sha256'],'source_design_sha256':file_digest(source/'data/design.json'),
        'readability_data_sha256':rd['data_sha256'],'readability_design_sha256':file_digest(cal/'data/design.json'),
        'data_sha256':file_digest(data),'contexts':160,'queries':320,'groups':list(range(8)),
        'confirm_scored':False,'cells':CELLS,
        'context_hashes':[{k:r[k] for k in ('id','cell','context_sha256','sources')} for r in rows]}
    (out/'design.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='context_hashes'}))


def collect(args):
    import torch
    from tasks.cbf_ttt import _load_model
    from tasks.cbf_memory_value import build_session,score_query,state_digest,weight_digest
    from tasks.cbf_selective import measure
    data = Path(args.data);rows = [json.loads(x) for x in data.read_text().splitlines()];validate(rows)
    rows = [r for r in rows if (r['group']//2+r['group']%2)%2==args.shard]
    rows.sort(key=lambda r:(r['cell'] not in ('original10','rebuilt_bridge'),r['group'],r['id'],r['cell']))
    out = Path(args.output)
    if out.exists():raise FileExistsError(out)
    torch.manual_seed(211);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model = _load_model(args.model,'cuda','bfloat16')
    if model.config.ttt_chunk!=4096:raise ValueError('wrong checkpoint chunk size')
    before = weight_digest(model);versions = [p._version for p in model.parameters()];started = time.perf_counter()
    with out.open('x') as sink:
        for i,r in enumerate(rows):
            torch.cuda.empty_cache()
            session,profile = measure(lambda:build_session(model,r,'none'))
            parent = state_digest(session.cache);queries = {}
            for name,q in r['queries'].items():
                score,p = measure(lambda:score_query(session,q,r['choice_ids'],'full_kv',{}))
                queries[name] = {**score,**p}
            if state_digest(session.cache)!=parent or session.cache.cbf_memory:raise RuntimeError('parent changed or memory not empty')
            row = {k:r[k] for k in ('id','cell','group','domain','variant','context_sha256','active_colors')}
            row.update(queries=queries,rollout=profile,parent_unchanged=True,model=args.model,weights_sha256=before,data_sha256=file_digest(data))
            sink.write(json.dumps(row)+'\n');sink.flush()
            print(json.dumps({'completed':i+1,'total':len(rows),'id':r['id'],'cell':r['cell']}),flush=True)
            del session
    if before!=weight_digest(model) or versions!=[p._version for p in model.parameters()]:raise RuntimeError('weights changed')
    Path(str(out)+'.audit.json').write_text(json.dumps({'rows':len(rows),'rows_sha256':file_digest(out),
        'weights_sha256':before,'backbone_unchanged':True,'seconds':time.perf_counter()-started})+'\n')


def aggregate(rows):
    result = {}
    for cell in CELLS:
        part = [r for r in rows if r['cell']==cell]
        result[cell] = {}
        for name in ('target','anchor'):
            qs = [r['queries'][name] for r in part]
            result[cell][name] = {k:statistics.fmean(q[k] for q in qs) for k in ('correct','nll','greedy_correct')}
            result[cell][name]['count'] = len(qs)
    return result


def summarize(args):
    data = Path(args.data);scenes = [json.loads(x) for x in data.read_text().splitlines()];validate(scenes)
    expected = {(r['id'],r['cell']):r for r in scenes};rows = [];audits = []
    for shard,path in enumerate(args.inputs):
        p = Path(path);a = json.loads(Path(str(p)+'.audit.json').read_text());part = list(map(json.loads,p.read_text().splitlines()))
        if a['rows_sha256']!=file_digest(p) or a['rows']!=80 or len(part)!=80 or not a['backbone_unchanged']:raise ValueError('shard audit failed')
        if any((r['group']//2+r['group']%2)%2!=shard or r['weights_sha256']!=a['weights_sha256'] for r in part):raise ValueError('shard identity')
        rows+=part;audits.append(a)
    if len(rows)!=160 or {(r['id'],r['cell']) for r in rows}!=set(expected):raise ValueError('incomplete/duplicate results')
    if len({r['model'] for r in rows})!=1 or len({r['weights_sha256'] for r in rows})!=1:raise ValueError('mixed model')
    profiles = [];confusion = {c:{n:[[0]*8 for _ in range(8)] for n in ('target','anchor')} for c in CELLS}
    distractor_errors = {c:{n:{'errors':0,'prediction_is_active_color':0} for n in ('target','anchor')} for c in CELLS}
    for r in rows:
        scene = expected[r['id'],r['cell']]
        if r['data_sha256']!=file_digest(data) or not r['parent_unchanged'] or any(r[k]!=scene[k] for k in ('group','domain','variant','context_sha256','active_colors')):raise ValueError('row mismatch')
        if set(r['queries'])!={'target','anchor'}:raise ValueError('query missing')
        profiles.append(r['rollout'])
        for name,v in r['queries'].items():
            q = scene['queries'][name];scores = v['choice_nll']
            if len(scores)!=8 or not all(math.isfinite(x) for x in scores):raise ValueError('invalid logits')
            pred = min(range(8),key=lambda j:scores[j])
            if abs(v['nll']-scores[q['label']])>1e-6 or v['prediction']!=pred or v['correct']!=int(pred==q['label']) or v['greedy_correct']!=int(v['greedy_id']==q['answer_id']):raise ValueError('score mismatch')
            confusion[r['cell']][name][q['label']][pred]+=1
            if not v['correct']:
                counts = distractor_errors[r['cell']][name];counts['errors']+=1
                counts['prediction_is_active_color']+=int(pred in scene['active_colors'])
            profiles.append(v)
    if any(not math.isfinite(p[k]) or p[k]<=0 for p in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):raise ValueError('invalid profile')
    refs = {}
    for cell,paths in [('original10',args.reference),('rebuilt_bridge',args.readability_reference)]:
        old = [r for p in paths for r in map(json.loads,Path(p).read_text().splitlines())]
        refs[cell] = {r['id']:r for r in old if (r.get('regime')=='stable' if cell=='original10' else r['cell']=='long_far/dual/target_last')}
    errors = []
    for r in rows:
        if r['cell'] not in refs:continue
        old = refs[r['cell']][r['id']]
        if any(r[k]!=old[k] for k in ('model','weights_sha256')):raise ValueError('bridge model mismatch')
        for name,a in r['queries'].items():
            b = old['queries']['full_kv']['self'][name] if r['cell']=='original10' else old['queries'][name]['qa']
            errors.append(abs(a['nll']-b['nll']))
            if any(a[k]!=b[k] for k in ('prediction','greedy_id')):raise ValueError('bridge predictions differ')
    if len(errors)!=64 or max(errors)>1e-5:raise ValueError('bridge scores differ')
    means = aggregate(rows);domains = {d:aggregate([r for r in rows if r['domain']==d]) for d in ('fineweb','longcrawl')}
    groups = {str(g):aggregate([r for r in rows if r['group']==g]) for g in range(8)}
    pairs = [('removal','natural0','original10'),('position_pre','natural_late2','natural_before2'),
        ('position_post','natural_late2','natural_early2'),('load2','natural_late2','natural0'),
        ('load4','natural_late4','natural_late2'),('load8','natural_after8','natural_late4'),
        ('load10','original10','natural_after8'),('filler0','newline0','natural0')]
    contrasts = {label:{'first':a,'second':b,'group_differences':{g:{name:{k:v[a][name][k]-v[b][name][k]
        for k in ('correct','nll','greedy_correct')} for name in ('target','anchor')} for g,v in groups.items()}} for label,a,b in pairs}
    contrasts['filler_interaction'] = {'definition':'(newline_late2-newline0)-(natural_late2-natural0)',
        'group_differences':{g:{name:{k:(v['newline_late2'][name][k]-v['newline0'][name][k])-(v['natural_late2'][name][k]-v['natural0'][name][k])
        for k in ('correct','nll','greedy_correct')} for name in ('target','anchor')} for g,v in groups.items()}}
    passed = min(table['natural0'][name]['correct'] for table in (means,*domains.values()) for name in ('target','anchor'))>=.75
    report = {'protocol':'record_interference_v1','contexts':160,'queries':320,'data_sha256':file_digest(data),
        'weights_sha256':rows[0]['weights_sha256'],'model':rows[0]['model'],'bridge_checks':64,'bridge_max_error':max(errors),
        'means':means,'domain_means':domains,'group_means':groups,'contrasts':contrasts,
        'confusion_label_by_prediction':confusion,'distractor_color_errors':distractor_errors,
        'natural0_readability_passed':passed,'confirm_scored':False,'backbone_unchanged':True,
        'profile':{'peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),
            'peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),'shard_seconds':[a['seconds'] for a in audits],
            'rollout_mean_seconds':statistics.fmean(r['rollout']['seconds'] for r in rows),
            'query_mean_seconds':statistics.fmean(v['seconds'] for r in rows for v in r['queries'].values())},
        'scope':'Observed dev stable only. Fixed record slots, full KV, M=0; no V/F or independent confirmation.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('group_means','contrasts','confusion_label_by_prediction')}))


def main():
    p = argparse.ArgumentParser(description=__doc__);sub = p.add_subparsers(dest='command',required=True)
    b = sub.add_parser('build');b.add_argument('--source',required=True);b.add_argument('--readability',required=True)
    b.add_argument('--tokenizer',required=True);b.add_argument('--output',required=True)
    for name in ('collect','summarize'):
        a = sub.add_parser(name);a.add_argument('--data',required=True);a.add_argument('--output',required=True)
        if name=='collect':a.add_argument('--model',required=True);a.add_argument('--shard',type=int,choices=(0,1),required=True)
        else:
            a.add_argument('--inputs',nargs=2,required=True);a.add_argument('--reference',nargs=2,required=True)
            a.add_argument('--readability-reference',nargs=2,required=True)
    args = p.parse_args();{'build':build,'collect':collect,'summarize':summarize}[args.command](args)


if __name__=='__main__':main()
