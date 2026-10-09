"""Build source-isolated paired six-chunk facts for memory-value validation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

COLORS = ('red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple')
REGIMES = ('stable', 'correction', 'recent1_distractor', 'recent2_distractor')
PROTOCOL = 'memory_value_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return ' '.join(text.split())


def domain(group):
    return 'fineweb' if group % 8 in (0, 1, 6, 7) else 'longcrawl'


def make_scenes(groups, tokenizer, chunk_size=4096):
    if len(groups) != 24:
        raise ValueError('expected 24 source groups')
    choices = [tokenizer.encode(' '+x, add_special_tokens=False) for x in COLORS]
    if any(len(x) != 1 for x in choices) or len({x[0] for x in choices}) != 8:
        raise ValueError('colors must be distinct single tokens')
    rng = random.Random(209)
    permutation = list(range(8)); rng.shuffle(permutation)
    rows = []
    for g, source in enumerate(groups):
        if source['domain'] != domain(g) or len(source['chunks']) != 6:
            raise ValueError('source layout mismatch')
        keys = random.Random(208+g).sample(range(100000, 999999), 14)
        anchor, target = f'device-{keys[0]}', f'device-{keys[1]}'
        old = permutation[(g+2) % 8]
        anchor_label = permutation[(g+1) % 8]
        for regime in REGIMES:
            for variant in (0, 1):
                label = permutation[(g+4*variant) % 8]
                chunks, records = [], []
                for c, background in enumerate(source['chunks'], 1):
                    entries = [(f'device-{keys[2*c]}', permutation[(g+c+3) % 8]),
                               (f'device-{keys[2*c+1]}', permutation[(g+c+5) % 8])]
                    if c == 2:
                        entries[0] = (anchor, anchor_label)
                    target_chunk = {'stable': 2, 'correction': 6,
                                    'recent1_distractor': 4, 'recent2_distractor': 5}[regime]
                    if c == target_chunk:
                        entries[1] = (target, label)
                    elif regime == 'correction' and c == 2:
                        entries[1] = (target, old)
                    suffix = '\n\nSession records. For a repeated device, the latest record replaces the earlier record.\n'
                    suffix += ''.join(f'Record: The access color for {key} is {COLORS[value]}.\n' for key, value in entries)
                    ids = tokenizer.encode(suffix, add_special_tokens=False)
                    if len(ids) >= chunk_size or len(background) < chunk_size:
                        raise ValueError('insufficient background or oversized record')
                    chunks.append(background[:chunk_size-len(ids)]+ids)
                    records.append({'chunk': c, 'offset': chunk_size-len(ids), 'entries': entries})
                queries = {}
                for name, key, value in (('target', target, label), ('anchor', anchor, anchor_label)):
                    text = f'Question: What is the latest access color recorded for {key} in this session?\nAnswer:'
                    queries[name] = {'ids': tokenizer.encode(text, add_special_tokens=False),
                                     'label': value, 'answer_id': choices[value][0], 'key': key}
                rows.append({'protocol': PROTOCOL, 'id': f'g{g:02d}_{regime}_{variant}',
                    'twin_id': f'g{g:02d}_{regime}_{1-variant}', 'group': g,
                    'split': 'dev' if g < 8 else 'confirm', 'domain': source['domain'],
                    'regime': regime, 'variant': variant, 'sources': source['sources'],
                    'chunks': chunks, 'chunk_size': chunk_size, 'records': records,
                    'queries': queries, 'old_label': old, 'choice_ids': [x[0] for x in choices]})
    validate(rows)
    return rows


def validate(rows):
    indexed = {r['id']: r for r in rows}
    if len(rows) != 192 or len(indexed) != 192:
        raise ValueError('expected 192 unique scenes')
    owners = {}
    for r in rows:
        if r['protocol'] != PROTOCOL or r['split'] != ('dev' if r['group'] < 8 else 'confirm'):
            raise ValueError('protocol/split mismatch')
        if r['domain'] != domain(r['group']):
            raise ValueError('domain mismatch')
        for source in r['sources']:
            if source in owners and owners[source] != r['group']:
                raise ValueError('cross-group source reuse')
            owners[source] = r['group']
        if len(r['chunks']) != 6 or any(len(c) != r['chunk_size'] for c in r['chunks']):
            raise ValueError('invalid chunk shape')
        twin = indexed[r['twin_id']]
        if twin['twin_id'] != r['id'] or any(twin[k] != r[k] for k in ('group','split','domain','regime','sources','choice_ids')):
            raise ValueError('invalid twin identity')
        if r['queries']['target']['label'] == twin['queries']['target']['label']:
            raise ValueError('twins share target label')
        if sum(x != y for a, b in zip(r['chunks'], twin['chunks']) for x, y in zip(a, b)) != 1:
            raise ValueError('twins must differ in exactly one token')
        for name, q in r['queries'].items():
            if q['ids'] != twin['queries'][name]['ids'] or q['answer_id'] != r['choice_ids'][q['label']]:
                raise ValueError('query or answer mismatch')
            history = [(record['chunk'], label) for record in r['records'] for key,label in record['entries'] if key == q['key']]
            if not history or history[-1][1] != q['label']:
                raise ValueError('ground truth differs from latest record')
            expected = 2 if name == 'anchor' else dict(stable=2, correction=6, recent1_distractor=4, recent2_distractor=5)[r['regime']]
            if history[-1][0] != expected:
                raise ValueError('incorrect fact placement')
            if name == 'target' and r['regime'] == 'correction':
                if history != [(2,r['old_label']),(6,q['label'])] or r['old_label'] == q['label']:
                    raise ValueError('invalid correction')
    for split, size in (('dev',8),('confirm',16)):
        subset = [r for r in rows if r['split'] == split]
        if len({r['group'] for r in subset}) != size:
            raise ValueError('incorrect group count')
        for regime in REGIMES:
            for dataset in ('fineweb','longcrawl'):
                counts = Counter(r['queries']['target']['label'] for r in subset if r['regime']==regime and r['domain']==dataset)
                if set(counts) != set(range(8)) or len(set(counts.values())) != 1:
                    raise ValueError('unbalanced labels within domain')


def excluded_values(paths):
    """Conservatively exclude hashes and source identifiers anywhere in old manifests."""
    excluded = set()
    def visit(x):
        if isinstance(x, dict):
            for key,value in x.items():
                if key in ('sha256','source_id','id','url','normalized_sha256') and isinstance(value,str):
                    excluded.add(value)
                visit(value)
        elif isinstance(x,list):
            for value in x: visit(value)
    for path in paths:
        visit(json.loads(Path(path).read_text()))
    return excluded


def publisher_rows(parquet, columns):
    offsets=[]; offset=0
    for rg in range(parquet.num_row_groups):
        offsets.append(offset); offset+=parquet.metadata.row_group(rg).num_rows
    for rg in reversed(range(parquet.num_row_groups)):
        index=offsets[rg]
        for batch in parquet.iter_batches(batch_size=16,row_groups=[rg],columns=columns):
            for row in batch.to_pylist():
                yield index,row
                index+=1


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    manifests = sorted(Path(args.old_root).glob('cbf_ttt*/**/*.meta.json'))
    manifests = [p for p in manifests if out not in p.parents]
    excluded = excluded_values(manifests)
    pool, seen = [], set()
    for dataset,path,quota,needed in (('fineweb',args.fineweb,96,4096),('longcrawl',args.longcrawl,32,6*4096)):
        parquet = pq.ParquetFile(path)
        columns = ['text']+[k for k in ('id','url') if k in parquet.schema.names]
        if 'text' not in parquet.schema.names: raise ValueError('publisher text column required')
        count = 0
        for index,row in publisher_rows(parquet,columns):
            text = row['text']
            if not isinstance(text,str): continue
            sha = digest(text.encode()); norm_sha = digest(normalize(text).encode())
            if any(x in excluded or x in seen for x in (sha,norm_sha,row.get('id'),row.get('url')) if x): continue
            ids = tok.encode(text,add_special_tokens=False)
            if len(ids) < needed: continue
            chunks = [ids[i:i+4096] for i in range(0,needed,4096)]
            anchors = [normalize(tok.decode(c[start:start+128]))[:192] for c in chunks for start in (256,2048)]
            if any(len(a)<96 for a in anchors): continue
            pool.append({'domain':dataset,'sha256':sha,'normalized_sha256':norm_sha,
                'source_id':row.get('id') or row.get('url') or f'{dataset}:{index}',
                'parquet':path,'row_index':index,'chunks':chunks,'anchors':anchors,'matches':[]})
            seen.update((sha,norm_sha)); count+=1
            if count%16==0: print(json.dumps({'domain':dataset,'candidates':count}),flush=True)
            if count == quota: break
        if count != quota: raise RuntimeError(f'candidate pool too small: {dataset}={count}')
    # A C automaton scans all long anchors in one pass over the 1B-token corpus.
    anchor_owner = {}
    for i,p in enumerate(pool):
        for a in p['anchors']: anchor_owner.setdefault(a,set()).add(i)
    import ahocorasick
    pattern=ahocorasick.Automaton()
    for anchor,owners in anchor_owner.items(): pattern.add_word(anchor,tuple(owners))
    pattern.make_automaton()
    def scan(text,source):
        hits=set()
        for _,owners in pattern.iter(normalize(text)):
            hits.update(owners)
        for i in hits:
            if source not in pool[i]['matches']: pool[i]['matches'].append(source)
    n=0
    with Path(args.training_data).open() as stream:
        for n,line in enumerate(stream,1):
            r=json.loads(line); scan(r.get('content_split',r.get('text','')),'training')
            if n%20000==0: print(json.dumps({'scanned_training_rows':n}),flush=True)
    if not n: raise RuntimeError('training overlap scan empty')
    # Audit available old raw documents and scene contexts, not result-label files.
    old_files=set()
    for glob in ('cbf_ttt*/**/documents.jsonl','cbf_ttt*/**/scenarios.jsonl','cbf_ttt*/**/scenes.jsonl','cbf_ttt*/**/episodes.jsonl','cbf_ttt*/pilot*scenarios.jsonl'):
        old_files.update(p for p in Path(args.old_root).glob(glob) if out not in p.parents)
    def texts(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('text','content_split') and isinstance(v,str): yield v
                elif k in ('ids','context_ids') and isinstance(v,list) and v and isinstance(v[0],int): yield tok.decode(v)
                elif k=='prefix' and isinstance(v,list):
                    for chunk in v:
                        if isinstance(chunk,list): yield tok.decode(chunk)
                else: yield from texts(v)
        elif isinstance(x,list):
            for v in x: yield from texts(v)
    for path in sorted(old_files):
        for line in path.open():
            for text in texts(json.loads(line)): scan(text,'old:'+str(path))
    eligible={name:[p for p in pool if p['domain']==name and not p['matches']] for name in ('fineweb','longcrawl')}
    if len(eligible['fineweb'])<72 or len(eligible['longcrawl'])<12:
        raise RuntimeError('insufficient nonoverlapping sources; protocol unchanged')
    rng=random.Random(208)
    for p in eligible.values(): rng.shuffle(p)
    groups=[]; cursors={'fineweb':0,'longcrawl':0}
    for g in range(24):
        name=domain(g); size=6 if name=='fineweb' else 1; start=cursors[name]
        chosen=eligible[name][start:start+size]; cursors[name]+=size
        groups.append({'domain':name,'sources':[p['sha256'] for p in chosen],
                       'chunks':[c for p in chosen for c in p['chunks']]})
    scenes=make_scenes(groups,tok)
    data=out/'scenes.jsonl'; data.write_text(''.join(json.dumps(r)+'\n' for r in scenes))
    metadata={'protocol':PROTOCOL,'training_rows_scanned':n,'training_data':args.training_data,'training_sha256':file_digest(args.training_data),
        'data_sha256':digest(data.read_bytes()),'tokenizer':args.tokenizer,
        'source_files':{name:{'path':path,'sha256':file_digest(path)} for name,path in (('fineweb',args.fineweb),('longcrawl',args.longcrawl))},
        'old_manifests':[str(p) for p in manifests],'old_context_files':[str(p) for p in sorted(old_files)],
        'candidate_sources':[{k:v for k,v in p.items() if k not in ('chunks','anchors')} for p in pool],
        'groups':[{k:v for k,v in p.items() if k!='chunks'} for p in groups],
        'counts':{'dev_groups':8,'confirm_groups':16,'scenes':192},
        'audit_scope':'normalized source hashes plus two 96-192 character interior anchors per background chunk; not semantic dedup'}
    (out/'design.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'built':192,'eligible':{k:len(v) for k,v in eligible.items()}}),flush=True)


def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fineweb','longcrawl','training-data','tokenizer','old-root','output'): p.add_argument('--'+key,required=True)
    prepare(p.parse_args())


if __name__=='__main__': main()
