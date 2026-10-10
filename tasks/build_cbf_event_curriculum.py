"""Build deterministic event-stream supervision; no oracle fast-weight labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from collections import Counter

PROTOCOL = 'cbf_event_curriculum_v1'
FAMILIES = ('retain', 'revoke_last1', 'revoke_last2', 'reset')
SPLITS = ('train', 'dev', 'test')
UNKNOWN = 'unknown'
TEMPLATES = {
    'train': ('Batch {batch} records these assignments:', '{key} = {value}.',
              'Withdraw batch {batch}; its assignments no longer apply.',
              'Start a new session. All assignments from earlier sessions expire.',
              'What is the current value for {key}? Reply with the value only; use unknown if none exists.'),
    'dev': ('Recorded transaction {batch}:', 'For {key}, the value is {value}.',
            'Transaction {batch} has been invalidated in full.',
            'The session has changed; every earlier assignment is now invalid.',
            'Give the currently valid value of {key}, or unknown if it has no valid assignment.'),
    'test': ('Accepted record set {batch}:', 'Assign {value} to {key}.',
             'Cancel all entries issued in record set {batch}.',
             'A fresh session begins here. Entries preceding this boundary are void.',
             'Return only the active value associated with {key}; return unknown when it is unset.'),
}
RULE = ('Records arrive in chronological order. The latest assignment in a still-valid batch wins. '
        'Canceling a batch reveals the previous valid assignment if one exists. '
        'A new session expires all earlier batches. Batch identifiers are references, not values.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def replay(events):
    """Semantic truth only: this does not simulate neural weight removal."""
    batches = []
    for event in events:
        if event['op'] == 'reset':
            batches = []
        elif event['op'] == 'revoke':
            batches = [(b, facts) for b, facts in batches if b != event['batch']]
        elif event['op'] == 'assign':
            if any(b == event['batch'] for b, _ in batches):
                raise ValueError('duplicate batch')
            batches.append((event['batch'], event['facts']))
        else:
            raise ValueError('unknown event operation')
    state = {}
    for _, facts in batches:
        state.update(facts)
    return state


def render(events, split):
    header, fact, revoke, reset, _ = TEMPLATES[split]
    lines = []
    for e in events:
        if e['op'] == 'reset':lines.append(reset)
        elif e['op'] == 'revoke':lines.append(revoke.format(batch=e['batch']))
        else:
            lines.append(header.format(batch=e['batch']))
            lines.extend(fact.format(key=k, value=v) for k,v in e['facts'].items())
    return '\n'.join(lines)


def generate_group(seed, split, group):
    # Hash-derived namespaces are opaque and do not spell out train/dev/test in model text.
    group_id = hashlib.sha256(f'{seed}/{split}/{group}'.encode()).hexdigest()[:16]
    rng = random.Random(int(group_id,16))
    anchor = 'asset_'+group_id[:8]
    keys = ['unit_'+group_id[8:]+f'_{i}' for i in range(3)]
    ids = ['batch_'+hashlib.sha256(f'{group_id}/{i}'.encode()).hexdigest()[:10] for i in range(8)]
    values = ['code_'+x for x in rng.sample([f'{i:05d}' for i in range(10000,99999)], 40)]
    # Nest schedules before observing any model response; all family/twin variants share them.
    order = rng.sample(range(4,9),3)
    outputs = []
    for family in FAMILIES:
        for decisions in (1,2,3):
            schedule = sorted(order[:decisions])
            for twin in (0,1):
                blocks=[]; events=[]; questions=[]; answers=[]; audits=[]
                for block in range(1,9):
                    current=[]
                    if block in schedule:
                        if family == 'reset':current.append({'op':'reset'})
                        elif family.startswith('revoke_'):
                            lag=1 if family=='revoke_last1' else 2
                            current.append({'op':'revoke','batch':ids[block-lag-1]})
                    facts = ({anchor:values[twin], **{k:values[2+i] for i,k in enumerate(keys)}}
                             if block==1 else {keys[(block-2)%3]:values[4+block]})
                    # Unqueried identifiers vary too; no repeated natural-language padding.
                    facts['sensor_'+group_id+f'_{block}']=values[15+block]
                    current.append({'op':'assign','batch':ids[block-1],'facts':facts})
                    events.extend(current)
                    blocks.append({'index':block,'text':render(current,split)})
                    if block in schedule:
                        # Query the just-written key, a revoked/other key, and a protected old anchor.
                        past_key = keys[(block-3 if family!='revoke_last2' else block-4)%3]
                        for role,key in (('current',keys[(block-2)%3]),('historical',past_key),('anchor',anchor)):
                            qid=f'b{block}_{role}'
                            questions.append({'id':qid,'after_block':block,'text':TEMPLATES[split][4].format(key=key)})
                            answers.append({'id':qid,'answer':replay(events).get(key,UNKNOWN),'role':role,'key':key})
                    audits.append(copy.deepcopy(current))
                identifier=f'{group_id}.{family}.{decisions}.{twin}'
                inputs={'protocol':PROTOCOL,'id':identifier,'group_id':group_id,'split':split,
                        'rules':RULE,'blocks':blocks,'decision_boundaries':schedule,'queries':questions}
                targets={'id':identifier,'answers':answers,'family':family,'decisions':decisions,'twin':twin,
                         'semantic_events':audits,'token_weight_oracle_available':False}
                outputs.append((inputs,targets))
    return outputs


def validate(pairs):
    """Audit causal truth, disjoint identities, render parity and complete twin groups."""
    if not pairs:raise ValueError('empty corpus')
    groups={};names={s:set() for s in SPLITS};counts=Counter()
    index={i['id']:(i,t) for i,t in pairs}
    if len(index)!=len(pairs):raise ValueError('duplicate episode')
    for inp,target in pairs:
        split=inp['split'];gid=inp['group_id']
        if inp['protocol']!=PROTOCOL or target['id']!=inp['id'] or split not in SPLITS:
            raise ValueError('identity mismatch')
        if gid in groups and groups[gid]!=split:raise ValueError('group leakage')
        groups[gid]=split
        if len(inp['blocks'])!=8 or len(target['semantic_events'])!=8:raise ValueError('block count')
        if len(inp['decision_boundaries'])!=target['decisions'] or len(inp['queries'])!=3*target['decisions']:
            raise ValueError('decision/query count')
        if target['token_weight_oracle_available']:raise ValueError('fabricated weight oracle')
        if set(inp)!=set(('protocol','id','group_id','split','rules','blocks','decision_boundaries','queries')):
            raise ValueError('unexpected model input metadata')
        truth={a['id']:a for a in target['answers']}
        if len(truth)!=len(inp['queries']) or set(truth)!={q['id'] for q in inp['queries']}:raise ValueError('answer matrix')
        all_events=[]
        for block,events in zip(inp['blocks'],target['semantic_events']):
            if block['text']!=render(events,split):raise ValueError('render mismatch')
            all_events+=events
            for event in events:
                if event['op']=='assign':names[split].update(event['facts'])
            state=replay(all_events)
            for q in inp['queries']:
                if q['after_block']!=block['index']:continue
                a=truth[q['id']]
                if a['answer']!=state.get(a['key'],UNKNOWN):raise ValueError('incorrect causal answer')
                if q['text']!=TEMPLATES[split][4].format(key=a['key']):raise ValueError('query mismatch')
                if a['answer']!=UNKNOWN and a['answer'] in q['text']:raise ValueError('query leaks value')
        # Twin perturbation belongs to this split and changes one initial assignment only.
        twin_id=inp['id'].rsplit('.',1)[0]+'.'+str(1-target['twin'])
        if twin_id not in index:raise ValueError('missing twin')
        other,other_t=index[twin_id]
        if other['split']!=split or other['queries']!=inp['queries']:raise ValueError('twin leakage/query change')
        if other_t['semantic_events'][1:]!=target['semantic_events'][1:]:raise ValueError('twin altered later events')
        a=target['semantic_events'][0][0]['facts'];b=other_t['semantic_events'][0][0]['facts']
        if set(a)!=set(b) or sum(a[k]!=b[k] for k in a)!=1:raise ValueError('twin must change one fact')
        counts[split,target['family'],target['decisions']]+=1
    for a in SPLITS:
        for b in SPLITS:
            if a!=b and names[a]&names[b]:raise ValueError('entity leakage')
    for gid in groups:
        members=[t for i,t in pairs if i['group_id']==gid]
        if len(members)!=24 or {(t['family'],t['decisions'],t['twin']) for t in members}!={(f,d,t) for f in FAMILIES for d in (1,2,3) for t in (0,1)}:
            raise ValueError('incomplete group matrix')
    return {'episodes':len(pairs),'groups':len(groups),'counts':{s:sum(i['split']==s for i,t in pairs) for s in SPLITS},
            'queries':sum(len(i['queries']) for i,t in pairs),'causal_answers_verified':True,
            'entities_disjoint':True,'templates_disjoint':True,'oracle_actions_generated':False}


def supervision_records(inp, target):
    """Causal training examples: future blocks and private events never enter input."""
    answers = {a['id']:a['answer'] for a in target['answers']}
    for q in inp['queries']:
        yield {'id':inp['id']+'.'+q['id'], 'group_id':inp['group_id'], 'split':inp['split'],
               'write_blocks':[b['text'] for b in inp['blocks'] if b['index']<=q['after_block']],
               'read_query':inp['rules']+'\n'+q['text'], 'answer':answers[q['id']],
               'decision_boundaries':[b for b in inp['decision_boundaries'] if b<=q['after_block']],
               'loss_scope':'answer_tokens_only', 'required_primary_readout':'fresh_kv_preserve_fast_memory'}


def warmup_records(pairs):
    """Single-block fact recall before training any forgetting policy."""
    for inp,target in pairs:
        if target['family']!='retain' or target['decisions']!=1:continue
        facts=target['semantic_events'][0][0]['facts']
        for key,value in list(facts.items())[:4]:
            yield {'id':inp['group_id']+'.'+str(target['twin'])+'.'+key,
                   'group_id':inp['group_id'],'split':inp['split'],
                   'write_blocks':[inp['blocks'][0]['text']],
                   'read_query':inp['rules']+'\n'+TEMPLATES[inp['split']][4].format(key=key),
                   'answer':value,'decision_boundaries':[], 'loss_scope':'answer_tokens_only',
                   'required_primary_readout':'fresh_kv_preserve_fast_memory'}


def materialize(seed, counts, output):
    if set(counts)!=set(SPLITS) or min(counts.values())<1:raise ValueError('positive counts for all splits required')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    pairs=[p for split in SPLITS for g in range(counts[split]) for p in generate_group(seed,split,g)]
    audit=validate(pairs); output.mkdir(parents=True)
    manifest={'protocol':PROTOCOL,'seed':seed,'group_counts':counts,'audit':audit,'files_sha256':{},
              'training_status':'not_trained','tokenization_status':'semantic_text_only',
              'notice':'IDs/split/group are join metadata, never model features. Read only prefix through after_block. Targets and semantic events are teacher-only. No native fixed-chunk packing is implied.'}
    for split in SPLITS:
        part=[(i,t) for i,t in pairs if i['split']==split]
        for suffix,rows in (('inputs',[i for i,t in part]),('targets',[t for i,t in part]),
                            ('supervision',[r for i,t in part for r in supervision_records(i,t)]),
                            ('warmup',list(warmup_records(part)))):
            p=output/f'{split}.{suffix}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest['files_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest['templates_sha256']=digest(TEMPLATES)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261011)
    for split,default in (('train',8),('dev',2),('test',2)):p.add_argument('--'+split+'-groups',type=int,default=default)
    a=p.parse_args();m=materialize(a.seed,{s:getattr(a,s+'_groups') for s in SPLITS},a.output);print(json.dumps(m,indent=2))
