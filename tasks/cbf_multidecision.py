"""Frozen multi-decision forgetting search on an audited subset of existing scenes."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.cbf_selective import action_name, audit_sources

GROUPS = (0, 1, 4, 5)
GRID = tuple(itertools.product((0., .5, 1.), repeat=3))
KEEP = (1., 1., 1.)
ANCHORS = ((0., 0., 1.), (0., 1., 1.), (1., 1., 0.), (1., 0., 1.),
           (0., 0., .5), (0., .5, 1.), (1., 1., .5), (1., .5, 1.))


def plan_id(actions):
    return 'p_' + ''.join(str(int(2*a)) for point in range(4, 8) for a in actions.get(point, KEEP))


def decode_plan(name):
    if not name.startswith('p_') or len(name) != 14 or set(name[2:]) - set('012'):
        raise ValueError('invalid plan')
    return {point: tuple(int(c)/2 for c in name[2+3*(point-4):5+3*(point-4)]) for point in range(4, 8)}


def group_design(group):
    later = random.Random(108+1009*group).sample([5, 6, 7], 2)
    result = {'schedules': {}, 'families': {}}
    for k in (1, 2, 3):
        points = sorted([4]+later[:k-1])
        result['schedules'][str(k)] = points
        globals_ = [plan_id(dict(zip(points, [(a,)*3 for a in values])))
                    for values in itertools.product((0., .5, 1.), repeat=k)]
        if k == 1:
            local = [plan_id({4: a}) for a in GRID]
        else:
            rng = random.Random(9108+1009*group+97*k)
            anchors = [plan_id(dict.fromkeys(points, a)) for a in ANCHORS]
            random_plans = []
            while len(random_plans) < 16:
                sequence = [rng.choice(GRID) for _ in points]
                name = plan_id(dict(zip(points, sequence)))
                if len(set(sequence)) < 2 or all(a[0] == a[1] == a[2] for a in sequence):
                    continue
                if name not in random_plans and name not in anchors:
                    random_plans.append(name)
            # Predeclared order: equal-budget k=2 covers constants and varying actions.
            extras = [x for pair in zip(anchors, random_plans[:8]) for x in pair] + random_plans[8:]
            local = [plan_id(dict.fromkeys(points, (a,)*3)) for a in (0., .5, 1.)]+extras
        result['families'][str(k)] = {
            'global_exact': globals_, 'local_sample': local,
            'local_matched': local[:3**k] if k > 1 else [],
            'union': sorted(set(globals_+local))}
    result['all_plans'] = sorted({p for f in result['families'].values() for p in f['union']})
    return result


def build(args):
    raw = Path(args.data).read_bytes()
    scenes = [json.loads(line) for line in raw.splitlines()]
    metadata = json.loads(Path(args.data+'.meta.json').read_text())
    source_audit = audit_sources(metadata, scenes)
    selected = [s for s in scenes if s['group'] in GROUPS]
    if len(scenes) != 32 or len(selected) != 16 or metadata['overlap_matching_rows']:
        raise ValueError('expected audited 32 scenes and fixed 16-scene subset')
    design = {'protocol': 'multidecision_v1', 'data_sha256': hashlib.sha256(raw).hexdigest(),
              'selected_groups': GROUPS, 'scene_ids': [s['id'] for s in selected],
              'source_audit': source_audit, 'groups': {str(g): group_design(g) for g in GROUPS},
              'scope': 'Previously observed scenes; historical split names are not fresh held-out data.'}
    path = Path(args.output)
    with path.open('x') as stream:
        json.dump(design, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'scenes': len(selected), 'new_plans_per_scene':
                     {g: len(d['all_plans'])-27 for g, d in design['groups'].items()}}))


def rollout(session, scene, name):
    actions = decode_plan(name)
    session.commit_cohorts(actions[4])
    losses = []
    for point, future in enumerate(scene['future'], 5):
        session.advance(future['ids'], actions[point])
        length = session.cache.get_seq_length()
        losses.append(session.score_answer(future['query_ids'], future['answer_ids']))
        if session.cache.get_seq_length() != length:
            raise RuntimeError('query contaminated trajectory')
    if not all(math.isfinite(x) for x in losses):
        raise RuntimeError('nonfinite rollout')
    return losses


def fixed_control(model, scene, name):
    from cbf_ttt.runtime import CBFSession
    from cbf_ttt.selective import CohortSession
    session = CBFSession(model) if name == 'none' else CohortSession(model)
    losses = []
    chunks = scene['prefix']+[f['ids'] for f in scene['future']]
    for point, chunk in enumerate(chunks, 1):
        if name == 'none':
            session._forward(chunk, collect=False)
        elif name == 'clear':
            session.advance(chunk, (0.,)*3)
        else:
            raise ValueError(name)
        if point >= 5:
            f = scene['future'][point-5]
            losses.append(session.score_answer(f['query_ids'], f['answer_ids']))
    return losses


def collect(args):
    import torch
    from cbf_ttt.selective import CohortSession
    from tasks.cbf_selective import measure
    from tasks.cbf_ttt import _load_model
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(args.data).read_bytes()
    design = json.loads(Path(args.design).read_text())
    if design['data_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('design data mismatch')
    for g in GROUPS:
        if design['groups'][str(g)] != group_design(g):
            raise ValueError('changed search design')
    scenes = [json.loads(line) for line in raw.splitlines()]
    scenes = [s for s in scenes if s['id'] in design['scene_ids'] and s['group'] % 2 == args.shard]
    if args.smoke:
        scenes = scenes[:1]
    refs = {r['id']: r for f in args.reference for r in map(json.loads, Path(f).read_text().splitlines())}
    torch.manual_seed(108)
    model = _load_model(args.model, 'cuda', 'bfloat16')
    versions = [p._version for p in model.parameters()]
    with output.open('x') as sink:
        for scene in scenes:
            started = time.perf_counter()
            ref = refs[scene['id']]
            if ref['data_sha256'] != design['data_sha256'] or ref['model'] != args.model or ref['smoke']:
                raise ValueError('reference source/checkpoint mismatch')
            if scene['chunk_size'] != model.config.ttt_chunk:
                raise ValueError('chunk mismatch')
            torch.cuda.empty_cache()
            prefix = CohortSession(model)
            for chunk in scene['prefix'][:-1]:
                prefix.advance(chunk)
            prefix._forward(scene['prefix'][-1], collect=True)
            d = design['groups'][str(scene['group'])]
            results = {}
            # Reuse all 27 audited single-point trajectories; three globals bridge the path.
            for a in GRID:
                results[plan_id({4: a})] = {**ref['results'][action_name(a)], 'reused': True}
            checks = []
            for a in (0., .5, 1.):
                name = plan_id({4: (a,)*3})
                losses, profile = measure(lambda: rollout(prefix.clone(), scene, name))
                checks.append({'plan': name, 'max_error': max(abs(x-y) for x, y in
                    zip(losses, results[name]['losses'])), **profile})
            if max(c['max_error'] for c in checks) > 1e-5:
                raise RuntimeError('single-decision reference mismatch')
            new = [p for p in d['all_plans'] if p not in results]
            if args.smoke:
                new = list(dict.fromkeys(p for k in ('2', '3') for p in
                    (d['families'][k]['global_exact'][:2]+d['families'][k]['local_sample'][3:5]) if p not in results))
            for index, name in enumerate(new):
                losses, profile = measure(lambda: rollout(prefix.clone(), scene, name))
                results[name] = {'losses': losses, 'nll': statistics.fmean(losses), 'reused': False, **profile}
                print(json.dumps({'scene': scene['id'], 'completed': index+1, 'total': len(new),
                                  'plan': name, **profile}), flush=True)
            del prefix
            controls = {name: {**ref['results']['fixed_'+name], 'reused': True}
                        for name in ('global_half', 'window2', 'window3')}
            for name in ('clear', 'none'):
                losses, profile = measure(lambda: fixed_control(model, scene, name))
                controls[name] = {'losses': losses, 'nll': statistics.fmean(losses), 'reused': False, **profile}
            error = max(abs(a-b) for a, b in zip(controls['clear']['losses'], ref['results']['fixed_global_clear']['losses']))
            if error > 1e-5:
                raise RuntimeError('every-step clear reference mismatch')
            sink.write(json.dumps({'id': scene['id'], 'group': scene['group'], 'regime': scene['regime'],
                'split': scene['split'], 'data_sha256': design['data_sha256'], 'model': args.model,
                'results': results, 'controls': controls, 'reference_checks': checks,
                'clear_reference_max_error': error, 'scene_seconds': time.perf_counter()-started,
                'smoke': args.smoke})+'\n')
            sink.flush()
    if versions != [p._version for p in model.parameters()]:
        raise RuntimeError('backbone changed')
    Path(str(output)+'.audit.json').write_text(json.dumps({'backbone_unchanged': True, 'scenes': len(scenes)})+'\n')


def summarize(args):
    design = json.loads(Path(args.design).read_text())
    rows = [r for f in args.inputs for r in map(json.loads, Path(f).read_text().splitlines())]
    if len(rows) != 16 or {r['id'] for r in rows} != set(design['scene_ids']):
        raise ValueError('incomplete/duplicate scene results')
    if len({r['model'] for r in rows}) != 1:
        raise ValueError('mixed checkpoints')
    for path in args.inputs:
        if not json.loads(Path(path+'.audit.json').read_text())['backbone_unchanged']:
            raise ValueError('backbone audit failed')
    stats, profiles = [], []
    for row in rows:
        d = design['groups'][str(row['group'])]
        if row['smoke'] or row['data_sha256'] != design['data_sha256'] or set(row['results']) != set(d['all_plans']):
            raise ValueError('smoke, data mismatch, or missing plans')
        if set(row['controls']) != {'global_half','window2','window3','clear','none'}:
            raise ValueError('missing controls')
        if max(row['clear_reference_max_error'], *(x['max_error'] for x in row['reference_checks'])) > 1e-5:
            raise ValueError('reference error')
        for value in list(row['results'].values())+list(row['controls'].values()):
            if len(value['losses']) != 3 or not all(math.isfinite(v) for v in value['losses']):
                raise ValueError('invalid loss')
            if abs(statistics.fmean(value['losses'])-value['nll']) > 1e-10:
                raise ValueError('invalid aggregate loss')
            if not value['reused']:
                if any(not math.isfinite(value[k]) or value[k] <= 0 for k in ('seconds','peak_allocated_gib','peak_reserved_gib')):
                    raise ValueError('invalid resource record')
                profiles.append(value)
        v = {'group': row['group'], 'split': row['split'], 'regime': row['regime']}
        def best(names):
            name = min(names, key=lambda n: row['results'][n]['nll'])
            return name, row['results'][name]['nll']
        for k in ('1','2','3'):
            f = d['families'][k]
            v['g'+k+'_plan'], v['g'+k] = best(f['global_exact'])
            v['u'+k+'_plan'], v['u'+k] = best(f['union'])
            if k != '1':
                v['m'+k+'_plan'], v['m'+k] = best(f['local_matched'])
        if v['g2'] > v['g1']+1e-5 or v['g3'] > v['g2']+1e-5:
            raise ValueError('nested exhaustive global search violated')
        v.update({'fixed_'+n: r['nll'] for n,r in row['controls'].items()})
        v['retain_all'] = row['results'][plan_id({})]['nll']
        v['strong'] = min(v['retain_all'], *(r['nll'] for r in row['controls'].values()))
        v['temporal_gain'] = v['g1']-v['g3']
        v['local_union_gain'] = v['g3']-v['u3']
        v['matched_local_gain'] = v['g3']-v['m3']
        v['vs_strong_gain'] = v['strong']-v['u3']
        stats.append(v)
    keys = [k for k,v in stats[0].items() if isinstance(v, (float,int)) and k != 'group']
    def means(items):
        return {k: statistics.fmean(r[k] for r in items) for k in keys}
    group_means = {str(g): means([r for r in stats if r['group']==g]) for g in GROUPS}
    report = {'protocol':'multidecision_v1', 'scenes':16, 'source_groups':4, 'data_sha256':design['data_sha256'],
              'means':means(stats), 'group_means':group_means,
              'regime_means':{r:means([x for x in stats if x['regime']==r]) for r in sorted({x['regime'] for x in stats})},
              'historical_split_means':{s:means([x for x in stats if x['split']==s]) for s in ('pilot','confirm')},
              'selected_plans':[ {k:v for k,v in r.items() if k.endswith('_plan') or k in ('group','regime')} for r in stats],
              'profile':{'new_scored_trajectories':len(profiles), 'mean_trajectory_seconds':statistics.fmean(p['seconds'] for p in profiles),
                         'max_peak_allocated_gib':max(p['peak_allocated_gib'] for p in profiles),
                         'max_peak_reserved_gib':max(p['peak_reserved_gib'] for p in profiles),
                         'summed_scene_seconds':sum(r['scene_seconds'] for r in rows)},
              'single_point_reused_trajectories':16*27, 'reference_checks':16*4,
              'reference_max_error':max(max(r['clear_reference_max_error'], *(x['max_error'] for x in r['reference_checks'])) for r in rows),
              'backbone_unchanged':True}
    report['temporal_signal'] = (report['means']['temporal_gain'] > .005 and
        sum(g['temporal_gain'] > .005 for g in group_means.values()) >= 3 and
        sum(r['temporal_gain'] > .005 for r in report['regime_means'].values()) >= 2)
    report['selective_signal_beyond_controls'] = (report['means']['matched_local_gain'] > .005 and
        report['means']['vs_strong_gain'] > .005 and
        sum(g['matched_local_gain'] > .005 and g['vs_strong_gain'] > .005 for g in group_means.values()) >= 3)
    report['scope'] = 'Exhaustive global oracle; restricted/sample local search. Previously observed diagnostic data, not policy generalization.'
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build'); b.add_argument('--data',required=True); b.add_argument('--output',required=True)
    c=sub.add_parser('collect')
    for key in ('data','design','model','output'):
        c.add_argument('--'+key,required=True)
    c.add_argument('--reference',nargs='+',required=True)
    c.add_argument('--shard',type=int,choices=(0,1),required=True)
    c.add_argument('--smoke',action='store_true')
    s=sub.add_parser('summarize'); s.add_argument('--design',required=True); s.add_argument('--inputs',nargs='+',required=True); s.add_argument('--output',required=True)
    args=p.parse_args(); {'build':build,'collect':collect,'summarize':summarize}[args.command](args)


if __name__=='__main__':
    main()
