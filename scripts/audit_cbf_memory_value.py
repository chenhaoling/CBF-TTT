"""Independent completion/source audit for a staged memory-value run."""
import argparse
from collections import Counter
import datetime
import json
import math
import statistics
from pathlib import Path

from tasks.build_cbf_memory_value import validate, file_digest, domain
from tasks.cbf_memory_value import read_results, q_summary


def audit(root):
    root=Path(root)
    design=json.loads((root/'data/design.json').read_text())
    data=root/'data/scenes.jsonl';scenes=[json.loads(x) for x in data.read_text().splitlines()]
    validate(scenes)
    assert file_digest(data)==design['data_sha256']
    groups=design['groups'];assert len(groups)==24
    selected=[source for g in groups for source in g['sources']]
    assert len(selected)==len(set(selected))==84
    pool={r['sha256']:r for r in design['candidate_sources']}
    docs=[pool[s] for s in selected]
    assert len({r['normalized_sha256'] for r in docs})==84
    assert len({r['source_id'] for r in docs})==84
    assert not any(r['matches'] for r in docs)
    assert design['training_rows_scanned']>0 and design['old_context_files'] and design['old_manifests']
    for g,item in enumerate(groups):
        assert item['domain']==domain(g)
        assert len(item['sources'])==(6 if domain(g)=='fineweb' else 1)
        assert {r['sources'][0] for r in scenes if r['group']==g}=={item['sources'][0]}
        assert all(r['sources']==item['sources'] for r in scenes if r['group']==g)
    qrows=read_results([str(root/f'q_{i}.jsonl') for i in (0,1)],scenes,'q')
    q=json.loads((root/'q_summary.json').read_text());assert q['Q']==q_summary(qrows)
    status=(root/'terminal_status.txt').read_text().strip()
    assert (root/'completed_at.txt').exists() and not (root/'failed_exit_code.txt').exists()
    if status=='stopped_by_Q':
        assert not q['Q']['passed']
        assert not any(root.glob('smoke_*.jsonl')) and not any(root.glob('dev_*.jsonl'))
        assert not any(root.glob('confirm_*.jsonl'))
    else:
        assert q['Q']['passed']
        smoke=json.loads((root/'smoke_summary.json').read_text())
        repeat=json.loads((root/'repeat_summary.json').read_text())
        read_results([str(root/f'smoke_{i}.jsonl') for i in (0,1)],scenes,'smoke')
        read_results([str(root/f'repeat_{i}.jsonl') for i in (0,1)],scenes,'repeat')
        if status=='stopped_by_resource_or_replay':
            assert not (smoke['resource_passed'] and repeat['repeat_passed'])
        else:
            assert smoke['resource_passed'] and repeat['repeat_passed']
            read_results([str(root/f'dev_{i}.jsonl') for i in (0,1)],scenes,'dev')
            dev=json.loads((root/'dev_summary.json').read_text())
            if status=='stopped_by_V_dev':
                assert not any(x['passed'] for x in dev['V'].values())
                assert not any(root.glob('confirm_*.jsonl'))
            else:
                assert status=='V_F_complete' and any(x['passed'] for x in dev['V'].values())
                read_results([str(root/f'confirm_{i}.jsonl') for i in (0,1)],scenes,'confirm')
    phases={}
    weights=set()
    for stage in ('q','smoke','repeat','dev','confirm'):
        paths=[root/f'{stage}_{i}.jsonl' for i in (0,1)]
        if not all(p.exists() for p in paths):continue
        rows=read_results([str(p) for p in paths],scenes,stage)
        profiles=[r['rollout'] for r in rows]+[v for r in rows for mode in r['queries'].values() for intervention in mode.values() for v in intervention.values()]
        assert all(math.isfinite(v[k]) and v[k]>0 for v in profiles for k in ('seconds','peak_allocated_gib','peak_reserved_gib'))
        weights.update(r['weights_sha256'] for r in rows)
        stagesec=[json.loads(Path(str(p)+'.audit.json').read_text())['seconds'] for p in paths]
        phases[stage]={'rows':len(rows),'query_forwards':len(profiles)-len(rows),
            'rollout_mean_seconds':sum(r['rollout']['seconds'] for r in rows)/len(rows),
            'peak_allocated_gib':max(v['peak_allocated_gib'] for v in profiles),
            'peak_reserved_gib':max(v['peak_reserved_gib'] for v in profiles),
            'shard_wall_seconds':stagesec,
            'rows_sha256':{p.name:file_digest(p) for p in paths}}
    assert len(weights)==1
    start=(root/'started_at.txt').read_text().strip();end=(root/'completed_at.txt').read_text().strip()
    strata={}
    for dataset in ('fineweb','longcrawl'):
        strata[dataset]={}
        for regime in ('stable','correction','recent1_distractor','recent2_distractor'):
            subset=[r for r in qrows if r['domain']==dataset and r['regime']==regime]
            strata[dataset][regime]={name:{k:statistics.fmean(r['queries']['full_kv']['self'][name][k] for r in subset) for k in ('correct','nll','greedy_correct')} for name in ('target','anchor')}
    return {'audit_passed':True,'status':status,'started_at':start,'completed_at':end,
        'wall_seconds':(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds(),
        'code_commit':(root/'code_commit.txt').read_text().strip(),
        'data_sha256':design['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
        'weights_sha256':next(iter(weights)),'selected_sources':84,'selected_source_overlap':0,
        'source_domain_counts':dict(Counter(d['domain'] for d in docs)),
        'training_rows_scanned':design['training_rows_scanned'],
        'old_manifest_count':len(design['old_manifests']),'old_context_file_count':len(design['old_context_files']),
        'phases':phases,'Q':q['Q'],'Q_strata':strata,'confirm_scored':bool(list(root.glob('confirm_*.jsonl')))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.root)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
