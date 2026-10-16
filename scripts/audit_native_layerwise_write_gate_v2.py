"""Independent artifact audit for native layerwise admission V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


PROTOCOL = "native_layerwise_fastweight_admission_v2"
LAYERS = (0, 6, 12, 18, 24, 30, 35)
SINGLETONS = tuple(f"admit_layer_{layer}" for layer in LAYERS)
ALL_ACTIONS = ("reject_all", *SINGLETONS, "admit_all")


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def action_J(action):
    related = [row["answer_nll"] for row in action["queries"] if row["role"] == "related"]
    protected = [row["answer_nll"] for row in action["queries"] if row["role"] == "protected"]
    if len(related) != 2 or len(protected) != 3: raise ValueError("query role contract")
    return .5 * statistics.fmean(related) + .5 * statistics.fmean(protected)


def safe_action(row):
    reject = row["actions"]["reject_all"]
    eligible = [name for name in SINGLETONS if row["actions"][name]["J"] < reject["J"]
                and row["actions"][name]["protected_accuracy"] >= reject["protected_accuracy"]]
    return min(eligible, key=lambda name: row["actions"][name]["J"]) if eligible else "reject_all"


def bootstrap(values, draws=20_000, seed=2026101603):
    rng=random.Random(seed); n=len(values)
    samples=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return [samples[int(.025*draws)],samples[min(draws-1,int(.975*draws))]]


def audit(args):
    root, v1_root = Path(args.root), Path(args.v1_root)
    paths=sorted(root.glob(f"{args.stage}_shard*.jsonl")); rows=load(paths)
    expected=36 if args.stage=="smoke" else 144
    if len(rows)!=expected or len({(r['group_id'],r['boundary']) for r in rows})!=expected:
        raise ValueError("row coverage")
    v1_paths=sorted(v1_root.glob(f"{args.stage}_shard*_labels.jsonl")); v1=load(v1_paths)
    v1_index={(r['group_id'],r['boundary']):r for r in v1}
    features=load(sorted(v1_root.glob(f"{args.stage}_shard*_features.jsonl")))
    feature_index={(r['group_id'],r['boundary']):r for r in features}
    sidecars=[]
    for path in paths:
        item=json.loads(Path(str(path)+'.audit.json').read_text())
        if item['rows_sha256']!=digest(path) or not item['backbone_unchanged']:
            raise ValueError("sidecar mismatch")
        sidecars.append(item)
    for row in rows:
        key=(row['group_id'],row['boundary']); source=v1_index[key]
        if row['v1_label_sha256']!=canonical(source) or row['feature_sha256']!=canonical(feature_index[key]):
            raise ValueError("V1 dependency row mismatch")
        if set(row['actions'])!=set(ALL_ACTIONS): raise ValueError("action matrix")
        if any(abs(action_J(value)-value['J'])>1e-9 for value in row['actions'].values()):
            raise ValueError("action J mismatch")
        if row['actions']['reject_all']!=source['actions']['reject'] or row['actions']['admit_all']!=source['actions']['admit']:
            raise ValueError("V1 reference action changed")
        for layer, name in zip(LAYERS,SINGLETONS):
            item=row['composition'][name]
            expected_sources={str(value):('admit' if value==layer else 'reject') for value in LAYERS}
            if item['sources']!=expected_sources: raise ValueError("singleton sources")
            for value in LAYERS:
                donor=item['sources'][str(value)]
                if item['layer_sha256'][str(value)]!=row['base_layer_sha256'][donor][str(value)]:
                    raise ValueError("singleton layer hash")
        if row['branch_audit']['parent_fast_sha256']!=source['branch_audit']['parent_fast_sha256'] \
                or row['branch_audit']['reject_fast_sha256']!=source['branch_audit']['reject_fast_sha256'] \
                or row['branch_audit']['admit_fast_sha256']!=source['branch_audit']['admit_fast_sha256']:
            raise ValueError("branch replay hash")
        if any(not math.isfinite(row['resource'][name]) or row['resource'][name]<=0
               for name in ('seconds','peak_allocated_gib','peak_reserved_gib')):
            raise ValueError("resource")
    choices=[safe_action(row) for row in rows]; groups=sorted({r['group_id'] for r in rows})
    gains=[statistics.fmean(r['actions']['reject_all']['J']-r['actions'][safe_action(r)]['J']
                            for r in rows if r['group_id']==group) for group in groups]
    summary=json.loads((root/f"{args.stage}_summary.json").read_text())
    recomputed={
        'headroom_fraction':sum(max(r['actions']['reject_all']['J']-r['actions'][name]['J'] for name in SINGLETONS)>1e-4 for r in rows)/len(rows),
        'safe_oracle_gain_vs_reject':statistics.fmean(gains),
        'safe_oracle_gain_group_bootstrap_95ci':bootstrap(gains),
        'safe_oracle_actions':{name:choices.count(name) for name in ALL_ACTIONS if choices.count(name)},
    }
    for name,value in recomputed.items():
        if summary[name]!=value: raise ValueError(f"summary mismatch {name}")
    result={'protocol':PROTOCOL+'_independent_audit','stage':args.stage,'passed':True,
            'groups':len(groups),'labels':len(rows),'plan_sha256':digest(Path(args.plan)),
            'v1_audit_sha256':digest(v1_root/f"{args.stage}_independent_audit.json"),
            'result_files':{path.name:digest(path) for path in paths},
            'summary_sha256':digest(root/f"{args.stage}_summary.json"),'recomputed':recomputed,
            'test_opened':False}
    (root/f"{args.stage}_independent_audit.json").write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--v1-root',required=True)
    p.add_argument('--plan',required=True);p.add_argument('--stage',choices=('smoke','pilot'),required=True)
    a=p.parse_args();print(json.dumps(audit(a),indent=2))


if __name__=='__main__':main()
