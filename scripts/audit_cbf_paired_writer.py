"""Independently audit paired source isolation, fixed budgets and dev stage gates."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from tasks.build_cbf_paired_facts import validate


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recompute(records, gate, raw_gate):
    controls = {"none": "none", "raw": f"raw_{raw_gate}", "twin": f"twin_{gate}", "train_mean": f"train_mean_{gate}"}
    writer = f"writer_{gate}"
    mean = statistics.mean(r["policies"][writer]["nll"] for r in records)
    accuracy = statistics.mean(r["policies"][writer]["correct"] for r in records)
    gains = {name: statistics.mean(r["policies"][p]["nll"] for r in records)-mean for name, p in controls.items()}
    accuracy_gains = {name: accuracy-statistics.mean(r["policies"][p]["correct"] for r in records) for name, p in controls.items()}
    groups = {r["group_id"] for r in records}
    count = sum(all(statistics.mean(r["policies"][p]["nll"]-r["policies"][writer]["nll"]
                                   for r in records if r["group_id"] == group) > .005
                    for p in ("none", f"twin_{gate}")) for group in groups)
    return {"mean_nll": mean, "accuracy": accuracy, "min_gain": min(gains.values()),
            "gain_vs_controls": gains, "accuracy_gain_vs_controls": accuracy_gains,
            "improved_groups": count, "groups": len(groups),
            "passed": min(gains.values()) > .005 and count >= math.ceil(len(groups)/2)
                      and accuracy >= .25 and min(accuracy_gains.values()) >= .125}


def equal(actual, expected):
    for key, value in actual.items():
        if isinstance(value, dict):
            equal(value, expected[key])
        elif isinstance(value, float):
            if not math.isfinite(value) or not math.isclose(value, expected[key], abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"recomputed {key} differs")
        elif value != expected[key]:
            raise ValueError(f"recomputed {key} differs")


def audit(root, prior):
    episodes = rows(root/"episodes.jsonl")
    validate(episodes)
    sources = {r["group_id"] for r in episodes}
    excluded = {r["source_id"] for path in prior for r in read(path)["accepted"]}
    if len(sources) != 48 or sources & excluded:
        raise ValueError("source overlap or count mismatch")
    if sources != {r["source_id"] for r in read(root/"documents.jsonl.meta.json")["accepted"]}:
        raise ValueError("source provenance mismatch")
    split_ids = {s: {r["id"] for r in episodes if r["split"] == s} for s in ("train", "dev", "test")}
    if {s: len(ids) for s, ids in split_ids.items()} != {"train": 64, "dev": 16, "test": 16}:
        raise ValueError("split budget mismatch")
    dataset_hash = hashlib.sha256((root/"episodes.jsonl").read_bytes()).hexdigest()
    manifests = [read(root/"features"/f"manifest_shard{i}.json") for i in (0, 1)]
    features = [r for m in manifests for r in m["records"]]
    if len(features) != 96 or {r["id"] for r in features} != {r["id"] for r in episodes} or any(m["episodes_sha256"] != dataset_hash for m in manifests):
        raise ValueError("candidate manifests differ from dataset")
    selections, orders, initials = {}, {}, {}
    for objective in ("paired", "nll"):
        directory = root/f"train_{objective}"
        selection = read(directory/"selection.json")
        history = read(directory/"history.json")
        best = max((c for h in history for c in h["decisions"]),
                   key=lambda c: (c["min_gain"], -c["mean_nll"], -c["epoch"], -c["gate"]))
        if best != selection["best"] or selection["smoke"] or selection["episodes_sha256"] != dataset_hash or selection["objective"] != objective:
            raise ValueError("selection identity or dev selection mismatch")
        selected_rows = read(directory/"selected_dev.json")
        if {r["id"] for r in selected_rows} != split_ids["dev"]:
            raise ValueError("non-dev source scored during selection")
        initial = read(directory/"initial_dev.json")
        raw_gate = min((.5, 1.), key=lambda g: statistics.mean(r["policies"][f"raw_{g}"]["nll"] for r in initial))
        if raw_gate != selection["raw_gate"]:
            raise ValueError("raw gate mismatch")
        for r in initial:
            for g in (.5, 1.):
                if abs(r["policies"][f"raw_{g}"]["nll"]-r["policies"][f"writer_{g}"]["nll"]) > 1e-6:
                    raise ValueError("initial writer differs from original")
        result = recompute(selected_rows, best["gate"], raw_gate)
        equal(result, best)
        if result["passed"] != selection["passed_dev_gate"]:
            raise ValueError("dev gate mismatch")
        steps = rows(directory/"steps.jsonl")
        if len(steps) != 320 or [h["epoch"] for h in history] != list(range(6)):
            raise ValueError("training budget mismatch")
        for epoch in range(1, 6):
            counts = Counter(r["id"] for r in steps if r["epoch"] == epoch)
            if dict(counts) != {key: 1 for key in split_ids["train"]}:
                raise ValueError("train epoch source coverage mismatch")
        orders[objective] = [r["id"] for r in steps]
        initials[objective] = history[0]["mean_nll"]
        selections[objective] = {"selected_epoch": best["epoch"], "selected_gate": best["gate"], **result}
    if orders["paired"] != orders["nll"]:
        raise ValueError("ablation data order differs")
    equal(initials["paired"], initials["nll"])
    test_exists = (root/"test").exists()
    if test_exists != selections["paired"]["passed"]:
        raise ValueError("test execution inconsistent with dev gate")
    if test_exists:
        report = read(root/"test/summary.json")
        for objective in ("paired", "nll"):
            tested = rows(root/"test"/f"{objective}_rows.jsonl")
            if {r["id"] for r in tested} != split_ids["test"]:
                raise ValueError("test sources differ")
            selection = read(root/f"train_{objective}/selection.json")
            equal(recompute(tested, selection["best"]["gate"], selection["raw_gate"]), report[objective]["decision"])
    return {"sources": 48, "prior_sources": len(excluded), "prior_overlap": 0,
            "split_episodes": {s: len(ids) for s, ids in split_ids.items()}, "twins_single_token_difference": True,
            "balanced_labels": True, "feature_records": len(features), "steps_per_objective": 320,
            "same_training_order_and_initialization": True, "dev_recomputed": selections,
            "test_scored": test_exists, "episodes_sha256": dataset_hash,
            "full_kv_sanity": read(root/"sanity/summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior-metadata", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.prior_metadata)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
