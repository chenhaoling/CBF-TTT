"""Independently check writer source splits, dev selection and test aggregation."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit(root, prior_metadata):
    documents = read(root / "documents.jsonl.meta.json")
    manifest = read(root / "episodes.jsonl.meta.json")
    papers = {row["source_id"] for row in documents["accepted"]}
    prior = {row["source_id"] for path in prior_metadata for row in read(path)["accepted"]}
    if len(papers) != 96 or papers & prior:
        raise ValueError("new source count or prior-paper overlap mismatch")
    splits = {split: set(ids) for split, ids in manifest["split_source_ids"].items()}
    if {key: len(ids) for key, ids in splits.items()} != {"train": 64, "dev": 16, "test": 16} or (
        set.union(*splits.values()) != papers or sum(map(len, splits.values())) != len(papers)
    ):
        raise ValueError("invalid or overlapping source splits")
    features = [row for index in (0, 1)
                for row in read(root / "features" / f"manifest_shard{index}.json")["records"]]
    if len(features) != 96 or {row["id"] for row in features} != papers or (
        len({row["feature_sha256"] for row in features}) != 96
    ):
        raise ValueError("missing/duplicate candidate cache provenance")
    selected = read(root / "train_seed123/selection.json")
    history = read(root / "train_seed123/history.json")
    expected = min(((epoch["dev_means"][f"writer_{gate}"], epoch["epoch"], gate)
                    for epoch in history for gate in (0.5, 1.0)), key=lambda value: value[0])
    if (selected["best"]["mean_nll"], selected["best"]["epoch"], selected["best"]["gate"]) != expected:
        raise ValueError("checkpoint selection does not match dev history")
    dev = rows_as_dict(read(root / "train_seed123/selected_dev.json"))
    baseline = rows_as_dict(read(root / "train_seed123/baseline_dev.json"))
    if set(dev) != splits["dev"] or set(baseline) != splits["dev"]:
        raise ValueError("dev rows differ from source split")
    gate = selected["best"]["gate"]
    dev_mean = statistics.mean(row[f"writer_{gate}"] for row in dev.values())
    if not math.isclose(dev_mean, selected["best"]["mean_nll"], abs_tol=1e-12):
        raise ValueError("dev selected loss mismatch")
    if any(abs(row[f"writer_{g}"] - row[f"raw_{g}"]) > 1e-6
           for row in baseline.values() for g in (0.5, 1.0)):
        raise ValueError("initial writer does not reproduce original candidate")
    raw_gate = min((0.5, 1.0), key=lambda g: statistics.mean(row[f"raw_{g}"] for row in baseline.values()))
    gain_none = statistics.mean(row["none"] for row in baseline.values()) - dev_mean
    gain_raw = statistics.mean(row[f"raw_{raw_gate}"] for row in baseline.values()) - dev_mean
    improved_dev = sum(baseline[key]["none"] - dev[key][f"writer_{gate}"] > 0.005 for key in dev)
    passed_dev = gain_none > 0.005 and gain_raw > 0.005 and improved_dev >= 8
    if selected["passed_dev_gate"] != passed_dev or selected["raw_gate"] != raw_gate:
        raise ValueError("dev gate or original-update selection mismatch")
    steps = rows(root / "train_seed123/steps.jsonl")
    if len(steps) != 320 or any(sum(row["epoch"] == e for row in steps) != 64 for e in range(1, 6)):
        raise ValueError("training budget mismatch")
    if {row["id"] for row in steps} != splits["train"]:
        raise ValueError("non-train source used for optimization")
    if any({row["id"] for row in steps if row["epoch"] == e} != splits["train"] for e in range(1, 6)):
        raise ValueError("incomplete training epoch")
    tested = rows_as_dict(rows(root / "test_seed123/test_rows.jsonl"))
    report = read(root / "test_seed123/summary.json")
    if not selected["passed_dev_gate"] or set(tested) != splits["test"] or (
        report["checkpoint_sha256"] != selected["checkpoint_sha256"] or report["gate"] != gate
    ):
        raise ValueError("test protocol/identity mismatch")
    for policy, value in report["mean_nll"].items():
        actual = statistics.mean(row[policy] for row in tested.values())
        if not math.isfinite(actual) or not math.isclose(value, actual, abs_tol=1e-12):
            raise ValueError("test aggregate mismatch")
    gains = {name: report["mean_nll"][name] - report["mean_nll"]["writer"]
             for name in ("none", "raw", "mismatched", "train_mean")}
    improved = sum(row["none"] - row["writer"] > 0.005 for row in tested.values())
    passed = improved >= 8 and min(gains.values()) > 0.005
    if passed != report["passed_test_gate"] or improved != report["improved_papers"]:
        raise ValueError("test gate mismatch")
    return {"source_papers": len(papers), "excluded_prior_papers": len(prior), "prior_overlap": 0,
            "split_counts": {key: len(value) for key, value in splits.items()},
            "feature_records": len(features), "train_steps": len(steps),
            "selected_epoch": selected["best"]["epoch"], "selected_gate": gate,
            "dev_gate_recomputed": passed_dev,
            "initial_writer_matches_raw": True, "test_means_recomputed": True,
            "test_gate_recomputed": passed, "writer_gain_vs_controls": gains,
            "summary_sha256": hashlib.sha256((root / "test_seed123/summary.json").read_bytes()).hexdigest()}


def rows_as_dict(values):
    result = {row["id"]: row for row in values}
    if len(result) != len(values):
        raise ValueError("duplicate source result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior-metadata", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.prior_metadata)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
