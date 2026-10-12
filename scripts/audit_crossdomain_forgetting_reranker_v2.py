"""Independent artifact audit for cross-domain forgetting reranker V2."""

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS,
    MAX_KEYS,
    paired_comparison,
    summarize_seed_metrics,
)
from tasks.train_crossdomain_forgetting_reranker import development_gate
from tasks.train_crossdomain_forgetting_reranker_v2 import (
    PROTOCOL,
    load_v1_records,
)
from tasks.train_crossdomain_forgetting_reranker import load_previous


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def oracle_metrics(rows):
    def correct(row):
        delete = row["probability"] >= row["threshold"]
        return (not delete and row["target"] == MAX_KEYS) or (
            delete and row["top"] == row["target"])
    local = [row for row in rows if row["target"] < MAX_KEYS]
    kinds = ("primary_initial", "hard_retain", "primary_update")
    return {
        "candidate_top1_accuracy": statistics.fmean(
            row["top"] == row["target"] for row in local),
        "action_accuracy": statistics.fmean(correct(row) for row in rows),
        "by_event_kind": {kind: {
            "accuracy": statistics.fmean(correct(row) for row in rows
                                         if row["event_kind"] == kind),
            "events": sum(row["event_kind"] == kind for row in rows),
        } for kind in kinds},
    }


def audit(root, zsre_root, counterfact_root, previous_root, v1_root):
    root = Path(root); zsre_root = Path(zsre_root); counterfact_root = Path(counterfact_root)
    previous_root = Path(previous_root); v1_root = Path(v1_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "observed_development_only": summary["data_status"] == "observed_development_only"
                                     and manifest["counterfact_status"]
                                     == "observed_development_only",
        "final_sources_sealed": not manifest["longmemeval_scored"]
                                and not manifest["locomo_scored"],
        "zsre_manifest": manifest["zsre_manifest_sha256"] == sha(zsre_root / "manifest.json"),
        "counterfact_manifest": manifest["counterfact_manifest_sha256"]
                                == sha(counterfact_root / "manifest.json"),
        "v1_summary": manifest["v1_summary_sha256"] == sha(v1_root / "summary.json"),
        "previous_summary": manifest["previous_summary_sha256"]
                            == sha(previous_root / "summary.json"),
        "checkpoint_hashes": all(training["checkpoint_sha256"][name] == sha(root / name)
                                 for name in training["checkpoint_sha256"]),
    }
    current_records, current_metrics = {}, {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"counterfact_mixed_oof_v2_seed{seed}.jsonl"
        rows = read_rows(path); current_records[str(seed)] = rows
        current_metrics[str(seed)] = aggregate(rows)
        checks[f"raw_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
        checks[f"raw_identity_{seed}"] = all(
            row["split"] == "counterfact_observed_mixed_oof_v2"
            and row["seed"] == seed for row in rows)
    causal = summarize_seed_metrics(current_metrics)
    checks["causal_recomputed"] = causal == summary["mixed_counterfact_oof"]["metrics"]

    oracle_path = root / "counterfact_mixed_oof_oracle_events.jsonl"
    rows = read_rows(oracle_path)
    checks["oracle_hash"] = training["raw_rows_sha256"][oracle_path.name] == sha(oracle_path)
    recomputed_oracle = oracle_metrics(rows)
    checks["oracle_recomputed"] = recomputed_oracle == summary["mixed_counterfact_oof"]["oracle_state"]
    fold_map = {key: int(value) for key, value in training["counterfact_folds"].items()}
    checks["oracle_fold_identity"] = all(
        row["phase"] == "outer_oof" and fold_map[row["session_id"]] == row["fold"]
        and row["threshold"] == training["selection"][str(row["fold"])]["threshold"]
        for row in rows)
    checks["oracle_event_identity"] = len(rows) == len({
        (row["session_id"], row["group_id"], row["event_kind"]) for row in rows})

    v1_records, _ = load_v1_records(v1_root)
    previous_records, previous_summary = load_previous(previous_root)
    v1_comparison = paired_comparison(current_records, v1_records, 11301)
    previous_comparison = paired_comparison(current_records, previous_records, 11311)
    checks["v1_comparison_recomputed"] = v1_comparison == summary["v2_vs_v1"]
    checks["previous_comparison_recomputed"] = previous_comparison == summary["v2_vs_previous"]
    fixed_retain = previous_summary["policies"]["fixed_retain"]["mean_accuracy"]
    checks["gate_recomputed"] = development_gate(
        causal, recomputed_oracle, fixed_retain) == summary["development_gate"]
    checks["paired_units"] = v1_comparison["paired_subject_units"] == 785 \
        and previous_comparison["paired_subject_units"] == 785
    return {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "manifest_sha256": sha(root / "training_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--zsre-root", required=True)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.zsre_root, args.counterfact_root,
                   args.previous_root, args.v1_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
