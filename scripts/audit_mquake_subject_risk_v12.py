"""Independent artifact and metric audit for MQuAKE subject-risk V12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.train_mquake_robust_calibration_v11 import subject_folds
from tasks.train_mquake_subject_risk_v12 import (
    EPOCH,
    FOLDS,
    PAIR_CACHE_LIMIT,
    RISK_THRESHOLDS,
    SAFETY_LCB_FLOOR,
    SAFETY_MACRO_FLOOR,
    SHORTLIST,
    candidate_metrics,
    cluster_metrics,
    release_gate,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def selected_threshold(grid):
    rows = []
    for key, details in grid.items():
        rows.append({"threshold": float(key), **details})
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["cluster"]["pair_macro"],
                row["metric"]["pair_state_accuracy"],
                row["metric"]["stale_deletion"],
                row["cluster"]["safety_lcb95"],
                -abs(row["threshold"] - .5),
            ),
        )
    else:
        selected = max(
            rows,
            key=lambda row: (
                row["cluster"]["safety_lcb95"],
                row["cluster"]["safety_macro"],
                row["cluster"]["pair_macro"],
                row["metric"]["pair_state_accuracy"],
                -abs(row["threshold"] - .5),
            ),
        )
    return selected["threshold"], bool(feasible)


def audit(root, data_root, source_root):
    import torch

    root = Path(root)
    data_root = Path(data_root)
    source_root = Path(source_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    records = read_jsonl(root / "oof_controller.jsonl")
    candidates = read_jsonl(root / "oof_candidate_rows.jsonl")
    random_records = read_jsonl(root / "oof_random_matched.jsonl")
    checkpoint = torch.load(
        root / "mquake_subject_risk_crossfit.pt", map_location="cpu", weights_only=False
    )
    metric = aggregate(records)
    cluster = cluster_metrics(records)
    candidate = candidate_metrics(candidates)
    comparison = paired_bootstrap(records, random_records, 37192)
    fold_metrics = {
        str(fold): aggregate(
            [row for row in records if checkpoint["folds"][row["subject"]] == fold]
        )
        for fold in range(FOLDS)
    }
    gate = release_gate(metric, candidate, cluster, fold_metrics, comparison)
    recompute_ok = (
        metric == summary["oof"]
        and cluster == summary["oof_cluster"]
        and candidate == summary["oof_candidate"]
        and comparison == summary["oof_vs_random"]
        and fold_metrics == summary["fold_metrics"]
        and gate == summary["oof_gate"]
    )
    thresholds_ok = all(
        selected_threshold(training["folds"][str(fold)]["models"][str(seed)]["grid"])
        == (
            training["folds"][str(fold)]["models"][str(seed)]["threshold"],
            training["folds"][str(fold)]["models"][str(seed)]["calibration_feasible"],
        )
        for fold in range(FOLDS)
        for seed in (32101, 32102, 32103)
    )
    train, _ = load_split(data_root, "train")
    tune, _ = load_split(data_root, "tune")
    episodes = train + tune
    expected_folds, expected_loads = subject_folds(episodes)
    identities_ok = (
        len(records) == len({row["episode_id"] for row in records}) == len(episodes)
        and len(candidates) == sum(row["length"] - 1 for row in episodes)
        and checkpoint["folds"] == expected_folds
        and training["fold_loads"] == expected_loads
    )
    deletion_count = sum(
        decision["deleted"] is not None
        for row in records for decision in row["decisions"]
    )
    random_count = sum(
        decision["deleted"] is not None
        for row in random_records for decision in row["decisions"]
    )
    source_ok = (
        manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
        and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
        and manifest["source_training_sha256"] == sha(source_root / "training.json")
        and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
        and training["checkpoint_sha256"]
        == sha(root / "mquake_subject_risk_crossfit.pt")
        and manifest["task_source_sha256"]
        == sha(Path(__file__).parents[1] / "tasks/train_mquake_subject_risk_v12.py")
    )
    locks_ok = (
        summary["protocol"] == manifest["protocol"] == "mquake_subject_risk_v12"
        and manifest["folds"] == FOLDS
        and manifest["candidate_epoch"] == manifest["gate_epoch"] == EPOCH
        and manifest["shortlist"] == SHORTLIST
        and manifest["risk_thresholds"] == list(RISK_THRESHOLDS)
        and manifest["safety_macro_floor"] == SAFETY_MACRO_FLOOR
        and manifest["safety_lcb_floor"] == SAFETY_LCB_FLOOR
        and manifest["pair_cache_limit"] == PAIR_CACHE_LIMIT
        and training["pair_cache"]["limit"] == PAIR_CACHE_LIMIT
        and manifest["candidate_subject_equal_loss"]
        and manifest["gate_subject_equal_loss"]
        and manifest["nested_train_calibrate_validate"]
        and not manifest["refit_after_calibration"]
        and not manifest["v10_development_used"]
        and not summary["v10_development_loaded"]
        and not manifest["mquake_cf_used"]
        and not manifest["longmemeval_used"]
        and not manifest["locomo_used"]
        and not summary["mquake_cf_used"]
        and not summary["longmemeval_used"]
        and not summary["locomo_scored"]
        and len(manifest["git_commit"]) == 40
    )
    checks = {
        "oof_metrics_cluster_candidate_comparison_and_gate": recompute_ok,
        "cluster_risk_threshold_selection": thresholds_ok,
        "subject_fold_assignment_and_identities": identities_ok,
        "matched_random_deletions": deletion_count == random_count
        == training["matched_random_deletions"],
        "raw_output_hashes": training["raw_rows_sha256"]
        == {path.name: sha(path) for path in root.glob("*.jsonl")},
        "source_and_checkpoint_hashes": source_ok,
        "protocol_training_and_access_locks": locks_ok,
    }
    return {
        "protocol": "mquake_subject_risk_v12_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "training_manifest_sha256": sha(root / "training_manifest.json"),
        "recomputed_oof_gate": gate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
