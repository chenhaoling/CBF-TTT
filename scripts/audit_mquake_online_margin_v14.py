"""Independent artifact and metric audit for MQuAKE online margin V14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.evaluate_mquake_ensemble_calibration_v13 import EXPECTED_V12_HASHES
from tasks.train_mquake_online_margin_v14 import (
    EPOCH,
    GATE_SEEDS,
    MARGIN_THRESHOLDS,
    PROTOCOL,
    SAFETY_LCB_FLOOR,
    SAFETY_MACRO_FLOOR,
    SCORE_SCALE_FLOOR,
)
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_subject_risk_v12 import (
    PAIR_CACHE_LIMIT,
    candidate_metrics,
    cluster_metrics,
    release_gate,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def selected_threshold(grid):
    rows = [{"threshold": float(key), **details} for key, details in grid.items()]
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            row["metric"]["stale_deletion"], row["cluster"]["safety_lcb95"],
            -abs(row["threshold"] - 2.),
        ))
    else:
        selected = max(rows, key=lambda row: (
            row["cluster"]["safety_lcb95"], row["cluster"]["safety_macro"],
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            -abs(row["threshold"] - 2.),
        ))
    return selected["threshold"], bool(feasible)


def audit(root, data_root, source_root, v12_root):
    import torch

    root, data_root = Path(root), Path(data_root)
    source_root, v12_root = Path(source_root), Path(v12_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    records = read_jsonl(root / "oof_controller.jsonl")
    candidates = read_jsonl(root / "oof_candidate_rows.jsonl")
    random_records = read_jsonl(root / "oof_random_matched.jsonl")
    checkpoint = torch.load(root / "mquake_online_margin_crossfit.pt",
                            map_location="cpu", weights_only=False)
    metric = aggregate(records)
    cluster = cluster_metrics(records)
    candidate = candidate_metrics(candidates)
    comparison = paired_bootstrap(records, random_records, 40192)
    fold_metrics = {str(fold): aggregate([
        row for row in records if checkpoint["folds"][row["subject"]] == fold
    ]) for fold in range(FOLDS)}
    gate = release_gate(metric, candidate, cluster, fold_metrics, comparison)
    pair_checks = all(gate["checks"][key] for key in (
        "stale_deletion_at_least_90", "pair_state_at_least_90",
        "subject_macro_pair_at_least_85", "minimum_fold_pair_at_least_85",
    ))
    safety_checks = all(gate["checks"][key] for key in (
        "current_retention_at_least_99", "non_target_retention_at_least_97",
        "hard_retain_at_least_97", "cluster_safety_lcb_at_least_95",
    ))
    recompute_ok = (
        metric == summary["oof"] and cluster == summary["oof_cluster"]
        and candidate == summary["oof_candidate"]
        and comparison == summary["oof_vs_random"]
        and fold_metrics == summary["fold_metrics"] and gate == summary["oof_gate"]
        and summary["relative_margin_hypothesis_supported"] == (pair_checks and safety_checks)
    )
    thresholds_ok = all(
        selected_threshold(training["folds"][str(fold)]["grid"]) == (
            training["folds"][str(fold)]["threshold"],
            training["folds"][str(fold)]["calibration_feasible"],
        )
        and checkpoint["selected_thresholds"][str(fold)]
        == training["folds"][str(fold)]["threshold"]
        for fold in range(FOLDS)
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
    deletion_count = sum(d["deleted"] is not None for r in records for d in r["decisions"])
    random_count = sum(d["deleted"] is not None for r in random_records for d in r["decisions"])
    actual_v12 = {name: sha(v12_root / name) for name in EXPECTED_V12_HASHES}
    source_ok = (
        actual_v12 == EXPECTED_V12_HASHES
        and manifest["v12_hashes"] == EXPECTED_V12_HASHES
        and training["source_v12_hashes"] == EXPECTED_V12_HASHES
        and checkpoint["v12_checkpoint_sha256"]
        == EXPECTED_V12_HASHES["mquake_subject_risk_crossfit.pt"]
        and manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
        and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
        and manifest["source_training_sha256"] == sha(source_root / "training.json")
        and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
        and training["checkpoint_sha256"] == sha(root / "mquake_online_margin_crossfit.pt")
        and manifest["task_source_sha256"]
        == sha(Path(__file__).parents[1] / "tasks/train_mquake_online_margin_v14.py")
    )
    history_ok = all(
        decision["history_size"] == max(0, decision["step"] - 1)
        for row in records for decision in row["decisions"]
    )
    locks_ok = (
        summary["protocol"] == manifest["protocol"] == PROTOCOL
        and manifest["folds"] == FOLDS
        and manifest["gate_seeds"] == list(GATE_SEEDS)
        and manifest["gate_epoch"] == EPOCH
        and manifest["margin_thresholds"] == list(MARGIN_THRESHOLDS)
        and manifest["safety_macro_floor"] == SAFETY_MACRO_FLOOR
        and manifest["safety_lcb_floor"] == SAFETY_LCB_FLOOR
        and manifest["score_scale_floor"] == SCORE_SCALE_FLOOR
        and manifest["pair_cache_limit"] == PAIR_CACHE_LIMIT
        and training["pair_cache"]["limit"] == PAIR_CACHE_LIMIT
        and manifest["subject_equal_episode_listwise_loss"]
        and manifest["member_fit_iqr_normalization"]
        and manifest["online_history_uses_only_prior_scores"]
        and manifest["online_history_reset_each_episode"]
        and manifest["first_scored_action_cannot_delete"]
        and not manifest["position_feature"] and not manifest["length_feature"]
        and not manifest["subject_identity_feature"] and not manifest["future_score_feature"]
        and manifest["nested_train_calibrate_validate"]
        and not manifest["refit_after_calibration"]
        and not manifest["v10_development_used"]
        and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
        and not manifest["locomo_used"] and not summary["mquake_cf_used"]
        and not summary["longmemeval_used"] and not summary["locomo_scored"]
        and len(manifest["git_commit"]) == 40
    )
    checks = {
        "oof_metrics_cluster_candidate_comparison_and_gate": recompute_ok,
        "online_margin_threshold_selection": thresholds_ok,
        "subject_fold_assignment_and_identities": identities_ok,
        "matched_random_deletions": deletion_count == random_count
        == training["matched_random_deletions"],
        "causal_history_sizes": history_ok,
        "raw_output_hashes": training["raw_rows_sha256"]
        == {path.name: sha(path) for path in root.glob("*.jsonl")},
        "source_v12_and_checkpoint_hashes": source_ok,
        "protocol_training_online_and_access_locks": locks_ok,
    }
    return {"protocol": "mquake_online_margin_v14_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_oof_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--v12-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.v12_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
