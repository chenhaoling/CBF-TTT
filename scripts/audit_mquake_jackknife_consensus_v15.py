"""Independent artifact and causal replay audit for MQuAKE V15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.evaluate_mquake_ensemble_calibration_v13 import EXPECTED_V12_HASHES
from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.train_mquake_jackknife_consensus_v15 import (
    FPR_TARGETS,
    INNER_MEMBERS,
    PROTOCOL,
    QUORUM,
)
from tasks.train_mquake_online_margin_v14 import (
    EPOCH,
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
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def selected_inner(grid):
    feasible = [row for row in grid if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            row["metric"]["stale_deletion"], row["cluster"]["safety_lcb95"],
            -row["target_fpr"],
        ))
    else:
        selected = max(grid, key=lambda row: (
            row["cluster"]["safety_lcb95"], row["cluster"]["safety_macro"],
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            -row["target_fpr"],
        ))
    return selected["threshold"], selected["target_fpr"], bool(feasible)


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
    checkpoint = torch.load(root / "mquake_jackknife_consensus_crossfit.pt",
                            map_location="cpu", weights_only=False)
    metric = aggregate(records); cluster = cluster_metrics(records)
    candidate = candidate_metrics(candidates)
    comparison = paired_bootstrap(records, random_records, 42192)
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
        and summary["jackknife_consensus_supported"] == (pair_checks and safety_checks)
    )

    thresholds_ok = True
    for outer in range(FOLDS):
        details = training["folds"][str(outer)]["inner"]
        expected_inner = {str(value) for value in range(FOLDS) if value != outer}
        thresholds_ok &= set(details) == expected_inner
        thresholds_ok &= set(checkpoint["inner_gate_members"][str(outer)]) == expected_inner
        thresholds_ok &= set(checkpoint["selected_thresholds"][str(outer)]) == expected_inner
        for inner, row in details.items():
            selected = selected_inner(row["grid"])
            thresholds_ok &= selected == (
                row["threshold"], row["target_fpr"], row["calibration_feasible"]
            )
            thresholds_ok &= checkpoint["selected_thresholds"][str(outer)][inner] \
                == row["threshold"]
            thresholds_ok &= len(checkpoint["inner_gate_members"][str(outer)][inner]) \
                == INNER_MEMBERS

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
    replay_ok = True
    for record in records:
        outer = str(checkpoint["folds"][record["subject"]])
        expected_thresholds = [
            training["folds"][outer]["inner"][inner]["threshold"]
            for inner in sorted(training["folds"][outer]["inner"], key=int)
        ]
        for decision in record["decisions"]:
            expected_history = max(0, decision["step"] - 1)
            empty = decision["step"] == 0
            replay_ok &= decision["quorum"] == QUORUM
            replay_ok &= decision["vote_count"] == sum(decision["votes"])
            replay_ok &= all(value == expected_history
                             for value in decision["history_sizes"])
            replay_ok &= len(decision["history_sizes"]) == len(expected_thresholds)
            replay_ok &= decision["thresholds"] == expected_thresholds
            replay_ok &= len(decision["votes"]) == (0 if empty else len(expected_thresholds))
            replay_ok &= len(decision["scores"]) == (0 if empty else len(expected_thresholds))
            replay_ok &= len(decision["margins"]) == (0 if empty else len(expected_thresholds))
            replay_ok &= ((decision["deleted"] is not None)
                          == (decision["vote_count"] >= QUORUM))

    deletion_count = sum(decision["deleted"] is not None
                         for row in records for decision in row["decisions"])
    random_count = sum(decision["deleted"] is not None
                       for row in random_records for decision in row["decisions"])
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
        and training["checkpoint_sha256"]
        == sha(root / "mquake_jackknife_consensus_crossfit.pt")
        and manifest["task_source_sha256"] == sha(
            Path(__file__).parents[1] / "tasks/train_mquake_jackknife_consensus_v15.py"
        )
    )
    locks_ok = (
        summary["protocol"] == manifest["protocol"] == PROTOCOL
        and manifest["folds"] == FOLDS
        and manifest["inner_members"] == INNER_MEMBERS
        and manifest["quorum"] == QUORUM
        and manifest["gate_epoch"] == EPOCH
        and manifest["fpr_targets"] == list(FPR_TARGETS)
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
        and manifest["nested_subject_jackknife"]
        and not manifest["inner_models_refit_after_calibration"]
        and not manifest["position_feature"] and not manifest["length_feature"]
        and not manifest["subject_identity_feature"]
        and not manifest["relation_identity_feature"]
        and not manifest["future_score_feature"]
        and not manifest["v10_development_used"]
        and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
        and not manifest["locomo_used"] and not summary["mquake_cf_used"]
        and not summary["longmemeval_used"] and not summary["locomo_scored"]
        and len(manifest["git_commit"]) == 40
    )
    checks = {
        "oof_metrics_cluster_candidate_comparison_and_gate": recompute_ok,
        "nested_inner_threshold_selection_and_checkpoint": thresholds_ok,
        "subject_fold_assignment_and_identities": identities_ok,
        "three_of_four_exact_causal_replay": replay_ok,
        "matched_random_deletions": deletion_count == random_count
        == training["matched_random_deletions"],
        "raw_output_hashes": training["raw_rows_sha256"]
        == {path.name: sha(path) for path in root.glob("*.jsonl")},
        "source_v12_and_checkpoint_hashes": source_ok,
        "protocol_training_and_access_locks": locks_ok,
    }
    return {
        "protocol": "mquake_jackknife_consensus_v15_audit",
        "audit_passed": all(checks.values()), "checks": checks,
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
