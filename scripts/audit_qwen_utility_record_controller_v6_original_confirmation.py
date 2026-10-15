"""Independent exact-rerun audit of V6 original-source confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v6_original_confirmation import (
    AUDIT_PROTOCOL,
    PROTOCOL,
    evaluate,
)
from tasks.train_qwen_utility_record_controller_v5_residual import SEEDS


def finite_metric(value: dict) -> bool:
    return (
        value["queries"] > 0
        and 0 <= value["accuracy"] <= 1
        and 0 <= value["unrelated_accuracy"] <= 1
        and 0 <= value["deletion_fraction"] <= 1
        and math.isfinite(value["mean_answer_nll"])
        and value["added_unrelated_errors"] >= 0
    )


def audit(result_path: Path, data_root: Path, feature_root: Path,
          label_root: Path, training_root: Path,
          release_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    rerun = evaluate(SimpleNamespace(
        data_root=str(data_root), feature_root=str(feature_root),
        label_root=str(label_root), training_root=str(training_root),
        release_root=str(release_root),
    ))
    controllers = [result["controllers"][str(seed)] for seed in SEEDS]
    physics = result["baselines"][result["release_baseline"]]
    oracle = result["safe_single_fact_key_oracle"]
    pooled = result["pooled_three_seed"]
    gate = result["original_source_confirmation_gate"]
    independent_gate = {
        "every_seed_gain_above_zero": all(
            row["evaluation"]["accuracy"] - physics["accuracy"] > 0
            for row in controllers
        ),
        "mean_seed_gain_at_least_point_02": (
            sum(row["evaluation"]["accuracy"] - physics["accuracy"]
                for row in controllers) / len(controllers) >= .02
        ),
        "pooled_group_bootstrap_lower_above_zero": (
            pooled["group_bootstrap_95ci"][0] > 0
        ),
        "relation_cluster_bootstrap_lower_above_zero": (
            pooled["relation_cluster_bootstrap_95ci"][0] > 0
        ),
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0
            for row in controllers
        ),
        "mean_oracle_gap_closure_at_least_point_30": (
            pooled["mean_oracle_gap_closure"] >= .30
        ),
        "every_mechanism_nonnegative": (
            min(pooled["mean_per_mechanism_gain"].values()) >= 0
        ),
        "upstream_audits_passed": all(
            json.loads(path.read_text()).get("passed")
            for path in (
                data_root / "data_audit.json",
                label_root / "labels_audit.json",
                feature_root / "features_audit.json",
                training_root / "training_audit.json",
                release_root / "result_audit.json",
            )
        ),
    }
    interval_keys = (
        "group_bootstrap_95ci", "relation_cluster_bootstrap_95ci",
    )
    checks = {
        "protocol_plan_and_partition": (
            result["protocol"] == PROTOCOL
            and result["plan"]
            == "QWEN_UTILITY_RECORD_CONTROLLER_V6_ORIGINAL_SOURCE_CONFIRMATION_PLAN.md"
            and result["partition"] == "lama_pararel_original_source_confirmation"
            and result["groups"] == 64
        ),
        "upstream_hashes_locked": (
            result["data_audit_sha256"]
            == sha256(data_root / "data_audit.json")
            and result["label_audit_sha256"]
            == sha256(label_root / "labels_audit.json")
            and result["feature_audit_sha256"]
            == sha256(feature_root / "features_audit.json")
            and result["training_audit_sha256"]
            == sha256(training_root / "training_audit.json")
            and result["templama_release_result_sha256"]
            == sha256(release_root / "result.json")
            and result["templama_release_audit_sha256"]
            == sha256(release_root / "result_audit.json")
            and independent_gate["upstream_audits_passed"]
        ),
        "metrics_and_uncertainty_finite": (
            all(finite_metric(row) for row in result["baselines"].values())
            and all(finite_metric(row["evaluation"]) for row in controllers)
            and finite_metric(result["majority_controller"])
            and finite_metric(oracle)
            and all(
                len(pooled[key]) == 2
                and pooled[key][0] <= pooled[key][1]
                and all(math.isfinite(value) for value in pooled[key])
                for key in interval_keys
            )
        ),
        "one_action_and_matched_random_controls": all(
            0 <= row["evaluation"]["residual_deletion_precision"] <= 1
            and 0 <= row["evaluation"]["residual_deletion_fraction"] <= 1
            and row["evaluation"]["residual_deletions"] >= 0
            and row["fact_key_deletion_matched_random"]["draws"] == 20
            and math.isfinite(row["fact_key_deletion_matched_random"]["accuracy_mean"])
            for row in controllers
        ),
        "confirmation_gate_independently_reconstructed": (
            gate["checks"] == independent_gate
            and gate["passed_pending_independent_audit"]
            == all(independent_gate.values())
        ),
        "deterministic_exact_rerun": copy.deepcopy(result) == copy.deepcopy(rerun),
        "original_confirmation_open_external_sealed": (
            result["templama_test_loaded"] is True
            and result["counterfactual_targets_available_to_controller"] is False
            and result["parameters_frozen_before_original_source_outcomes"] is True
            and result["v1_confirmation_loaded"] is True
            and result["external_benchmarks_loaded"] is False
            and result["next_stage"]
            == "independent_v6_original_source_result_audit"
        ),
    }
    audit_passed = all(checks.values())
    confirmation_passed = audit_passed and all(independent_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": audit_passed,
        "confirmation_checks": independent_gate,
        "confirmation_passed": confirmation_passed,
        "result_sha256": sha256(result_path),
        "mean_controller_gain": pooled["mean_gain"],
        "pooled_group_bootstrap_95ci": pooled["group_bootstrap_95ci"],
        "mean_oracle_gap_closure": pooled["mean_oracle_gap_closure"],
        "next_stage": (
            "v6_external_long_memory_evaluation"
            if confirmation_passed else "stop_v6_original_source_without_repair"
        ),
        "terminal_status": (
            "v6_original_source_confirmation_audit_passed"
            if confirmation_passed else
            "v6_original_source_result_audit_passed_confirmation_failed"
            if audit_passed else "v6_original_source_result_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.result, args.data_root, args.feature_root, args.label_root,
        args.training_root, args.release_root,
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
