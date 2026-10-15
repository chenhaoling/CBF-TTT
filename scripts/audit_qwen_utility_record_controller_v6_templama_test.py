"""Independent exact-rerun audit of one-time V6 TempLAMA test confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v6_templama_test import (
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
          development_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    rerun = evaluate(SimpleNamespace(
        data_root=str(data_root), feature_root=str(feature_root),
        label_root=str(label_root), training_root=str(training_root),
        development_root=str(development_root),
    ))
    controllers = [result["controllers"][str(seed)] for seed in SEEDS]
    physics = result["baselines"][result["release_baseline"]]
    oracle = result["safe_single_fact_key_oracle"]
    pooled = result["pooled_three_seed"]
    gate = result["confirmation_gate"]
    independent_gate = {
        "every_seed_gain_above_zero": all(
            row["evaluation"]["accuracy"] - physics["accuracy"] > 0
            for row in controllers
        ),
        "mean_seed_gain_at_least_point_015": (
            sum(row["evaluation"]["accuracy"] - physics["accuracy"]
                for row in controllers) / len(controllers) >= .015
        ),
        "pooled_group_bootstrap_lower_above_zero": (
            pooled["group_bootstrap_95ci"][0] > 0
        ),
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0
            for row in controllers
        ),
        "mean_oracle_gap_closure_at_least_point_30": (
            pooled["mean_oracle_gap_closure"] >= .30
        ),
        "no_mechanism_below_minus_point_02": (
            min(pooled["mean_per_mechanism_gain"].values()) >= -.02
        ),
        "upstream_audits_passed": all(
            json.loads(path.read_text()).get("passed")
            for path in (
                data_root / "data_audit.json",
                label_root / "test_labels_audit.json",
                feature_root / "test_feature_audit.json",
                training_root / "training_audit.json",
                development_root / "result_audit.json",
            )
        ),
    }
    interval_keys = (
        "group_bootstrap_95ci", "relation_cluster_bootstrap_95ci",
        "background_multiway_bootstrap_95ci",
        "background_heldout_group_bootstrap_95ci",
    )
    checks = {
        "protocol_plan_and_partition": (
            result["protocol"] == PROTOCOL
            and result["plan"]
            == "QWEN_UTILITY_RECORD_CONTROLLER_V6_FACT_KEY_SAFETY_PLAN.md"
            and result["partition"] == "templama_official_test_confirmation"
            and result["groups"] == 256
        ),
        "upstream_hashes_locked": (
            result["data_audit_sha256"]
            == sha256(data_root / "data_audit.json")
            and result["label_audit_sha256"]
            == sha256(label_root / "test_labels_audit.json")
            and result["feature_audit_sha256"]
            == sha256(feature_root / "test_feature_audit.json")
            and result["training_audit_sha256"]
            == sha256(training_root / "training_audit.json")
            and result["development_result_sha256"]
            == sha256(development_root / "result.json")
            and result["development_audit_sha256"]
            == sha256(development_root / "result_audit.json")
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
        "test_open_other_confirmations_sealed": (
            result["templama_test_loaded"] is True
            and result["counterfactual_targets_available_to_controller"] is False
            and result["test_parameters_frozen_before_label_outcomes"] is True
            and result["v1_confirmation_loaded"] is False
            and result["external_benchmarks_loaded"] is False
            and result["next_stage"]
            == "independent_v6_templama_test_result_audit"
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
            "v6_original_source_confirmation_and_external_evaluation"
            if confirmation_passed else "stop_v6_without_test_repair"
        ),
        "terminal_status": (
            "templama_v6_test_confirmation_audit_passed"
            if confirmation_passed else
            "templama_v6_test_result_audit_passed_confirmation_failed"
            if audit_passed else "templama_v6_test_result_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.result, args.data_root, args.feature_root, args.label_root,
        args.training_root, args.development_root,
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
