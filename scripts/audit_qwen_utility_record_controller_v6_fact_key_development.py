"""Independent exact-rerun audit of V6 fact-key development evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
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
          label_root: Path, training_root: Path, v5_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    rerun = evaluate(SimpleNamespace(
        data_root=str(data_root), feature_root=str(feature_root),
        label_root=str(label_root), training_root=str(training_root),
        v5_root=str(v5_root),
    ))
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    training_summary_path = training_root / "summary.json"
    v5_result_path = v5_root / "result.json"
    v5_audit_path = v5_root / "result_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    v5_audit = json.loads(v5_audit_path.read_text())
    controllers = [result["controllers"][str(seed)] for seed in SEEDS]
    physics = result["lifecycle_physics"]
    oracle = result["safe_single_fact_key_oracle"]
    pooled = result["pooled_three_seed"]
    release = result["release_candidate_gate"]
    independent_release = {
        "safe_single_fact_key_oracle_headroom_at_least_point_03": (
            oracle["accuracy"] - physics["accuracy"] >= .03
        ),
        "every_seed_gain_at_least_point_01": all(
            row["evaluation"]["accuracy"] - physics["accuracy"] >= .01
            for row in controllers
        ),
        "mean_seed_gain_at_least_point_02": (
            sum(row["evaluation"]["accuracy"] - physics["accuracy"]
                for row in controllers) / len(controllers) >= .02
        ),
        "pooled_group_bootstrap_lower_above_zero": (
            pooled["group_bootstrap_95ci"][0] > 0
        ),
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0
            for row in controllers
        ),
        "mean_oracle_gap_closure_at_least_point_40": (
            pooled["mean_oracle_gap_closure"] >= .40
        ),
        "overwrite_and_stale_mean_gains_nonnegative": (
            pooled["mean_per_mechanism_gain"]["semantic_overwrite"] >= 0
            and pooled["mean_per_mechanism_gain"]["stale_repetition"] >= 0
        ),
        "relation_robustness": (
            pooled["macro_relation_gain"] > 0
            and pooled["nonnegative_relations"] >= 8
            and pooled["worst_relation_gain"] >= -.02
        ),
        "upstream_audits_passed": (
            feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and training_audit.get("protocol") == TRAINING_AUDIT_PROTOCOL
            and training_audit.get("passed") is True
            and training_audit.get("summary_sha256")
            == sha256(training_summary_path)
            and v5_audit.get("passed") is True
            and v5_audit.get("release_passed") is False
            and v5_audit.get("result_sha256") == sha256(v5_result_path)
        ),
    }
    interval_keys = (
        "group_bootstrap_95ci", "relation_cluster_bootstrap_95ci",
        "background_multiway_bootstrap_95ci",
        "background_heldout_group_bootstrap_95ci",
    )
    localization = result["collateral_localization"]
    checks = {
        "protocol_plan_and_hashes_locked": (
            result["protocol"] == PROTOCOL
            and result["plan"]
            == "QWEN_UTILITY_RECORD_CONTROLLER_V6_FACT_KEY_SAFETY_PLAN.md"
            and result["feature_audit_sha256"] == sha256(feature_audit_path)
            and result["training_audit_sha256"] == sha256(training_audit_path)
            and result["training_summary_sha256"] == sha256(training_summary_path)
            and result["v5_result_sha256"] == sha256(v5_result_path)
            and result["v5_audit_sha256"] == sha256(v5_audit_path)
            and result["label_summary_sha256"]
            == sha256(label_root / "summary.json")
            and result["data_manifest_sha256"]
            == sha256(data_root / "manifest.json")
        ),
        "fact_key_projection_and_error_localization": (
            result["semantic_key_candidates"] > 0
            and result["semantic_key_candidates"] <= result["candidate_records"]
            and localization["all_v5_error_boundaries_cross_fact_key"] is True
            and all(len(localization["v5_subject_only"][str(seed)]) > 0
                    for seed in SEEDS)
            and all(
                sum(detail["added_unrelated_errors"] for detail in
                    localization["v6_fact_key"][str(seed)])
                == controllers[index]["evaluation"]["added_unrelated_errors"]
                for index, seed in enumerate(SEEDS)
            )
        ),
        "metrics_and_uncertainty_finite": (
            finite_metric(physics)
            and finite_metric(oracle)
            and finite_metric(result["majority_controller"])
            and all(finite_metric(row["evaluation"]) for row in controllers)
            and all(
                len(pooled[key]) == 2
                and pooled[key][0] <= pooled[key][1]
                and all(math.isfinite(value) for value in pooled[key])
                for key in interval_keys
            )
        ),
        "matched_random_controls_complete": all(
            row["fact_key_deletion_matched_random"]["draws"] == 20
            and math.isfinite(row["fact_key_deletion_matched_random"]["accuracy_mean"])
            for row in controllers
        ),
        "release_gate_independently_reconstructed": (
            release["checks"] == independent_release
            and release["passed_pending_independent_audit"]
            == all(independent_release.values())
        ),
        "deterministic_exact_rerun": copy.deepcopy(result) == copy.deepcopy(rerun),
        "heldouts_sealed": (
            result["partition"] == "templama_validation_full_development_adaptive"
            and result["development_adaptive"] is True
            and result["counterfactual_targets_available_to_controller"] is False
            and result["templama_test_loaded"] is False
            and result["external_benchmarks_loaded"] is False
            and result["next_stage"] == "independent_v6_development_result_audit"
        ),
        "upstream_audits_passed": independent_release["upstream_audits_passed"],
    }
    audit_passed = all(checks.values())
    release_passed = audit_passed and all(independent_release.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": audit_passed,
        "release_checks": independent_release,
        "release_passed": release_passed,
        "result_sha256": sha256(result_path),
        "mean_controller_gain": pooled["mean_gain"],
        "pooled_group_bootstrap_95ci": pooled["group_bootstrap_95ci"],
        "mean_oracle_gap_closure": pooled["mean_oracle_gap_closure"],
        "next_stage": (
            "templama_v6_test_construction"
            if release_passed else "stop_v6_without_opening_test"
        ),
        "terminal_status": (
            "templama_v6_development_release_audit_passed"
            if release_passed else
            "templama_v6_development_audit_passed_release_failed"
            if audit_passed else
            "templama_v6_development_result_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--v5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.result, args.data_root, args.feature_root, args.label_root,
        args.training_root, args.v5_root,
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
