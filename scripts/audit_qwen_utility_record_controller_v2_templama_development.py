"""Independent audit of the frozen-controller TempLAMA development result."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_qwen_utility_record_controller_v1_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL,
    EXPECTED_TRAINING_AUDIT_SHA256,
    FROZEN_CONFIGS,
    PROTOCOL,
    evaluate,
)
from tasks.train_qwen_utility_record_controller_v1 import SEEDS


def finite_metric(value: dict) -> bool:
    scalar_keys = (
        "accuracy", "mean_answer_nll", "unrelated_accuracy",
        "deletion_fraction", "deletion_precision",
        "binary_teacher_action_agreement", "binary_teacher_record_agreement",
    )
    bounded = (
        "accuracy", "unrelated_accuracy", "deletion_fraction",
        "deletion_precision", "binary_teacher_action_agreement",
        "binary_teacher_record_agreement",
    )
    return (
        all(math.isfinite(float(value[key])) for key in scalar_keys)
        and all(0 <= value[key] <= 1 for key in bounded)
        and value["queries"] > 0
        and value["added_unrelated_errors"] >= 0
        and all(math.isfinite(float(item))
                for item in value["per_mechanism_accuracy"].values())
        and all(math.isfinite(float(item))
                for item in value["per_relation_accuracy"].values())
        and all(
            row["queries"] > 0
            and math.isfinite(float(row["accuracy"]))
            and math.isfinite(float(row["mean_answer_nll"]))
            for row in value["per_query_role"].values()
        )
    )


def deterministic_view(value: dict) -> dict:
    output = copy.deepcopy(value)
    for row in output["controllers"].values():
        row.pop("prediction_seconds", None)
    return output


def audit(result_path: Path, data_root: Path, feature_root: Path,
          label_root: Path, training_root: Path, pilot_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    rerun = evaluate(SimpleNamespace(
        data_root=str(data_root),
        feature_root=str(feature_root),
        label_root=str(label_root),
        training_root=str(training_root),
        pilot_root=str(pilot_root),
    ))
    controllers = [result["controllers"][str(seed)] for seed in SEEDS]
    baseline_name = result["strongest_safe_deterministic_baseline"]
    baseline = result["baselines"][baseline_name]
    oracle = result["safe_binary_oracle"]
    pooled = result["pooled_three_seed"]
    release = result["release_candidate_gate"]
    independent_release = {
        "safe_binary_oracle_headroom_at_least_point_03": (
            oracle["accuracy"] - baseline["accuracy"] >= .03
        ),
        "every_seed_gain_at_least_point_01": all(
            row["evaluation"]["accuracy"] - baseline["accuracy"] >= .01
            for row in controllers
        ),
        "mean_seed_gain_at_least_point_02": (
            sum(row["evaluation"]["accuracy"] - baseline["accuracy"]
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
        "upstream_audits_and_training_lock_passed": (
            feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and training_audit.get("protocol") == TRAINING_AUDIT_PROTOCOL
            and training_audit.get("passed") is True
            and sha256(training_audit_path) == EXPECTED_TRAINING_AUDIT_SHA256
        ),
    }
    interval_keys = (
        "group_bootstrap_95ci", "relation_cluster_bootstrap_95ci",
        "background_multiway_bootstrap_95ci",
        "background_heldout_group_bootstrap_95ci",
    )
    checks = {
        "upstream_audits_and_hashes_locked": (
            result["protocol"] == PROTOCOL
            and result["feature_audit_sha256"] == sha256(feature_audit_path)
            and result["label_summary_sha256"]
            == sha256(label_root / "summary.json")
            and result["training_audit_sha256"] == sha256(training_audit_path)
            and result["data_manifest_sha256"]
            == sha256(data_root / "manifest.json")
            and independent_release["upstream_audits_and_training_lock_passed"]
        ),
        "frozen_configs_exact": all(
            result["controllers"][str(seed)]["frozen_config"]
            == FROZEN_CONFIGS[seed] for seed in SEEDS
        ),
        "baseline_controller_oracle_metrics_finite": (
            all(finite_metric(row) for row in result["baselines"].values())
            and all(finite_metric(row["evaluation"]) for row in controllers)
            and finite_metric(result["majority_controller"])
            and finite_metric(oracle)
        ),
        "uncertainty_and_background_diagnostics": (
            result["groups"] == 256
            and pooled["background_heldout_groups"] == 9
            and all(
                len(pooled[key]) == 2
                and all(math.isfinite(float(value)) for value in pooled[key])
                and pooled[key][0] <= pooled[key][1]
                for key in interval_keys
            )
        ),
        "deletion_matched_random_complete": all(
            row["deletion_matched_random"]["draws"] == 20
            and math.isfinite(row["deletion_matched_random"]["accuracy_mean"])
            for row in controllers
        ),
        "release_gate_independently_reconstructed": (
            release["checks"] == independent_release
            and release["passed_pending_independent_audit"]
            == all(independent_release.values())
        ),
        "deterministic_exact_rerun_except_latency": (
            deterministic_view(result) == deterministic_view(rerun)
        ),
        "heldouts_sealed": (
            result["templama_test_loaded"] is False
            and result["v1_confirmation_loaded"] is False
            and result["external_benchmarks_loaded"] is False
            and result["next_stage"] == "independent_development_result_audit"
        ),
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
        "development_groups": result["groups"],
        "strongest_safe_deterministic_baseline": baseline_name,
        "mean_controller_gain": pooled["mean_gain"],
        "pooled_group_bootstrap_95ci": pooled["group_bootstrap_95ci"],
        "mean_oracle_gap_closure": pooled["mean_oracle_gap_closure"],
        "next_stage": (
            "templama_cross_source_test_construction"
            if release_passed else "stop_v2_without_opening_test"
        ),
        "terminal_status": (
            "templama_development_release_audit_passed"
            if release_passed else
            "templama_development_audit_passed_release_failed"
            if audit_passed else
            "templama_development_result_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.result, args.data_root, args.feature_root, args.label_root,
        args.training_root, args.pilot_root,
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
