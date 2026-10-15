"""Reproducibility and leakage audit for the controller V1 development diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_qwen_utility_record_controller_v1_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v1_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    CONFORMAL_LEVELS,
    PROTOCOL,
    RISK_PENALTIES,
    UTILITY_THRESHOLDS,
    development_split,
    evaluate,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.train_qwen_utility_record_controller_v1 import SEEDS


AUDIT_PROTOCOL = "qwen_utility_record_controller_v1_development_diagnostic_audit"


def finite_metric(metric: dict) -> bool:
    scalar_keys = (
        "accuracy",
        "mean_answer_nll",
        "unrelated_accuracy",
        "deletion_fraction",
        "binary_teacher_action_agreement",
    )
    return (
        all(math.isfinite(metric[key]) for key in scalar_keys)
        and all(0.0 <= metric[key] <= 1.0 for key in (
            "accuracy", "unrelated_accuracy", "deletion_fraction",
            "binary_teacher_action_agreement",
        ))
        and metric["queries"] > 0
        and metric["added_unrelated_errors"] >= 0
        and all(math.isfinite(value)
                for value in metric["per_mechanism_accuracy"].values())
    )


def locally_select_safe(results: dict[str, dict]) -> str | None:
    safe = [name for name, row in results.items()
            if row["added_unrelated_errors"] == 0]
    if not safe:
        return None
    return max(safe, key=lambda name: (
        results[name]["accuracy"], -results[name]["mean_answer_nll"], name
    ))


def audit(result_path: Path, feature_root: Path, label_root: Path,
          training_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    labels = []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
    all_groups = {row["source_group"] for row in labels}
    calibration = set(result["calibration_groups"])
    evaluation = set(result["evaluation_groups"])
    expected_calibration, expected_evaluation = development_split(list(all_groups))

    expected_grid = {
        f"u{utility}_p{penalty}_c{level}"
        for utility in UTILITY_THRESHOLDS
        for penalty in RISK_PENALTIES
        for level in CONFORMAL_LEVELS
    }
    controller_grid_valid = True
    controller_metrics_valid = True
    controller_selection_valid = True
    for seed in SEEDS:
        row = result["controllers"][str(seed)]
        grid = row["calibration_grid"]
        controller_grid_valid &= set(grid) == expected_grid
        controller_metrics_valid &= all(finite_metric(metric) for metric in grid.values())
        selected = locally_select_safe(grid)
        controller_selection_valid &= row["selected"] == selected
        if selected is not None:
            controller_selection_valid &= row["selected_calibration"] == grid[selected]
            controller_metrics_valid &= finite_metric(row["evaluation"])
            interval = row["gain_over_strongest_baseline_group_bootstrap_95ci"]
            controller_metrics_valid &= (
                len(interval) == 2 and all(math.isfinite(value) for value in interval)
                and interval[0] <= interval[1]
            )

    baselines = result["baselines_calibration"]
    strongest = locally_select_safe(baselines)
    baseline_metrics_valid = (
        all(finite_metric(metric) for metric in baselines.values())
        and all(finite_metric(metric)
                for metric in result["baselines_evaluation"].values())
        and strongest == result["strongest_safe_deterministic_baseline"]
    )

    rerun = evaluate(SimpleNamespace(
        feature_root=str(feature_root),
        label_root=str(label_root),
        training_root=str(training_root),
    ))
    release = result["release_gate"]
    checks = {
        "upstream_audits_locked": (
            feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and training_audit.get("protocol") == TRAINING_AUDIT_PROTOCOL
            and training_audit.get("passed") is True
            and result["feature_audit_sha256"] == sha256(feature_audit_path)
            and result["training_audit_sha256"] == sha256(training_audit_path)
            and result["label_summary_sha256"] == sha256(label_root / "summary.json")
        ),
        "development_partition_exact": (
            len(all_groups) == 64 and len(calibration) == len(evaluation) == 32
            and not calibration & evaluation
            and calibration | evaluation == all_groups
            and calibration == expected_calibration
            and evaluation == expected_evaluation
        ),
        "baseline_metrics_and_selection": baseline_metrics_valid,
        "controller_grid_complete": controller_grid_valid,
        "controller_metrics_finite": controller_metrics_valid,
        "controller_selection_reconstructed": controller_selection_valid,
        "deterministic_exact_rerun": result == rerun,
        "registered_release_gate_enforced": (
            result["formal_development_oracle_gain"] < 0.10
            and release["checks"]["oracle_headroom_at_least_point_10"] is False
            and release["passed"] is False
            and result["next_stage"] == "prospective_v2"
        ),
        "confirmation_and_external_sealed": (
            result["confirmation_loaded"] is False
            and result["external_benchmarks_loaded"] is False
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "development_groups": len(all_groups),
        "calibration_groups": len(calibration),
        "evaluation_groups": len(evaluation),
        "result_sha256": sha256(result_path),
        "next_stage": "prospective_v2" if passed else "stop",
        "terminal_status": (
            "controller_v1_development_diagnostic_audit_passed" if passed
            else "controller_v1_development_diagnostic_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.result, args.feature_root, args.label_root, args.training_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
