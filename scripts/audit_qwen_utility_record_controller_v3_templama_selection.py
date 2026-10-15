"""Independent audit of V3 train-only locality calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_qwen_utility_record_controller_v1_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v3_templama_pilot_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    CONFORMAL_LEVELS,
    RISK_PENALTIES,
    UTILITY_THRESHOLDS,
)
from tasks.select_qwen_utility_record_controller_v3_templama import (
    AUDIT_PROTOCOL,
    EXPECTED_TRAINING_AUDIT_SHA256,
    LOCALITY_RULES,
    PROTOCOL,
    select,
)
from tasks.train_qwen_utility_record_controller_v1 import SEEDS


def locally_select(candidates: dict[str, dict]) -> str | None:
    eligible = [name for name, value in candidates.items()
                if value["added_unrelated_errors"] == 0]
    if not eligible:
        return None
    rule_order = {name: index for index, name in enumerate(LOCALITY_RULES)}
    return max(eligible, key=lambda name: (
        candidates[name]["accuracy"],
        candidates[name]["deletion_precision"],
        -candidates[name]["deletion_fraction"],
        -rule_order[candidates[name]["config"]["locality_rule"]],
        candidates[name]["config"]["utility_threshold"],
        candidates[name]["config"]["risk_penalty"],
        -candidates[name]["config"]["conformal_level"],
        name,
    ))


def audit(result_path: Path, feature_root: Path, label_root: Path,
          training_root: Path) -> dict:
    result = json.loads(result_path.read_text())
    feature_audit_path = feature_root / "pilot_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    expected_count = (len(UTILITY_THRESHOLDS) * len(RISK_PENALTIES)
                      * len(CONFORMAL_LEVELS) * len(LOCALITY_RULES))
    grid_valid = True
    selection_valid = True
    metrics_valid = True
    for seed in SEEDS:
        row = result["controllers"][str(seed)]
        candidates = row["candidates"]
        grid_valid &= len(candidates) == expected_count
        for value in candidates.values():
            config = value["config"]
            grid_valid &= (
                config["utility_threshold"] in UTILITY_THRESHOLDS
                and config["risk_penalty"] in RISK_PENALTIES
                and config["conformal_level"] in CONFORMAL_LEVELS
                and config["locality_rule"] in LOCALITY_RULES
                and close(
                    config["risk_threshold"],
                    row["conformal_risk_thresholds"][
                        str(config["conformal_level"])
                    ],
                )
            )
            metrics_valid &= all(math.isfinite(float(value[key])) for key in (
                "accuracy", "mean_answer_nll", "deletion_fraction",
                "deletion_precision", "unrelated_accuracy",
            ))
        selected = locally_select(candidates)
        selection_valid &= row["selected"] == selected
        if selected is not None:
            selection_valid &= row["selected_config"] == candidates[selected]["config"]
            selection_valid &= row["selected_metric"] == candidates[selected]
            selection_valid &= candidates[selected]["added_unrelated_errors"] == 0
    rerun = select(SimpleNamespace(
        feature_root=str(feature_root),
        label_root=str(label_root),
        training_root=str(training_root),
    ))
    checks = {
        "upstream_audits_locked": (
            result["protocol"] == PROTOCOL
            and feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and training_audit.get("protocol") == TRAINING_AUDIT_PROTOCOL
            and training_audit.get("passed") is True
            and result["feature_audit_sha256"] == sha256(feature_audit_path)
            and result["label_summary_sha256"]
            == sha256(label_root / "summary.json")
            and result["training_audit_sha256"] == sha256(training_audit_path)
            and sha256(training_audit_path) == EXPECTED_TRAINING_AUDIT_SHA256
        ),
        "complete_registered_grid": grid_valid,
        "finite_candidate_metrics": metrics_valid,
        "independent_safe_selection": selection_valid,
        "deterministic_exact_rerun": result == rerun,
        "train_only_and_heldouts_sealed": (
            result["partition"] == "templama_train_resource_pilot"
            and result["source_groups"] == 72
            and result["templama_validation_loaded"] is False
            and result["templama_test_loaded"] is False
            and result["v1_confirmation_loaded"] is False
            and result["external_benchmarks_loaded"] is False
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "result_sha256": sha256(result_path),
        "selected": {
            str(seed): result["controllers"][str(seed)]["selected"]
            for seed in SEEDS
        },
        "next_stage": (
            "templama_v3_validation_evaluation" if passed else "stop_v3"
        ),
        "terminal_status": (
            "templama_v3_train_selection_audit_passed" if passed
            else "templama_v3_train_selection_audit_failed"
        ),
    }


def close(left, right, tolerance=1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.result, args.feature_root, args.label_root, args.training_root
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
