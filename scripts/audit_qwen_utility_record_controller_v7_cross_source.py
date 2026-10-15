"""Independent checkpoint/result audit for V7 cross-source training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    SEEDS,
    predict_state,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    AUDIT_PROTOCOL,
    PROTOCOL,
    add_arguments,
    development_gate,
    domain_metric,
    load_inputs,
)


def close(left, right, tolerance=1e-8):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
    return left == right


def audit(args) -> dict:
    root = Path(args.training_root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    train_domains, development, audit_hashes = load_inputs(args)
    reconstructed = []
    checkpoint_checks = []
    for seed in SEEDS:
        seed_summary = next(row for row in summary["seeds"] if row["seed"] == seed)
        checkpoint_path = root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        predictions = {
            name: predict_state(
                checkpoint["state_dict"], domain["examples"],
                checkpoint["means"], checkpoint["stds"],
            ) for name, domain in development.items()
        }
        domains = {
            name: domain_metric(
                domain, predictions[name], checkpoint["selected_config"]
            ) for name, domain in development.items()
        }
        reconstructed_row = {
            **seed_summary,
            "selected_metric": {
                "config": checkpoint["selected_config"], "domains": domains,
            },
        }
        reconstructed.append(reconstructed_row)
        checkpoint_checks.append(
            checkpoint.get("protocol") == PROTOCOL
            and checkpoint.get("seed") == seed
            and tuple(checkpoint.get("feature_names", ())) == FEATURE_NAMES
            and checkpoint.get("input_dim") == len(FEATURE_NAMES)
            and checkpoint.get("upstream_audit_hashes") == audit_hashes
            and checkpoint.get("training_domains") == sorted(train_domains)
            and checkpoint.get("selected_config") == seed_summary["selected_config"]
            and sha256(checkpoint_path) == seed_summary["checkpoint_sha256"]
            and close(
                reconstructed_row["selected_metric"],
                seed_summary["selected_metric"],
            )
        )
    gate = development_gate(reconstructed, development, True)
    checks = {
        "protocol_plan_and_summary_passed": (
            summary.get("protocol") == PROTOCOL
            and summary.get("plan")
            == "QWEN_UTILITY_RECORD_CONTROLLER_V7_CROSS_SOURCE_PLAN.md"
            and summary.get("passed")
        ),
        "upstream_audit_hashes_reconstructed": (
            summary.get("upstream_audit_hashes") == audit_hashes
        ),
        "all_checkpoint_hashes_and_states_locked": all(checkpoint_checks),
        "selected_domain_metrics_reconstructed": all(
            close(a["selected_metric"], b["selected_metric"])
            for a, b in zip(reconstructed, summary["seeds"])
        ),
        "development_gate_reconstructed": close(
            gate, summary["development_gate"]
        ),
        "finite_selected_metrics": all(
            math.isfinite(value)
            for row in reconstructed
            for domain in row["selected_metric"]["domains"].values()
            for value in (
                domain["accuracy"], domain["mean_answer_nll"],
                domain["gain_over_lifecycle_physics"],
                domain["oracle_gap_closure"],
            )
        ),
        "heldout_sources_sealed": (
            not summary["templama_test_loaded"]
            and not summary["lama_confirmation_loaded"]
            and not summary["longmemeval_loaded"]
            and not summary["locomo_loaded"]
        ),
    }
    passed = all(checks.values())
    release = passed and gate["passed_pending_independent_audit"]
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "development_checks": gate["checks"],
        "development_release_passed": release,
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": {
            str(seed): sha256(root / f"seed_{seed}" / "controller.pt")
            for seed in SEEDS
        },
        "mean_gain": gate["mean_gain"],
        "mean_oracle_gap_closure": gate["mean_oracle_gap_closure"],
        "next_stage": (
            "register_v7_locomo_confirmation" if release
            else "stop_v7_cross_source_iteration"
        ),
        "terminal_status": (
            "v7_cross_source_training_audit_passed_released" if release
            else "v7_cross_source_training_audit_passed_gate_failed" if passed
            else "v7_cross_source_training_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
