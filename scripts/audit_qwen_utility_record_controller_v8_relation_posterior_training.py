"""Independent checkpoint/result audit for V8 relation-posterior training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features import (
    FEATURE_NAMES,
)
from tasks.train_qwen_utility_record_controller_v5_residual import SEEDS
from tasks.train_qwen_utility_record_controller_v7_cross_source import domain_metric
from tasks.train_qwen_utility_record_controller_v8_relation_posterior import (
    AUDIT_PROTOCOL,
    PROTOCOL,
    add_arguments,
    extended_gate,
    head_metrics,
    load_inputs,
    predict_state,
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
    train_domains, development, hashes = load_inputs(args)
    v7_diagnostic = json.loads(Path(args.v7_diagnostic).read_text())
    reconstructed, checkpoint_checks = [], []
    for seed in SEEDS:
        observed = next(row for row in summary["seeds"] if row["seed"] == seed)
        path = root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(path)
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
        ranking = {
            name: head_metrics(domain["examples"], predictions[name])
            for name, domain in development.items()
        }
        row = {
            **observed,
            "selected_metric": {
                "config": checkpoint["selected_config"], "domains": domains,
            },
            "head_metrics": ranking,
        }
        reconstructed.append(row)
        checkpoint_checks.append(
            checkpoint.get("protocol") == PROTOCOL
            and checkpoint.get("seed") == seed
            and tuple(checkpoint.get("feature_names", ())) == FEATURE_NAMES
            and checkpoint.get("input_dim") == len(FEATURE_NAMES)
            and checkpoint.get("upstream_hashes") == hashes
            and checkpoint.get("selected_config") == observed["selected_config"]
            and sha256(path) == observed["checkpoint_sha256"]
            and close(row["selected_metric"], observed["selected_metric"])
            and close(row["head_metrics"], observed["head_metrics"])
        )
    gate = extended_gate(reconstructed, development, v7_diagnostic)
    checks = {
        "protocol_plan_and_summary_passed": (
            summary.get("protocol") == PROTOCOL
            and summary.get("plan")
            == "QWEN_UTILITY_RECORD_CONTROLLER_V8_RELATION_POSTERIOR_PLAN.md"
            and summary.get("passed")
        ),
        "upstream_hashes_reconstructed": summary.get("upstream_hashes") == hashes,
        "all_checkpoint_hashes_and_states_locked": all(checkpoint_checks),
        "selected_metrics_and_head_diagnostics_reconstructed": all(
            close(a["selected_metric"], b["selected_metric"])
            and close(a["head_metrics"], b["head_metrics"])
            for a, b in zip(reconstructed, summary["seeds"])
        ),
        "development_gate_reconstructed": close(gate, summary["development_gate"]),
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
        "heldout_sources_sealed": all(
            not summary[name] for name in (
                "templama_test_loaded", "lama_confirmation_loaded",
                "longmemeval_loaded", "locomo_loaded",
            )
        ),
    }
    passed = all(checks.values())
    release = passed and gate["passed_pending_independent_audit"]
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": passed,
        "checks": checks,
        "development_checks": gate["checks"],
        "development_release_passed": release,
        "mean_gain": gate["mean_gain"],
        "mean_oracle_gap_closure": gate["mean_oracle_gap_closure"],
        "lama_gain_over_v7": gate["lama_gain_over_v7"],
        "lama_benefit_average_precision_gain": gate[
            "lama_benefit_average_precision_gain"
        ],
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": {
            str(seed): sha256(root / f"seed_{seed}" / "controller.pt")
            for seed in SEEDS
        },
        "next_stage": (
            "register_v8_locomo_confirmation" if release
            else "stop_v8_relation_posterior_iteration"
        ),
        "terminal_status": (
            "v8_relation_posterior_audit_passed_released" if release
            else "v8_relation_posterior_audit_passed_gate_failed" if passed
            else "v8_relation_posterior_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(); add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
