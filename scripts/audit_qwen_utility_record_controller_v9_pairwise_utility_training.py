"""Independent checkpoint/result audit for V9 pairwise utility training."""

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
from tasks.train_qwen_utility_record_controller_v5_residual import SEEDS
from tasks.train_qwen_utility_record_controller_v7_cross_source import domain_metric
from tasks.train_qwen_utility_record_controller_v9_pairwise_utility import (
    ARMS,
    AUDIT_PROTOCOL,
    PROJECTION_DIM,
    PROTOCOL,
    add_arguments,
    load_inputs,
    make_model,
    predict_state,
    prediction_metrics,
    v9_gate,
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


def risk_pair_isolated(checkpoint: dict) -> bool:
    import torch

    model = make_model(
        len(FEATURE_NAMES), checkpoint["embedding_dim"],
        ARMS[checkpoint["arm"]]["use_pair"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    base = torch.zeros((3, len(FEATURE_NAMES)), dtype=torch.float32)
    first = torch.zeros((3, 2, checkpoint["embedding_dim"]), dtype=torch.float32)
    second = torch.randn((3, 2, checkpoint["embedding_dim"]),
                         generator=torch.Generator().manual_seed(24117))
    with torch.inference_mode():
        risk_first = model(base, first)[1]
        risk_second = model(base, second)[1]
    return torch.equal(risk_first, risk_second)


def audit(args) -> dict:
    root = Path(args.training_root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    _, development, _, hashes = load_inputs(args)
    v8_audit = json.loads(Path(args.v8_training_audit).read_text())
    reconstructed = {}; checkpoint_checks = []; isolation_checks = []
    checkpoint_hashes = {}
    for arm in ARMS:
        reconstructed[arm] = []
        checkpoint_hashes[arm] = {}
        for seed in SEEDS:
            observed = next(
                row for row in summary["arms"][arm] if row["seed"] == seed
            )
            path = root / arm / f"seed_{seed}" / "controller.pt"
            checkpoint = load_checkpoint(path)
            checkpoint_hashes[arm][str(seed)] = sha256(path)
            predictions = {
                name: predict_state(
                    checkpoint["state_dict"], domain["examples"],
                    checkpoint["means"], checkpoint["stds"],
                    checkpoint["embedding_dim"], arm,
                ) for name, domain in development.items()
            }
            domains = {
                name: domain_metric(
                    domain, predictions[name], checkpoint["selected_config"]
                ) for name, domain in development.items()
            }
            metrics = {
                name: prediction_metrics(domain["examples"], predictions[name])
                for name, domain in development.items()
            }
            row = {
                **observed,
                "selected_metric": {
                    "config": checkpoint["selected_config"], "domains": domains,
                },
                "prediction_metrics": metrics,
            }
            reconstructed[arm].append(row)
            checkpoint_checks.append(
                checkpoint.get("protocol") == PROTOCOL
                and checkpoint.get("arm") == arm
                and checkpoint.get("seed") == seed
                and tuple(checkpoint.get("feature_names", ())) == FEATURE_NAMES
                and checkpoint.get("input_dim") == len(FEATURE_NAMES)
                and checkpoint.get("projection_dim") == PROJECTION_DIM
                and checkpoint.get("upstream_hashes") == hashes
                and checkpoint.get("arm_config") == ARMS[arm]
                and sha256(path) == observed["checkpoint_sha256"]
                and close(row["selected_metric"], observed["selected_metric"])
                and close(row["prediction_metrics"], observed["prediction_metrics"])
            )
            isolation_checks.append(risk_pair_isolated(checkpoint))
    gate = v9_gate(reconstructed["pair_rank"], development, v8_audit)
    checks = {
        "protocol_plan_and_summary_passed": (
            summary.get("protocol") == PROTOCOL
            and summary.get("plan")
            == "QWEN_UTILITY_RECORD_CONTROLLER_V9_PAIRWISE_UTILITY_PLAN.md"
            and summary.get("passed")
        ),
        "upstream_hashes_reconstructed": summary.get("upstream_hashes") == hashes,
        "all_arm_checkpoint_hashes_and_states_locked": all(checkpoint_checks),
        "all_metrics_reconstructed": all(
            close(a["selected_metric"], b["selected_metric"])
            and close(a["prediction_metrics"], b["prediction_metrics"])
            for arm in ARMS
            for a, b in zip(reconstructed[arm], summary["arms"][arm])
        ),
        "risk_numerically_pair_isolated": all(isolation_checks),
        "development_gate_reconstructed": close(gate, summary["development_gate"]),
        "finite_selected_metrics": all(
            math.isfinite(value)
            for arm in ARMS for row in reconstructed[arm]
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
        "protocol": AUDIT_PROTOCOL, "passed": passed, "checks": checks,
        "development_checks": gate["checks"],
        "development_release_passed": release,
        "mean_gain": gate["mean_gain"],
        "mean_oracle_gap_closure": gate["mean_oracle_gap_closure"],
        "lama_gain_over_v8": gate["lama_gain_over_v8"],
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": checkpoint_hashes,
        "next_stage": (
            "register_v9_locomo_confirmation" if release
            else "stop_v9_pairwise_controller_iteration"
        ),
        "terminal_status": (
            "v9_pairwise_utility_audit_passed_released" if release
            else "v9_pairwise_utility_audit_passed_gate_failed" if passed
            else "v9_pairwise_utility_audit_failed"
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
