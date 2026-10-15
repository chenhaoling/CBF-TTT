"""Diagnose the preregistered V7 cross-source development-gate failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    CHECKPOINT_EPOCHS,
    SEEDS,
    fit,
    normalization,
    predict_state,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    add_arguments,
    evaluate_state,
    load_inputs,
)


PROTOCOL = "qwen_utility_record_controller_v7_failure_diagnostic"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V7_FAILURE_DIAGNOSTIC_PLAN.md"
EXPECTED_V7_AUDIT_PROTOCOL = (
    "qwen_utility_record_controller_v7_cross_source_training_audit"
)


def rank_metrics(labels: list[int], scores: list[float]) -> dict:
    if len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("rank metrics require aligned examples and two classes")
    positive = sum(labels)
    negative = len(labels) - positive
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    auroc = (
        rank_sum - positive * (positive + 1) / 2
    ) / (positive * negative)

    ordered = sorted(zip(scores, labels), key=lambda item: -item[0])
    seen_positive = seen_total = index = 0
    average_precision = 0.0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positive = sum(label for _, label in ordered[index:end])
        seen_positive += group_positive
        seen_total += end - index
        average_precision += (
            group_positive / positive * seen_positive / seen_total
        )
        index = end
    return {"auroc": auroc, "average_precision": average_precision}


def expected_calibration_error(labels: list[int], scores: list[float],
                               bins: int = 10) -> float:
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1)
        ]
        if not members:
            continue
        confidence = statistics.fmean(scores[index] for index in members)
        frequency = statistics.fmean(labels[index] for index in members)
        error += len(members) / len(labels) * abs(confidence - frequency)
    return error


def probability_metrics(labels: list[int], scores: list[float]) -> dict:
    value = {
        "examples": len(labels),
        "positives": sum(labels),
        "prevalence": statistics.fmean(labels),
        "brier": statistics.fmean(
            (score - label) ** 2 for score, label in zip(scores, labels)
        ),
        "ece_10": expected_calibration_error(labels, scores),
    }
    if set(labels) == {0, 1}:
        value.update(rank_metrics(labels, scores))
    return value


def head_diagnostic(examples: list[dict], predictions: dict,
                    head: str, target: str) -> dict:
    def metric(rows):
        labels = [int(row[target]) for row in rows]
        scores = [predictions[(
            row["episode_id"], row["boundary_index"], row["slot"]
        )][head] for row in rows]
        return probability_metrics(labels, scores)

    mechanisms = sorted({row["mechanism"] for row in examples})
    return {
        "overall": metric(examples),
        "per_mechanism": {
            mechanism: metric([
                row for row in examples if row["mechanism"] == mechanism
            ])
            for mechanism in mechanisms
        },
    }


def best_domain_candidate(candidates: dict, domain: str) -> dict:
    feasible = [
        (name, row) for name, row in candidates.items()
        if row["domains"][domain]["added_unrelated_errors"] == 0
    ]
    if not feasible:
        raise ValueError(f"no zero-collateral candidate for {domain}")

    def key(item):
        name, row = item
        value = row["domains"][domain]
        config = row["config"]
        return (
            value["gain_over_lifecycle_physics"],
            value["residual_deletion_precision"],
            -value["residual_deletion_fraction"],
            -config["epoch"], config["utility_threshold"],
            config["risk_penalty"], -config["conformal_level"], name,
        )

    name, row = max(feasible, key=key)
    return {
        "selected": name,
        "config": row["config"],
        "metric": row["domains"][domain],
        "feasible_candidates": len(feasible),
    }


def joint_diagnostics(v7_root: Path, development: dict) -> tuple[dict, dict]:
    rankings = {}
    ceilings = {}
    hashes = {}
    for seed in SEEDS:
        path = v7_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(path)
        hashes[str(seed)] = sha256(path)
        seed_rankings = {}
        predictions = {}
        for domain_name, domain in development.items():
            prediction = predict_state(
                checkpoint["state_dict"], domain["examples"],
                checkpoint["means"], checkpoint["stds"],
            )
            predictions[domain_name] = prediction
            seed_rankings[domain_name] = {
                "benefit": head_diagnostic(
                    domain["examples"], prediction,
                    "benefit", "benefit_target",
                ),
                "risk": head_diagnostic(
                    domain["examples"], prediction,
                    "risk", "risk_target",
                ),
            }
        candidates = evaluate_state(
            checkpoint["state_dict"], checkpoint["means"],
            checkpoint["stds"], development,
            checkpoint["selected_config"]["epoch"],
        )
        rankings[str(seed)] = seed_rankings
        ceilings[str(seed)] = {
            domain: best_domain_candidate(candidates, domain)
            for domain in development
        }
    return {
        "ranking_and_calibration": rankings,
        "domain_specific_policy_ceiling": ceilings,
    }, hashes


def source_transfer(train_domains: dict, development: dict, device) -> dict:
    matrix = {"templama": {}, "lama": {}}
    for training_domain, train in train_domains.items():
        for seed in SEEDS:
            fitted = fit(train["examples"], seed, device)
            all_candidates = {}
            for epoch in CHECKPOINT_EPOCHS:
                all_candidates.update(evaluate_state(
                    fitted["states"][epoch], fitted["means"], fitted["stds"],
                    development, epoch,
                ))
            matrix[training_domain][str(seed)] = {
                domain: best_domain_candidate(all_candidates, domain)
                for domain in development
            }
    return matrix


def source_shift(train_domains: dict, development: dict) -> dict:
    combined = (
        train_domains["templama"]["examples"]
        + train_domains["lama"]["examples"]
    )
    means, stds = normalization(combined)

    def standardized(rows):
        return [[
            (value - mean) / std
            for value, mean, std in zip(row["features"], means, stds)
        ] for row in rows]

    train_values = {
        name: standardized(domain["examples"])
        for name, domain in train_domains.items()
    }
    centroids = {
        name: [
            statistics.fmean(row[index] for row in values)
            for index in range(len(FEATURE_NAMES))
        ] for name, values in train_values.items()
    }
    labels, scores = [], []
    per_domain = {}
    for domain_name, domain in development.items():
        values = standardized(domain["examples"])
        domain_scores = []
        for row in values:
            temp_distance = sum(
                (value - centroid) ** 2
                for value, centroid in zip(row, centroids["templama"])
            )
            lama_distance = sum(
                (value - centroid) ** 2
                for value, centroid in zip(row, centroids["lama"])
            )
            domain_scores.append(temp_distance - lama_distance)
        labels.extend([int(domain_name == "lama")] * len(values))
        scores.extend(domain_scores)
        per_domain[domain_name] = {
            "examples": len(values),
            "mean_source_score": statistics.fmean(domain_scores),
        }
    shifts = sorted((
        {
            "feature": name,
            "standardized_mean_difference": (
                centroids["lama"][index] - centroids["templama"][index]
            ),
            "absolute_standardized_mean_difference": abs(
                centroids["lama"][index] - centroids["templama"][index]
            ),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ), key=lambda row: (-row["absolute_standardized_mean_difference"], row["feature"]))
    return {
        "development_source_auroc": rank_metrics(labels, scores)["auroc"],
        "per_domain": per_domain,
        "top_feature_shifts": shifts[:10],
    }


def aggregate_ceiling(rows: dict, training_domain: str | None,
                      evaluation_domain: str) -> dict:
    selected = (
        rows if training_domain is None else rows[training_domain]
    )
    values = [
        selected[str(seed)][evaluation_domain]["metric"] for seed in SEEDS
    ]
    return {
        "mean_gain": statistics.fmean(
            value["gain_over_lifecycle_physics"] for value in values
        ),
        "mean_oracle_gap_closure": statistics.fmean(
            value["oracle_gap_closure"] for value in values
        ),
        "zero_collateral_all_seeds": all(
            value["added_unrelated_errors"] == 0 for value in values
        ),
    }


def diagnose_class(joint_ceiling: dict, transfer: dict) -> tuple[str, dict]:
    joint = aggregate_ceiling(joint_ceiling, None, "lama")
    lama_only = aggregate_ceiling(transfer, "lama", "lama")
    selection_limited = (
        joint["mean_gain"] >= .015
        and joint["mean_oracle_gap_closure"] >= .50
        and joint["zero_collateral_all_seeds"]
    )
    negative_transfer = (
        not selection_limited
        and lama_only["mean_gain"] >= .015
        and lama_only["mean_oracle_gap_closure"] >= .50
        and lama_only["zero_collateral_all_seeds"]
        and lama_only["mean_gain"] >= joint["mean_gain"] + .005
    )
    diagnosis = (
        "selection_calibration_limited" if selection_limited
        else "negative_transfer_limited" if negative_transfer
        else "representation_ranking_limited"
    )
    return diagnosis, {
        "joint_lama_domain_specific_ceiling": joint,
        "lama_only_lama_domain_specific_ceiling": lama_only,
        "selection_calibration_condition": selection_limited,
        "negative_transfer_condition": negative_transfer,
    }


def diagnose(args) -> dict:
    import torch

    started = time.perf_counter()
    v7_root = Path(args.v7_training_root)
    v7_summary_path = v7_root / "summary.json"
    v7_audit_path = Path(args.v7_audit)
    v7_summary = json.loads(v7_summary_path.read_text())
    v7_audit = json.loads(v7_audit_path.read_text())
    if not (
        v7_summary.get("passed")
        and v7_audit.get("passed")
        and v7_audit.get("protocol") == EXPECTED_V7_AUDIT_PROTOCOL
        and not v7_audit.get("development_release_passed")
    ):
        raise ValueError("V7 must be audited and development-gate failed")
    train_domains, development, upstream_hashes = load_inputs(args)
    joint, checkpoint_hashes = joint_diagnostics(v7_root, development)
    if checkpoint_hashes != v7_audit["checkpoint_sha256"]:
        raise ValueError("locked V7 checkpoint hashes changed")
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    transfer = source_transfer(train_domains, development, device)
    shift = source_shift(train_domains, development)
    diagnosis, evidence = diagnose_class(
        joint["domain_specific_policy_ceiling"], transfer
    )
    transfer_aggregate = {
        train: {
            evaluation: aggregate_ceiling(transfer, train, evaluation)
            for evaluation in development
        } for train in transfer
    }
    checks = {
        "v7_integrity_audit_passed_gate_failed": True,
        "checkpoint_hashes_locked": checkpoint_hashes
        == v7_audit["checkpoint_sha256"],
        "registered_support_loaded": (
            len(train_domains["templama"]["examples"]) == 4392
            and len(train_domains["lama"]["examples"]) == 4684
            and len(development["templama"]["examples"]) == 2204
            and len(development["lama"]["examples"]) == 588
        ),
        "all_transfer_cells_zero_collateral": all(
            row["zero_collateral_all_seeds"]
            for values in transfer_aggregate.values() for row in values.values()
        ),
        "finite_diagnostics": math.isfinite(
            shift["development_source_auroc"]
        ),
        "heldout_sources_sealed": True,
    }
    return {
        "protocol": PROTOCOL,
        "plan": PLAN,
        "passed": all(checks.values()),
        "checks": checks,
        "v7_summary_sha256": sha256(v7_summary_path),
        "v7_audit_sha256": sha256(v7_audit_path),
        "v7_checkpoint_sha256": checkpoint_hashes,
        "upstream_audit_hashes": upstream_hashes,
        **joint,
        "source_transfer_matrix": transfer,
        "source_transfer_aggregate": transfer_aggregate,
        "covariate_shift": shift,
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "templama_test_loaded": False,
        "lama_confirmation_loaded": False,
        "longmemeval_loaded": False,
        "locomo_loaded": False,
        "seconds": time.perf_counter() - started,
        "next_stage": f"register_v8_{diagnosis}",
        "terminal_status": "v7_failure_diagnostic_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--v7-training-root", required=True)
    parser.add_argument("--v7-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = diagnose(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
