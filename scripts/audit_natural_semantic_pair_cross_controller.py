"""Independently audit the frozen natural-semantic pair cross controller.

The auditor never imports the training implementation.  It reconstructs every
reported ensemble OOF metric from the saved row-level predictions, verifies
that those rows are an exact keyed view of the frozen relation-pair corpus,
checks the nested-fold structure, and binds the result to its audited feature
dependencies and serialized controller checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "natural_semantic_pair_cross_controller_v1"
AUDIT_PROTOCOL = "audit_natural_semantic_pair_cross_controller_v1"
EXPECTED_SUMMARY_SHA256 = (
    "e1ea1b2b97b4d69a0d088de9e9d474ce8dec876ccdaab659da5ae531fcb368ff"
)
EXPECTED_PAIR_MANIFEST_SHA256 = (
    "66b21a27d437f250d66f0b3daee6dc3869c89954c1f211d5093c600c06162d01"
)
EXPECTED_PAIRS_SHA256 = (
    "770638be7fd406abf89e544770ada8f5a1c73278a0b771b85db301d0a62780b4"
)
EXPECTED_CROSS_AUDIT_SHA256 = (
    "b0ff99e28a13e8744527b86ccf9b70a20504e9b1e6ccbf403effb3fd482cfcc3"
)
EXPECTED_ZERO_SHOT_SUMMARY_SHA256 = (
    "fa24986f598da28f24523ffd2c71f7e3a817123e8a40680362012e48942de152"
)
EXPECTED_ROWS_SHA256 = (
    "2f4592f8d8fa244b97d7e19bf8303d79ff192f7d05915d475190c13a3fa23d49"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "f7f6508011faaa8342f134b3d0e1261aab0d6491359337bc490ca4c02819b4a0"
)
EXPECTED_FEATURES = (
    "symmetric_cross_embedding_2560",
    "zero_shot_yes_no_margin",
    "character_trigram_jaccard",
)
EXPECTED_SEEDS = (22031, 22037, 22043)
FOLDS = 5


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def binary_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions):
        raise ValueError("invalid binary metric inputs")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    positive, negative = tp + fn, tn + fp
    recall = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "rows": len(labels), "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "positive_recall": recall, "negative_specificity": specificity,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)
               if precision + recall else 0.0),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def rank_metrics(labels, scores):
    if not labels or len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ValueError("rank metrics require two classes")
    positive, negative = sum(labels), len(labels) - sum(labels)
    ascending = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ascending):
        end = index + 1
        while end < len(ascending) and ascending[end][0] == ascending[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ascending[index:end])
        index = end
    auroc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    descending = sorted(zip(scores, labels), key=lambda item: -item[0])
    seen_positive = seen_total = index = 0
    average_precision = 0.0
    while index < len(descending):
        end = index + 1
        while end < len(descending) and descending[end][0] == descending[index][0]:
            end += 1
        group_positive = sum(label for _, label in descending[index:end])
        seen_positive += group_positive
        seen_total += end - index
        average_precision += group_positive / positive * seen_positive / seen_total
        index = end
    return {"auroc": auroc, "average_precision": average_precision}


def complete_metrics(rows):
    labels = [int(row["same_relation"]) for row in rows]
    values = binary_metrics(labels, [int(row["prediction"]) for row in rows])
    if set(labels) == {0, 1}:
        values.update(rank_metrics(labels, [float(row["decision_score"]) for row in rows]))
    return values


def close_tree(left, right, tolerance=1e-12):
    """Compare metric trees while allowing JSON int/float representation."""
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(close_tree(left[key], right[key], tolerance) for key in left))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) \
            and abs(float(left) - float(right)) <= tolerance
    return left == right


def recompute_gate(summary, rows, per_fold, by_kind):
    selections = summary["selection"]
    return {
        "complete_nested_oof_coverage": (
            len(rows) == summary["rows"]
            and len({row["pair_id"] for row in rows}) == len(rows)),
        "all_inner_seed_and_ensemble_constraints_feasible": all(
            value["ensemble_decision"]["constraints_feasible"]
            and all(seed["selected_decision"]["constraints_feasible"]
                    for seed in value["seeds"].values())
            for value in selections.values()),
        "ensemble_balanced_accuracy_at_least_point_80":
            complete_metrics(rows)["balanced_accuracy"] >= .80,
        "ensemble_auroc_at_least_point_85":
            complete_metrics(rows)["auroc"] >= .85,
        "all_fold_balanced_accuracy_at_least_point_70": all(
            value["balanced_accuracy"] >= .70 for value in per_fold.values()),
        "all_seed_balanced_accuracy_at_least_point_75": all(
            value["balanced_accuracy"] >= .75
            for value in summary["per_seed_oof"].values()),
        "lexical_hard_negative_specificity_at_least_point_80":
            by_kind["lexical_hard_negative"]["negative_specificity"] >= .80,
    }


def audit(controller_root, pair_root, cross_root, zero_root):
    import torch

    controller_root = Path(controller_root)
    pair_root, cross_root, zero_root = map(Path, (pair_root, cross_root, zero_root))
    result_root = controller_root / "results"
    summary_path = result_root / "summary.json"
    rows_path = result_root / "oof_cross_controller.jsonl"
    checkpoint_path = result_root / "cross_controllers.pt"
    summary = json.loads(summary_path.read_text())
    rows = read_jsonl(rows_path)
    pairs = read_jsonl(pair_root / "pairs.jsonl")
    source = {row["pair_id"]: row for row in pairs}

    finite_rows = all(
        row["prediction"] in (0, 1)
        and 0 <= float(row["utility_probability"]) <= 1
        and 0 <= float(row["risk_probability"]) <= 1
        and math.isfinite(float(row["decision_score"]))
        for row in rows)
    exact_source_view = (
        len(source) == len(pairs) == len(rows)
        and set(source) == {row["pair_id"] for row in rows}
        and all(
            (int(row["outer_fold"]), row["kind"], int(row["same_relation"]))
            == (int(source[row["pair_id"]]["outer_fold"]),
                source[row["pair_id"]]["kind"],
                int(source[row["pair_id"]]["same_relation"]))
            for row in rows))
    sorted_unique = (
        [row["pair_id"] for row in rows] == sorted(row["pair_id"] for row in rows)
        and len({row["pair_id"] for row in rows}) == len(rows))

    ensemble = complete_metrics(rows)
    per_fold = {str(fold): complete_metrics(
        [row for row in rows if int(row["outer_fold"]) == fold])
        for fold in range(FOLDS)}
    by_kind = {kind: complete_metrics([row for row in rows if row["kind"] == kind])
               for kind in sorted({row["kind"] for row in rows})}
    gate = recompute_gate(summary, rows, per_fold, by_kind)
    selections = summary["selection"]
    nested_structure = (
        set(selections) == {str(fold) for fold in range(FOLDS)}
        and all(
            value["outer_fold"] == fold
            and value["inner_fold"] == (fold + 1) % FOLDS
            and value["train_folds"]
                == sorted(set(range(FOLDS)) - {fold, (fold + 1) % FOLDS})
            and set(value["seeds"]) == {str(seed) for seed in EXPECTED_SEEDS}
            for fold, value in ((fold, selections[str(fold)])
                                for fold in range(FOLDS))))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_structure = all((
        checkpoint["protocol"] == PROTOCOL,
        tuple(checkpoint["feature_spec"]) == EXPECTED_FEATURES,
        int(checkpoint["feature_dim"]) == 2562,
        tuple(checkpoint["seeds"]) == EXPECTED_SEEDS,
        set(checkpoint["folds"]) == {str(fold) for fold in range(FOLDS)},
    ))

    dependencies = summary["dependencies"]
    checks = {
        "frozen_summary_sha256": sha(summary_path) == EXPECTED_SUMMARY_SHA256,
        "frozen_pair_manifest_sha256": (
            sha(pair_root / "manifest.json") == EXPECTED_PAIR_MANIFEST_SHA256),
        "frozen_pair_rows_sha256": sha(pair_root / "pairs.jsonl") == EXPECTED_PAIRS_SHA256,
        "audited_cross_features_sha256": (
            sha(cross_root / "audit.json") == EXPECTED_CROSS_AUDIT_SHA256
            and json.loads((cross_root / "audit.json").read_text())["audit_passed"] is True),
        "frozen_zero_shot_summary_sha256": (
            sha(zero_root / "results/summary.json") == EXPECTED_ZERO_SHOT_SUMMARY_SHA256),
        "reported_dependencies_reproduced": dependencies == {
            "pair_manifest_sha256": EXPECTED_PAIR_MANIFEST_SHA256,
            "pairs_sha256": EXPECTED_PAIRS_SHA256,
            "cross_audit_sha256": EXPECTED_CROSS_AUDIT_SHA256,
            "zero_shot_summary_sha256": EXPECTED_ZERO_SHOT_SUMMARY_SHA256,
        },
        "runner_and_terminal_completed": (
            (controller_root / "runner_exit_code.txt").read_text().strip() == "0"
            and (controller_root / "terminal_status.txt").read_text().strip()
                == "completed_natural_semantic_pair_cross_controller"),
        "oof_rows_sha256": (
            sha(rows_path) == EXPECTED_ROWS_SHA256
            == summary["files_sha256"][rows_path.name]),
        "checkpoint_sha256": (
            sha(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256
            == summary["files_sha256"][checkpoint_path.name]),
        "checkpoint_structure": checkpoint_structure,
        "exact_pair_key_fold_kind_and_label_view": exact_source_view,
        "sorted_unique_complete_oof": sorted_unique,
        "finite_bounded_row_values": finite_rows,
        "ensemble_metrics_recomputed": close_tree(ensemble, summary["ensemble_oof"]),
        "per_fold_metrics_recomputed": close_tree(per_fold, summary["per_fold_ensemble"]),
        "per_kind_metrics_recomputed": close_tree(by_kind, summary["by_kind"]),
        "nested_fold_and_seed_structure": nested_structure,
        "controller_gate_recomputed": (
            gate == summary["controller_gate"]["checks"]
            and summary["controller_gate"]["passed"] == all(gate.values())),
        "failed_gate_and_registered_stop_honored": (
            summary["controller_gate"]["passed"] is False
            and summary["next_stage"]
                == "stop_learned_semantic_controller_iteration_on_observed_sources"),
        "no_model_outcome_or_sealed_benchmark_access": (
            summary["model_outcomes_loaded"] is False
            and summary["sealed_benchmarks_loaded"] is False),
        "relation_ids_are_supervision_only": (
            summary["relation_ids_used_as_training_labels_only"] is True),
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "rows": len(rows),
        "recomputed": {
            "ensemble_oof": ensemble,
            "per_fold_ensemble": per_fold,
            "by_kind": by_kind,
            "controller_gate": gate,
        },
        "files_sha256": {
            "summary.json": sha(summary_path),
            rows_path.name: sha(rows_path),
            checkpoint_path.name: sha(checkpoint_path),
        },
        "model_outcomes_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_natural_semantic_pair_cross_controller_audit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--cross-root", required=True)
    parser.add_argument("--zero-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.controller_root, args.pair_root, args.cross_root, args.zero_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
