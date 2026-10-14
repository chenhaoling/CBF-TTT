"""Independently audit the failed FewRel controller development release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    binary_metrics, rank_metrics, select_threshold,
)
from tasks.extract_fewrel_selective_pair_embeddings import (
    PAIR_AUDIT_SHA256, PAIR_MANIFEST_SHA256, development_pairs, load_embeddings,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_fewrel_selective_controller import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, FEATURES, PROTOCOL, SEEDS,
    cluster_false_deletion_upper, relation_bootstrap_draws,
)


EXPECTED_SUMMARY_SHA256 = (
    "ff5a36e551a3155ed3877fc97f80d54d1bc4f543153e3895e475cecb7674c129"
)
EXPECTED_EMBEDDING_AUDIT_SHA256 = (
    "9fd6a12b5a1c28f676aa1febb44af30c9dcd654c6b3941afaaf1f2c79a396857"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "815a3621f015d2b88e1a0bef35527f1a05c11ff3dd16eb77cb03bb12b2eaeacb"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "d4741e104b09026178d1b8aab0ffc8c8c62acc03549daf0d480ead220f9d46b2"
)


def close_tree(left, right, tolerance=1e-12):
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(close_tree(left[key], right[key], tolerance) for key in left))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) \
            and abs(float(left) - float(right)) <= tolerance
    return left == right


def complete_metric(labels, predictions, scores):
    values = binary_metrics(labels, predictions)
    values.update(rank_metrics(labels, scores))
    values["false_deletion_rate"] = 1 - values["negative_specificity"]
    return values


def recompute_baseline(train_rows, dev_rows, field):
    train_labels = [int(row["same_relation"]) for row in train_rows]
    train_scores = [float(row[field]) for row in train_rows]
    threshold, train_metric, feasible = select_threshold(train_labels, train_scores)
    dev_labels = [int(row["same_relation"]) for row in dev_rows]
    dev_scores = [float(row[field]) for row in dev_rows]
    dev_metric = complete_metric(
        dev_labels, [int(score >= threshold) for score in dev_scores], dev_scores)
    return {"field": field, "threshold_selected_on_train": threshold,
            "train_constraints_feasible": feasible, "train_metrics": train_metric,
            "development_metrics": dev_metric}


def audit(controller_root, pair_root, embedding_root):
    import torch

    controller_root, pair_root, embedding_root = map(
        Path, (controller_root, pair_root, embedding_root))
    result_root = controller_root / "results"
    summary_path = result_root / "summary.json"
    predictions_path = result_root / "development_predictions.jsonl"
    checkpoint_path = result_root / "fewrel_selective_controllers.pt"
    summary = json.loads(summary_path.read_text())
    rows = load_jsonl(predictions_path)
    all_pairs = load_jsonl(pair_root / "data/pairs.jsonl")
    pairs = development_pairs(all_pairs)
    development = [row for row in pairs if row["partition"] == "development"]
    source = {row["pair_id"]: row for row in development}

    exact_view = (
        len(source) == len(rows) == 9554
        and set(source) == {row["pair_id"] for row in rows}
        and all((row["partition"], row["kind"], row["left_relation_id"],
                 row["right_relation_id"], int(row["same_relation"]))
                == (source[row["pair_id"]]["partition"],
                    source[row["pair_id"]]["kind"],
                    source[row["pair_id"]]["left_relation_id"],
                    source[row["pair_id"]]["right_relation_id"],
                    int(source[row["pair_id"]]["same_relation"]))
                for row in rows))
    finite = all(
        0 <= float(row["utility_probability"]) <= 1
        and 0 <= float(row["risk_probability"]) <= 1
        and math.isfinite(float(row["full_score"]))
        and math.isfinite(float(row["selective_score"]))
        and row["full_prediction"] in (0, 1)
        and row["selective_action"] in ("delete", "retain", "defer")
        for row in rows)

    labels = [int(row["same_relation"]) for row in rows]
    full_metric = complete_metric(
        labels, [int(row["full_prediction"]) for row in rows],
        [float(row["full_score"]) for row in rows])
    ensemble = summary["ensemble"]
    full_match = close_tree(full_metric, ensemble["full_binary"]["metrics"])
    selective_predictions = [int(row["selective_action"] == "delete") for row in rows]
    delete_tp = sum(label and prediction for label, prediction in
                    zip(labels, selective_predictions))
    delete_fp = sum(not label and prediction for label, prediction in
                    zip(labels, selective_predictions))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    precision = delete_tp / (delete_tp + delete_fp) if delete_tp + delete_fp else 0.0
    draw_spec = relation_bootstrap_draws(
        [row["left_relation_id"] for row in rows], BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    upper = cluster_false_deletion_upper(
        labels, selective_predictions, [row["left_relation_id"] for row in rows], draw_spec)
    delete = ensemble["selective"]["delete"]
    selective_match = all((
        delete["tp"] == delete_tp, delete["fp"] == delete_fp,
        delete["tn"] == negatives - delete_fp,
        delete["fn"] == positives - delete_tp,
        abs(delete["positive_recall"] - delete_tp / positives) <= 1e-12,
        abs(delete["deletion_precision"] - precision) <= 1e-12,
        abs(delete["false_deletion_relation_bootstrap_upper_95"] - upper) <= 1e-12,
        ensemble["selective"]["action_counts"]
            == {action: sum(row["selective_action"] == action for row in rows)
                for action in ("delete", "retain", "defer")},
    ))

    pair_ids, partitions, _, margins = load_embeddings(embedding_root)
    if pair_ids != [row["pair_id"] for row in pairs]:
        raise ValueError("FewRel audit feature order mismatch")
    baseline_rows = [dict(row, zero_shot_margin=float(margins[index]))
                     for index, row in enumerate(pairs)]
    train_rows = [row for row in baseline_rows if row["partition"] == "train"]
    dev_rows = [row for row in baseline_rows if row["partition"] == "development"]
    baselines = {
        "zero_shot_margin": recompute_baseline(
            train_rows, dev_rows, "zero_shot_margin"),
        "character_trigram_jaccard": recompute_baseline(
            train_rows, dev_rows, "character_trigram_jaccard"),
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_structure = all((
        checkpoint["protocol"] == PROTOCOL,
        tuple(checkpoint["feature_spec"]) == FEATURES,
        checkpoint["feature_dim"] == 2562,
        tuple(checkpoint["seeds"]) == SEEDS,
        set(checkpoint["controllers"]) == {str(seed) for seed in SEEDS},
        checkpoint["ensemble_full_binary"] == ensemble["full_binary"],
        checkpoint["ensemble_selective"] == ensemble["selective"],
    ))
    full_checks = summary["development_gate"]["full_binary"]["checks"]
    selective_checks = summary["development_gate"]["selective"]["checks"]
    gate_match = (
        full_checks == {
            "ensemble_constraints_feasible":
                ensemble["full_binary"]["constraints_feasible"],
            "all_seed_constraints_feasible": all(
                value["full_binary"]["constraints_feasible"]
                for value in summary["per_seed"].values()),
            "ensemble_auroc_at_least_point_85": full_metric["auroc"] >= .85,
            "ensemble_balanced_accuracy_at_least_point_80":
                full_metric["balanced_accuracy"] >= .80,
            "all_seed_auroc_at_least_point_85": all(
                value["full_binary"]["metrics"]["auroc"] >= .85
                for value in summary["per_seed"].values()),
            "all_seed_balanced_accuracy_at_least_point_80": all(
                value["full_binary"]["metrics"]["balanced_accuracy"] >= .80
                for value in summary["per_seed"].values()),
        }
        and selective_checks == {
            "ensemble_selective_gate": ensemble["selective"]["gate"]["passed"],
            "all_seed_selective_gates": all(
                value["selective"]["gate"]["passed"]
                for value in summary["per_seed"].values()),
        })

    checks = {
        "frozen_summary_sha256": sha(summary_path) == EXPECTED_SUMMARY_SHA256,
        "frozen_audited_dependencies": (
            sha(pair_root / "data/manifest.json") == PAIR_MANIFEST_SHA256
            and sha(pair_root / "audit.json") == PAIR_AUDIT_SHA256
            and sha(embedding_root / "audit.json") == EXPECTED_EMBEDDING_AUDIT_SHA256
            and json.loads((embedding_root / "audit.json").read_text())["audit_passed"]),
        "reported_dependencies_reproduced": summary["dependencies"] == {
            "pair_manifest_sha256": PAIR_MANIFEST_SHA256,
            "pair_audit_sha256": PAIR_AUDIT_SHA256,
            "embedding_audit_sha256": EXPECTED_EMBEDDING_AUDIT_SHA256,
        },
        "runner_and_terminal_completed": (
            (controller_root / "runner_exit_code.txt").read_text().strip() == "0"
            and (controller_root / "terminal_status.txt").read_text().strip()
                == "completed_fewrel_selective_controller"),
        "prediction_sha256": (
            sha(predictions_path) == EXPECTED_PREDICTIONS_SHA256
            == summary["files_sha256"][predictions_path.name]),
        "checkpoint_sha256": (
            sha(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256
            == summary["files_sha256"][checkpoint_path.name]),
        "checkpoint_structure": checkpoint_structure,
        "exact_development_key_label_relation_view": exact_view,
        "finite_bounded_predictions": finite,
        "ensemble_full_metrics_recomputed": full_match,
        "ensemble_selective_metrics_and_bootstrap_recomputed": selective_match,
        "baselines_recomputed": close_tree(baselines, summary["baselines"]),
        "development_gate_recomputed": gate_match,
        "failed_gate_and_stop_honored": (
            summary["development_gate"]["passed_for_internal_test"] is False
            and summary["next_stage"] == "stop_fewrel_controller_branch"),
        "test_and_external_benchmarks_remain_sealed": all((
            summary["internal_test_features_loaded"] is False,
            summary["internal_test_scored"] is False,
            summary["external_test_features_loaded"] is False,
            summary["external_test_scored"] is False,
            summary["sealed_benchmarks_loaded"] is False,
        )),
    }
    return {
        "protocol": "audit_fewrel_selective_controller_development_v1",
        "audit_passed": all(checks.values()), "checks": checks,
        "recomputed": {"full_binary": full_metric,
                       "selective_delete": {
                           "tp": delete_tp, "fp": delete_fp,
                           "positive_recall": delete_tp / positives,
                           "deletion_precision": precision,
                           "false_deletion_relation_bootstrap_upper_95": upper,
                       }, "baselines": baselines},
        "summary_sha256": sha(summary_path),
        "predictions_sha256": sha(predictions_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "internal_test_scored": False, "external_test_scored": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_fewrel_selective_controller_audit",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.controller_root, args.pair_root, args.embedding_root)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
