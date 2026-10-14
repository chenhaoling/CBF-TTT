"""Independently reconstruct DocRED projection development results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import rank_metrics
from tasks.extract_docred_relation_space_embeddings import load_embeddings
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_docred_relation_space_projection import (
    EPOCHS, EMBEDDING_AUDIT_SHA256, LOSS_WEIGHTS, MARGIN, PAIR_BATCH_SIZE,
    PROTOCOL, RANK, SEEDS, STATEMENTS_PER_RELATION, STEPS_PER_EPOCH,
    TEMPERATURE, LEARNING_RATE, WEIGHT_DECAY, metric_set, project,
    relation_metrics, scores_for_pairs, selective_operating_point,
    verify_release,
)


def close(left, right, tolerance=1e-9):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
    return left == right


def audit(args):
    import torch

    data_root, embedding_root, result_root = map(
        Path, (args.data_root, args.embedding_root, args.result_root))
    dependencies = verify_release(data_root, embedding_root)
    summary_path = result_root / "summary.json"
    checkpoint_path = result_root / "projection_checkpoints.pt"
    predictions_path = result_root / "development_predictions.jsonl"
    summary = json.loads(summary_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    predictions = load_jsonl(predictions_path)
    manifest = json.loads((data_root / "data/manifest.json").read_text())
    pairs = load_jsonl(data_root / "data/pairs.jsonl")
    dev_pairs = [row for row in pairs if row["partition"] == "relation_development"]
    records, vectors = load_embeddings(embedding_root)
    dev_rows = [(row, vectors[index]) for index, row in enumerate(records)
                if row["item_type"] == "statement"
                and row["partition"] == "relation_development"]
    dev_vectors = torch.stack([vector for _, vector in dev_rows]).float()
    dev_index = {row["item_id"]: index for index, (row, _) in enumerate(dev_rows)}
    labels = [int(row["same_relation"]) for row in dev_pairs]
    relations = [row["left_relation_id"] for row in dev_pairs]
    raw_scores = scores_for_pairs(dev_vectors, dev_pairs, dev_index)
    trigram_scores = [float(row["character_trigram_jaccard"]) for row in dev_pairs]
    raw = metric_set(labels, raw_scores)
    trigram = metric_set(labels, trigram_scores)

    seed_scores, seeds_match = [], True
    for seed in SEEDS:
        artifact = checkpoint["controllers"][str(seed)]
        scores = scores_for_pairs(
            dev_vectors, dev_pairs, dev_index, artifact["statement_weight"].float())
        metrics = metric_set(labels, scores)
        recorded = summary["seeds"][str(seed)]
        seeds_match &= all((
            artifact["selected_epoch"] == recorded["selected_epoch"],
            artifact["selected_epoch"] in EPOCHS,
            close(metrics, recorded["selected_metrics"]),
            close(rank_metrics(labels, scores),
                  next(row["rank_metrics"] for row in recorded["epoch_rank_metrics"]
                       if row["epoch"] == artifact["selected_epoch"])),
        ))
        seed_scores.append(torch.tensor(scores))
    ensemble_scores = torch.stack(seed_scores).mean(0).tolist()
    ensemble = metric_set(labels, ensemble_scores)
    selective = selective_operating_point(labels, ensemble_scores, relations)
    per_relation = relation_metrics(dev_pairs, labels, ensemble_scores)
    relation_macro = {
        name: sum(value[name] for value in per_relation.values()) / len(per_relation)
        for name in ("auroc", "average_precision")}
    hard_indices = [index for index, row in enumerate(dev_pairs)
                    if row["kind"] == "lexical_hard_negative"]
    hard_specificity = sum(ensemble_scores[index] < ensemble["threshold"]
                           for index in hard_indices) / len(hard_indices)
    gate_checks = {
        "ensemble_auroc_at_least_point_85": ensemble["metrics"]["auroc"] >= .85,
        "ensemble_balanced_accuracy_at_least_point_80": (
            ensemble["metrics"]["balanced_accuracy"] >= .80),
        "every_seed_auroc_at_least_point_82": all(
            metric_set(labels, scores.tolist())["metrics"]["auroc"] >= .82
            for scores in seed_scores),
        "full_or_selective_safety": (
            ensemble["full_safety_feasible"] or selective["feasible"]),
        "balanced_accuracy_gain_over_raw_at_least_point_05": (
            ensemble["metrics"]["balanced_accuracy"]
            - raw["metrics"]["balanced_accuracy"] >= .05),
        "every_relation_auroc_at_least_point_60": min(
            value["auroc"] for value in per_relation.values()) >= .60,
    }
    performance_gate = {"checks": gate_checks, "passed": all(gate_checks.values())}

    reconstructed_predictions = []
    for index, pair in enumerate(dev_pairs):
        reconstructed_predictions.append({
            "pair_id": pair["pair_id"], "kind": pair["kind"],
            "left_relation_id": pair["left_relation_id"],
            "right_relation_id": pair["right_relation_id"],
            "same_relation": labels[index], "raw_cosine": raw_scores[index],
            "character_trigram_jaccard": trigram_scores[index],
            "ensemble_projected_cosine": ensemble_scores[index],
        })
    expected_hyperparameters = {
        "rank": RANK, "temperature": TEMPERATURE, "margin": MARGIN,
        "statements_per_relation": STATEMENTS_PER_RELATION,
        "steps_per_epoch": STEPS_PER_EPOCH, "pair_batch_size": PAIR_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss_weights": LOSS_WEIGHTS, "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
    }
    expected_next = ("independent_audit_then_relation_source_test"
                     if performance_gate["passed"] else "stop_representation_branch")
    checks = {
        "frozen_audited_dependencies": (
            dependencies == summary["dependencies"]
            and dependencies["embedding_audit_sha256"] == EMBEDDING_AUDIT_SHA256),
        "checkpoint_protocol_rank_relations_and_seeds": all((
            checkpoint["protocol"] == PROTOCOL, checkpoint["rank"] == RANK,
            checkpoint["relation_order"]
                == manifest["partitions"]["representation_train"],
            checkpoint["seeds"] == list(SEEDS),
            set(checkpoint["controllers"]) == {str(value) for value in SEEDS},
        )),
        "registered_hyperparameters": summary["hyperparameters"] == expected_hyperparameters,
        "artifact_hashes": summary["files_sha256"] == {
            "development_predictions.jsonl": sha(predictions_path),
            "projection_checkpoints.pt": sha(checkpoint_path)},
        "selected_seed_scores_and_metrics_recomputed": seeds_match,
        "baselines_recomputed": close(
            summary["baselines"], {"raw_statement_cosine": raw,
                                   "character_trigram_jaccard": trigram}),
        "ensemble_and_selective_recomputed": (
            close(summary["ensemble"], ensemble)
            and close(summary["selective"], selective)
            and close(checkpoint["ensemble_threshold"], ensemble["threshold"])
            and close(checkpoint["selective_operating_point"], selective)),
        "relation_metrics_recomputed": (
            close(summary["per_relation"], per_relation)
            and close(summary["relation_macro"], relation_macro)
            and close(summary["lexical_hard_negative_specificity"], hard_specificity)),
        "development_predictions_recomputed": close(
            predictions, reconstructed_predictions),
        "development_gate_and_next_stage_recomputed": (
            summary["development_performance_gate"] == performance_gate
            and summary["next_stage"] == expected_next),
        "source_test_and_external_benchmarks_remain_sealed": all((
            summary["relation_source_test_feature_extracted"] is False,
            summary["fewrel_internal_test_loaded"] is False,
            summary["fewrel_external_test_loaded"] is False,
            summary["sealed_benchmarks_loaded"] is False,
        )),
    }
    result = {
        "protocol": "audit_docred_relation_space_projection_development_v1",
        "audit_passed": all(checks.values()), "checks": checks,
        "summary_sha256": sha(summary_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "predictions_sha256": sha(predictions_path),
        "development_performance_gate": performance_gate,
        "ensemble": ensemble, "selective": selective,
        "relation_macro": relation_macro,
        "minimum_relation_auroc": min(value["auroc"] for value in per_relation.values()),
        "relation_source_test_feature_extracted": False,
        "fewrel_internal_test_loaded": False, "fewrel_external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "completed_docred_relation_space_projection_audit",
    }
    write_json(Path(args.output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
