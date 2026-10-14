"""Diagnose anisotropy in frozen semantic embeddings without test fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS,
    binary_metrics,
    load_embeddings,
    rank_metrics,
    select_threshold,
)
from tasks.train_cbf_event_writer import write_json
from tasks.train_natural_semantic_projection import verify_dependencies


PROTOCOL = "natural_semantic_embedding_geometry_diagnostic_v1"
ARMS = ("raw", "train_centered", "train_diagonal_standardized")


def surface_folds(pairs):
    mapping = {}
    for row in pairs:
        fold = int(row["outer_fold"])
        for key in ("left_surface_id", "right_surface_id"):
            surface_id = row[key]
            mapping.setdefault(surface_id, set()).add(fold)
    return mapping


def transform(embeddings, train_indices, arm):
    import torch
    if arm == "raw":
        values = embeddings
    else:
        train = embeddings[train_indices]
        mean = train.mean(dim=0)
        values = embeddings - mean
        if arm == "train_diagonal_standardized":
            std = train.std(dim=0, unbiased=False).clamp_min(1e-4)
            values = values / std
        elif arm != "train_centered":
            raise ValueError(arm)
    return torch.nn.functional.normalize(values, p=2, dim=-1)


def score_pairs(values, pairs, index):
    import torch
    left = torch.tensor([index[row["left_surface_id"]] for row in pairs],
                        device=values.device)
    right = torch.tensor([index[row["right_surface_id"]] for row in pairs],
                         device=values.device)
    return (values[left] * values[right]).sum(-1).cpu().tolist()


def run(args):
    import torch
    dependencies, _ = verify_dependencies(
        args.pair_root, args.embedding_root, args.raw_root)
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    surface_ids, matrix = load_embeddings(args.embedding_root)
    index = {value: position for position, value in enumerate(surface_ids)}
    folds = surface_folds(pairs)
    if set(folds) != set(surface_ids):
        raise ValueError("diagnostic requires complete surface-fold ownership")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = matrix.to(device)
    started = time.perf_counter()
    results = {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        train_folds = set(range(FOLDS)) - {outer, inner}
        # Exclude a surface from the normalization statistics if any identical
        # occurrence belongs to inner or outer folds. This makes the geometry
        # diagnostic stricter than the registered relation split.
        train_indices = torch.tensor([
            index[surface_id] for surface_id in surface_ids
            if folds[surface_id] <= train_folds], device=device)
        inner_pairs = [row for row in pairs if int(row["outer_fold"]) == inner]
        outer_pairs = [row for row in pairs if int(row["outer_fold"]) == outer]
        inner_labels = [int(row["same_relation"]) for row in inner_pairs]
        outer_labels = [int(row["same_relation"]) for row in outer_pairs]
        fold_result = {}
        for arm in ARMS:
            values = transform(matrix, train_indices, arm)
            inner_scores = score_pairs(values, inner_pairs, index)
            threshold, inner_metric, feasible = select_threshold(
                inner_labels, inner_scores)
            outer_scores = score_pairs(values, outer_pairs, index)
            outer_metric = binary_metrics(
                outer_labels, [int(score >= threshold) for score in outer_scores])
            outer_metric.update(rank_metrics(outer_labels, outer_scores))
            by_kind = {}
            for kind in sorted({row["kind"] for row in outer_pairs}):
                selected = [position for position, row in enumerate(outer_pairs)
                            if row["kind"] == kind]
                by_kind[kind] = binary_metrics(
                    [outer_labels[position] for position in selected],
                    [int(outer_scores[position] >= threshold) for position in selected])
            fold_result[arm] = {
                "threshold": threshold, "inner_constraints_feasible": feasible,
                "inner_metrics": inner_metric, "outer_metrics": outer_metric,
                "outer_by_kind": by_kind,
            }
        results[str(outer)] = {
            "outer_fold": outer, "inner_fold": inner,
            "train_folds": sorted(train_folds),
            "training_surfaces": len(train_indices), "arms": fold_result,
        }
    aggregate = {}
    for arm in ARMS:
        metrics = [results[str(fold)]["arms"][arm]["outer_metrics"]
                   for fold in range(FOLDS)]
        aggregate[arm] = {
            key: statistics.fmean(value[key] for value in metrics)
            for key in ("accuracy", "balanced_accuracy", "positive_recall",
                        "negative_specificity", "auroc", "average_precision")}
        aggregate[arm]["all_inner_constraints_feasible"] = all(
            results[str(fold)]["arms"][arm]["inner_constraints_feasible"]
            for fold in range(FOLDS))
        aggregate[arm]["minimum_fold_auroc"] = min(
            value["auroc"] for value in metrics)
    best = max(ARMS, key=lambda arm: aggregate[arm]["balanced_accuracy"])
    result = {
        "protocol": PROTOCOL,
        "folds": results,
        "aggregate_fold_macro": aggregate,
        "best_arm": best,
        "best_minus_raw_balanced_accuracy": (
            aggregate[best]["balanced_accuracy"]
            - aggregate["raw"]["balanced_accuracy"]),
        "diagnosis": ("anisotropy_correction_has_material_value"
                      if aggregate[best]["balanced_accuracy"]
                      - aggregate["raw"]["balanced_accuracy"] >= .05
                      else "pooling_or_supervision_requires_revision"),
        "dependencies": dependencies,
        "surface_fold_overlap": {
            "unique_surfaces": len(folds),
            "surfaces_in_multiple_folds": sum(len(value) > 1 for value in folds.values()),
            "maximum_folds_per_surface": max(map(len, folds.values())),
            "fraction_in_multiple_folds": (
                sum(len(value) > 1 for value in folds.values()) / len(folds)),
        },
        "training_fold_statistics_only": True,
        "model_outcomes_loaded": False,
        "sealed_benchmarks_loaded": False,
        "resources": {"seconds": time.perf_counter() - started,
                      "device": str(device)},
        "terminal_status": "completed_natural_semantic_embedding_geometry_diagnostic",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
