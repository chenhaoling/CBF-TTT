"""Evaluate nested-OOF raw-Qwen cosine on frozen semantic relation pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_raw_embedding_pairs_v1"
PAIR_MANIFEST_SHA256 = (
    "66b21a27d437f250d66f0b3daee6dc3869c89954c1f211d5093c600c06162d01"
)
PAIR_ROWS_SHA256 = (
    "770638be7fd406abf89e544770ada8f5a1c73278a0b771b85db301d0a62780b4"
)
EMBEDDING_AUDIT_SHA256 = (
    "4ff61768ac749b6479f9209acb39b04463fb25b6e48a7d3aa811ca223e981649"
)
FOLDS = 5


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError("invalid binary predictions")
    tp = sum(label == 1 and prediction == 1
             for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0
             for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1
             for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0
             for label, prediction in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "rows": len(labels), "positive": sum(labels), "negative": len(labels) - sum(labels),
        "accuracy": (tp + tn) / len(labels), "balanced_accuracy": (recall + specificity) / 2,
        "positive_recall": recall, "negative_specificity": specificity,
        "precision": precision, "f1": (2 * precision * recall / (precision + recall)
                                        if precision + recall else 0.0),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def rank_metrics(labels, scores):
    if not labels or len(set(labels)) != 2:
        raise ValueError("rank metrics require both labels")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    ordered = sorted(zip(scores, labels), key=lambda value: value[0])
    rank_sum, index = 0.0, 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    auroc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    descending = sorted(zip(scores, labels), key=lambda value: -value[0])
    seen_positive, seen_total, average_precision, index = 0, 0, 0.0, 0
    while index < len(descending):
        end = index + 1
        while end < len(descending) and descending[end][0] == descending[index][0]:
            end += 1
        group_positive = sum(label for _, label in descending[index:end])
        seen_positive += group_positive
        seen_total += end - index
        average_precision += group_positive / positives * seen_positive / seen_total
        index = end
    return {"auroc": auroc, "average_precision": average_precision}


def threshold_candidates(scores):
    values = sorted(set(scores))
    if len(values) == 1:
        return [values[0]]
    candidates = [math.nextafter(values[0], -math.inf)]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    candidates.append(math.nextafter(values[-1], math.inf))
    return candidates


def select_threshold(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise ValueError("invalid threshold inputs")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("threshold selection requires both labels")

    def from_counts(tp, fp):
        fn, tn = positives - tp, negatives - fp
        recall, specificity = tp / positives, tn / negatives
        precision = tp / (tp + fp) if tp + fp else 0.0
        return {
            "rows": len(labels), "positive": positives, "negative": negatives,
            "accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (recall + specificity) / 2,
            "positive_recall": recall, "negative_specificity": specificity,
            "precision": precision,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    groups = []
    for score, label in sorted(zip(scores, labels), reverse=True):
        if not groups or score != groups[-1][0]:
            groups.append([score, 0, 0])
        groups[-1][1 + int(label)] += 1  # index 1: negatives, 2: positives
    rows = [(math.nextafter(groups[0][0], math.inf), from_counts(0, 0))]
    tp = fp = 0
    for index, (score, group_negative, group_positive) in enumerate(groups):
        tp += group_positive; fp += group_negative
        threshold = ((score + groups[index + 1][0]) / 2
                     if index + 1 < len(groups)
                     else math.nextafter(score, -math.inf))
        rows.append((threshold, from_counts(tp, fp)))
    feasible = [value for value in rows
                if value[1]["positive_recall"] >= .80
                and value[1]["negative_specificity"] >= .95]
    pool = feasible or rows
    selected = max(pool, key=lambda value: (
        value[1]["balanced_accuracy"], value[1]["f1"],
        value[1]["negative_specificity"], -value[0]))
    return selected[0], selected[1], bool(feasible)


def load_embeddings(root):
    import torch
    root = Path(root)
    rows = []
    for shard in range(2):
        bundle = torch.load(root / "embeddings" / f"shard_{shard}.pt",
                            map_location="cpu", weights_only=True)
        rows.extend(zip(bundle["surface_indices"], bundle["surface_ids"],
                        bundle["embeddings"].float()))
    rows.sort(key=lambda value: value[0])
    if [value[0] for value in rows] != list(range(len(rows))):
        raise ValueError("embedding indices are not complete")
    ids = [value[1] for value in rows]
    matrix = torch.stack([value[2] for value in rows])
    if len(set(ids)) != len(ids) or not torch.isfinite(matrix).all():
        raise ValueError("invalid frozen embedding bundle")
    return ids, matrix


def evaluate(args):
    import torch
    pair_root, embedding_root = Path(args.pair_root), Path(args.embedding_root)
    manifest_path, pairs_path = pair_root / "manifest.json", pair_root / "pairs.jsonl"
    audit_path = embedding_root / "audit.json"
    audit = json.loads(audit_path.read_text())
    if any((
        sha(manifest_path) != PAIR_MANIFEST_SHA256,
        sha(pairs_path) != PAIR_ROWS_SHA256,
        sha(audit_path) != EMBEDDING_AUDIT_SHA256,
        audit.get("audit_passed") is not True,
        (embedding_root / "runner_exit_code.txt").read_text().strip() != "0",
        (embedding_root / "terminal_status.txt").read_text().strip()
            != "completed_natural_semantic_surface_embeddings",
    )):
        raise ValueError("raw cosine requires the frozen audited releases")
    pairs = load_jsonl(pairs_path)
    surface_ids, embeddings = load_embeddings(embedding_root)
    indices = {value: index for index, value in enumerate(surface_ids)}
    left = torch.tensor([indices[row["left_surface_id"]] for row in pairs])
    right = torch.tensor([indices[row["right_surface_id"]] for row in pairs])
    scores = (embeddings[left] * embeddings[right]).sum(dim=1).tolist()
    labels = [int(row["same_relation"]) for row in pairs]
    if any(not math.isfinite(value) or not -1.0001 <= value <= 1.0001
           for value in scores):
        raise ValueError("invalid cosine score")

    fold_results, output_rows = {}, []
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        inner_indices = [index for index, row in enumerate(pairs)
                         if int(row["outer_fold"]) == inner]
        outer_indices = [index for index, row in enumerate(pairs)
                         if int(row["outer_fold"]) == outer]
        threshold, inner_metric, feasible = select_threshold(
            [labels[index] for index in inner_indices],
            [scores[index] for index in inner_indices])
        outer_labels = [labels[index] for index in outer_indices]
        outer_scores = [scores[index] for index in outer_indices]
        outer_metric = binary_metrics(
            outer_labels, [int(score >= threshold) for score in outer_scores])
        outer_metric.update(rank_metrics(outer_labels, outer_scores))
        kind_metrics = {}
        for kind in sorted({pairs[index]["kind"] for index in outer_indices}):
            selected = [index for index in outer_indices if pairs[index]["kind"] == kind]
            kind_labels = [labels[index] for index in selected]
            kind_scores = [scores[index] for index in selected]
            kind_metrics[kind] = binary_metrics(
                kind_labels, [int(score >= threshold) for score in kind_scores])
        fold_results[str(outer)] = {
            "inner_fold": inner, "selected_threshold": threshold,
            "inner_constraints_feasible": feasible,
            "inner_metrics": inner_metric, "outer_metrics": outer_metric,
            "outer_by_kind": kind_metrics,
        }
        for index in outer_indices:
            row = pairs[index]
            output_rows.append({
                "pair_id": row["pair_id"], "outer_fold": outer,
                "kind": row["kind"], "same_relation": labels[index],
                "raw_cosine": scores[index], "threshold": threshold,
                "prediction": int(scores[index] >= threshold),
            })
    output_rows.sort(key=lambda row: row["pair_id"])
    aggregate = binary_metrics(
        [row["same_relation"] for row in output_rows],
        [row["prediction"] for row in output_rows])
    aggregate.update(rank_metrics(
        [row["same_relation"] for row in output_rows],
        [row["raw_cosine"] for row in output_rows]))
    by_fold = {fold: value["outer_metrics"] for fold, value in fold_results.items()}
    checks = {
        "complete_nested_oof_coverage": (
            len(output_rows) == len(pairs)
            and len({row["pair_id"] for row in output_rows}) == len(pairs)),
        "all_inner_thresholds_constraint_feasible": all(
            value["inner_constraints_feasible"] for value in fold_results.values()),
        "all_outer_folds_above_chance_balanced_accuracy": all(
            value["balanced_accuracy"] > .5 for value in by_fold.values()),
        "all_outer_folds_auroc_above_point_75": all(
            value["auroc"] >= .75 for value in by_fold.values()),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "oof_raw_cosine.jsonl"
    with rows_path.open("x") as sink:
        for row in output_rows:
            sink.write(json.dumps(row) + "\n")
    result = {
        "protocol": PROTOCOL,
        "rows": len(output_rows),
        "aggregate": aggregate,
        "folds": fold_results,
        "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": {
            "pair_manifest_sha256": sha(manifest_path),
            "pairs_sha256": sha(pairs_path),
            "embedding_audit_sha256": sha(audit_path),
        },
        "oof_rows_sha256": sha(rows_path),
        "relation_ids_used_for_scoring_or_threshold_selection": False,
        "model_outcomes_loaded": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": "train_rank16_semantic_projection",
        "terminal_status": "completed_natural_semantic_raw_embedding_pair_baseline",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    result = evaluate(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
