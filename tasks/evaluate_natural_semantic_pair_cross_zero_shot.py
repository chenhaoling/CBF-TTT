"""Evaluate nested-OOF zero-shot yes/no margins from the pair cross-encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS, PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
    binary_metrics, rank_metrics, select_threshold,
)
from tasks.extract_natural_semantic_pair_cross_embeddings import load_cross_embeddings
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_pair_cross_zero_shot_v1"
CROSS_AUDIT_SHA256 = (
    "b0ff99e28a13e8744527b86ccf9b70a20504e9b1e6ccbf403effb3fd482cfcc3"
)


def verify_release(pair_root, cross_root):
    pair_root, cross_root = Path(pair_root), Path(cross_root)
    audit = json.loads((cross_root / "audit.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(cross_root / "audit.json") == CROSS_AUDIT_SHA256,
        audit.get("audit_passed") is True,
        audit.get("rows") == 63782,
        (cross_root / "runner_exit_code.txt").read_text().strip() == "0",
        (cross_root / "terminal_status.txt").read_text().strip()
            == "completed_natural_semantic_pair_cross_embeddings",
    )
    if not all(checks):
        raise ValueError("zero-shot cross score requires frozen audited features")
    return {"pair_manifest_sha256": sha(pair_root / "manifest.json"),
            "pairs_sha256": sha(pair_root / "pairs.jsonl"),
            "cross_audit_sha256": sha(cross_root / "audit.json")}


def run(args):
    dependencies = verify_release(args.pair_root, args.cross_root)
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    pair_ids, _, margins = load_cross_embeddings(args.cross_root)
    if pair_ids != [row["pair_id"] for row in pairs]:
        raise ValueError("cross feature pair order mismatch")
    labels = [int(row["same_relation"]) for row in pairs]
    scores = margins.tolist()
    fold_results, rows = {}, []
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
            outer_labels, [int(value >= threshold) for value in outer_scores])
        outer_metric.update(rank_metrics(outer_labels, outer_scores))
        fold_results[str(outer)] = {
            "outer_fold": outer, "inner_fold": inner, "threshold": threshold,
            "inner_constraints_feasible": feasible,
            "inner_metrics": inner_metric, "outer_metrics": outer_metric,
        }
        for index in outer_indices:
            rows.append({
                "pair_id": pairs[index]["pair_id"], "outer_fold": outer,
                "kind": pairs[index]["kind"], "same_relation": labels[index],
                "yes_minus_no_margin": scores[index], "threshold": threshold,
                "prediction": int(scores[index] >= threshold),
            })
    rows.sort(key=lambda row: row["pair_id"])
    aggregate = binary_metrics([row["same_relation"] for row in rows],
                               [row["prediction"] for row in rows])
    aggregate.update(rank_metrics([row["same_relation"] for row in rows],
                                  [row["yes_minus_no_margin"] for row in rows]))
    by_kind = {kind: binary_metrics(
        [row["same_relation"] for row in rows if row["kind"] == kind],
        [row["prediction"] for row in rows if row["kind"] == kind])
        for kind in sorted({row["kind"] for row in rows})}
    checks = {
        "complete_nested_oof_coverage": (
            len(rows) == len(pairs) and len({row["pair_id"] for row in rows}) == len(pairs)),
        "all_inner_constraints_feasible": all(
            value["inner_constraints_feasible"] for value in fold_results.values()),
        "aggregate_balanced_accuracy_at_least_point_75":
            aggregate["balanced_accuracy"] >= .75,
        "aggregate_auroc_at_least_point_80": aggregate["auroc"] >= .80,
        "all_fold_balanced_accuracy_at_least_point_65": all(
            value["outer_metrics"]["balanced_accuracy"] >= .65
            for value in fold_results.values()),
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "oof_zero_shot_margin.jsonl"
    with rows_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    result = {
        "protocol": PROTOCOL, "rows": len(rows), "aggregate": aggregate,
        "folds": fold_results, "by_kind": by_kind,
        "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": dependencies, "oof_rows_sha256": sha(rows_path),
        "relation_ids_used": False, "model_outcomes_loaded": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": "train_pair_cross_controller",
        "terminal_status": "completed_natural_semantic_pair_cross_zero_shot",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--cross-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
