"""Threshold-free failure slices for the sealed FewRel development result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.build_fewrel_selective_forgetting import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import binary_metrics, rank_metrics
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


CONTROLLER_SUMMARY_SHA256 = (
    "ff5a36e551a3155ed3877fc97f80d54d1bc4f543153e3895e475cecb7674c129"
)


def complete(rows):
    labels = [int(row["same_relation"]) for row in rows]
    scores = [float(row["full_score"]) for row in rows]
    predictions = [int(row["full_prediction"]) for row in rows]
    metric = binary_metrics(labels, predictions)
    if set(labels) == {0, 1}:
        metric.update(rank_metrics(labels, scores))
    return metric


def diagnose(controller_root, output):
    controller_root, output = Path(controller_root), Path(output)
    summary_path = controller_root / "results/summary.json"
    if sha(summary_path) != CONTROLLER_SUMMARY_SHA256:
        raise ValueError("FewRel controller release changed")
    rows = load_jsonl(controller_root / "results/development_predictions.jsonl")
    relations = sorted({row["left_relation_id"] for row in rows})
    by_relation = {relation: complete(
        [row for row in rows if row["left_relation_id"] == relation])
        for relation in relations}
    by_kind = {kind: complete([row for row in rows if row["kind"] == kind])
               for kind in sorted({row["kind"] for row in rows})}
    selective_by_kind = {}
    for kind in sorted({row["kind"] for row in rows}):
        values = [row for row in rows if row["kind"] == kind]
        delete = [row for row in values if row["selective_action"] == "delete"]
        selective_by_kind[kind] = {
            "rows": len(values), "delete": len(delete),
            "delete_rate": len(delete) / len(values),
            "correct_delete": sum(row["same_relation"] for row in delete),
            "false_delete": sum(not row["same_relation"] for row in delete),
        }
    relation_ba = [value["balanced_accuracy"] for value in by_relation.values()]
    relation_auc = [value["auroc"] for value in by_relation.values()]
    result = {
        "protocol": "fewrel_selective_controller_failure_diagnostic_v1",
        "rows": len(rows), "relations": len(relations),
        "relation_macro": {
            "balanced_accuracy": statistics.fmean(relation_ba),
            "auroc": statistics.fmean(relation_auc),
            "minimum_balanced_accuracy": min(relation_ba),
            "maximum_balanced_accuracy": max(relation_ba),
            "minimum_auroc": min(relation_auc), "maximum_auroc": max(relation_auc),
        },
        "by_relation": by_relation, "by_kind": by_kind,
        "selective_by_kind": selective_by_kind,
        "interpretation_keys": {
            "random_negative_specificity":
                by_kind["random_negative"]["negative_specificity"],
            "lexical_hard_negative_specificity":
                by_kind["lexical_hard_negative"]["negative_specificity"],
            "same_subject_hard_negative_rows":
                by_kind.get("same_subject_hard_negative", {}).get("rows", 0),
            "positive_recall": by_kind["same_relation_positive"]["positive_recall"],
        },
        "controller_summary_sha256": sha(summary_path),
        "internal_test_loaded": False, "external_test_loaded": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": "new_source_relation_description_representation_training",
        "terminal_status": "completed_fewrel_controller_failure_diagnostic",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(diagnose(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
