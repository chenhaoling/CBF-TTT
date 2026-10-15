"""Apply the frozen S1 one-shot SemEval development gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.select_structured_fact_comparator_s1 import (
    baseline_row, fixed_delete_metrics, relation_surface, summarize, trigram_similarity,
)


PROTOCOL = "qwen_structured_fact_comparator_s1_development_v1"
SCORE_PROTOCOL = "qwen_structured_fact_comparator_s1_scores_v1"
SELECTION_PROTOCOL = "qwen_structured_fact_comparator_s1_selection_v1"


def load_shards(root: str | Path, mode: str, expected_rows: int = 6140):
    root = Path(root)
    rows, audits = [], []
    for shard in range(2):
        path = root / f"shard_{shard}.jsonl"
        audit_path = path.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text())
        if (audit["protocol"] != SCORE_PROTOCOL or audit["split"] != "development"
                or audit["mode"] != mode or audit["shard"] != shard
                or (mode == "candidate" and audit.get("candidate_kernel") != "cached")
                or audit["num_shards"] != 2 or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError(f"invalid S1 development {mode} shard {shard}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        audits.append(audit)
    rows.sort(key=lambda row: row["index"])
    if len(rows) != expected_rows or [row["index"] for row in rows] != list(range(expected_rows)):
        raise ValueError(f"incomplete S1 development {mode} rows")
    return rows, audits


def action_specificity(rows: list[dict], threshold: float, kind: str) -> float:
    subset = [row for row in rows if row["kind"] == kind]
    return statistics.fmean(not (row["prediction"]["operation"] != "retain"
                                 and row["margin"] >= threshold) for row in subset)


def metric_summary(rows: list[dict], threshold: float) -> dict:
    result = summarize(rows)
    result["delete"] = fixed_delete_metrics(rows, threshold)
    result["same_entity_other_relation_specificity"] = action_specificity(
        rows, threshold, "same_entity_other_relation")
    return result


def model_free_development(raw_rows: list[dict], selection: dict) -> dict:
    scored = [(row, trigram_similarity(row["left_text"], row["right_text"]))
              for row in raw_rows]
    trigram_threshold = selection["summaries"]["trigram"]["delete"]["threshold"]
    trigram_rows = [baseline_row(row, similarity >= trigram_threshold, similarity)
                    for row, similarity in scored]
    trigram = metric_summary(trigram_rows, trigram_threshold)
    exact_rows = [baseline_row(row, relation_surface(row["left_text"])
                              == relation_surface(row["right_text"]), similarity)
                  for row, similarity in scored]
    exact = metric_summary(exact_rows, 0.0)
    return {"exact_lexical": exact, "trigram": trigram}


def gate(selected: dict, generation_validity: float, baseline: dict) -> dict:
    nonadapted = [value for value in baseline.values()
                  if value["delete"]["precision"] >= .99]
    best_recall = max((value["delete"]["recall"] for value in nonadapted), default=0.0)
    checks = {
        "json_validity_at_least_99_5": generation_validity >= .995,
        "exact_four_field_at_least_85": selected["exact_four_field_accuracy"] >= .85,
        "same_relation_recall_at_least_80": selected["same_relation"]["recall"] >= .80,
        "same_relation_specificity_at_least_95": selected["same_relation"]["specificity"] >= .95,
        "delete_precision_at_least_99": selected["delete"]["precision"] >= .99,
        "delete_recall_at_least_25": selected["delete"]["recall"] >= .25,
        "hard_relation_specificity_at_least_98":
            selected["same_entity_other_relation_specificity"] >= .98,
        "relation_macro_balanced_accuracy_at_least_85":
            selected["relation_macro_balanced_accuracy"] >= .85,
        "delete_recall_gain_at_least_10_points": selected["delete"]["recall"]
            - best_recall >= .10,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "best_nonadapted_delete_recall_at_99_precision": best_recall,
            "selected_delete_recall_gain": selected["delete"]["recall"] - best_recall}


def evaluate(args):
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    if (selection["protocol"] != SELECTION_PROTOCOL
            or selection["development_loaded"] is not False
            or selection["semeval_test_loaded"] is not False):
        raise ValueError("invalid S1 frozen selection")
    selected_rows, selected_audits = load_shards(args.selected_root, "candidate")
    zero_rows, zero_audits = load_shards(args.zero_shot_root, "candidate")
    generation_rows, generation_audits = load_shards(args.generation_root, "generation")
    aligned = [row["pair_id"] for row in selected_rows]
    if ([row["pair_id"] for row in zero_rows] != aligned
            or [row["pair_id"] for row in generation_rows] != aligned):
        raise ValueError("S1 development outputs are not aligned")
    selection_hash = sha(selection_path)
    if any(audit["development_selection_sha256"] != selection_hash
           for audit in selected_audits + zero_audits + generation_audits):
        raise ValueError("S1 development used a different selection")
    if any(audit["adapter"] != selection["selected_adapter"]
           for audit in selected_audits + generation_audits):
        raise ValueError("S1 selected development adapter mismatch")
    if any(audit["adapter"] is not None for audit in zero_audits):
        raise ValueError("S1 zero-shot development unexpectedly used an adapter")

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    raw_path = data_root / "development.jsonl"
    if (sha(data_root / "manifest.json") != selection["data_manifest_sha256"]
            or sha(raw_path) != manifest["files_sha256"]["development.jsonl"]):
        raise ValueError("S1 development data changed")
    raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    if [row["pair_id"] for row in raw_rows] != aligned:
        raise ValueError("S1 development raw rows are not aligned")

    selected = metric_summary(selected_rows, selection["selected_threshold"])
    zero_threshold = selection["summaries"]["zero_shot"]["delete"]["threshold"]
    zero = metric_summary(zero_rows, zero_threshold)
    baselines = {"zero_shot": zero, **model_free_development(raw_rows, selection)}
    validity = statistics.fmean(row["valid"] for row in generation_rows)
    generation_exact = statistics.fmean(
        row["valid"] and row["prediction"] == row["gold"] for row in generation_rows)
    release = gate(selected, validity, baselines)
    result = {
        "protocol": PROTOCOL, "selected_arm": selection["selected_arm"],
        "selected_adapter": selection["selected_adapter"],
        "selected_threshold": selection["selected_threshold"],
        "selected": selected, "baselines": baselines,
        "generation": {"rows": len(generation_rows), "validity": validity,
                       "exact_four_field_accuracy": generation_exact},
        "gate": release, "selection_sha256": selection_hash,
        "score_audit_sha256": {
            "selected": [sha(Path(args.selected_root) / f"shard_{i}.audit.json") for i in range(2)],
            "zero_shot": [sha(Path(args.zero_shot_root) / f"shard_{i}.audit.json") for i in range(2)],
            "generation": [sha(Path(args.generation_root) / f"shard_{i}.audit.json") for i in range(2)],
        },
        "development_loaded_once": True, "semeval_test_loaded": False,
        "terminal_status": "passed_development" if release["passed"]
        else "stopped_by_development",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selected-root", required=True)
    parser.add_argument("--zero-shot-root", required=True)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--output", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
