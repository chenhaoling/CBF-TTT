"""Independently reconstruct the one-shot S1 development result and gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.audit_structured_fact_comparator_s1_selection import (
    baseline, binary, close, similarity, summary, surface,
)
from tasks.pack_cbf_event_warmup import sha


def load(root, mode):
    rows = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "development" or audit["mode"] != mode
                or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError("invalid development shard")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(6140)):
        raise ValueError("development rows incomplete")
    return rows


def fixed(rows, threshold_value):
    value = summary(rows)
    value["delete"] = {"threshold": threshold_value, **binary(
        [row["gold"]["operation"] != "retain" for row in rows],
        [row["prediction"]["operation"] != "retain" and row["margin"] >= threshold_value
         for row in rows]), "feasible": True}
    hard = [row for row in rows if row["kind"] == "same_entity_other_relation"]
    value["same_entity_other_relation_specificity"] = statistics.fmean(
        not (row["prediction"]["operation"] != "retain" and row["margin"] >= threshold_value)
        for row in hard)
    return value


def model_free(raw, selection):
    scored = [(row, similarity(row["left_text"], row["right_text"])) for row in raw]
    tri_threshold = selection["summaries"]["trigram"]["delete"]["threshold"]
    tri_rows = [baseline(row, score >= tri_threshold, score) for row, score in scored]
    exact_rows = [baseline(row, surface(row["left_text"]) == surface(row["right_text"]), score)
                  for row, score in scored]
    return {"trigram": fixed(tri_rows, tri_threshold),
            "exact_lexical": fixed(exact_rows, 0.)}


def release_gate(selected, validity, baselines):
    safe = [value["delete"]["recall"] for value in baselines.values()
            if value["delete"]["precision"] >= .99]
    best = max(safe, default=0.)
    checks = {
        "json_validity_at_least_99_5": validity >= .995,
        "exact_four_field_at_least_85": selected["exact_four_field_accuracy"] >= .85,
        "same_relation_recall_at_least_80": selected["same_relation"]["recall"] >= .8,
        "same_relation_specificity_at_least_95": selected["same_relation"]["specificity"] >= .95,
        "delete_precision_at_least_99": selected["delete"]["precision"] >= .99,
        "delete_recall_at_least_25": selected["delete"]["recall"] >= .25,
        "hard_relation_specificity_at_least_98":
            selected["same_entity_other_relation_specificity"] >= .98,
        "relation_macro_balanced_accuracy_at_least_85":
            selected["relation_macro_balanced_accuracy"] >= .85,
        "delete_recall_gain_at_least_10_points": selected["delete"]["recall"] - best >= .10,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "best_nonadapted_delete_recall_at_99_precision": best,
            "selected_delete_recall_gain": selected["delete"]["recall"] - best}


def audit(args):
    result_path, selection_path = Path(args.result), Path(args.selection)
    produced, selection = json.loads(result_path.read_text()), json.loads(selection_path.read_text())
    selected_rows = load(args.selected_root, "candidate")
    zero_rows = load(args.zero_shot_root, "candidate")
    generation = load(args.generation_root, "generation")
    if not ([row["pair_id"] for row in selected_rows]
            == [row["pair_id"] for row in zero_rows]
            == [row["pair_id"] for row in generation]):
        raise ValueError("development outputs misaligned")
    raw = [json.loads(line) for line in
           (Path(args.data_root) / "development.jsonl").read_text().splitlines() if line]
    if [row["pair_id"] for row in raw] != [row["pair_id"] for row in selected_rows]:
        raise ValueError("development source misaligned")
    reconstructed_selected = fixed(selected_rows, selection["selected_threshold"])
    reconstructed_zero = fixed(
        zero_rows, selection["summaries"]["zero_shot"]["delete"]["threshold"])
    baselines = {"zero_shot": reconstructed_zero, **model_free(raw, selection)}
    validity = statistics.fmean(row["valid"] for row in generation)
    generation_exact = statistics.fmean(row["valid"] and row["prediction"] == row["gold"]
                                        for row in generation)
    release = release_gate(reconstructed_selected, validity, baselines)
    checks = {
        "selected": close(reconstructed_selected, produced["selected"]),
        "baselines": close(baselines, produced["baselines"]),
        "generation": close({"rows": len(generation), "validity": validity,
                             "exact_four_field_accuracy": generation_exact},
                            produced["generation"]),
        "gate": close(release, produced["gate"]),
        "terminal": produced["terminal_status"]
            == ("passed_development" if release["passed"] else "stopped_by_development"),
        "selection": produced["selection_sha256"] == sha(selection_path),
        "sealed": produced["development_loaded_once"] is True
            and produced["semeval_test_loaded"] is False,
    }
    passed = all(checks.values())
    audit_result = {
        "protocol": "qwen_structured_fact_comparator_s1_development_audit_v1",
        "passed": passed, "checks": checks, "result_sha256": sha(result_path),
        "development_rows_reconstructed": len(raw), "gate_passed": release["passed"],
        "semeval_test_loaded": False,
    }
    if not passed:
        raise RuntimeError(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selected-root", required=True)
    parser.add_argument("--zero-shot-root", required=True)
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = audit(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
