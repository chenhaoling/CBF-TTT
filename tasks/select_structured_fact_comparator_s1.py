"""Select the S1 checkpoint and confidence threshold on calibration only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_structured_fact_comparator_s1_selection_v1"
SCORE_PROTOCOL = "qwen_structured_fact_comparator_s1_scores_v1"
FIELDS = ("same_entity", "same_relation", "same_value", "operation")


def load_arm(root: str | Path, name: str, expected_rows: int = 26805) -> tuple[list[dict], list[dict]]:
    root = Path(root) / name
    rows, audits = [], []
    for shard in range(2):
        path = root / f"shard_{shard}.jsonl"
        audit_path = path.with_suffix(".audit.json")
        audit = json.loads(audit_path.read_text())
        if (audit["protocol"] != SCORE_PROTOCOL or audit["split"] != "calibration"
                or audit["mode"] != "candidate" or audit["shard"] != shard
                or audit["num_shards"] != 2 or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError(f"invalid S1 calibration shard: {name}/{shard}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        audits.append(audit)
    rows.sort(key=lambda row: row["index"])
    if len(rows) != expected_rows or [row["index"] for row in rows] != list(range(expected_rows)):
        raise ValueError(f"incomplete S1 calibration arm: {name}")
    if len({row["pair_id"] for row in rows}) != expected_rows:
        raise ValueError(f"duplicate S1 calibration rows: {name}")
    return rows, audits


def binary_counts(gold: list[bool], prediction: list[bool]) -> dict:
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def select_threshold(rows: list[dict], minimum_precision: float = .99) -> dict:
    gold = [row["gold"]["operation"] != "retain" for row in rows]
    ordered = sorted(((row["margin"], index) for index, row in enumerate(rows)
                      if row["prediction"]["operation"] != "retain"), reverse=True)
    prediction = [False] * len(rows)
    candidates = [(math.inf, binary_counts(gold, prediction))]
    cursor = 0
    while cursor < len(ordered):
        margin = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == margin:
            prediction[ordered[cursor][1]] = True
            cursor += 1
        candidates.append((margin, binary_counts(gold, prediction)))
    feasible = [(threshold, metrics) for threshold, metrics in candidates
                if metrics["precision"] >= minimum_precision]
    threshold, metrics = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]))
    return {"threshold": threshold, **metrics,
            "feasible": metrics["precision"] >= minimum_precision}


def relation_macro(rows: list[dict]) -> float:
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation"]].append(row)
    values = []
    for subset in by_relation.values():
        metrics = binary_counts([row["gold"]["same_relation"] for row in subset],
                                [row["prediction"]["same_relation"] for row in subset])
        values.append((metrics["recall"] + metrics["specificity"]) / 2)
    return statistics.fmean(values)


def summarize(rows: list[dict]) -> dict:
    exact = statistics.fmean(all(row["gold"][field] == row["prediction"][field]
                                 for field in FIELDS) for row in rows)
    relation = binary_counts([row["gold"]["same_relation"] for row in rows],
                             [row["prediction"]["same_relation"] for row in rows])
    by_kind = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = statistics.fmean(
            row["gold"]["operation"] == row["prediction"]["operation"] for row in subset)
    return {"rows": len(rows), "exact_four_field_accuracy": exact,
            "same_relation": relation, "relation_macro_balanced_accuracy": relation_macro(rows),
            "by_kind_operation_accuracy": by_kind, "delete": select_threshold(rows)}


def select(args):
    names = [value.strip() for value in args.arms.split(",") if value.strip()]
    if "zero_shot" not in names or len(names) != 5:
        raise ValueError("S1 selection requires zero-shot plus four checkpoints")
    summaries, audits = {}, {}
    pair_ids = None
    for name in names:
        rows, arm_audits = load_arm(args.score_root, name)
        ids = [row["pair_id"] for row in rows]
        if pair_ids is not None and ids != pair_ids:
            raise ValueError("S1 calibration arms are not row-aligned")
        pair_ids = ids
        summaries[name] = summarize(rows)
        audits[name] = arm_audits
    adapter_names = [name for name in names if name != "zero_shot"]
    selected_name = max(adapter_names, key=lambda name: (
        summaries[name]["delete"]["feasible"], summaries[name]["delete"]["recall"],
        summaries[name]["exact_four_field_accuracy"],
        summaries[name]["relation_macro_balanced_accuracy"], -summaries[name]["delete"]["threshold"]))
    selected_adapter = audits[selected_name][0]["adapter"]
    data_hash = audits[selected_name][0]["data"]["manifest_sha256"]
    if any(shard["data"]["manifest_sha256"] != data_hash
           for arm in audits.values() for shard in arm):
        raise ValueError("S1 calibration arms use different data")
    result = {
        "protocol": PROTOCOL, "selected_arm": selected_name,
        "selected_adapter": selected_adapter,
        "selected_threshold": summaries[selected_name]["delete"]["threshold"],
        "minimum_delete_precision": .99, "summaries": summaries,
        "score_audits_sha256": {name: [sha(Path(args.score_root) / name
                                                   / f"shard_{shard}.audit.json")
                                        for shard in range(2)] for name in names},
        "data_manifest_sha256": data_hash, "calibration_rows": len(pair_ids),
        "development_loaded": False, "semeval_test_loaded": False,
        "terminal_status": "selected_before_development",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--output", required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
