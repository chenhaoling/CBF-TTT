"""Select the S1 checkpoint and confidence threshold on calibration only."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_structured_fact_comparator_s1_selection_v1"
SCORE_PROTOCOL = "qwen_structured_fact_comparator_s1_scores_v1"
FIELDS = ("same_entity", "same_relation", "same_value", "operation")
TAG_PATTERNS = {name: re.compile(rf"\[{name}\](.*?)\[/{name}\]")
                for name in ("SUBJECT", "VALUE")}


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
    initial_threshold = ordered[0][0] + 1.0 if ordered else 1.0
    candidates = [(initial_threshold, binary_counts(gold, prediction))]
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


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def tag_value(text: str, name: str) -> str:
    values = TAG_PATTERNS[name].findall(text)
    if len(values) != 1:
        raise ValueError(f"invalid {name} tags")
    return normalize(values[0])


def relation_surface(text: str) -> str:
    for name, pattern in TAG_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return normalize(text)


def trigram_similarity(left: str, right: str) -> float:
    def grams(value):
        value = f"  {value}  "
        return {value[index:index + 3] for index in range(max(1, len(value) - 2))}
    a, b = grams(relation_surface(left)), grams(relation_surface(right))
    return len(a & b) / max(1, len(a | b))


def baseline_row(row: dict, relation_same: bool, margin: float) -> dict:
    entity = tag_value(row["left_text"], "SUBJECT") == tag_value(row["right_text"], "SUBJECT")
    value = tag_value(row["left_text"], "VALUE") == tag_value(row["right_text"], "VALUE")
    operation = ("duplicate" if entity and relation_same and value else
                 "update" if entity and relation_same else "retain")
    return {"pair_id": row["pair_id"], "kind": row["kind"],
            "relation": row["left_relation_audit"],
            "gold": {field: bool(row[field]) for field in FIELDS[:-1]}
                    | {"operation": row["operation"]},
            "prediction": {"same_entity": entity, "same_relation": relation_same,
                           "same_value": value, "operation": operation},
            "margin": margin}


def fixed_delete_metrics(rows: list[dict], threshold: float = 0.0) -> dict:
    gold = [row["gold"]["operation"] != "retain" for row in rows]
    prediction = [row["prediction"]["operation"] != "retain" and row["margin"] >= threshold
                  for row in rows]
    return {"threshold": threshold, **binary_counts(gold, prediction), "feasible": True}


def calibrate_model_free(rows: list[dict]) -> dict:
    scored = [(row, trigram_similarity(row["left_text"], row["right_text"])) for row in rows]
    candidates_for_threshold = []
    for row, similarity in scored:
        # Expose every exact-subject item as a potential deletion; the threshold
        # alone decides whether its relation is compatible.
        same_entity = tag_value(row["left_text"], "SUBJECT") == tag_value(
            row["right_text"], "SUBJECT")
        candidate = baseline_row(row, same_entity, similarity)
        candidates_for_threshold.append(candidate)
    selected = select_threshold(candidates_for_threshold)
    trigram_rows = [baseline_row(row, similarity >= selected["threshold"], similarity)
                    for row, similarity in scored]
    trigram_summary = summarize(trigram_rows)
    trigram_summary["delete"] = fixed_delete_metrics(trigram_rows, selected["threshold"])
    exact_rows = [baseline_row(row, relation_surface(row["left_text"])
                              == relation_surface(row["right_text"]), similarity)
                  for row, similarity in scored]
    exact_summary = summarize(exact_rows)
    exact_summary["delete"] = fixed_delete_metrics(exact_rows)
    return {"exact_lexical": exact_summary, "trigram": trigram_summary}


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
    data_root = Path(args.data_root)
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    calibration_path = data_root / "calibration.jsonl"
    if (sha(data_root / "manifest.json") != audits["zero_shot"][0]["data"]["manifest_sha256"]
            or sha(calibration_path) != data_manifest["files_sha256"]["calibration.jsonl"]):
        raise ValueError("S1 model-free calibration data changed")
    calibration_rows = [json.loads(line) for line in calibration_path.read_text().splitlines()
                        if line]
    if [row["pair_id"] for row in calibration_rows] != pair_ids:
        raise ValueError("S1 model-free calibration rows are not aligned")
    summaries.update(calibrate_model_free(calibration_rows))
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
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--output", required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
