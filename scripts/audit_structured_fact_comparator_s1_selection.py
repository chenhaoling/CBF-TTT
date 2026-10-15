"""Independently reconstruct the S1 calibration selection before development."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


ARMS = ("zero_shot", "epoch_0_25", "epoch_0_50", "epoch_1_00", "epoch_2_00")
FIELDS = ("same_entity", "same_relation", "same_value", "operation")
PATTERNS = {name: re.compile(rf"\[{name}\](.*?)\[/{name}\]")
            for name in ("SUBJECT", "VALUE")}


def close(left, right, tolerance=1e-12):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance)
                                               for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def binary(gold, prediction):
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.,
            "recall": tp / (tp + fn) if tp + fn else 1.,
            "specificity": tn / (tn + fp) if tn + fp else 1.}


def threshold(rows):
    gold = [row["gold"]["operation"] != "retain" for row in rows]
    choices = sorted(((row["margin"], index) for index, row in enumerate(rows)
                      if row["prediction"]["operation"] != "retain"), reverse=True)
    prediction = [False] * len(rows)
    candidates = [(choices[0][0] + 1 if choices else 1., binary(gold, prediction))]
    cursor = 0
    while cursor < len(choices):
        value = choices[cursor][0]
        while cursor < len(choices) and choices[cursor][0] == value:
            prediction[choices[cursor][1]] = True
            cursor += 1
        candidates.append((value, binary(gold, prediction)))
    feasible = [(value, metrics) for value, metrics in candidates if metrics["precision"] >= .99]
    value, metrics = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]))
    return {"threshold": value, **metrics, "feasible": True}


def summary(rows):
    relation = binary([row["gold"]["same_relation"] for row in rows],
                      [row["prediction"]["same_relation"] for row in rows])
    groups = defaultdict(list)
    for row in rows:
        groups[row["relation"]].append(row)
    macro = statistics.fmean((lambda value: (value["recall"] + value["specificity"]) / 2)(
        binary([row["gold"]["same_relation"] for row in subset],
               [row["prediction"]["same_relation"] for row in subset]))
        for subset in groups.values())
    kinds = {kind: statistics.fmean(row["gold"]["operation"]
                                    == row["prediction"]["operation"]
                                    for row in rows if row["kind"] == kind)
             for kind in sorted({row["kind"] for row in rows})}
    return {"rows": len(rows),
            "exact_four_field_accuracy": statistics.fmean(
                all(row["gold"][field] == row["prediction"][field] for field in FIELDS)
                for row in rows),
            "same_relation": relation, "relation_macro_balanced_accuracy": macro,
            "by_kind_operation_accuracy": kinds, "delete": threshold(rows)}


def normalized(value):
    return " ".join(value.casefold().split())


def visible(text, name):
    values = PATTERNS[name].findall(text)
    if len(values) != 1:
        raise ValueError("invalid S1 tag")
    return normalized(values[0])


def surface(text):
    for name, pattern in PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return normalized(text)


def similarity(left, right):
    def grams(value):
        value = f"  {value}  "
        return {value[index:index + 3] for index in range(max(1, len(value) - 2))}
    a, b = grams(surface(left)), grams(surface(right))
    return len(a & b) / max(1, len(a | b))


def baseline(raw, relation_same, margin):
    entity = visible(raw["left_text"], "SUBJECT") == visible(raw["right_text"], "SUBJECT")
    value = visible(raw["left_text"], "VALUE") == visible(raw["right_text"], "VALUE")
    operation = ("duplicate" if entity and relation_same and value else
                 "update" if entity and relation_same else "retain")
    return {"kind": raw["kind"], "relation": raw["left_relation_audit"],
            "gold": {field: bool(raw[field]) for field in FIELDS[:-1]}
                    | {"operation": raw["operation"]},
            "prediction": {"same_entity": entity, "same_relation": relation_same,
                           "same_value": value, "operation": operation}, "margin": margin}


def fixed_delete(rows, value):
    result = binary([row["gold"]["operation"] != "retain" for row in rows],
                    [row["prediction"]["operation"] != "retain" and row["margin"] >= value
                     for row in rows])
    return {"threshold": value, **result, "feasible": True}


def model_free(raw):
    scored = [(row, similarity(row["left_text"], row["right_text"])) for row in raw]
    possible = [baseline(row, visible(row["left_text"], "SUBJECT")
                         == visible(row["right_text"], "SUBJECT"), score)
                for row, score in scored]
    selected = threshold(possible)
    tri_rows = [baseline(row, score >= selected["threshold"], score) for row, score in scored]
    tri = summary(tri_rows); tri["delete"] = fixed_delete(tri_rows, selected["threshold"])
    exact_rows = [baseline(row, surface(row["left_text"]) == surface(row["right_text"]), score)
                  for row, score in scored]
    exact = summary(exact_rows); exact["delete"] = fixed_delete(exact_rows, 0.)
    return {"exact_lexical": exact, "trigram": tri}


def load_arm(root, name):
    rows = []
    for shard in range(2):
        path = root / name / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "calibration" or audit["semeval_test_loaded"] is not False
                or audit.get("candidate_kernel") != "cached"):
            raise ValueError("invalid calibration audit")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(26805)):
        raise ValueError("calibration arm incomplete")
    return rows


def audit(score_root, data_root):
    score_root, data_root = Path(score_root), Path(data_root)
    selection_path = score_root / "selection.json"
    selected = json.loads(selection_path.read_text())
    reconstructed, pair_ids = {}, None
    arms = {}
    for name in ARMS:
        rows = load_arm(score_root, name)
        arms[name] = rows
        ids = [row["pair_id"] for row in rows]
        if pair_ids is not None and ids != pair_ids:
            raise ValueError("calibration arms misaligned")
        pair_ids = ids
        reconstructed[name] = summary(rows)
    raw = [json.loads(line) for line in (data_root / "calibration.jsonl").read_text().splitlines()
           if line]
    if [row["pair_id"] for row in raw] != pair_ids:
        raise ValueError("calibration raw rows misaligned")
    reconstructed.update(model_free(raw))
    adapter_names = ARMS[1:]
    selected_name = max(adapter_names, key=lambda name: (
        reconstructed[name]["delete"]["feasible"], reconstructed[name]["delete"]["recall"],
        reconstructed[name]["exact_four_field_accuracy"],
        reconstructed[name]["relation_macro_balanced_accuracy"],
        -reconstructed[name]["delete"]["threshold"]))
    checks = {
        "summaries": close(reconstructed, selected["summaries"]),
        "selection": selected_name == selected["selected_arm"],
        "threshold": close(reconstructed[selected_name]["delete"]["threshold"],
                           selected["selected_threshold"]),
        "rows": selected["calibration_rows"] == 26805,
        "sealed": selected["development_loaded"] is False
            and selected["semeval_test_loaded"] is False,
        "data": selected["data_manifest_sha256"] == sha(data_root / "manifest.json"),
    }
    passed = all(checks.values())
    result = {"protocol": "qwen_structured_fact_comparator_s1_selection_audit_v1",
              "passed": passed, "checks": checks, "selection_sha256": sha(selection_path),
              "selected_arm": selected_name, "calibration_rows_reconstructed": len(raw),
              "development_loaded": False, "semeval_test_loaded": False}
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.score_root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
