"""Independently reconstruct S2 calibration selection and retrieval metrics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_natural_relation_reranker_s2_selection_v1"
ARMS = ("zero_shot", "epoch_0_25", "epoch_0_50", "epoch_1_00")


def close(left, right, tolerance=1e-10):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
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


def weighted_delete(gold, prediction):
    groups = len(gold) // 2
    predicted_positive = sum(prediction[key] for key, value in gold.items() if value)
    predicted_negative = sum(prediction[key] for key, value in gold.items() if not value)
    tp, fn, fp = 2 * predicted_positive, 2 * (groups - predicted_positive), predicted_negative
    tn = 3 * groups - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.,
            "recall": tp / (tp + fn) if tp + fn else 1.,
            "specificity": tn / (tn + fp) if tn + fp else 1.}


def threshold(rows, scores):
    gold = {row["pair_id"]: bool(row["same_relation"]) for row in rows}
    ordered = sorted((scores[key], key) for key in gold)[::-1]
    prediction = {key: False for key in gold}
    candidates = [(ordered[0][0] + 1., weighted_delete(gold, prediction))]
    cursor = 0
    while cursor < len(ordered):
        value = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == value:
            prediction[ordered[cursor][1]] = True
            cursor += 1
        candidates.append((value, weighted_delete(gold, prediction)))
    value, metrics = max((item for item in candidates if item[1]["precision"] >= .99),
                         key=lambda item: (item[1]["recall"], item[1]["precision"], item[0]))
    return value, {"threshold": value, **metrics, "feasible": True}


def summary(rows, scenarios, scores):
    selected_threshold, delete = threshold(rows, scores)
    prediction = {row["pair_id"]: scores[row["pair_id"]] >= selected_threshold for row in rows}
    relation = binary([bool(row["same_relation"]) for row in rows],
                      [prediction[row["pair_id"]] for row in rows])
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["left_relation_audit"]].append(row)
    macro = statistics.fmean((value["recall"] + value["specificity"]) / 2
                             for subset in by_relation.values()
                             for value in [binary(
                                 [bool(row["same_relation"]) for row in subset],
                                 [prediction[row["pair_id"]] for row in subset])])
    kinds = {}
    for kind in sorted({row["kind"] for row in scenarios}):
        subset = [row for row in scenarios if row["kind"] == kind]
        correct = []
        for row in subset:
            relation_value = prediction[row["relation_pair_id"]]
            action = ("duplicate" if row["same_entity"] and relation_value
                      and row["same_value"] else "update"
                      if row["same_entity"] and relation_value else "retain")
            correct.append(action == row["operation"])
        kinds[kind] = statistics.fmean(correct)
    return {"rows": len(rows), "threshold": selected_threshold, "delete": delete,
            "same_relation": relation, "relation_macro_balanced_accuracy": macro,
            "by_kind_operation_accuracy": kinds}


def load_scored(root, arm, expected):
    values = []
    for shard in range(2):
        path = Path(root) / arm / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "calibration" or audit["candidate_kernel"] != "cached"
                or audit["shard"] != shard or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError("invalid calibration score shard")
        values.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    values.sort(key=lambda row: row["index"])
    if [row["index"] for row in values] != list(range(expected)):
        raise ValueError("score rows incomplete")
    return values


def embeddings(root, raw_pairs):
    import numpy as np

    values = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.npz"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "calibration" or audit["shard"] != shard
                or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError("invalid calibration embeddings")
        with np.load(path) as data:
            values.extend(zip(data["indices"].tolist(), data["source_ids"].tolist(),
                              data["relations"].tolist(), data["embeddings"].astype("float32")))
    values.sort()
    source = {}
    for row in raw_pairs:
        for side in ("left", "right"):
            source[row[f"{side}_source_row_sha256"]] = row[f"{side}_relation_audit"]
    expected_ids = sorted(source)
    if [item[0] for item in values] != list(range(len(expected_ids))) or [item[1] for item in values] != expected_ids:
        raise ValueError("embedding order mismatch")
    matrix = np.stack([item[3] for item in values]); matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    relations = np.asarray([item[2] for item in values])
    return expected_ids, relations, matrix


def retrieval(ids, relations, matrix, query_ids):
    import numpy as np

    lookup = {value: index for index, value in enumerate(ids)}
    hits = {4: 0, 8: 0, 16: 0}; reciprocal = []
    for start in range(0, len(query_ids), 256):
        query = np.asarray([lookup[value] for value in query_ids[start:start + 256]])
        scores = matrix[query] @ matrix.T; scores[np.arange(len(query)), query] = -np.inf
        top = np.argpartition(-scores, 15, axis=1)[:, :16]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)
        truth = relations[top] == relations[query, None]
        for k in hits: hits[k] += int(truth[:, :k].any(1).sum())
        reciprocal.extend(1. / (truth.argmax(1) + 1) * truth.any(1))
    return {"queries": len(query_ids), **{f"recall_at_{k}": hits[k] / len(query_ids) for k in hits},
            "mrr_at_16": statistics.fmean(reciprocal)}


def audit(args):
    result_path = Path(args.selection)
    produced = json.loads(result_path.read_text())
    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    pair_path = data_root / "calibration_relation_pairs.jsonl"
    scenario_path = data_root / "calibration_scenarios.jsonl"
    rows = [json.loads(line) for line in pair_path.read_text().splitlines() if line]
    scenarios = [json.loads(line) for line in scenario_path.read_text().splitlines() if line]
    summaries, score_rows = {}, {}
    for arm in ARMS:
        scored = load_scored(args.score_root, arm, len(rows)); score_rows[arm] = scored
        if [row["pair_id"] for row in scored] != [row["pair_id"] for row in rows]:
            raise ValueError("score pair order mismatch")
        summaries[arm] = summary(rows, scenarios,
                                 {row["pair_id"]: row["relation_score"] for row in scored})
    summaries["trigram"] = summary(
        rows, scenarios, {row["pair_id"]: row["trigram_similarity"] for row in rows})
    ids, relations, matrix = embeddings(args.embedding_root, rows)
    lookup = {value: index for index, value in enumerate(ids)}
    cosine = {row["pair_id"]: float(matrix[lookup[row["left_source_row_sha256"]]]
                                              @ matrix[lookup[row["right_source_row_sha256"]]])
              for row in rows}
    summaries["frozen_qwen_cosine"] = summary(rows, scenarios, cosine)
    query_ids = list(dict.fromkeys(row["left_source_row_sha256"] for row in rows))
    retrieval_value = retrieval(ids, relations, matrix, query_ids)
    feasible_k = [k for k in (4, 8, 16) if retrieval_value[f"recall_at_{k}"] >= .95]
    selected_k = min(feasible_k) if feasible_k else max(
        (4, 8, 16), key=lambda k: (retrieval_value[f"recall_at_{k}"], -k))
    adapter_arms = list(ARMS[1:])
    selected_arm = max(adapter_arms, key=lambda arm: (
        summaries[arm]["delete"]["recall"],
        summaries[arm]["relation_macro_balanced_accuracy"], -adapter_arms.index(arm)))
    checks = {
        "protocol": produced["protocol"] == PROTOCOL,
        "data": produced["data_manifest_sha256"] == sha(data_root / "manifest.json")
            and produced["data_audit_sha256"] == sha(data_root / "data_audit.json")
            and sha(pair_path) == manifest["files_sha256"][pair_path.name]
            and sha(scenario_path) == manifest["files_sha256"][scenario_path.name],
        "summaries": close(summaries, produced["summaries"]),
        "retrieval": close(retrieval_value, produced["retrieval"]),
        "selection": selected_arm == produced["selected_arm"]
            and close(summaries[selected_arm]["threshold"], produced["selected_threshold"])
            and selected_k == produced["selected_retrieval_k"],
        "sealed": produced["development_loaded"] is False
            and produced["semeval_test_loaded"] is False,
    }
    passed = all(checks.values())
    result = {"protocol": "qwen_natural_relation_reranker_s2_selection_audit_v1",
              "passed": passed, "checks": checks, "selected_arm": selected_arm,
              "selected_retrieval_k": selected_k,
              "calibration_rows_reconstructed": len(rows),
              "selection_sha256": sha(result_path), "development_loaded": False,
              "semeval_test_loaded": False}
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = audit(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
