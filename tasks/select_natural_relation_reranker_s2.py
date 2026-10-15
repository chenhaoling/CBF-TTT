"""Select S2 reranker checkpoint, safety threshold and retrieval k on calibration."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.extract_natural_relation_reranker_s2_embeddings import (
    PROTOCOL as EMBEDDING_PROTOCOL, unique_mentions,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.score_natural_relation_reranker_s2 import PROTOCOL as SCORE_PROTOCOL
from tasks.train_natural_relation_reranker_s2 import DATA_PROTOCOL


PROTOCOL = "qwen_natural_relation_reranker_s2_selection_v1"
TRAIN_AUDIT_PROTOCOL = "qwen_natural_relation_reranker_s2_training_audit_v1"


def binary(gold: list[bool], prediction: list[bool]) -> dict:
    tp = sum(a and b for a, b in zip(gold, prediction))
    fp = sum(not a and b for a, b in zip(gold, prediction))
    fn = sum(a and not b for a, b in zip(gold, prediction))
    tn = sum(not a and not b for a, b in zip(gold, prediction))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def delete_counts(pair_gold: dict[str, bool], pair_predictions: dict[str, bool]) -> dict:
    groups = len(pair_gold) // 2
    predicted_positive = sum(pair_predictions[pair_id] for pair_id, gold in pair_gold.items()
                             if gold)
    predicted_negative = sum(pair_predictions[pair_id] for pair_id, gold in pair_gold.items()
                             if not gold)
    tp = 2 * predicted_positive
    fn = 2 * (groups - predicted_positive)
    fp = predicted_negative
    tn = 3 * groups - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0}


def select_threshold(rows: list[dict], scores: dict[str, float], minimum_precision=.99) -> dict:
    pair_gold = {row["pair_id"]: bool(row["same_relation"]) for row in rows}
    ordered = sorted(((scores[pair_id], pair_id) for pair_id in pair_gold), reverse=True)
    prediction = {pair_id: False for pair_id in pair_gold}
    initial = ordered[0][0] + 1.0 if ordered else 1.0
    candidates = [(initial, delete_counts(pair_gold, prediction))]
    cursor = 0
    while cursor < len(ordered):
        threshold = ordered[cursor][0]
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            prediction[ordered[cursor][1]] = True
            cursor += 1
        candidates.append((threshold, delete_counts(pair_gold, prediction)))
    feasible = [(threshold, value) for threshold, value in candidates
                if value["precision"] >= minimum_precision]
    threshold, value = max(feasible, key=lambda item: (
        item[1]["recall"], item[1]["precision"], item[0]))
    return {"threshold": threshold, **value, "feasible": True}


def relation_macro(rows: list[dict], prediction: dict[str, bool]) -> float:
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["left_relation_audit"]].append(row)
    values = []
    for subset in by_relation.values():
        metrics = binary([bool(row["same_relation"]) for row in subset],
                         [prediction[row["pair_id"]] for row in subset])
        values.append((metrics["recall"] + metrics["specificity"]) / 2)
    return statistics.fmean(values)


def summarize(rows: list[dict], scenarios: list[dict], scores: dict[str, float]) -> dict:
    selected = select_threshold(rows, scores)
    prediction = {row["pair_id"]: scores[row["pair_id"]] >= selected["threshold"]
                  for row in rows}
    relation = binary([bool(row["same_relation"]) for row in rows],
                      [prediction[row["pair_id"]] for row in rows])
    by_kind = {}
    for kind in sorted({row["kind"] for row in scenarios}):
        subset = [row for row in scenarios if row["kind"] == kind]
        correct = 0
        for row in subset:
            relation_value = prediction[row["relation_pair_id"]]
            predicted_operation = ("duplicate" if row["same_entity"] and relation_value
                                   and row["same_value"] else "update"
                                   if row["same_entity"] and relation_value else "retain")
            correct += predicted_operation == row["operation"]
        by_kind[kind] = correct / len(subset)
    return {"rows": len(rows), "threshold": selected["threshold"],
            "delete": selected, "same_relation": relation,
            "relation_macro_balanced_accuracy": relation_macro(rows, prediction),
            "by_kind_operation_accuracy": by_kind}


def load_arm(root: str | Path, name: str, expected_rows: int):
    root = Path(root) / name
    rows, audits = [], []
    for shard in range(2):
        path = root / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit.get("protocol") != SCORE_PROTOCOL or audit.get("split") != "calibration"
                or audit.get("candidate_kernel") != "cached" or audit.get("shard") != shard
                or audit.get("num_shards") != 2 or audit.get("semeval_test_loaded") is not False
                or sha(path) != audit.get("output_sha256")):
            raise ValueError(f"invalid S2 calibration shard: {name}/{shard}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        audits.append(audit)
    rows.sort(key=lambda row: row["index"])
    if len(rows) != expected_rows or [row["index"] for row in rows] != list(range(expected_rows)):
        raise ValueError(f"incomplete S2 calibration arm: {name}")
    return rows, audits


def load_embeddings(root: str | Path, raw_pairs: list[dict]):
    import numpy as np

    root = Path(root)
    arrays, audits = [], []
    for shard in range(2):
        path = root / f"shard_{shard}.npz"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit.get("protocol") != EMBEDDING_PROTOCOL or audit.get("split") != "calibration"
                or audit.get("shard") != shard or audit.get("num_shards") != 2
                or audit.get("semeval_test_loaded") is not False
                or sha(path) != audit.get("output_sha256")):
            raise ValueError(f"invalid S2 calibration embedding shard {shard}")
        with np.load(path) as value:
            arrays.extend(zip(value["indices"].tolist(), value["source_ids"].tolist(),
                              value["relations"].tolist(), value["embeddings"].astype("float32")))
        audits.append(audit)
    arrays.sort(key=lambda value: value[0])
    expected = unique_mentions(raw_pairs)
    if [value[0] for value in arrays] != list(range(len(expected))):
        raise ValueError("S2 calibration embedding indices incomplete")
    if [value[1] for value in arrays] != [row["source_id"] for row in expected]:
        raise ValueError("S2 calibration embedding source order mismatch")
    matrix = np.stack([value[3] for value in arrays])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return expected, matrix, audits


def retrieval_metrics(mentions: list[dict], matrix, query_ids: list[str], ks=(4, 8, 16)) -> dict:
    import numpy as np

    id_to_index = {row["source_id"]: index for index, row in enumerate(mentions)}
    relations = np.asarray([row["relation"] for row in mentions])
    hits = {k: 0 for k in ks}
    reciprocal = []
    for start in range(0, len(query_ids), 256):
        indices = np.asarray([id_to_index[value] for value in query_ids[start:start + 256]])
        scores = matrix[indices] @ matrix.T
        scores[np.arange(len(indices)), indices] = -np.inf
        top = np.argpartition(-scores, min(max(ks) - 1, len(mentions) - 1), axis=1)[:, :max(ks)]
        top_scores = np.take_along_axis(scores, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
        truth = relations[top] == relations[indices, None]
        for k in ks:
            hits[k] += int(truth[:, :k].any(1).sum())
        reciprocal.extend(1.0 / (truth.argmax(1) + 1) * truth.any(1))
    return {"queries": len(query_ids),
            **{f"recall_at_{k}": hits[k] / len(query_ids) for k in ks},
            "mrr_at_16": statistics.fmean(reciprocal)}


def select_retrieval_k(metrics: dict, ks=(4, 8, 16)) -> int:
    feasible = [k for k in ks if metrics[f"recall_at_{k}"] >= .95]
    if feasible:
        return min(feasible)
    return max(ks, key=lambda k: (metrics[f"recall_at_{k}"], -k))


def select(args):
    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest.get("protocol") != DATA_PROTOCOL or manifest.get("semeval_test_parsed") is not False:
        raise ValueError("invalid S2 data manifest")
    pair_path, scenario_path = (data_root / "calibration_relation_pairs.jsonl",
                                data_root / "calibration_scenarios.jsonl")
    if (sha(pair_path) != manifest["files_sha256"][pair_path.name]
            or sha(scenario_path) != manifest["files_sha256"][scenario_path.name]):
        raise ValueError("S2 calibration data changed")
    raw_pairs = [json.loads(line) for line in pair_path.read_text().splitlines() if line]
    scenarios = [json.loads(line) for line in scenario_path.read_text().splitlines() if line]
    names = [value.strip() for value in args.arms.split(",") if value.strip()]
    if names != ["zero_shot", "epoch_0_25", "epoch_0_50", "epoch_1_00"]:
        raise ValueError("unexpected S2 calibration arms")
    summaries, audits = {}, {}
    for name in names:
        scored, arm_audits = load_arm(args.score_root, name, len(raw_pairs))
        if [row["pair_id"] for row in scored] != [row["pair_id"] for row in raw_pairs]:
            raise ValueError(f"S2 score row mismatch: {name}")
        score_map = {row["pair_id"]: row["relation_score"] for row in scored}
        summaries[name] = summarize(raw_pairs, scenarios, score_map)
        audits[name] = arm_audits
    summaries["trigram"] = summarize(
        raw_pairs, scenarios, {row["pair_id"]: row["trigram_similarity"] for row in raw_pairs})

    mentions, matrix, embedding_audits = load_embeddings(args.embedding_root, raw_pairs)
    id_to_index = {row["source_id"]: index for index, row in enumerate(mentions)}
    cosine_scores = {row["pair_id"]: float(
        matrix[id_to_index[row["left_source_row_sha256"]]]
        @ matrix[id_to_index[row["right_source_row_sha256"]]]) for row in raw_pairs}
    summaries["frozen_qwen_cosine"] = summarize(raw_pairs, scenarios, cosine_scores)
    query_ids = list(dict.fromkeys(row["left_source_row_sha256"] for row in raw_pairs))
    retrieval = retrieval_metrics(mentions, matrix, query_ids)
    retrieval_k = select_retrieval_k(retrieval)

    adapter_names = names[1:]
    selected_name = max(adapter_names, key=lambda name: (
        summaries[name]["delete"]["recall"],
        summaries[name]["relation_macro_balanced_accuracy"],
        -adapter_names.index(name)))
    selected_adapter = audits[selected_name][0]["adapter"]
    training_audit = json.loads(Path(args.training_audit).read_text())
    if training_audit.get("protocol") != TRAIN_AUDIT_PROTOCOL or training_audit.get("passed") is not True:
        raise ValueError("S2 selection requires passed training audit")
    result = {
        "protocol": PROTOCOL, "selected_arm": selected_name,
        "selected_adapter": selected_adapter,
        "selected_threshold": summaries[selected_name]["threshold"],
        "selected_retrieval_k": retrieval_k, "minimum_delete_precision": .99,
        "summaries": summaries, "retrieval": retrieval,
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_root / "data_audit.json"),
        "training_audit_sha256": sha(args.training_audit),
        "score_audits_sha256": {name: [sha(Path(args.score_root) / name
                                                  / f"shard_{shard}.audit.json")
                                       for shard in range(2)] for name in names},
        "embedding_audits_sha256": [sha(Path(args.embedding_root)
                                           / f"shard_{shard}.audit.json") for shard in range(2)],
        "calibration_relation_pairs": len(raw_pairs),
        "development_loaded": False, "semeval_test_loaded": False,
        "terminal_status": "selected_before_development",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--output", required=True)
    select(parser.parse_args())


if __name__ == "__main__":
    main()
