"""Independently reconstruct the one-shot S2 PubMed development gate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.audit_natural_relation_reranker_s2_selection import (
    binary, close, retrieval, weighted_delete,
)
from tasks.pack_cbf_event_warmup import sha


def load_scores(root, expected, selection_hash, adapter):
    rows = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "development" or audit["shard"] != shard
                or audit["candidate_kernel"] != "cached" or audit["adapter"] != adapter
                or audit["development_selection_sha256"] != selection_hash
                or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError("invalid development score")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(expected)):
        raise ValueError("development scores incomplete")
    return rows


def load_embeddings(root, raw, selection_hash):
    import numpy as np

    values = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.npz"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit["split"] != "development" or audit["shard"] != shard
                or audit["data"]["selection_sha256"] != selection_hash
                or audit["semeval_test_loaded"] is not False
                or sha(path) != audit["output_sha256"]):
            raise ValueError("invalid development embeddings")
        with np.load(path) as data:
            values.extend(zip(data["indices"].tolist(), data["source_ids"].tolist(),
                              data["relations"].tolist(), data["embeddings"].astype("float32")))
    values.sort()
    sources = {}
    for row in raw:
        for side in ("left", "right"):
            sources[row[f"{side}_source_row_sha256"]] = row[f"{side}_relation_audit"]
    ids = sorted(sources)
    if [value[0] for value in values] != list(range(len(ids))) or [value[1] for value in values] != ids:
        raise ValueError("development embedding order mismatch")
    matrix = np.stack([value[3] for value in values]); matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    return ids, np.asarray([value[2] for value in values]), matrix


def fixed(rows, scenarios, scores, threshold):
    prediction = {row["pair_id"]: scores[row["pair_id"]] >= threshold for row in rows}
    gold = {row["pair_id"]: bool(row["same_relation"]) for row in rows}
    relation = binary(list(gold.values()), [prediction[key] for key in gold])
    by_relation = defaultdict(list)
    for row in rows: by_relation[row["left_relation_audit"]].append(row)
    macro = statistics.fmean((value["recall"] + value["specificity"]) / 2
                             for subset in by_relation.values()
                             for value in [binary([bool(row["same_relation"]) for row in subset],
                                                  [prediction[row["pair_id"]] for row in subset])])
    kinds, hard = {}, None
    for kind in sorted({row["kind"] for row in scenarios}):
        subset = [row for row in scenarios if row["kind"] == kind]
        correct, deletes = [], []
        for row in subset:
            relation_value = prediction[row["relation_pair_id"]]
            action = ("duplicate" if row["same_entity"] and relation_value and row["same_value"]
                      else "update" if row["same_entity"] and relation_value else "retain")
            correct.append(action == row["operation"]); deletes.append(action != "retain")
        kinds[kind] = statistics.fmean(correct)
        if kind == "same_entity_other_relation": hard = statistics.fmean(not x for x in deletes)
    return {"rows": len(rows), "threshold": threshold,
            "delete": {"threshold": threshold, **weighted_delete(gold, prediction), "feasible": True},
            "same_relation": relation, "relation_macro_balanced_accuracy": macro,
            "same_entity_other_relation_specificity": hard,
            "by_kind_operation_accuracy": kinds}


def release(selected, baselines, retrieval_value):
    safe = [value["delete"]["recall"] for value in baselines.values()
            if value["delete"]["precision"] >= .99]
    best = max(safe, default=0.)
    cosine = baselines["frozen_qwen_cosine"]["delete"]["recall"]
    checks = {
        "retrieval_recall_at_16_at_least_95": retrieval_value["recall_at_16"] >= .95,
        "same_relation_recall_at_least_80": selected["same_relation"]["recall"] >= .8,
        "same_relation_specificity_at_least_95": selected["same_relation"]["specificity"] >= .95,
        "relation_macro_balanced_accuracy_at_least_85": selected["relation_macro_balanced_accuracy"] >= .85,
        "delete_precision_at_least_99": selected["delete"]["precision"] >= .99,
        "delete_recall_at_least_50": selected["delete"]["recall"] >= .5,
        "delete_false_positives_at_most_4": selected["delete"]["fp"] <= 4,
        "hard_relation_specificity_at_least_98": selected["same_entity_other_relation_specificity"] >= .98,
        "delete_recall_gain_at_least_10_points": selected["delete"]["recall"] - best >= .1,
        "cosine_recall_gain_at_least_5_points": selected["delete"]["recall"] - cosine >= .05,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "best_safe_nonadapted_delete_recall": best,
            "selected_delete_recall_gain": selected["delete"]["recall"] - best,
            "selected_cosine_recall_gain": selected["delete"]["recall"] - cosine}


def audit(args):
    result_path, selection_path = Path(args.result), Path(args.selection)
    produced, selection = json.loads(result_path.read_text()), json.loads(selection_path.read_text())
    selection_hash = sha(selection_path)
    data_root = Path(args.data_root)
    pairs = [json.loads(line) for line in
             (data_root / "development_relation_pairs.jsonl").read_text().splitlines() if line]
    scenarios = [json.loads(line) for line in
                 (data_root / "development_scenarios.jsonl").read_text().splitlines() if line]
    selected_rows = load_scores(args.selected_root, len(pairs), selection_hash,
                                selection["selected_adapter"])
    zero_rows = load_scores(args.zero_shot_root, len(pairs), selection_hash, None)
    if not ([row["pair_id"] for row in selected_rows]
            == [row["pair_id"] for row in zero_rows]
            == [row["pair_id"] for row in pairs]):
        raise ValueError("development rows misaligned")
    selected = fixed(pairs, scenarios, {row["pair_id"]: row["relation_score"]
                                        for row in selected_rows}, selection["selected_threshold"])
    zero = fixed(pairs, scenarios, {row["pair_id"]: row["relation_score"] for row in zero_rows},
                 selection["summaries"]["zero_shot"]["threshold"])
    trigram = fixed(pairs, scenarios, {row["pair_id"]: row["trigram_similarity"] for row in pairs},
                    selection["summaries"]["trigram"]["threshold"])
    ids, relations, matrix = load_embeddings(args.embedding_root, pairs, selection_hash)
    lookup = {value: index for index, value in enumerate(ids)}
    cosine_scores = {row["pair_id"]: float(matrix[lookup[row["left_source_row_sha256"]]]
                                                   @ matrix[lookup[row["right_source_row_sha256"]]])
                     for row in pairs}
    cosine = fixed(pairs, scenarios, cosine_scores,
                   selection["summaries"]["frozen_qwen_cosine"]["threshold"])
    query_ids = list(dict.fromkeys(row["left_source_row_sha256"] for row in pairs))
    retrieval_value = retrieval(ids, relations, matrix, query_ids)
    baselines = {"zero_shot": zero, "trigram": trigram, "frozen_qwen_cosine": cosine}
    gate = release(selected, baselines, retrieval_value)
    checks = {
        "selected": close(selected, produced["selected"]),
        "baselines": close(baselines, produced["baselines"]),
        "retrieval": close(retrieval_value, produced["retrieval"]),
        "gate": close(gate, produced["gate"]),
        "terminal": produced["terminal_status"]
            == ("passed_development" if gate["passed"] else "stopped_by_development"),
        "selection": produced["selection_sha256"] == selection_hash,
        "sealed": produced["development_loaded_once"] is True
            and produced["semeval_test_loaded"] is False,
    }
    passed = all(checks.values())
    result = {"protocol": "qwen_natural_relation_reranker_s2_development_audit_v1",
              "passed": passed, "checks": checks, "gate_passed": gate["passed"],
              "development_relation_pairs_reconstructed": len(pairs),
              "result_sha256": sha(result_path), "semeval_test_loaded": False}
    if not passed: raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selected-root", required=True)
    parser.add_argument("--zero-shot-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); value = audit(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
