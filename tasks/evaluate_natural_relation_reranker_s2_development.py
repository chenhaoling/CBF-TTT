"""Apply the frozen S2 one-shot PubMed development gate."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tasks.extract_natural_relation_reranker_s2_embeddings import unique_mentions
from tasks.pack_cbf_event_warmup import sha
from tasks.score_natural_relation_reranker_s2 import PROTOCOL as SCORE_PROTOCOL
from tasks.select_natural_relation_reranker_s2 import (
    PROTOCOL as SELECTION_PROTOCOL, binary, delete_counts, retrieval_metrics,
)


PROTOCOL = "qwen_natural_relation_reranker_s2_development_v1"


def load_scores(root: str | Path, expected: int, selection_hash: str,
                expected_adapter: str | None) -> list[dict]:
    values = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.jsonl"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit.get("protocol") != SCORE_PROTOCOL or audit.get("split") != "development"
                or audit.get("candidate_kernel") != "cached" or audit.get("shard") != shard
                or audit.get("num_shards") != 2 or audit.get("semeval_test_loaded") is not False
                or audit.get("development_selection_sha256") != selection_hash
                or audit.get("adapter") != expected_adapter
                or sha(path) != audit.get("output_sha256")):
            raise ValueError("invalid S2 development score shard")
        values.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    values.sort(key=lambda row: row["index"])
    if len(values) != expected or [row["index"] for row in values] != list(range(expected)):
        raise ValueError("incomplete S2 development scores")
    return values


def load_embeddings(root: str | Path, raw_pairs: list[dict], selection_hash: str):
    import numpy as np

    values = []
    for shard in range(2):
        path = Path(root) / f"shard_{shard}.npz"
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        if (audit.get("split") != "development" or audit.get("shard") != shard
                or audit.get("num_shards") != 2 or audit.get("semeval_test_loaded") is not False
                or audit["data"].get("selection_sha256") != selection_hash
                or sha(path) != audit.get("output_sha256")):
            raise ValueError("invalid S2 development embedding shard")
        with np.load(path) as data:
            values.extend(zip(data["indices"].tolist(), data["source_ids"].tolist(),
                              data["relations"].tolist(), data["embeddings"].astype("float32")))
    values.sort()
    expected = unique_mentions(raw_pairs)
    if ([row[0] for row in values] != list(range(len(expected)))
            or [row[1] for row in values] != [row["source_id"] for row in expected]):
        raise ValueError("S2 development embeddings incomplete")
    matrix = np.stack([row[3] for row in values])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return expected, matrix


def fixed_summary(rows: list[dict], scenarios: list[dict], scores: dict[str, float],
                  threshold: float) -> dict:
    prediction = {row["pair_id"]: scores[row["pair_id"]] >= threshold for row in rows}
    gold = {row["pair_id"]: bool(row["same_relation"]) for row in rows}
    relation = binary(list(gold.values()), [prediction[key] for key in gold])
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["left_relation_audit"]].append(row)
    macro = statistics.fmean((value["recall"] + value["specificity"]) / 2
                             for subset in by_relation.values()
                             for value in [binary(
                                 [bool(row["same_relation"]) for row in subset],
                                 [prediction[row["pair_id"]] for row in subset])])
    kinds, hard_specificity = {}, None
    for kind in sorted({row["kind"] for row in scenarios}):
        subset = [row for row in scenarios if row["kind"] == kind]
        correct, deleted = [], []
        for row in subset:
            relation_value = prediction[row["relation_pair_id"]]
            action = ("duplicate" if row["same_entity"] and relation_value
                      and row["same_value"] else "update"
                      if row["same_entity"] and relation_value else "retain")
            correct.append(action == row["operation"]); deleted.append(action != "retain")
        kinds[kind] = statistics.fmean(correct)
        if kind == "same_entity_other_relation":
            hard_specificity = statistics.fmean(not value for value in deleted)
    return {"rows": len(rows), "threshold": threshold,
            "delete": {"threshold": threshold, **delete_counts(gold, prediction), "feasible": True},
            "same_relation": relation, "relation_macro_balanced_accuracy": macro,
            "same_entity_other_relation_specificity": hard_specificity,
            "by_kind_operation_accuracy": kinds}


def gate(selected: dict, baselines: dict, retrieval: dict) -> dict:
    safe = [value["delete"]["recall"] for value in baselines.values()
            if value["delete"]["precision"] >= .99]
    best_safe = max(safe, default=0.)
    cosine = baselines["frozen_qwen_cosine"]["delete"]["recall"]
    checks = {
        "retrieval_recall_at_16_at_least_95": retrieval["recall_at_16"] >= .95,
        "same_relation_recall_at_least_80": selected["same_relation"]["recall"] >= .80,
        "same_relation_specificity_at_least_95": selected["same_relation"]["specificity"] >= .95,
        "relation_macro_balanced_accuracy_at_least_85":
            selected["relation_macro_balanced_accuracy"] >= .85,
        "delete_precision_at_least_99": selected["delete"]["precision"] >= .99,
        "delete_recall_at_least_50": selected["delete"]["recall"] >= .50,
        "delete_false_positives_at_most_4": selected["delete"]["fp"] <= 4,
        "hard_relation_specificity_at_least_98":
            selected["same_entity_other_relation_specificity"] >= .98,
        "delete_recall_gain_at_least_10_points":
            selected["delete"]["recall"] - best_safe >= .10,
        "cosine_recall_gain_at_least_5_points":
            selected["delete"]["recall"] - cosine >= .05,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "best_safe_nonadapted_delete_recall": best_safe,
            "selected_delete_recall_gain": selected["delete"]["recall"] - best_safe,
            "selected_cosine_recall_gain": selected["delete"]["recall"] - cosine}


def evaluate(args):
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    selection_hash = sha(selection_path)
    if (selection.get("protocol") != SELECTION_PROTOCOL
            or selection.get("development_loaded") is not False
            or selection.get("semeval_test_loaded") is not False):
        raise ValueError("invalid frozen S2 selection")
    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    pair_path = data_root / "development_relation_pairs.jsonl"
    scenario_path = data_root / "development_scenarios.jsonl"
    if (sha(data_root / "manifest.json") != selection["data_manifest_sha256"]
            or sha(pair_path) != manifest["files_sha256"][pair_path.name]
            or sha(scenario_path) != manifest["files_sha256"][scenario_path.name]):
        raise ValueError("S2 development data changed")
    pairs = [json.loads(line) for line in pair_path.read_text().splitlines() if line]
    scenarios = [json.loads(line) for line in scenario_path.read_text().splitlines() if line]
    selected_scores = load_scores(args.selected_root, len(pairs), selection_hash,
                                  selection["selected_adapter"])
    zero_scores = load_scores(args.zero_shot_root, len(pairs), selection_hash, None)
    if ([row["pair_id"] for row in selected_scores]
            != [row["pair_id"] for row in zero_scores]
            or [row["pair_id"] for row in selected_scores]
            != [row["pair_id"] for row in pairs]):
        raise ValueError("S2 development rows not aligned")
    selected = fixed_summary(
        pairs, scenarios, {row["pair_id"]: row["relation_score"] for row in selected_scores},
        selection["selected_threshold"])
    zero = fixed_summary(
        pairs, scenarios, {row["pair_id"]: row["relation_score"] for row in zero_scores},
        selection["summaries"]["zero_shot"]["threshold"])
    trigram = fixed_summary(
        pairs, scenarios, {row["pair_id"]: row["trigram_similarity"] for row in pairs},
        selection["summaries"]["trigram"]["threshold"])
    mentions, matrix = load_embeddings(args.embedding_root, pairs, selection_hash)
    lookup = {row["source_id"]: index for index, row in enumerate(mentions)}
    cosine_scores = {row["pair_id"]: float(
        matrix[lookup[row["left_source_row_sha256"]]]
        @ matrix[lookup[row["right_source_row_sha256"]]]) for row in pairs}
    cosine = fixed_summary(pairs, scenarios, cosine_scores,
                           selection["summaries"]["frozen_qwen_cosine"]["threshold"])
    query_ids = list(dict.fromkeys(row["left_source_row_sha256"] for row in pairs))
    retrieval = retrieval_metrics(mentions, matrix, query_ids)
    baselines = {"zero_shot": zero, "trigram": trigram, "frozen_qwen_cosine": cosine}
    release = gate(selected, baselines, retrieval)
    result = {
        "protocol": PROTOCOL, "selected_arm": selection["selected_arm"],
        "selected_adapter": selection["selected_adapter"],
        "selected_threshold": selection["selected_threshold"],
        "selected_retrieval_k": selection["selected_retrieval_k"],
        "selected": selected, "baselines": baselines, "retrieval": retrieval,
        "gate": release, "selection_sha256": selection_hash,
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
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
