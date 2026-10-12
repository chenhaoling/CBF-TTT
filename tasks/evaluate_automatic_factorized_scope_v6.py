"""Evaluate automatic text-only entity/relation factorization for scope ranking."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256
from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_relation_supervised_scope_v5 import (
    RelationClassifier, ensemble_logits, load_v4, relation_feature,
)
from tasks.train_scoped_delta_ranker_v4 import load_corpus


PROTOCOL = "automatic_factorized_scope_v6"
LAMBDAS = (0., .5, 1., 2., 4., 8.)
WINDOW = 10


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def source_documents(path):
    path = Path(path)
    if sha(path) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    output = []
    for row in json.loads(path.read_text()):
        rewrite = row["requested_rewrite"]; subject = rewrite["subject"]
        texts = [rewrite["prompt"].format(subject), *row.get("paraphrase_prompts", [])]
        output.extend((subject, text) for text in texts)
    return output


def fit_idf(documents, excluded_subjects):
    frequency = Counter(); count = 0
    for subject, text in documents:
        if subject in excluded_subjects:
            continue
        frequency.update(set(tokens(text))); count += 1
    return {token: math.log((count + 1) / (value + 1)) + 1
            for token, value in frequency.items()}, count


def _weight(token, idf):
    return idf.get(token, math.log(max(2, len(idf))))


def automatic_pair_view(query, key, idf, window=WINDOW):
    """Return entity evidence and pair-specific relation windows from text only."""
    left, right = tokens(query), tokens(key)
    if not left or not right:
        raise ValueError("empty scope text")
    blocks = [block for block in SequenceMatcher(None, left, right, autojunk=False)
              .get_matching_blocks() if block.size]
    if blocks:
        block = max(blocks, key=lambda row: (
            sum(_weight(token, idf) for token in left[row.a:row.a + row.size]),
            row.size, -row.a, -row.b))
        q_start, k_start, size = block.a, block.b, block.size
    else:
        shared = set(left) & set(right)
        if shared:
            token = max(shared, key=lambda value: _weight(value, idf))
            q_start, k_start, size = left.index(token), right.index(token), 1
        else:
            q_start = k_start = 0; size = 0
    left_weight = sum(_weight(value, idf) for value in left)
    right_weight = sum(_weight(value, idf) for value in right)
    denominator = max(1e-8, min(left_weight, right_weight))
    common = Counter(left) & Counter(right)
    weighted_overlap = sum(count * _weight(value, idf)
                           for value, count in common.items()) / denominator
    block_mass = sum(_weight(value, idf) for value in left[q_start:q_start + size]) \
        / denominator if size else 0.
    entity_score = .7 * weighted_overlap + .3 * block_mass

    def relation_view(values, start):
        if size:
            merged = [*values[:start], "<ENTITY>", *values[start + size:]]
            marker = start
        else:
            merged = values; marker = len(values) // 2
        begin = max(0, marker - window); end = min(len(merged), marker + window + 1)
        return " ".join(merged[begin:end])

    return {"entity_score": entity_score,
            "query_relation": relation_view(left, q_start),
            "key_relation": relation_view(right, k_start),
            "matched_tokens": left[q_start:q_start + size],
            "weighted_overlap": weighted_overlap, "block_mass": block_mass}


def build_fold_views(groups, fold_map, documents):
    output, idf_sizes = {}, {}
    for fold in range(FOLDS):
        selected = [group for group in groups if fold_map[group["session_id"]] == fold]
        held_subjects = {group["query_subject"] for group in selected}
        idf, size = fit_idf(documents, held_subjects); idf_sizes[str(fold)] = size
        fold_rows = []
        for group in selected:
            candidates = [automatic_pair_view(group["query"], row["prompt"], idf)
                          for row in group["candidates"]]
            fold_rows.append({"group": group, "candidates": candidates})
        output[str(fold)] = fold_rows
    return output, idf_sizes


def relation_features(texts, embeddings):
    return {text: relation_feature(text, embeddings) for text in texts}


def fold_scores(fold_rows, features, artifacts):
    import torch

    output = []
    for row in fold_rows:
        relation, entity = [], []
        for candidate in row["candidates"]:
            logits = ensemble_logits([candidate["query_relation"], candidate["key_relation"]],
                                     features, artifacts)
            relation.append(float(logits[0].softmax(-1) @ logits[1].softmax(-1)))
            entity.append(candidate["entity_score"])
        output.append({"group": row["group"], "relation": torch.tensor(relation),
                       "entity": torch.tensor(entity)})
    return output


def predict(rows, weight, fold):
    output = []
    for row in rows:
        scores = (row["relation"].clamp_min(1e-12).log() + weight * row["entity"])
        target = row["group"]["target"]; prediction = int(scores.argmax())
        order = scores.argsort(descending=True).tolist()
        output.append({"session_id": row["group"]["session_id"],
                       "group_id": row["group"]["group_id"],
                       "target": target, "prediction": prediction,
                       "correct": int(prediction == target),
                       "reciprocal_rank": 1 / (order.index(target) + 1),
                       "weight": weight, "fold": fold})
    return output


def metrics(rows):
    return {"events": len(rows),
            "top1_accuracy": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows)}


def select_weight(rows):
    grid = {str(value): metrics(predict(rows, value, "inner")) for value in LAMBDAS}
    selected = max(LAMBDAS, key=lambda value: (
        grid[str(value)]["top1_accuracy"], grid[str(value)]["mrr"],
        -abs(value - 2.), -value))
    return selected, grid


def load_v5(root):
    import torch
    root = Path(root); training = json.loads((root / "training.json").read_text())
    checkpoint = root / "relation_classifiers.pt"
    if training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("V5 classifier hash mismatch")
    return torch.load(checkpoint, map_location="cpu", weights_only=False), \
        json.loads((root / "summary.json").read_text())


def load_rows(root, stem):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    path = root / stem
    if training["raw_rows_sha256"][path.name] != sha(path):
        raise ValueError("comparison trace hash mismatch")
    return {(row["session_id"], row["group_id"]): row
            for row in map(json.loads, path.read_text().splitlines())}


def compare(rows, baseline, seed):
    differences = [row["correct"] - baseline[(row["session_id"], row["group_id"])]["correct"]
                   for row in rows]
    return {"top1_gain": statistics.fmean(differences),
            "bootstrap_95ci": _bootstrap(differences, seed), "units": len(rows)}


def development_gate(metric, versus_v4, v5_top1):
    checks = {"top1_at_least_95": metric["top1_accuracy"] >= .95,
              "mrr_at_least_97": metric["mrr"] >= .97,
              "within_3pp_of_v5": metric["top1_accuracy"] >= v5_top1 - .03,
              "gain_over_v4_at_least_10pp": versus_v4["top1_gain"] >= .10,
              "gain_ci_positive": versus_v4["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus, corpus_manifest = load_corpus(args.corpus_root)
    groups = [row for row in corpus if row["source"] == "counterfact"
              and row["target"] < MAX_KEYS]
    fold_map = deterministic_folds({group["session_id"] for group in groups})
    documents = source_documents(args.counterfact_source)
    views, idf_sizes = build_fold_views(groups, fold_map, documents)
    relation_texts = sorted({candidate[field] for rows in views.values() for row in rows
                             for candidate in row["candidates"]
                             for field in ("query_relation", "key_relation")})
    v5_artifact, v5_summary = load_v5(args.v5_root)

    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": relation_texts, "value": ["unknown"]},
                  args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": relation_texts, "value": ["unknown"]},
        transforms["transforms"], source_summary["selected_config"]["regularizer"])
    features = relation_features(relation_texts, embeddings)
    del raw, model; torch.cuda.empty_cache()

    scores_by_fold = {str(fold): fold_scores(
        views[str(fold)], features, v5_artifact["artifacts"][str(fold)])
        for fold in range(FOLDS)}
    final_rows, selection = [], {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        weight, grid = select_weight(scores_by_fold[str(inner)])
        final_rows.extend(predict(scores_by_fold[str(outer)], weight, outer))
        selection[str(outer)] = {"inner_fold": inner, "weight": weight,
                                  "inner_grid": grid}
    final_rows.sort(key=lambda row: (row["session_id"], row["group_id"]))
    path = root / "counterfact_oof_automatic_factorized.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in final_rows))
    metric = metrics(final_rows)
    v4 = load_v4(args.v4_root)
    v5 = load_rows(args.v5_root, "counterfact_oof_relation_supervised.jsonl")
    versus_v4 = compare(final_rows, v4, 15201)
    versus_v5 = compare(final_rows, v5, 15211)
    v5_top1 = v5_summary["counterfact_oof"]["top1_accuracy"]
    gate = development_gate(metric, versus_v4, v5_top1)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "counterfact_oof": metric, "v6_vs_v4": versus_v4,
               "v6_vs_oracle_segmented_v5": versus_v5,
               "oracle_segmented_v5_top1": v5_top1,
               "development_gate": gate,
               "next_stage": "causal_automatic_scope_forgetting" if gate["passed"]
               else "revise_automatic_entity_scope",
               "terminal_status": "completed_automatic_factorized_scope_v6"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "unique_relation_texts": len(relation_texts),
                "source_documents": len(documents), "idf_training_documents": idf_sizes,
                "selection": selection, "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {path.name: sha(path)}}
    manifest = {"protocol": PROTOCOL,
                "corpus_manifest_sha256": sha(Path(args.corpus_root) / "manifest.json"),
                "counterfact_source_sha256": sha(args.counterfact_source),
                "v5_summary_sha256": sha(Path(args.v5_root) / "summary.json"),
                "v4_summary_sha256": sha(Path(args.v4_root) / "summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "lambdas": list(LAMBDAS), "window": WINDOW,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False,
                "chain_identity_in_features": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False}
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--counterfact-source", required=True)
    parser.add_argument("--v4-root", required=True)
    parser.add_argument("--v5-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

