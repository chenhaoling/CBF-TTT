"""Fuse OOF learned-entity evidence into the nested text-only scope calibrator."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import (
    build_fold_views, load_v5, relation_features, source_documents,
)
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_learned_entity_scope_v9 import entity_logits, pair_feature, pair_text
from tasks.train_relation_supervised_scope_v5 import load_v4
from tasks.train_scoped_delta_ranker_v4 import load_corpus
from tasks.train_text_only_scope_calibrator_v8 import (
    SEEDS, aggregate, choose_epoch, compare, feature_rows, fit_outer, load_trace,
    predict, records, tensorize,
)
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "entity_fusion_scope_v10"


def load_entity_heads(root):
    import torch
    root = Path(root); training = json.loads((root / "training.json").read_text())
    checkpoint = root / "entity_heads.pt"
    if training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("V9 entity checkpoint mismatch")
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def append_entity_features(rows, pair_table, entity_artifacts):
    import torch

    output = []
    for row in rows:
        group = row["group"]
        pair_values = torch.stack([pair_feature(group["query"], candidate["prompt"], pair_table)
                                   for candidate in group["candidates"]])
        logits = entity_logits(pair_values, entity_artifacts)
        probability = logits.sigmoid()
        additions = torch.stack((probability,
                                 probability.clamp_min(1e-12).log(), logits), dim=-1)
        output.append({"group": group, "base": row["base"],
                       "values": torch.cat((row["values"], additions), dim=-1)})
    return output


def development_gate(result, vs_v8, vs_v4):
    checks = {"top1_at_least_95": result["top1_accuracy"] >= .95,
              "mrr_at_least_97": result["mrr"] >= .97,
              "gain_v8_at_least_1pp": vs_v8["top1_gain"] >= .01,
              "gain_v8_ci_positive": vs_v8["bootstrap_95ci"][0] > 0,
              "gain_v4_at_least_12pp": vs_v4["top1_gain"] >= .12,
              "gain_v4_ci_positive": vs_v4["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus, _ = load_corpus(args.corpus_root)
    groups = [row for row in corpus if row["source"] == "counterfact"
              and row["target"] < MAX_KEYS]
    fold_map = deterministic_folds({row["session_id"] for row in groups})
    views, idf_sizes = build_fold_views(groups, fold_map, source_documents(args.counterfact_source))
    relation_texts = sorted({candidate[field] for rows in views.values() for row in rows
                             for candidate in row["candidates"]
                             for field in ("query_relation", "key_relation")})
    pair_names = sorted({pair_text(group["query"], candidate["prompt"])
                         for group in groups for candidate in group["candidates"]})
    v5_artifact, _ = load_v5(args.v5_root)
    entity_heads = load_entity_heads(args.v9_root)
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": relation_texts, "value": ["unknown"]},
                  args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": relation_texts, "value": ["unknown"]}, transforms["transforms"],
        source_summary["selected_config"]["regularizer"])
    relation_table = relation_features(relation_texts, embeddings)
    pair_table = extract_pair_features(model, tokenizer, pair_names, args.pair_batch_size)
    del raw, model; torch.cuda.empty_cache()

    rows = []
    for fold in range(FOLDS):
        base_rows = feature_rows(views[str(fold)], relation_table,
                                 v5_artifact["artifacts"][str(fold)])
        rows.extend(append_entity_features(base_rows, pair_table, entity_heads[str(fold)]))
    rows.sort(key=lambda row: (row["group"]["session_id"], row["group"]["group_id"]))
    values, base, counts, target = tensorize(rows)
    fold_for_index = [fold_map[row["group"]["session_id"]] for row in rows]
    final, selection, checkpoints = [], {}, {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        inner_train = [i for i, fold in enumerate(fold_for_index)
                       if fold not in (outer, inner)]
        tune = [i for i, fold in enumerate(fold_for_index) if fold == inner]
        outer_train = [i for i, fold in enumerate(fold_for_index) if fold != outer]
        test = [i for i, fold in enumerate(fold_for_index) if fold == outer]
        artifacts, epochs, curves = [], {}, {}
        for seed in SEEDS:
            epoch, curve = choose_epoch(values, base, counts, target,
                                         inner_train, tune, seed + outer * 100)
            artifacts.append(fit_outer(values, base, counts, target, outer_train,
                                       seed + outer * 100, epoch))
            epochs[str(seed)] = epoch; curves[str(seed)] = curve
        logits = predict(values, base, counts, test, artifacts)
        final.extend(records([rows[index]["group"] for index in test], logits, outer))
        selection[str(outer)] = {"inner_fold": inner, "epochs": epochs, "curves": curves}
        checkpoints[str(outer)] = artifacts
    final.sort(key=lambda row: (row["session_id"], row["group_id"]))
    path = root / "counterfact_oof_entity_fusion.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in final))
    result = aggregate(final)
    v8 = load_trace(args.v8_root, "counterfact_oof_text_calibrator.jsonl")
    v4 = load_v4(args.v4_root)
    vs_v8 = compare(final, v8, 18201); vs_v4 = compare(final, v4, 18211)
    gate = development_gate(result, vs_v8, vs_v4)
    checkpoint = root / "fusion_calibrators.pt"; torch.save(checkpoints, checkpoint)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "counterfact_oof": result, "v10_vs_v8": vs_v8, "v10_vs_v4": vs_v4,
               "development_gate": gate,
               "next_stage": "causal_automatic_scope_forgetting" if gate["passed"]
               else "token_level_entity_span_extractor",
               "terminal_status": "completed_entity_fusion_scope_v10"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "unique_relation_texts": len(relation_texts),
                "unique_entity_pairs": len(pair_names), "feature_dim": values.shape[-1],
                "selection": selection, "idf_training_documents": idf_sizes,
                "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {path.name: sha(path)},
                "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "counterfact_source_sha256": sha(args.counterfact_source),
                "corpus_manifest_sha256": sha(Path(args.corpus_root) / "manifest.json"),
                "v5_summary_sha256": sha(Path(args.v5_root) / "summary.json"),
                "v8_summary_sha256": sha(Path(args.v8_root) / "summary.json"),
                "v9_summary_sha256": sha(Path(args.v9_root) / "summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
                "epochs": [10, 25, 50], "rank": 16,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root", "counterfact-source", "v4-root", "v5-root", "v8-root",
                 "v9-root", "source-root", "output", "model", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

