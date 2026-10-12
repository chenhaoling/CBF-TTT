"""Full ZsRE forgetting controller with nested grouped OOF threshold selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_scope_resolver import extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names, load_calibration, replay_session
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS, oracle_event_groups, paired_comparison, replay_controller,
    summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import (
    THRESHOLDS, TopGateController, top_examples, train_fixed,
)


PROTOCOL = "zsre_counterfactual_forgetting_controller_loso_v2"
INNER_FOLDS = 5
FIXED_EPOCHS = 200


def grouped_folds(session_ids, outer_id, count=INNER_FOLDS):
    ranked = sorted(session_ids, key=lambda value: hashlib.sha256(
        f"{PROTOCOL}|{outer_id}|{value}".encode()).hexdigest())
    folds = [ranked[offset::count] for offset in range(count)]
    if any(not fold for fold in folds): raise ValueError("empty inner fold")
    return folds


def select_threshold(sessions, groups, values, scorer, outer_train_ids, outer_id,
                     controller_seed, readout_seed):
    records = {threshold: [] for threshold in THRESHOLDS}
    fold_manifest = []
    for validation_ids in grouped_folds(outer_train_ids, outer_id):
        validation = set(validation_ids)
        train_indices = [index for index, group in enumerate(groups)
                         if group["session_id"] in outer_train_ids
                         and group["session_id"] not in validation]
        train_groups = [groups[index] for index in train_indices]
        head, mean, std = train_fixed(train_groups, values[train_indices],
                                      controller_seed, FIXED_EPOCHS)
        held_sessions = [session for session in sessions if session["id"] in validation]
        for threshold in THRESHOLDS:
            controller = TopGateController(scorer, head, mean, std, threshold)
            records[threshold].extend(replay_controller(
                session, scorer, readout_seed, controller_seed, controller)
                for session in held_sessions)
        fold_manifest.append({"validation_sessions": sorted(validation),
                              "train_sessions": sorted(set(outer_train_ids) - validation)})
    metrics = {threshold: aggregate(rows) for threshold, rows in records.items()}
    selected = max(THRESHOLDS, key=lambda threshold: (
        metrics[threshold]["accuracy"], metrics[threshold]["action_accuracy"],
        -abs(threshold - .5), -threshold))
    grid = {str(threshold): {"query_accuracy": metrics[threshold]["accuracy"],
                             "action_accuracy": metrics[threshold]["action_accuracy"]}
            for threshold in THRESHOLDS}
    return selected, grid, fold_manifest


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, _ = load_calibration(args.data_root, DATA_PROTOCOL); groups = oracle_event_groups(sessions)
    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    names, pairs = all_names(sessions); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]},
                                         transforms_artifact["transforms"],
                                         source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    values, _ = top_examples(groups, scorer); session_ids = sorted({group["session_id"] for group in groups})
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    controller_records = {str(seed): [] for seed in CONTROLLER_SEEDS}
    folds, selected_thresholds = {}, {str(seed): [] for seed in CONTROLLER_SEEDS}
    for outer_id in session_ids:
        outer_train_ids = [value for value in session_ids if value != outer_id]
        train_indices = [index for index, group in enumerate(groups) if group["session_id"] != outer_id]
        train_groups = [groups[index] for index in train_indices]
        folds[outer_id] = {"outer_train_sessions": outer_train_ids, "seeds": {}}
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            readout_seed = GATE_SEEDS[replicate]
            threshold, grid, inner_manifest = select_threshold(
                sessions, groups, values, scorer, outer_train_ids, outer_id, seed, readout_seed)
            selected_thresholds[str(seed)].append(threshold)
            head, mean, std = train_fixed(train_groups, values[train_indices], seed, FIXED_EPOCHS)
            controller = TopGateController(scorer, head, mean, std, threshold)
            session = next(row for row in sessions if row["id"] == outer_id)
            controller_records[str(seed)].append(replay_controller(
                session, scorer, readout_seed, seed, controller))
            folds[outer_id]["seeds"][str(seed)] = {
                "selected_threshold": threshold, "inner_grid": grid,
                "inner_folds": inner_manifest}

    controller_metrics, raw_hashes = {}, {}
    for seed, records in controller_records.items():
        path = root / f"calibration_loso_oof_gate_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path); controller_metrics[seed] = aggregate(records)
    baseline_records, baseline_metrics = {}, {}
    for readout_seed in GATE_SEEDS:
        records = [replay_session(session, "cosine_local", scorer, readout_seed) for session in sessions]
        for record in records: record["split"] = "calibration_loso"
        baseline_records[str(readout_seed)] = records; baseline_metrics[str(readout_seed)] = aggregate(records)
        path = root / f"calibration_cosine_local_seed{readout_seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)

    final_heads, final_thresholds = {}, {}
    for seed in CONTROLLER_SEEDS:
        threshold = statistics.median(selected_thresholds[str(seed)])
        head, mean, std = train_fixed(groups, values, seed, FIXED_EPOCHS)
        final_heads[str(seed)] = head.state_dict(); final_thresholds[str(seed)] = threshold
    checkpoint = root / "counterfactual_forgetting_oof_gate.pt"
    torch.save({"heads": final_heads, "mean": mean, "std": std,
                "epochs": FIXED_EPOCHS, "thresholds": final_thresholds,
                "controller_seeds": CONTROLLER_SEEDS, "feature_dim": values.shape[-1]}, checkpoint)

    summary = {"protocol": PROTOCOL, "controller": summarize_seed_metrics(controller_metrics),
               "cosine_local": summarize_seed_metrics(baseline_metrics),
               "comparison": paired_comparison(controller_records, baseline_records, 9101),
               "dev_loaded": False, "test_loaded": False, "test_scored": False,
               "terminal_status": "completed_calibration_nested_grouped_oof"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "sessions": len(session_ids), "events_with_candidates": len(groups),
                "unique_texts": len(names), "unique_pairs": len(pairs), "feature_dim": values.shape[-1],
                "inner_folds": INNER_FOLDS, "fixed_epochs": FIXED_EPOCHS,
                "folds": folds, "selected_thresholds": selected_thresholds,
                "final_thresholds": final_thresholds, "embedding_hashes": embedding_hashes,
                "pair_features_sha256": table_hash(pair_features), "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": PROTOCOL, "data_protocol": DATA_PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_nested_grouped_oof", "controller_seeds": list(CONTROLLER_SEEDS),
                "readout_seeds": list(GATE_SEEDS), "inner_folds": INNER_FOLDS,
                "fixed_epochs": FIXED_EPOCHS, "thresholds": list(THRESHOLDS),
                "outer_folds": session_ids, "selection_objective": "causal_query_then_action_accuracy",
                "candidate_selector": "frozen_bi_cosine", "gate_features": ["bi_cosine", "lexical"],
                "chain_identity_in_features": False, "dev_loaded": False,
                "test_loaded": False, "test_scored": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128); args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
