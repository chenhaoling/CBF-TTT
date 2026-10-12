"""Efficient nested OOF action calibration for full ZsRE forgetting."""

from __future__ import annotations

import argparse
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
    THRESHOLDS, TopGateController, decision_metrics, top_examples, train_fixed,
)
from tasks.train_zsre_counterfactual_forgetting_controller_v2 import grouped_folds


PROTOCOL = "zsre_counterfactual_forgetting_controller_loso_v3"
INNER_FOLDS = 5
FIXED_EPOCHS = 200
TRAINING_SEED = CONTROLLER_SEEDS[0]


def select_action_threshold(groups, values, tops, outer_train_ids, outer_id):
    import torch

    held_groups, held_tops, held_logits, fold_manifest = [], [], [], []
    for validation_ids in grouped_folds(outer_train_ids, outer_id, INNER_FOLDS):
        validation = set(validation_ids)
        train_indices = [index for index, group in enumerate(groups)
                         if group["session_id"] in outer_train_ids
                         and group["session_id"] not in validation]
        validation_indices = [index for index, group in enumerate(groups)
                              if group["session_id"] in validation]
        train_groups = [groups[index] for index in train_indices]
        head, mean, std = train_fixed(train_groups, values[train_indices],
                                      TRAINING_SEED, FIXED_EPOCHS)
        with torch.no_grad(): logits = head((values[validation_indices] - mean) / std)
        held_groups.extend(groups[index] for index in validation_indices)
        held_tops.append(tops[validation_indices]); held_logits.append(logits)
        fold_manifest.append({"validation_sessions": sorted(validation),
                              "train_sessions": sorted(set(outer_train_ids) - validation)})
    held_tops, held_logits = torch.cat(held_tops), torch.cat(held_logits)
    grid = {threshold: decision_metrics(held_groups, held_tops, held_logits, threshold)
            for threshold in THRESHOLDS}
    selected = max(THRESHOLDS, key=lambda threshold: (
        grid[threshold]["accuracy"], -grid[threshold]["binary_nll"],
        -abs(threshold - .5), -threshold))
    serializable = {str(threshold): grid[threshold] for threshold in THRESHOLDS}
    return selected, serializable, fold_manifest


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
    values, tops = top_examples(groups, scorer); session_ids = sorted({g["session_id"] for g in groups})
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    controller_records = {str(seed): [] for seed in CONTROLLER_SEEDS}
    folds, selected_thresholds = {}, []
    for outer_id in session_ids:
        outer_train_ids = [value for value in session_ids if value != outer_id]
        train_indices = [index for index, group in enumerate(groups) if group["session_id"] != outer_id]
        train_groups = [groups[index] for index in train_indices]
        threshold, grid, inner_manifest = select_action_threshold(
            groups, values, tops, outer_train_ids, outer_id)
        selected_thresholds.append(threshold)
        head, mean, std = train_fixed(train_groups, values[train_indices],
                                      TRAINING_SEED, FIXED_EPOCHS)
        controller = TopGateController(scorer, head, mean, std, threshold)
        session = next(row for row in sessions if row["id"] == outer_id)
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            controller_records[str(seed)].append(replay_controller(
                session, scorer, GATE_SEEDS[replicate], seed, controller))
        folds[outer_id] = {"outer_train_sessions": outer_train_ids,
                           "selected_threshold": threshold, "inner_grid": grid,
                           "inner_folds": inner_manifest}

    controller_metrics, raw_hashes = {}, {}
    for seed, records in controller_records.items():
        path = root / f"calibration_loso_action_oof_gate_seed{seed}.jsonl"
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

    final_threshold = statistics.median(selected_thresholds)
    final_head, final_mean, final_std = train_fixed(groups, values, TRAINING_SEED, FIXED_EPOCHS)
    checkpoint = root / "counterfactual_forgetting_action_oof_gate.pt"
    torch.save({"head": final_head.state_dict(), "mean": final_mean, "std": final_std,
                "epochs": FIXED_EPOCHS, "threshold": final_threshold,
                "training_seed": TRAINING_SEED, "feature_dim": values.shape[-1]}, checkpoint)

    summary = {"protocol": PROTOCOL, "controller": summarize_seed_metrics(controller_metrics),
               "cosine_local": summarize_seed_metrics(baseline_metrics),
               "comparison": paired_comparison(controller_records, baseline_records, 9201),
               "dev_loaded": False, "test_loaded": False, "test_scored": False,
               "terminal_status": "completed_calibration_nested_action_oof"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "sessions": len(session_ids), "events_with_candidates": len(groups),
                "unique_texts": len(names), "unique_pairs": len(pairs), "feature_dim": values.shape[-1],
                "inner_folds": INNER_FOLDS, "fixed_epochs": FIXED_EPOCHS,
                "folds": folds, "selected_thresholds": selected_thresholds,
                "final_threshold": final_threshold, "embedding_hashes": embedding_hashes,
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
                "split": "calibration_nested_action_oof", "controller_seeds": list(CONTROLLER_SEEDS),
                "readout_seeds": list(GATE_SEEDS), "training_seed": TRAINING_SEED,
                "inner_folds": INNER_FOLDS, "fixed_epochs": FIXED_EPOCHS,
                "thresholds": list(THRESHOLDS), "outer_folds": session_ids,
                "selection_objective": "oracle_state_action_accuracy",
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
