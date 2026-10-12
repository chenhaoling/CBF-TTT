"""Nested-LOSO forgetting gate with subject-excluded calibration pair supervision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names, load_calibration, replay_session
from tasks.train_zsre_forgetting_controller import (
    CHECKPOINT_EPOCHS, CONTROLLER_SEEDS, choose_inner_tune, oracle_event_groups,
    paired_comparison, replay_controller, summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import (
    PROTOCOL as V2_PROTOCOL, THRESHOLDS, BinaryActionHead, TopGateController,
    decision_metrics, labels, numeric_feature, top_examples,
)


PROTOCOL = "zsre_forgetting_controller_loso_v3"
AUXILIARY_WEIGHT = .5


def load_contextual_calibration(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != "zsre_contextual_readout_v1":
        raise ValueError("contextual protocol mismatch")
    path = root / "calibration.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("contextual calibration hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def auxiliary_examples(sessions):
    output = []
    for session in sessions:
        for fact in session["facts"]:
            output.append({"subject": fact["subject"], "key": fact["prompt"],
                           "query": fact["rephrase_prompt"], "label": 1.0})
            output.extend({"subject": fact["subject"], "key": fact["prompt"],
                           "query": query, "label": 0.0}
                          for query in fact["locality_prompts"])
    return output


def subject_hash(subjects):
    return hashlib.sha256(json.dumps(sorted(subjects)).encode()).hexdigest()


def auxiliary_tensor(examples, scorer):
    import torch

    return torch.stack([numeric_feature(row["key"], row["query"], scorer) for row in examples])


def fit_normalizer(primary, auxiliary):
    import torch

    values = torch.cat((primary, auxiliary)) if len(auxiliary) else primary
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4)


def train_loss(head, primary, primary_labels, auxiliary, auxiliary_labels):
    import torch

    natural = torch.nn.functional.binary_cross_entropy_with_logits(head(primary), primary_labels)
    if not len(auxiliary): return natural
    extra = torch.nn.functional.binary_cross_entropy_with_logits(head(auxiliary), auxiliary_labels)
    return natural + AUXILIARY_WEIGHT * extra


def select_model(train_groups, tune_groups, train_raw, tune_raw, train_top, tune_top,
                 auxiliary_raw, auxiliary_labels, seed):
    import torch

    mean, std = fit_normalizer(train_raw, auxiliary_raw)
    train, tune = (train_raw - mean) / std, (tune_raw - mean) / std
    auxiliary = (auxiliary_raw - mean) / std
    head = BinaryActionHead(train.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    target = labels(train_groups); candidates, curve = [], []
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(head, train, target, auxiliary, auxiliary_labels)
        loss.backward(); optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            with torch.no_grad(): train_logits, tune_logits = head(train), head(tune)
            epoch_rows = []
            for threshold in THRESHOLDS:
                tune_metric = decision_metrics(tune_groups, tune_top, tune_logits, threshold)
                row = (tune_metric["accuracy"], -tune_metric["binary_nll"],
                       -abs(threshold - .5), -epoch, threshold, copy.deepcopy(head.state_dict()))
                candidates.append(row); epoch_rows.append(row)
            selected_epoch = max(epoch_rows, key=lambda row: row[:4])
            curve.append({"epoch": epoch,
                          "train": decision_metrics(train_groups, train_top, train_logits,
                                                    selected_epoch[4]),
                          "tune": decision_metrics(tune_groups, tune_top, tune_logits,
                                                   selected_epoch[4]),
                          "threshold": selected_epoch[4]})
    selected = max(candidates, key=lambda row: row[:4]); head.load_state_dict(selected[5])
    return head, -selected[3], selected[4], curve


def train_fixed(groups, raw, auxiliary_raw, auxiliary_labels, seed, epochs):
    import torch

    mean, std = fit_normalizer(raw, auxiliary_raw)
    values, auxiliary = (raw - mean) / std, (auxiliary_raw - mean) / std
    head = BinaryActionHead(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    target = labels(groups)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = train_loss(head, values, target, auxiliary, auxiliary_labels)
        loss.backward(); optimizer.step()
    return head, mean, std


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, _ = load_calibration(args.data_root); groups = oracle_event_groups(sessions)
    contextual_sessions, contextual_manifest = load_contextual_calibration(args.context_root)
    auxiliary = auxiliary_examples(contextual_sessions)
    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    hard_names, pairs = all_names(sessions)
    names = sorted(set(hard_names) | {text for row in auxiliary for text in (row["key"], row["query"])})
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]},
                                         transforms_artifact["transforms"],
                                         source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    values, tops = top_examples(groups, scorer); auxiliary_values = auxiliary_tensor(auxiliary, scorer)
    auxiliary_labels = torch.tensor([row["label"] for row in auxiliary])
    session_ids = sorted({group["session_id"] for group in groups})
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    session_subjects = {session["id"]: {chain["subject"] for chain in session["chains"]}
                        for session in sessions}
    controller_records = {str(seed): [] for seed in CONTROLLER_SEEDS}
    folds, selected_epochs, selected_thresholds = {}, {str(s): [] for s in CONTROLLER_SEEDS}, {
        str(s): [] for s in CONTROLLER_SEEDS}
    for outer_id in session_ids:
        outer = [i for i, group in enumerate(groups) if group["session_id"] == outer_id]
        remaining_ids = [value for value in session_ids if value != outer_id]
        tune_id = choose_inner_tune(remaining_ids, outer_id)
        inner_train = [i for i, group in enumerate(groups)
                       if group["session_id"] not in (outer_id, tune_id)]
        inner_tune = [i for i, group in enumerate(groups) if group["session_id"] == tune_id]
        outer_train = [i for i, group in enumerate(groups) if group["session_id"] != outer_id]
        inner_excluded = session_subjects[outer_id] | session_subjects[tune_id]
        outer_excluded = session_subjects[outer_id]
        inner_aux = [i for i, row in enumerate(auxiliary) if row["subject"] not in inner_excluded]
        outer_aux = [i for i, row in enumerate(auxiliary) if row["subject"] not in outer_excluded]
        folds[outer_id] = {"inner_train_sessions": sorted(set(remaining_ids) - {tune_id}),
                           "inner_tune_session": tune_id, "outer_train_sessions": sorted(remaining_ids),
                           "heldout_subject_hash": subject_hash(session_subjects[outer_id]),
                           "inner_auxiliary_examples": len(inner_aux),
                           "outer_auxiliary_examples": len(outer_aux), "seeds": {}}
        outer_groups = [groups[i] for i in outer_train]
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            _, epoch, threshold, curve = select_model(
                [groups[i] for i in inner_train], [groups[i] for i in inner_tune],
                values[inner_train], values[inner_tune], tops[inner_train], tops[inner_tune],
                auxiliary_values[inner_aux], auxiliary_labels[inner_aux], seed)
            selected_epochs[str(seed)].append(epoch); selected_thresholds[str(seed)].append(threshold)
            head, mean, std = train_fixed(outer_groups, values[outer_train], auxiliary_values[outer_aux],
                                          auxiliary_labels[outer_aux], seed, epoch)
            controller = TopGateController(scorer, head, mean, std, threshold)
            session = next(row for row in sessions if row["id"] == outer_id)
            controller_records[str(seed)].append(replay_controller(
                session, scorer, GATE_SEEDS[replicate], seed, controller))
            with torch.no_grad(): held_logits = head((values[outer] - mean) / std)
            folds[outer_id]["seeds"][str(seed)] = {"selected_epoch": epoch,
                "selected_threshold": threshold, "inner_curve": curve,
                "heldout_oracle_state": decision_metrics(
                    [groups[i] for i in outer], tops[outer], held_logits, threshold)}

    controller_metrics, raw_hashes = {}, {}
    for seed, records in controller_records.items():
        path = root / f"calibration_loso_augmented_gate_seed{seed}.jsonl"
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

    final_heads, final_epochs, final_thresholds = {}, {}, {}
    for seed in CONTROLLER_SEEDS:
        epoch = int(statistics.median(selected_epochs[str(seed)]))
        threshold = statistics.median(selected_thresholds[str(seed)])
        head, mean, std = train_fixed(groups, values, auxiliary_values, auxiliary_labels, seed, epoch)
        final_heads[str(seed)] = head.state_dict(); final_epochs[str(seed)] = epoch
        final_thresholds[str(seed)] = threshold
    checkpoint = root / "forgetting_augmented_gate.pt"
    torch.save({"heads": final_heads, "mean": mean, "std": std,
                "selected_epochs": final_epochs, "thresholds": final_thresholds,
                "controller_seeds": CONTROLLER_SEEDS, "feature_dim": values.shape[-1],
                "auxiliary_weight": AUXILIARY_WEIGHT}, checkpoint)

    summary = {"protocol": PROTOCOL, "controller": summarize_seed_metrics(controller_metrics),
               "cosine_local": summarize_seed_metrics(baseline_metrics),
               "comparison": paired_comparison(controller_records, baseline_records, 8901),
               "dev_loaded": False, "test_loaded": False, "test_scored": False,
               "terminal_status": "completed_calibration_nested_loso_v3"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "sessions": len(session_ids), "events_with_candidates": len(groups),
                "auxiliary_examples": len(auxiliary), "auxiliary_subjects": len({r["subject"] for r in auxiliary}),
                "unique_texts": len(names), "unique_pairs": len(pairs), "feature_dim": values.shape[-1],
                "folds": folds, "selected_epochs": selected_epochs,
                "selected_thresholds": selected_thresholds, "final_epochs": final_epochs,
                "final_thresholds": final_thresholds, "embedding_hashes": embedding_hashes,
                "pair_features_sha256": table_hash(pair_features), "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint),
                "context_manifest_sha256": sha(Path(args.context_root) / "manifest.json"),
                "context_calibration_sha256": sha(Path(args.context_root) / "calibration.jsonl"),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": PROTOCOL, "parent_protocol": V2_PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "context_manifest_sha256": sha(Path(args.context_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_nested_loso", "controller_seeds": list(CONTROLLER_SEEDS),
                "readout_seeds": list(GATE_SEEDS), "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "thresholds": list(THRESHOLDS), "outer_folds": session_ids,
                "candidate_selector": "frozen_bi_cosine", "gate_features": ["bi_cosine", "lexical"],
                "auxiliary_supervision": "contextual calibration exact-rephrase/locality",
                "auxiliary_weight": AUXILIARY_WEIGHT, "outer_subjects_excluded_from_auxiliary": True,
                "chain_identity_in_features": False, "dev_loaded": False,
                "test_loaded": False, "test_scored": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--context-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
