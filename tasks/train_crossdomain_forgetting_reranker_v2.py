"""Nested OOF V2 with full outer training and a balanced conflict gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import PROTOCOL as COUNTERFACT_PROTOCOL
from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as ZSRE_PROTOCOL
from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_counterfactual_forgetting_controller import load_split
from tasks.evaluate_zsre_hard_retain_forgetting import all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_forgetting_controller import MAX_KEYS, oracle_event_groups, paired_comparison
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_scope_resolver import extract_pair_features
from tasks.train_crossdomain_forgetting_reranker import (
    COSINE_TEMPERATURE,
    FOLDS,
    THRESHOLDS,
    DecomposedController,
    ResidualRanker,
    action_metrics,
    candidate_counts,
    cosine_values,
    deterministic_folds,
    development_gate,
    evaluate_records,
    feature_tensor,
    fit_normalizer,
    gate_features,
    load_previous,
    oracle_state_metrics,
    tag_sessions,
    targets,
)


PROTOCOL = "crossdomain_forgetting_reranker_v2"
RANK_EPOCHS = 10
GATE_EPOCHS = 25


def fit_fixed_policy(groups, values, cosines, counts, train_indices, pair_dim, seed):
    import torch

    mean, std = fit_normalizer(values, counts, train_indices)
    device = torch.device("cuda")
    train_groups = [groups[index] for index in train_indices]
    train_target = targets(train_groups); local = train_target < MAX_KEYS
    train_x = ((values[train_indices] - mean) / std).to(device)
    train_cosine = cosines[train_indices].to(device); train_counts = counts[train_indices].to(device)
    train_target = train_target.to(device); local = local.to(device)
    ranker = ResidualRanker(values.shape[-1], seed); ranker.linear.to(device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=.01, weight_decay=.1)
    for _ in range(RANK_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = ranker.logits(train_x, train_cosine, train_counts)
        loss = torch.nn.functional.cross_entropy(logits[local], train_target[local])
        loss.backward(); torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.); optimizer.step()
    ranker.linear.cpu()

    normalized = (values - mean) / std
    with torch.no_grad(): rank_logits = ranker.logits(normalized, cosines, counts)
    all_gate, _ = gate_features(normalized, rank_logits, counts)
    train_gate = all_gate[train_indices]
    gate_mean = train_gate.mean(0); gate_std = train_gate.std(0, unbiased=False).clamp_min(1e-4)
    gate_x = ((train_gate - gate_mean) / gate_std).to(device)
    desired_local = targets(train_groups).lt(MAX_KEYS).float().to(device)
    positives = desired_local.sum(); negatives = len(desired_local) - positives
    pos_weight = (negatives / positives).clamp_min(1.)
    gate = BinaryActionHead(gate_x.shape[-1], seed + 100); gate.linear.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.01, weight_decay=.1)
    for _ in range(GATE_EPOCHS):
        optimizer.zero_grad(set_to_none=True); logits = gate(gate_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, desired_local, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.); optimizer.step()
    gate.linear.cpu()
    return {
        "ranker": ranker.state_dict(), "rank_mean": mean, "rank_std": std,
        "gate": gate.state_dict(), "gate_mean": gate_mean, "gate_std": gate_std,
        "threshold": .5, "rank_epoch": RANK_EPOCHS, "gate_epoch": GATE_EPOCHS,
        "feature_dim": values.shape[-1], "pair_dim": pair_dim, "seed": seed,
        "gate_target": "semantic_conflict", "positive_weight": float(pos_weight.cpu()),
    }


def predict_oracle(groups, values, cosines, counts, indices, artifact, fold, phase):
    import torch

    ranker = ResidualRanker(artifact["feature_dim"], artifact["seed"])
    ranker.load_state_dict(artifact["ranker"])
    gate = BinaryActionHead(artifact["feature_dim"] + 4, artifact["seed"] + 100)
    gate.load_state_dict(artifact["gate"])
    selected = values[indices]
    normalized = (selected - artifact["rank_mean"]) / artifact["rank_std"]
    with torch.no_grad():
        logits = ranker.logits(normalized, cosines[indices], counts[indices])
        gate_raw, top = gate_features(normalized, logits, counts[indices])
        probability = gate((gate_raw - artifact["gate_mean"]) / artifact["gate_std"]).sigmoid()
    rows = []
    for group, candidate, score in zip((groups[index] for index in indices),
                                       top.tolist(), probability.tolist()):
        rows.append({"session_id": group["session_id"], "group_id": group["group_id"],
                     "event_kind": group["event_kind"], "target": group["target"],
                     "top": candidate, "probability": score, "fold": fold, "phase": phase})
    return rows


def metrics_from_rows(rows, threshold):
    correct, candidate = [], []
    by_kind = {kind: [] for kind in ("primary_initial", "hard_retain", "primary_update")}
    for row in rows:
        delete = row["probability"] >= threshold
        value = (not delete and row["target"] == MAX_KEYS) or (
            delete and row["top"] == row["target"])
        correct.append(value); by_kind[row["event_kind"]].append(value)
        if row["target"] < MAX_KEYS: candidate.append(row["top"] == row["target"])
    return {"candidate_top1_accuracy": statistics.fmean(candidate),
            "action_accuracy": statistics.fmean(correct),
            "by_event_kind": {kind: {"accuracy": statistics.fmean(values),
                                      "events": len(values)}
                              for kind, values in by_kind.items()}}


def choose_threshold(rows):
    grid = {threshold: metrics_from_rows(rows, threshold) for threshold in THRESHOLDS}
    selected = max(THRESHOLDS, key=lambda threshold: (
        min(grid[threshold]["by_event_kind"]["hard_retain"]["accuracy"],
            grid[threshold]["by_event_kind"]["primary_update"]["accuracy"]),
        grid[threshold]["action_accuracy"],
        grid[threshold]["by_event_kind"]["primary_update"]["accuracy"],
        -abs(threshold - .5)))
    return selected, {str(value): row for value, row in grid.items()}


def load_v1_records(root):
    from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS

    root = Path(root); training = json.loads((root / "training.json").read_text())
    records = {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"counterfact_mixed_oof_seed{seed}.jsonl"
        if training["raw_rows_sha256"][path.name] != sha(path):
            raise ValueError("V1 mixed-OOF trace hash mismatch")
        records[str(seed)] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return records, json.loads((root / "summary.json").read_text())


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    zsre = []
    for split in ("calibration", "dev", "test"):
        zsre.extend(tag_sessions(load_split(args.zsre_root, split), f"zsre_{split}"))
    counterfact, counterfact_manifest = load_transfer(args.counterfact_root)
    zsre_manifest = json.loads((Path(args.zsre_root) / "manifest.json").read_text())
    if counterfact_manifest["protocol"] != COUNTERFACT_PROTOCOL \
            or zsre_manifest["protocol"] != ZSRE_PROTOCOL:
        raise ValueError("data protocol mismatch")
    all_sessions = [*zsre, *counterfact]; groups = oracle_event_groups(all_sessions)
    zsre_ids = {session["id"] for session in zsre}
    counterfact_ids = {session["id"] for session in counterfact}
    names, pairs = all_names(all_sessions)

    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms["transforms"],
        source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    values = feature_tensor(groups, scorer); counts = candidate_counts(groups)
    pair_dim = next(iter(pair_features.values())).numel(); cosines = cosine_values(values, pair_dim)
    del model; torch.cuda.empty_cache()

    fold_map = deterministic_folds(counterfact_ids); artifacts, selection = {}, {}
    final_oracle_rows = []
    for outer in range(FOLDS):
        inner_rows = []
        remaining = [fold for fold in range(FOLDS) if fold != outer]
        for inner in remaining:
            train_ids = zsre_ids | {session_id for session_id, fold in fold_map.items()
                                    if fold not in (outer, inner)}
            validation_ids = {session_id for session_id, fold in fold_map.items()
                              if fold == inner}
            train_indices = [index for index, group in enumerate(groups)
                             if group["session_id"] in train_ids]
            validation_indices = [index for index, group in enumerate(groups)
                                  if group["session_id"] in validation_ids]
            artifact = fit_fixed_policy(
                groups, values, cosines, counts, train_indices, pair_dim,
                11101 + outer * 10 + inner)
            inner_rows.extend(predict_oracle(
                groups, values, cosines, counts, validation_indices,
                artifact, inner, "inner_oof"))
        threshold, grid = choose_threshold(inner_rows)
        outer_train_ids = zsre_ids | {session_id for session_id, fold in fold_map.items()
                                      if fold != outer}
        outer_indices = [index for index, group in enumerate(groups)
                         if group["session_id"] in outer_train_ids]
        outer_test_indices = [index for index, group in enumerate(groups)
                              if group["session_id"] in counterfact_ids
                              and fold_map[group["session_id"]] == outer]
        artifact = fit_fixed_policy(
            groups, values, cosines, counts, outer_indices, pair_dim, 11201 + outer)
        artifact["threshold"] = threshold; artifacts[outer] = artifact
        torch.save(artifact, root / f"mixed_oof_fold{outer}.pt")
        rows = predict_oracle(
            groups, values, cosines, counts, outer_test_indices,
            artifact, outer, "outer_oof")
        final_oracle_rows.extend(rows)
        selection[str(outer)] = {
            "threshold": threshold, "inner_grid": grid,
            "outer_test_sessions": sorted(session_id for session_id, fold in fold_map.items()
                                          if fold == outer),
            "outer_train_sessions_hash": hashlib.sha256(json.dumps(sorted(
                session_id for session_id in outer_train_ids)).encode()).hexdigest(),
        }

    oracle_path = root / "counterfact_mixed_oof_oracle_events.jsonl"
    oracle_path.write_text("".join(json.dumps(row) + "\n" for row in final_oracle_rows))
    oracle_metrics = metrics_from_rows(
        final_oracle_rows, 1.1)  # correctness below is recomputed with each fold threshold
    # The helper above accepts one threshold; outer rows already have calibrated
    # probabilities but thresholds differ, so recompute exact fold-wise decisions.
    for row in final_oracle_rows: row["threshold"] = artifacts[row["fold"]]["threshold"]
    def row_correct(row):
        delete = row["probability"] >= row["threshold"]
        return (not delete and row["target"] == MAX_KEYS) or (
            delete and row["top"] == row["target"])
    oracle_metrics["action_accuracy"] = statistics.fmean(row_correct(row) for row in final_oracle_rows)
    for kind in oracle_metrics["by_event_kind"]:
        rows = [row for row in final_oracle_rows if row["event_kind"] == kind]
        oracle_metrics["by_event_kind"][kind] = {
            "accuracy": statistics.fmean(row_correct(row) for row in rows), "events": len(rows)}
    # Rewrite with explicit frozen threshold for independent audit.
    oracle_path.write_text("".join(json.dumps(row) + "\n" for row in final_oracle_rows))

    records, metrics, raw_hashes = evaluate_records(
        counterfact, scorer, lambda session_id: artifacts[fold_map[session_id]],
        "counterfact_observed_mixed_oof_v2", root, "counterfact_mixed_oof_v2")
    raw_hashes[oracle_path.name] = sha(oracle_path)
    v1_records, v1_summary = load_v1_records(args.v1_root)
    old_records, old_summary = load_previous(args.previous_root)
    fixed_retain_query = old_summary["policies"]["fixed_retain"]["mean_accuracy"]
    gate = development_gate(metrics, oracle_metrics, fixed_retain_query)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    summary = {
        "protocol": PROTOCOL, "data_status": "observed_development_only",
        "mixed_counterfact_oof": {"metrics": metrics, "oracle_state": oracle_metrics},
        "v1_mixed_oof": v1_summary["mixed_counterfact_oof"]["metrics"],
        "previous_frozen_controller": old_summary["policies"]["learned"],
        "fixed_retain": old_summary["policies"]["fixed_retain"],
        "v2_vs_v1": paired_comparison(records, v1_records, 11301),
        "v2_vs_previous": paired_comparison(records, old_records, 11311),
        "development_gate": gate,
        "terminal_status": "completed_crossdomain_reranker_v2_development",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "sessions": {"zsre": len(zsre), "counterfact": len(counterfact)},
        "groups": {"total": len(groups), "counterfact": len(final_oracle_rows)},
        "unique_texts": len(names), "unique_pairs": len(pairs),
        "feature_dim": values.shape[-1], "pair_dim": pair_dim,
        "counterfact_folds": fold_map, "selection": selection,
        "embedding_hashes": embedding_hashes,
        "pair_features_sha256": table_hash(pair_features),
        "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": {f"mixed_oof_fold{fold}.pt": sha(
            root / f"mixed_oof_fold{fold}.pt") for fold in range(FOLDS)},
    }
    manifest = {
        "protocol": PROTOCOL, "zsre_protocol": ZSRE_PROTOCOL,
        "counterfact_protocol": COUNTERFACT_PROTOCOL,
        "zsre_manifest_sha256": sha(Path(args.zsre_root) / "manifest.json"),
        "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
        "v1_summary_sha256": sha(Path(args.v1_root) / "summary.json"),
        "previous_summary_sha256": sha(Path(args.previous_root) / "summary.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "folds": FOLDS,
        "rank_epochs": RANK_EPOCHS, "gate_epochs": GATE_EPOCHS,
        "thresholds": list(THRESHOLDS), "threshold_objective": [
            "min_hard_retain_primary_update", "overall_action", "primary_update"],
        "gate_target": "semantic_conflict", "gate_class_balanced": True,
        "cosine_temperature": COSINE_TEMPERATURE, "current_write_gate": 1,
        "chain_identity_in_features": False,
        "counterfact_status": "observed_development_only",
        "longmemeval_scored": False, "locomo_scored": False,
    }
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-root", required=True)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
