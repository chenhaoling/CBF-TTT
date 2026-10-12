"""Causal replay of V2 weights with inner-OOF constraint calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_hard_retain_forgetting import all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import (
    development_gate,
    evaluate_records,
    load_previous,
)
from tasks.train_crossdomain_forgetting_reranker_v2 import load_v1_records
from tasks.train_zsre_forgetting_controller import paired_comparison
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "crossdomain_forgetting_reranker_v2_constraint_calibration"
HARD_FLOOR = .85
UPDATE_FLOOR = .75


def select_constrained_threshold(grid):
    rows = [(float(key), value) for key, value in grid.items()]
    feasible = [(threshold, row) for threshold, row in rows
                if row["by_event_kind"]["hard_retain"]["accuracy"] >= HARD_FLOOR
                and row["by_event_kind"]["primary_update"]["accuracy"] >= UPDATE_FLOOR]
    if feasible:
        threshold, row = max(feasible, key=lambda item: (
            item[1]["action_accuracy"],
            item[1]["by_event_kind"]["primary_update"]["accuracy"],
            -abs(item[0] - .5)))
        return threshold, "feasible"
    threshold, row = min(rows, key=lambda item: (
        max(0., HARD_FLOOR - item[1]["by_event_kind"]["hard_retain"]["accuracy"])
        + max(0., UPDATE_FLOOR - item[1]["by_event_kind"]["primary_update"]["accuracy"]),
        -item[1]["action_accuracy"], abs(item[0] - .5)))
    return threshold, "minimum_constraint_deficit"


def oracle_metrics(rows, thresholds):
    def correct(row):
        delete = row["probability"] >= thresholds[str(row["fold"])]["threshold"]
        return (not delete and row["target"] == 8) or (delete and row["top"] == row["target"])
    local = [row for row in rows if row["target"] < 8]
    kinds = ("primary_initial", "hard_retain", "primary_update")
    return {"candidate_top1_accuracy": statistics.fmean(
                row["top"] == row["target"] for row in local),
            "action_accuracy": statistics.fmean(correct(row) for row in rows),
            "by_event_kind": {kind: {"accuracy": statistics.fmean(
                correct(row) for row in rows if row["event_kind"] == kind),
                "events": sum(row["event_kind"] == kind for row in rows)} for kind in kinds}}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source_run = Path(args.v2_root)
    source_summary = json.loads((source_run / "summary.json").read_text())
    source_training = json.loads((source_run / "training.json").read_text())
    source_manifest = json.loads((source_run / "training_manifest.json").read_text())
    thresholds = {}
    artifacts = {}
    for fold, selection in source_training["selection"].items():
        threshold, status = select_constrained_threshold(selection["inner_grid"])
        thresholds[fold] = {"threshold": threshold, "status": status}
        checkpoint = source_run / f"mixed_oof_fold{fold}.pt"
        if source_training["checkpoint_sha256"][checkpoint.name] != sha(checkpoint):
            raise ValueError("V2 checkpoint hash mismatch")
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        artifact["threshold"] = threshold; artifacts[int(fold)] = artifact

    sessions, data_manifest = load_transfer(args.counterfact_root)
    fold_map = {key: int(value) for key, value in source_training["counterfact_folds"].items()}
    source, scope = Path(args.source_root), Path(args.scope_root)
    context_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")

    names, pairs = all_names(sessions); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms["transforms"],
        context_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    records, metrics, raw_hashes = evaluate_records(
        sessions, scorer, lambda session_id: artifacts[fold_map[session_id]],
        "counterfact_observed_constraint_calibration", root,
        "counterfact_constraint_calibration")
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    oracle_path = source_run / "counterfact_mixed_oof_oracle_events.jsonl"
    if source_training["raw_rows_sha256"][oracle_path.name] != sha(oracle_path):
        raise ValueError("V2 oracle trace hash mismatch")
    oracle_rows = [json.loads(line) for line in oracle_path.read_text().splitlines() if line]
    oracle = oracle_metrics(oracle_rows, thresholds)
    v1_records, v1_summary = load_v1_records(args.v1_root)
    old_records, old_summary = load_previous(args.previous_root)
    fixed_retain_query = old_summary["policies"]["fixed_retain"]["mean_accuracy"]
    gate = development_gate(metrics, oracle, fixed_retain_query)
    summary = {
        "protocol": PROTOCOL,
        "data_status": "observed_development_only",
        "thresholds": thresholds,
        "metrics": metrics,
        "oracle_state": oracle,
        "v2_original": source_summary["mixed_counterfact_oof"]["metrics"],
        "v1_mixed_oof": v1_summary["mixed_counterfact_oof"]["metrics"],
        "constraint_vs_v2": paired_comparison(
            records, {str(seed): [json.loads(line) for line in (
                source_run / f"counterfact_mixed_oof_v2_seed{seed}.jsonl").read_text().splitlines()
                if line] for seed in (8601, 8602, 8603)}, 11401),
        "constraint_vs_v1": paired_comparison(records, v1_records, 11411),
        "constraint_vs_previous": paired_comparison(records, old_records, 11421),
        "development_gate": gate,
        "terminal_status": "completed_constraint_calibration_diagnostic",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "features": {"unique_texts": len(names), "unique_pairs": len(pairs),
                     "embedding_hashes": embedding_hashes,
                     "pair_features_sha256": table_hash(pair_features)},
        "raw_rows_sha256": raw_hashes,
        "v2_summary_sha256": sha(source_run / "summary.json"),
        "v2_training_sha256": sha(source_run / "training.json"),
        "v2_manifest_sha256": sha(source_run / "training_manifest.json"),
        "v2_oracle_sha256": sha(oracle_path),
    }
    manifest = {
        "protocol": PROTOCOL, "source_protocol": source_manifest["protocol"],
        "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "selection_rule": ["feasible_hard_85_update_75", "overall_action",
                           "primary_update", "proximity_to_0.5",
                           "minimum_constraint_deficit_fallback"],
        "weights_changed": False, "counterfact_status": "observed_development_only",
        "longmemeval_scored": False, "locomo_scored": False,
        "units": data_manifest["eligible_subjects"],
    }
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
