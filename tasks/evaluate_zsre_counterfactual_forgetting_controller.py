"""Frozen development gate and one-time test for ZsRE forgetting controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_scope_resolver import extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names, replay_session
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS, paired_comparison, replay_controller, summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead, TopGateController
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import PROTOCOL as SOURCE_PROTOCOL


PROTOCOL = "zsre_counterfactual_forgetting_controller_formal_v1"


def load_split(root, split):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != DATA_PROTOCOL: raise ValueError("data protocol mismatch")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]: raise ValueError("split hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def development_gate(controller, comparison):
    checks = {
        "query_gain_confident": comparison["query_bootstrap_95ci"][0] > 0,
        "action_gain_confident": comparison["action_bootstrap_95ci"][0] > 0,
        "overall_action": controller["mean_action_accuracy"] >= .96,
        "primary_initial": controller["by_event_kind"]["primary_initial"]["accuracy"] >= .98,
        "hard_retain": controller["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
        "primary_update": controller["by_event_kind"]["primary_update"]["accuracy"] >= .92,
    }
    return {"checks": checks, "passed": all(checks.values())}


def build_scorer(sessions, model, tokenizer, transforms, regularizer, gate_artifact, batch_size):
    names, pairs = all_names(sessions)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]},
                                         transforms, regularizer)
    pair_features = extract_pair_features(model, tokenizer, pairs, batch_size)
    return FrozenScopeScorer(embeddings, pair_features, gate_artifact), {
        "unique_texts": len(names), "unique_pairs": len(pairs),
        "embedding_hashes": embedding_hashes, "pair_features_sha256": table_hash(pair_features)}


def evaluate_split(root, split, sessions, scorer, head, mean, std, threshold, comparison_seed):
    controller_records, controller_metrics, raw_hashes = {}, {}, {}
    controller = TopGateController(scorer, head, mean, std, threshold)
    for replicate, seed in enumerate(CONTROLLER_SEEDS):
        records = [replay_controller(session, scorer, GATE_SEEDS[replicate], seed,
                                     controller, split=split) for session in sessions]
        controller_records[str(seed)] = records; controller_metrics[str(seed)] = aggregate(records)
        path = root / f"{split}_controller_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)
    baseline_records, baseline_metrics = {}, {}
    for seed in GATE_SEEDS:
        records = [replay_session(session, "cosine_local", scorer, seed) for session in sessions]
        for record in records: record["split"] = split
        baseline_records[str(seed)] = records; baseline_metrics[str(seed)] = aggregate(records)
        path = root / f"{split}_cosine_local_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)
    controller_summary = summarize_seed_metrics(controller_metrics)
    return {"controller": controller_summary,
            "cosine_local": summarize_seed_metrics(baseline_metrics),
            "comparison": paired_comparison(controller_records, baseline_records, comparison_seed)}, raw_hashes


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source_result = Path(args.controller_root)
    source_manifest = json.loads((source_result / "training_manifest.json").read_text())
    source_training = json.loads((source_result / "training.json").read_text())
    checkpoint = source_result / "counterfactual_forgetting_action_oof_gate.pt"
    if source_manifest["protocol"] != SOURCE_PROTOCOL or source_training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("controller source mismatch")
    controller_artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = BinaryActionHead(controller_artifact["feature_dim"], controller_artifact["training_seed"])
    head.load_state_dict(controller_artifact["head"])
    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training_base = json.loads((source / "training.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dev_sessions = load_split(args.data_root, "dev")
    dev_scorer, dev_features = build_scorer(
        dev_sessions, model, tokenizer, transforms_artifact["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size)
    dev, raw_hashes = evaluate_split(
        root, "dev", dev_sessions, dev_scorer, head, controller_artifact["mean"],
        controller_artifact["std"], controller_artifact["threshold"], 9301)
    dev_gate = development_gate(dev["controller"], dev["comparison"])

    test = test_features = None; test_scored = False
    if dev_gate["passed"]:
        test_sessions = load_split(args.data_root, "test")
        test_scorer, test_features = build_scorer(
            test_sessions, model, tokenizer, transforms_artifact["transforms"],
            source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size)
        test, test_hashes = evaluate_split(
            root, "test", test_sessions, test_scorer, head, controller_artifact["mean"],
            controller_artifact["std"], controller_artifact["threshold"], 9401)
        raw_hashes.update(test_hashes); test_scored = True
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()
    summary = {"protocol": PROTOCOL, "dev": dev, "development_gate": dev_gate,
               "test": test, "test_scored": test_scored,
               "terminal_status": "completed_formal_gated_evaluation"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "dev_features": dev_features, "test_features": test_features,
                "raw_rows_sha256": raw_hashes,
                "controller_summary_sha256": sha(source_result / "summary.json"),
                "controller_training_sha256": sha(source_result / "training.json"),
                "controller_manifest_sha256": sha(source_result / "training_manifest.json"),
                "controller_checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
                "data_protocol": DATA_PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "controller_threshold": controller_artifact["threshold"],
                "controller_epochs": controller_artifact["epochs"],
                "controller_seeds": list(CONTROLLER_SEEDS), "readout_seeds": list(GATE_SEEDS),
                "dev_loaded": True, "test_release_rule": "all six frozen development checks",
                "test_loaded": test_scored, "test_scored": test_scored}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--controller-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
