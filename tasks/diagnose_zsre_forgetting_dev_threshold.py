"""Causal dev-only threshold diagnostic for the frozen ZsRE forgetting head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.evaluate_zsre_counterfactual_forgetting_controller import (
    PROTOCOL as FORMAL_PROTOCOL,
    build_scorer,
    development_gate,
    load_split,
)
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS,
    paired_comparison,
    replay_controller,
    summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead, TopGateController
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import PROTOCOL as SOURCE_PROTOCOL


PROTOCOL = "zsre_counterfactual_forgetting_dev_threshold_v1"
THRESHOLDS = tuple(value / 10 for value in range(1, 10))


def threshold_key(value):
    return f"{value:.1f}"


def rank_threshold(row):
    """Prefer reliability, then query quality, then proximity to the frozen value."""
    metrics = row["controller"]
    return (
        metrics["mean_action_accuracy"],
        min(metrics["by_event_kind"][kind]["accuracy"]
            for kind in ("hard_retain", "primary_update")),
        metrics["mean_accuracy"],
        -abs(row["threshold"] - .4),
    )


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    formal = Path(args.formal_root); controller_root = Path(args.controller_root)
    formal_summary = json.loads((formal / "summary.json").read_text())
    formal_training = json.loads((formal / "training.json").read_text())
    formal_manifest = json.loads((formal / "training_manifest.json").read_text())
    if formal_summary["protocol"] != FORMAL_PROTOCOL or formal_summary["test_scored"] \
            or formal_manifest["test_loaded"]:
        raise ValueError("formal source must be the failed dev gate with sealed test")
    source_manifest = json.loads((controller_root / "training_manifest.json").read_text())
    source_training = json.loads((controller_root / "training.json").read_text())
    checkpoint = controller_root / "counterfactual_forgetting_action_oof_gate.pt"
    if source_manifest["protocol"] != SOURCE_PROTOCOL \
            or source_training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("controller source mismatch")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = BinaryActionHead(artifact["feature_dim"], artifact["training_seed"])
    head.load_state_dict(artifact["head"])

    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")

    baseline_records, baseline_metrics = {}, {}
    for seed in GATE_SEEDS:
        path = formal / f"dev_cosine_local_seed{seed}.jsonl"
        if formal_training["raw_rows_sha256"][path.name] != sha(path):
            raise ValueError("formal baseline hash mismatch")
        rows = read_rows(path); baseline_records[str(seed)] = rows
        baseline_metrics[str(seed)] = aggregate(rows)
    baseline_summary = summarize_seed_metrics(baseline_metrics)

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sessions = load_split(args.data_root, "dev")
    scorer, feature_meta = build_scorer(
        sessions, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size)

    grid, raw_hashes = {}, {}
    for threshold in THRESHOLDS:
        controller = TopGateController(scorer, head, artifact["mean"], artifact["std"], threshold)
        records_by_seed, metrics_by_seed = {}, {}
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            records = [replay_controller(session, scorer, GATE_SEEDS[replicate], seed,
                                         controller, split="development_threshold_diagnostic")
                       for session in sessions]
            records_by_seed[str(seed)] = records; metrics_by_seed[str(seed)] = aggregate(records)
            path = root / f"dev_threshold_{threshold_key(threshold)}_controller_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        controller_summary = summarize_seed_metrics(metrics_by_seed)
        comparison = paired_comparison(
            records_by_seed, baseline_records, 9501 + int(round(threshold * 10)))
        grid[threshold_key(threshold)] = {
            "threshold": threshold,
            "controller": controller_summary,
            "comparison": comparison,
            "development_gate": development_gate(controller_summary, comparison),
        }
    selected = max(grid.values(), key=rank_threshold)
    passing = [row["threshold"] for row in grid.values() if row["development_gate"]["passed"]]
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    summary = {
        "protocol": PROTOCOL,
        "diagnostic_split": "development",
        "baseline": baseline_summary,
        "grid": grid,
        "selected_by_reliability": selected,
        "thresholds_passing_original_gate": passing,
        "threshold_only_sufficient": bool(passing),
        "test_loaded": False,
        "test_scored": False,
        "terminal_status": "completed_dev_threshold_diagnostic",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "features": feature_meta,
        "raw_rows_sha256": raw_hashes,
        "formal_summary_sha256": sha(formal / "summary.json"),
        "formal_training_sha256": sha(formal / "training.json"),
        "formal_manifest_sha256": sha(formal / "training_manifest.json"),
        "controller_checkpoint_sha256": sha(checkpoint),
    }
    manifest = {
        "protocol": PROTOCOL,
        "formal_protocol": FORMAL_PROTOCOL,
        "controller_protocol": SOURCE_PROTOCOL,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "thresholds": list(THRESHOLDS),
        "selection_objective": ["action_accuracy", "worst_hard_or_update", "query_accuracy"],
        "controller_seeds": list(CONTROLLER_SEEDS),
        "readout_seeds": list(GATE_SEEDS),
        "dev_loaded": True,
        "test_loaded": False,
        "test_scored": False,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
