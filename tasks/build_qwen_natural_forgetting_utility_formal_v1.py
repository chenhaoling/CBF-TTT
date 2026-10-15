"""Full calibration counterfactual future-utility corpus for forgetting."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import EXPECTED as DATA_COUNTS, load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    BOOTSTRAP_SEED,
    EXPECTED,
    QwenOutcomeCache,
    UNIQUE_ACTIONS,
    bootstrap_session,
    build_label,
    quantile,
    registered_grid,
    scope_examples,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_utility_formal_v1"
PLAN = "QWEN_NATURAL_FORGETTING_UTILITY_FORMAL_V1_PLAN.md"


def aggregate(labels, resources):
    safe_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in labels)
    retain_correct = sum(row["actions"]["retain_all"]["correct"] for row in labels)
    queries = sum(row["actions"]["retain_all"]["queries"] for row in labels)
    gain = (safe_correct - retain_correct) / queries
    ci, session_gains = bootstrap_session(labels)
    selected = Counter(row["safe_oracle_action"] for row in labels)
    unrelated_errors = sum(max(
        0, row["actions"]["retain_all"]["unrelated_correct"]
        - row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
    ) for row in labels)
    seconds = [row["label_seconds"] for row in labels]
    max_allocated = max(row["peak_allocated_gib"] for row in labels)
    checks = {
        "exactly_65_sessions": len({row["session_id"] for row in labels}) == 65,
        "exactly_774_unique_labels": len(labels) == 774 and len({
            (row["session_id"], row["event_step"]) for row in labels
        }) == 774,
        "complete_unique_and_registered_grid": all(
            set(row["actions"]) == set(UNIQUE_ACTIONS) for row in labels
        ) and len(registered_grid()) == 15,
        "safe_oracle_gain_ci_lower_above_zero": ci[0] > 0,
        "nonretain_rate_at_least_point_10":
            statistics.fmean(row["safe_oracle_nonretain"] for row in labels) >= .10,
        "retain_all_at_least_50_labels": selected["retain_all"] >= 50,
        "scope_top1_a0_at_least_50_labels": selected["scope_top1_a0"] >= 50,
        "at_least_five_selected_actions": len(selected) >= 5,
        "no_action_above_point_90": max(selected.values()) / len(labels) <= .90,
        "zero_added_unrelated_errors": unrelated_errors == 0,
        "median_label_seconds_at_most_120": statistics.median(seconds) <= 120,
        "peak_allocated_gib_at_most_28": max_allocated <= 28,
    }
    return {
        "protocol": PROTOCOL, "stage": "calibration_formal",
        "sessions": len({row["session_id"] for row in labels}), "labels": len(labels),
        "registered_grid": registered_grid(), "unique_actions": list(UNIQUE_ACTIONS),
        "label_distribution": dict(sorted(selected.items())),
        "safe_oracle_nonretain_rate": statistics.fmean(
            row["safe_oracle_nonretain"] for row in labels),
        "retain_candidate_accuracy": retain_correct / queries,
        "safe_oracle_candidate_accuracy": safe_correct / queries,
        "safe_oracle_gain": gain,
        "safe_oracle_gain_session_bootstrap_95ci": ci,
        "session_gains": session_gains,
        "added_unrelated_errors": unrelated_errors,
        "resources": {
            "median_label_seconds": statistics.median(seconds),
            "p90_label_seconds": quantile(seconds, .90),
            "max_label_seconds": max(seconds),
            "peak_allocated_gib": max_allocated,
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in labels),
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "shards": resources,
        },
        "gate": {"checks": checks, "controller_training_authorized": all(checks.values())},
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    outputs = [Path(args.output), Path(args.resource_output)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("formal utility shard output exists")
    paths = {
        "data_manifest": Path(args.data_root) / "manifest.json",
        "v5_checkpoint": Path(args.v5_root) / "result" / "constrained_scope_gate.pt",
        "transforms": Path(args.context_root) / "transforms.pt",
        "scope": Path(args.scope_root) / "scope_gates.pt",
    }
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED:
        raise ValueError(f"sealed formal input mismatch: {observed}")
    _, sessions = load(args.data_root, "calibration")
    if len(sessions) != DATA_COUNTS["calibration"]["sessions"]:
        raise ValueError("formal calibration session count changed")
    selected = [row for index, row in enumerate(sessions)
                if index % args.num_shards == args.shard]
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    source_summary = json.loads((Path(args.context_root) / "summary.json").read_text())
    transforms = torch.load(paths["transforms"], map_location="cpu", weights_only=False)
    scope_artifact = torch.load(paths["scope"], map_location="cpu", weights_only=False)
    scorer, feature_meta = build_scorer(
        scope_examples(selected), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact,
        args.batch_size,
    )
    gate, threshold, reader_selected = load_v5_gate(
        paths["v5_checkpoint"], scope_artifact["heads"]["7301"]
    )
    if threshold != .30:
        raise ValueError("frozen V5 threshold changed")
    labels = []
    with Path(args.output).open("x") as sink, torch.no_grad():
        for session_index, session in enumerate(selected):
            cache = QwenOutcomeCache(model, tokenizer, session["candidates"])
            for step in range(len(session["events"])):
                label = build_label(session, step, scorer, gate, threshold, cache)
                label["protocol"] = PROTOCOL; label["shard"] = args.shard
                sink.write(json.dumps(label, ensure_ascii=False) + "\n"); sink.flush()
                labels.append(label)
            print(json.dumps({"shard": args.shard, "session": session["id"],
                              "session_index": session_index + 1,
                              "sessions": len(selected), "labels": len(labels)}), flush=True)
    resource = {
        "protocol": PROTOCOL, "stage": "calibration_formal", "shard": args.shard,
        "num_shards": args.num_shards, "sessions": len(selected), "labels": len(labels),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "labels_sha256": sha(args.output), "sealed_inputs": observed,
        "plan_sha256": sha(PLAN), "feature_meta": feature_meta,
        "reader_selected": reader_selected, "reader_threshold": threshold,
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(Path(args.resource_output), resource)
    return resource


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    labels, resources = [], []
    for shard in range(args.num_shards):
        label_path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in label_path.read_text().splitlines() if line]
        if resource["labels_sha256"] != sha(label_path) or resource["labels"] != len(part):
            raise ValueError("formal shard hash/count mismatch")
        labels.extend(part); resources.append(resource)
    result = aggregate(labels, resources)
    result.update({
        "label_files_sha256": {f"labels_shard{index}.jsonl": resource["labels_sha256"]
                               for index, resource in enumerate(resources)},
        "plan_sha256": sha(PLAN), "sealed_inputs": EXPECTED,
    })
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("data_root", "v5_root", "model", "tokenizer", "context_root",
                 "scope_root", "output", "resource_output"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=128)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
