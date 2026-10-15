"""Construct frozen-Qwen counterfactual labels for controller development."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v1_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
    GROUPS,
    PROTOCOL as DATA_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    OutcomeCache,
    VALUE_SCALE,
    bootstrap_group_gain,
    build_label,
    extract_embeddings,
    load_jsonl,
    percentile,
    unique_relation_texts,
    unrelated_effects,
)


PROTOCOL = "qwen_utility_record_controller_v1_development_labels"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v1_development_labels_audit"
EXPECTED_EPISODES = GROUPS * 2
EXPECTED_LABELS = EXPECTED_EPISODES * 3


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root = Path(args.data_root)
    manifest_path = data_root / "manifest.json"
    audit_path = data_root / "development_data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unexpected controller development data protocol")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL or not audit.get("passed"):
        raise ValueError("development data audit did not pass")
    if audit.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("development manifest changed after audit")
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    selected = [row for index, row in enumerate(episodes)
                if index % args.num_shards == args.shard]
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("development label shard output exists")

    torch.manual_seed(2610162)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = unique_relation_texts(selected)
    extraction_started = time.perf_counter()
    embeddings, token_lengths = extract_embeddings(
        model, tokenizer, texts, args.embedding_batch_size, args.max_relation_length
    )
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - extraction_started
    extraction_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_reserved = torch.cuda.max_memory_reserved() / 2**30

    started = time.perf_counter(); rows = []
    total_cache_hits = 0; total_cache_misses = 0
    with output.open("x") as sink, torch.inference_mode():
        for episode_index, episode in enumerate(selected):
            cache = OutcomeCache(model, tokenizer)
            for boundary_offset in range(len(episode["boundaries"])):
                row = build_label(episode, boundary_offset, embeddings, cache, True)
                row["protocol"] = PROTOCOL
                row["partition"] = "development"
                row["shard"] = args.shard
                sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            total_cache_hits += cache.hits; total_cache_misses += cache.misses
            del cache; torch.cuda.empty_cache()
            print(json.dumps({"shard": args.shard, "completed": episode_index + 1,
                              "total": len(selected)}), flush=True)
    label_seconds = time.perf_counter() - started
    resource = {
        "protocol": PROTOCOL,
        "partition": "development",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "episodes": len(selected),
        "labels": len(rows),
        "relation_texts": len(texts),
        "relation_token_min": min(token_lengths),
        "relation_token_max": max(token_lengths),
        "embedding_seconds": extraction_seconds,
        "label_seconds": label_seconds,
        "total_seconds": extraction_seconds + label_seconds,
        "overall_peak_allocated_gib": max(
            extraction_allocated, max(row["peak_allocated_gib"] for row in rows)
        ),
        "overall_peak_reserved_gib": max(
            extraction_reserved, max(row["peak_reserved_gib"] for row in rows)
        ),
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path),
        "model": model_meta,
        "tokenizer": args.tokenizer,
        "value_scale": VALUE_SCALE,
        "gpu": torch.cuda.get_device_name(),
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }
    resource_output.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    return resource


def summarize(args) -> dict:
    root, data_root = Path(args.root), Path(args.data_root)
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    expected = {(episode["episode_id"], boundary["boundary_index"])
                for episode in episodes for boundary in episode["boundaries"]}
    rows, resources = [], []
    for shard in range(args.num_shards):
        path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) or resource["labels"] != len(part):
            raise ValueError("development label shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    total_queries = sum(row["actions"][row["retain_action"]]["queries"] for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"] for row in rows)
    oracle_accuracy = oracle_correct / total_queries
    retain_accuracy = retain_correct / total_queries
    interval = bootstrap_group_gain(rows)
    unrelated = unrelated_effects(rows)
    label_times = [row["label_seconds"] for row in rows]
    per_mechanism = {}
    for mechanism in sorted({row["mechanism"] for row in rows}):
        part = [row for row in rows if row["mechanism"] == mechanism]
        queries = sum(row["actions"][row["retain_action"]]["queries"] for row in part)
        per_mechanism[mechanism] = {
            "labels": len(part),
            "safe_oracle_accuracy": sum(
                row["actions"][row["safe_oracle_action"]]["correct"] for row in part
            ) / queries,
            "retain_all_accuracy": sum(
                row["actions"][row["retain_action"]]["correct"] for row in part
            ) / queries,
        }
    checks = {
        "complete_unique_rows": len(rows) == EXPECTED_LABELS and identities == expected,
        "finite_rows": all(row["protocol"] == PROTOCOL and row["partition"] == "development"
                           and math.isfinite(row["label_seconds"]) and row["label_seconds"] > 0
                           for row in rows),
        "oracle_gain_ci_lower_above_zero": interval[0] > 0,
        "zero_added_unrelated_errors": unrelated["added_errors"] == 0,
        "development_data_locked": all(
            resource["data_manifest_sha256"] == sha256(data_root / "manifest.json")
            and resource["data_audit_sha256"]
            == sha256(data_root / "development_data_audit.json") for resource in resources
        ),
        "frozen_plain_qwen": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["value_scale"] == VALUE_SCALE for resource in resources
        ),
        "confirmation_and_external_sealed": all(
            not resource["confirmation_loaded"] and not resource["external_benchmarks_loaded"]
            for resource in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "partition": "development",
        "labels": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "future_queries": total_queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": unrelated["added_errors"],
        "additional_unrelated_correct": unrelated["additional_correct"],
        "safe_oracle_actions": dict(Counter(row["safe_oracle_action"] for row in rows)),
        "per_mechanism": per_mechanism,
        "label_time_seconds": {
            "mean": statistics.fmean(label_times), "median": statistics.median(label_times),
            "p90": percentile(label_times, .90), "p95": percentile(label_times, .95),
            "max": max(label_times),
        },
        "resource": {
            "two_gpu_critical_seconds": max(row["total_seconds"] for row in resources),
            "peak_allocated_gib": max(row["overall_peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["overall_peak_reserved_gib"] for row in resources),
        },
        "gate": {"checks": checks, "passed": passed},
        "passed": passed,
        "next_stage": "development_label_audit" if passed else "stop",
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--embedding-batch-size", type=int, default=16)
    collect_parser.add_argument("--max-relation-length", type=int, default=256)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps(collect(args), indent=2, sort_keys=True))
    else:
        result = summarize(args)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
