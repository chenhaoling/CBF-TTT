"""Label and summarize the TempLAMA V2 ternary resource pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
    PROTOCOL as DATA_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    OutcomeCache,
    VALUE_SCALE,
    build_label,
    extract_embeddings,
    load_jsonl,
    percentile,
    unique_relation_texts,
)


PROTOCOL = "qwen_utility_record_controller_v2_templama_resource_pilot"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v2_templama_resource_pilot_audit"
EXPECTED_GROUPS = 72
EXPECTED_EPISODES = 144
EXPECTED_LABELS = 432
FORMAL_GROUPS = 256
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610170


def bootstrap_group_gain(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    gains = {
        group: sum(value["safe_oracle_correct_gain"] for value in values)
        / sum(value["actions"][value["retain_action"]]["queries"] for value in values)
        for group, values in grouped.items()
    }
    groups = sorted(gains); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(gains[group] for group in sample))
    return [percentile(draws, .025), percentile(draws, .975)]


def half_strength_decision(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["safe_oracle_contains_half"]]
    unique = [
        row for row in selected
        if row["half_vs_endpoint_correct_gain"] > 0
        or (row["half_vs_endpoint_correct_gain"] == 0
            and row["half_vs_endpoint_nll_gain"] >= .01)
    ]
    changes_prediction = any(
        row["half_vs_endpoint_correct_gain"] != 0 for row in selected
    )
    requires_soft_branch = (
        len(unique) / len(rows) >= .05 and changes_prediction
    )
    return {
        "selected_labels": len(selected),
        "unique_margin_labels": len(unique),
        "unique_fraction": len(unique) / len(rows),
        "changes_prediction": changes_prediction,
        "requires_separate_soft_v2": requires_soft_branch,
        "primary_action_space": "ternary" if requires_soft_branch else "binary",
    }


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root = Path(args.data_root)
    manifest_path, audit_path = data_root / "manifest.json", data_root / "data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unexpected TempLAMA pilot data protocol")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL or not audit.get("passed"):
        raise ValueError("TempLAMA pilot data audit did not pass")
    if audit.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("TempLAMA pilot manifest changed after audit")
    episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    selected = [row for index, row in enumerate(episodes)
                if index % args.num_shards == args.shard]
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("TempLAMA pilot shard output exists")

    torch.manual_seed(202610170)
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

    started = time.perf_counter(); rows = []; hits = 0; misses = 0
    with output.open("x") as sink, torch.inference_mode():
        for episode_index, episode in enumerate(selected):
            cache = OutcomeCache(model, tokenizer)
            for boundary_offset in range(len(episode["boundaries"])):
                row = build_label(episode, boundary_offset, embeddings, cache, True)
                row["protocol"] = PROTOCOL; row["shard"] = args.shard
                sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            hits += cache.hits; misses += cache.misses
            del cache
            torch.cuda.empty_cache()
            print(json.dumps({"shard": args.shard, "completed": episode_index + 1,
                              "total": len(selected)}), flush=True)
    label_seconds = time.perf_counter() - started
    resource = {
        "protocol": PROTOCOL,
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
        "extraction_peak_allocated_gib": extraction_allocated,
        "extraction_peak_reserved_gib": extraction_reserved,
        "label_peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
        "label_peak_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
        "overall_peak_allocated_gib": max(
            extraction_allocated, max(row["peak_allocated_gib"] for row in rows)
        ),
        "overall_peak_reserved_gib": max(
            extraction_reserved, max(row["peak_reserved_gib"] for row in rows)
        ),
        "mean_label_seconds": statistics.fmean(row["label_seconds"] for row in rows),
        "cache_hits": hits,
        "cache_misses": misses,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path),
        "model": model_meta,
        "tokenizer": args.tokenizer,
        "value_scale": VALUE_SCALE,
        "gpu": torch.cuda.get_device_name(),
        "v1_confirmation_loaded": False,
        "development_loaded": False,
        "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    resource_output.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    return resource


def summarize(args) -> dict:
    root = Path(args.root); rows = []; resources = []
    for shard in range(args.num_shards):
        path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) or resource["labels"] != len(part):
            raise ValueError("TempLAMA pilot shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    if len(rows) != EXPECTED_LABELS or len(identities) != len(rows):
        raise ValueError("TempLAMA pilot label coverage mismatch")
    total_queries = sum(row["actions"][row["retain_action"]]["queries"] for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"] for row in rows)
    oracle_accuracy = oracle_correct / total_queries
    retain_accuracy = retain_correct / total_queries
    unrelated_deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"] for row in rows
    ]
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
            "optimal_actions": len({row["safe_oracle_action"] for row in part}),
        }
    half = half_strength_decision(rows)
    interval = bootstrap_group_gain(rows)
    label_times = [row["label_seconds"] for row in rows]
    critical_seconds = max(resource["total_seconds"] for resource in resources)
    output_bytes = sum((root / f"labels_shard{shard}.jsonl").stat().st_size
                       for shard in range(args.num_shards))
    projection = {
        "groups": FORMAL_GROUPS,
        "two_gpu_seconds": critical_seconds * FORMAL_GROUPS / EXPECTED_GROUPS,
        "output_gib": output_bytes / EXPECTED_GROUPS * FORMAL_GROUPS / 2**30,
    }
    checks = {
        "finite_complete_rows": all(
            math.isfinite(row["label_seconds"]) and row["label_seconds"] > 0
            for row in rows
        ),
        "oracle_gain_at_least_point_03": oracle_accuracy - retain_accuracy >= .03,
        "oracle_gain_ci_lower_above_zero": interval[0] > 0,
        "zero_added_unrelated_errors": all(delta >= 0 for delta in unrelated_deltas),
        "every_mechanism_has_two_actions": all(
            value["optimal_actions"] >= 2 for value in per_mechanism.values()
        ),
        "peak_reserved_below_24_gib": max(
            resource["overall_peak_reserved_gib"] for resource in resources
        ) < 24,
        "formal_projection_below_two_hours_and_50_gib": (
            projection["two_gpu_seconds"] < 2 * 3600
            and projection["output_gib"] < 50
        ),
        "heldouts_sealed": all(
            not resource["v1_confirmation_loaded"] and not resource["development_loaded"]
            and not resource["test_loaded"] and not resource["external_benchmarks_loaded"]
            for resource in resources
        ),
    }
    passed = all(checks.values())
    if not passed:
        next_stage = "stop_resource_pilot"
    elif half["requires_separate_soft_v2"]:
        next_stage = "register_soft_v2_before_formal"
    else:
        next_stage = "templama_replication_development_construction"
    return {
        "protocol": PROTOCOL,
        "labels": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "future_queries": total_queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": sum(max(0, -delta) for delta in unrelated_deltas),
        "additional_unrelated_correct": sum(max(0, delta) for delta in unrelated_deltas),
        "safe_oracle_actions": dict(Counter(row["safe_oracle_action"] for row in rows)),
        "per_mechanism": per_mechanism,
        "half_strength": half,
        "label_time_seconds": {
            "mean": statistics.fmean(label_times),
            "median": statistics.median(label_times),
            "p90": percentile(label_times, .90),
            "p95": percentile(label_times, .95),
            "max": max(label_times),
        },
        "resource": {
            "two_gpu_critical_seconds": critical_seconds,
            "peak_allocated_gib": max(r["overall_peak_allocated_gib"] for r in resources),
            "peak_reserved_gib": max(r["overall_peak_reserved_gib"] for r in resources),
            "formal_projection": projection,
        },
        "gate": {"checks": checks, "passed": passed},
        "next_stage": next_stage,
        "v1_confirmation_loaded": False,
        "development_loaded": False,
        "test_loaded": False,
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
