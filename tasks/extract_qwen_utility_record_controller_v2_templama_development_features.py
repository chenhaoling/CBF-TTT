"""Extract frozen V1-controller features on TempLAMA development."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
)
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES,
    base_pair_margin,
    candidate_features,
    required_embedding_texts,
    teacher_targets,
)
from tasks.label_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
    EXPECTED_LABELS,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    extract_embeddings,
    load_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v2_templama_development_features"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v2_templama_development_features_audit"


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, label_root = Path(args.data_root), Path(args.label_root)
    data_audit_path = data_root / "data_audit.json"
    label_audit_path = label_root / "development_labels_audit.json"
    data_audit = json.loads(data_audit_path.read_text())
    label_audit = json.loads(label_audit_path.read_text())
    if data_audit.get("protocol") != DATA_AUDIT_PROTOCOL or not data_audit.get("passed"):
        raise ValueError("development data audit did not pass")
    if label_audit.get("protocol") != LABEL_AUDIT_PROTOCOL or not label_audit.get("passed"):
        raise ValueError("development label audit did not pass")
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    label_path = label_root / f"labels_shard{args.shard}.jsonl"
    labels = load_jsonl(label_path)
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("development feature shard output exists")

    torch.manual_seed(2610162)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = required_embedding_texts(list(episode_by_id.values()), labels)
    started = time.perf_counter()
    embeddings, lengths = extract_embeddings(
        model, tokenizer, texts, args.embedding_batch_size, args.max_text_length
    )
    torch.cuda.synchronize()
    embedding_seconds = time.perf_counter() - started

    rows = []; pair_seconds = 0.0
    with output.open("x") as sink, torch.inference_mode():
        for index, label in enumerate(labels):
            episode = episode_by_id[label["episode_id"]]
            incoming = episode["events"][label["event_index"]]["record"]
            prior = {event["record"]["record_id"]: event["record"]
                     for event in episode["events"][:label["event_index"]]}
            pair_values = {}
            pair_started = time.perf_counter()
            for identity in label["candidate_record_ids"]:
                pair_values[(identity, incoming["record_id"])] = base_pair_margin(
                    model, tokenizer, prior[identity], incoming
                )
            pair_seconds += time.perf_counter() - pair_started
            features = candidate_features(episode, label, embeddings, pair_values)
            row = {
                "protocol": PROTOCOL,
                "episode_id": label["episode_id"],
                "source_group": label["source_group"],
                "boundary_index": label["boundary_index"],
                "event_index": label["event_index"],
                "candidate_record_ids": label["candidate_record_ids"],
                "feature_names": list(FEATURE_NAMES),
                "features": features,
                "targets": teacher_targets(label),
                "audit": {
                    "target_relation": episode["audit"]["target_relation"],
                    "mechanism": episode["mechanism"],
                    "twin": episode["twin"],
                },
                "partition": "templama_validation_development",
                "shard": args.shard,
            }
            sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            if (index + 1) % 25 == 0:
                print(json.dumps({"shard": args.shard, "completed": index + 1,
                                  "total": len(labels)}), flush=True)
    torch.cuda.synchronize()
    resource = {
        "protocol": PROTOCOL,
        "partition": "templama_validation_development",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "boundaries": len(rows),
        "record_examples": sum(len(row["features"]) for row in rows),
        "feature_dimension": len(FEATURE_NAMES),
        "embedding_texts": len(texts),
        "token_min": min(lengths),
        "token_max": max(lengths),
        "embedding_seconds": embedding_seconds,
        "pair_seconds": pair_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_audit_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "label_audit_sha256": sha256(label_audit_path),
        "label_shard_sha256": sha256(label_path),
        "model": model_meta,
        "tokenizer": args.tokenizer,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }
    resource_output.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    return resource


def summarize(args) -> dict:
    root = Path(args.root); rows = []; resources = []
    for shard in range(args.num_shards):
        path = root / f"features_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) or resource["boundaries"] != len(part):
            raise ValueError("development feature shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    features = [value for row in rows for candidate in row["features"] for value in candidate]
    deletes = [value for row in rows for value in row["targets"]["delete_targets"]]
    risks = [value for row in rows for value in row["targets"]["risk_targets"]]
    checks = {
        "complete_unique_boundaries": len(rows) == EXPECTED_LABELS and len(identities) == len(rows),
        "fixed_feature_schema": all(tuple(row["feature_names"]) == FEATURE_NAMES
                                    and all(len(candidate) == len(FEATURE_NAMES)
                                            for candidate in row["features"]) for row in rows),
        "finite_features": all(math.isfinite(value) for value in features),
        "valid_binary_targets": (
            bool(deletes) and bool(risks)
            and set(deletes) <= {0, 1} and set(risks) <= {0, 1}
        ),
        "locked_inputs": len({row["data_manifest_sha256"] for row in resources}) == 1
                         and len({row["label_audit_sha256"] for row in resources}) == 1,
        "heldouts_sealed": all(
            not row["templama_test_loaded"]
            and not row["v1_confirmation_loaded"]
            and not row["external_benchmarks_loaded"]
            for row in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "checks": checks,
        "passed": passed,
        "boundaries": len(rows),
        "record_examples": len(deletes),
        "feature_dimension": len(FEATURE_NAMES),
        "delete_fraction": statistics.fmean(deletes),
        "risk_fraction": statistics.fmean(risks),
        "delete_values": sorted(set(deletes)),
        "risk_values": sorted(set(risks)),
        "safe_binary_actions": dict(Counter(
            row["targets"]["safe_binary_action"] for row in rows
        )),
        "resource": {
            "two_gpu_critical_seconds": max(row["total_seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
        },
        "next_stage": "templama_development_feature_audit" if passed else "stop",
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--label-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--embedding-batch-size", type=int, default=16)
    collect_parser.add_argument("--max-text-length", type=int, default=256)
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
