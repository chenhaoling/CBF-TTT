"""Construct V11 labels on audited unseen-relation FewRel development episodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v11_fewrel_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL, EXPECTED_RELATIONS,
    GROUPS_PER_RELATION, PROTOCOL as DATA_PROTOCOL,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.pilot_qwen_utility_record_controller_v10_fewrel import (
    ALPHAS, MAX_ANSWER_TOKENS, _answer_coverage, single_candidate_benefit,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    OutcomeCache, VALUE_SCALE, action_grid, build_label, extract_embeddings,
    load_jsonl, percentile, retain_action_name, unique_relation_texts,
    unrelated_effects,
)


PROTOCOL = "qwen_utility_record_controller_v11_fewrel_development_labels"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v11_fewrel_development_labels_audit"
EXPECTED_GROUPS = EXPECTED_RELATIONS * GROUPS_PER_RELATION
EXPECTED_EPISODES = EXPECTED_GROUPS * 2
EXPECTED_LABELS = EXPECTED_EPISODES * 3
EXPECTED_LABELS_PER_RELATION = GROUPS_PER_RELATION * 2 * 3


def _verified_inputs(data_root: Path, data_audit_path: Path) \
        -> tuple[dict, dict, dict]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unexpected V11 FewRel development data protocol")
    if data_audit.get("protocol") != DATA_AUDIT_PROTOCOL \
            or not data_audit.get("passed") \
            or data_audit.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("V11 FewRel development data audit dependency failed")
    locks = {
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(data_audit_path),
    }
    return manifest, data_audit, locks


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root = Path(args.data_root)
    _, _, locks = _verified_inputs(data_root, Path(args.data_audit))
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError("unexpected V11 FewRel development episode count")
    selected = [row for index, row in enumerate(episodes)
                if index % args.num_shards == args.shard]
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("V11 FewRel development shard output exists")

    torch.manual_seed(202610183)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    coverage = _answer_coverage(selected, tokenizer)
    texts = unique_relation_texts(selected)
    extraction_started = time.perf_counter()
    embeddings, token_lengths = extract_embeddings(
        model, tokenizer, texts, args.embedding_batch_size,
        args.max_relation_length,
    )
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - extraction_started
    extraction_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_reserved = torch.cuda.max_memory_reserved() / 2**30

    rows, total_hits, total_misses = [], 0, 0
    started = time.perf_counter()
    with output.open("x") as sink, torch.inference_mode():
        for episode_index, episode in enumerate(selected):
            cache = OutcomeCache(model, tokenizer)
            for boundary_offset in range(len(episode["boundaries"])):
                row = build_label(
                    episode, boundary_offset, embeddings, cache, True,
                    alphas=ALPHAS,
                )
                row["protocol"] = PROTOCOL
                row["partition"] = "fewrel_development"
                row["target_relation"] = episode["audit"]["target_relation"]
                row["shard"] = args.shard
                sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            total_hits += cache.hits; total_misses += cache.misses
            del cache
            torch.cuda.empty_cache()
            print(json.dumps({"shard": args.shard,
                              "completed": episode_index + 1,
                              "total": len(selected)}), flush=True)
    label_seconds = time.perf_counter() - started
    resource = {
        "protocol": PROTOCOL, "partition": "fewrel_development",
        "shard": args.shard, "num_shards": args.num_shards,
        "episodes": len(selected), "labels": len(rows),
        "relation_texts": len(texts),
        "relation_token_min": min(token_lengths),
        "relation_token_max": max(token_lengths),
        "answer_coverage": coverage,
        "embedding_seconds": extraction_seconds,
        "label_seconds": label_seconds,
        "total_seconds": extraction_seconds + label_seconds,
        "overall_peak_allocated_gib": max(
            extraction_allocated,
            max(row["peak_allocated_gib"] for row in rows),
        ),
        "overall_peak_reserved_gib": max(
            extraction_reserved,
            max(row["peak_reserved_gib"] for row in rows),
        ),
        "mean_label_seconds": statistics.fmean(
            row["label_seconds"] for row in rows
        ),
        "cache_hits": total_hits, "cache_misses": total_misses,
        "rows_sha256": sha256(output), **locks,
        "model": model_meta, "tokenizer": args.tokenizer,
        "value_scale": VALUE_SCALE, "action_alphas": list(ALPHAS),
        "gpu": torch.cuda.get_device_name(),
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "heldout_benchmarks_loaded": False,
        "internal_test_loaded": False,
        "external_test_loaded": False,
    }
    write_json(resource_output, resource)
    return resource


def formal_gate(*, finite_complete: bool, peak_reserved_gib: float,
                answer_coverage: float, positive_fraction: float,
                oracle_headroom: float, relation_coverage: bool,
                zero_oracle_collateral: bool, heldouts_sealed: bool) -> dict:
    checks = {
        "finite_complete_auditable_labels": finite_complete,
        "peak_reserved_at_most_28_gib": peak_reserved_gib <= 28.0,
        "answer_token_coverage_at_least_point_95": answer_coverage >= .95,
        "positive_safe_benefit_candidates_at_least_point_05": (
            positive_fraction >= .05
        ),
        "safe_oracle_headroom_at_least_point_015": oracle_headroom >= .015,
        "complete_balanced_relation_coverage": relation_coverage,
        "safe_oracle_zero_added_unrelated_errors": zero_oracle_collateral,
        "heldout_sources_sealed": heldouts_sealed,
    }
    return {"checks": checks, "passed": all(checks.values())}


def summarize(args) -> dict:
    root, data_root = Path(args.root), Path(args.data_root)
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    expected = {(row["episode_id"], boundary["boundary_index"])
                for row in episodes for boundary in row["boundaries"]}
    rows, resources = [], []
    for shard in range(args.num_shards):
        path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) \
                or resource["labels"] != len(part):
            raise ValueError("V11 FewRel development shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    total_queries = sum(row["actions"][row["retain_action"]]["queries"]
                        for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"]
                         for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"]
                         for row in rows)
    oracle_accuracy, retain_accuracy = (
        oracle_correct / total_queries, retain_correct / total_queries
    )
    headroom = oracle_accuracy - retain_accuracy
    candidates = single_candidate_benefit(rows)
    answer_total = sum(row["answer_coverage"]["answers"] for row in resources)
    answer_covered = sum(row["answer_coverage"]["covered"] for row in resources)
    answer_coverage = answer_covered / answer_total
    relation_counts = Counter(row["target_relation"] for row in rows)
    mechanism_counts = Counter(row["mechanism"] for row in rows)
    unrelated = unrelated_effects(rows)
    finite = (
        len(rows) == EXPECTED_LABELS and identities == expected
        and all(math.isfinite(row["label_seconds"])
                and row["label_seconds"] > 0
                and row["retain_action"] == retain_action_name(row["candidate_count"])
                and set(row["actions"]) == {
                    action["name"] for action in action_grid(
                        row["candidate_count"], ALPHAS
                    )
                } for row in rows)
    )
    relation_coverage = (
        len(relation_counts) == EXPECTED_RELATIONS
        and set(relation_counts.values()) == {EXPECTED_LABELS_PER_RELATION}
    )
    sealed = all(not row["development_labels_loaded"]
                 and not row["development_model_scores_loaded"]
                 and not row["heldout_benchmarks_loaded"]
                 and not row["internal_test_loaded"]
                 and not row["external_test_loaded"]
                 for row in resources)
    peak_reserved = max(row["overall_peak_reserved_gib"] for row in resources)
    gate = formal_gate(
        finite_complete=finite, peak_reserved_gib=peak_reserved,
        answer_coverage=answer_coverage,
        positive_fraction=candidates["positive_safe_benefit_fraction"],
        oracle_headroom=headroom, relation_coverage=relation_coverage,
        zero_oracle_collateral=(unrelated["added_errors"] == 0
                                and unrelated["rowwise_barrier_passed"]),
        heldouts_sealed=sealed,
    )
    label_times = [row["label_seconds"] for row in rows]
    per_mechanism = {}
    for mechanism in sorted(mechanism_counts):
        part = [row for row in rows if row["mechanism"] == mechanism]
        queries = sum(row["actions"][row["retain_action"]]["queries"]
                      for row in part)
        per_mechanism[mechanism] = {
            "labels": len(part),
            "retain_accuracy": sum(
                row["actions"][row["retain_action"]]["correct"] for row in part
            ) / queries,
            "safe_oracle_accuracy": sum(
                row["actions"][row["safe_oracle_action"]]["correct"] for row in part
            ) / queries,
        }
    return {
        "protocol": PROTOCOL, "labels": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "future_queries": total_queries,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_accuracy": oracle_accuracy,
        "safe_oracle_headroom": headroom,
        "single_candidate_benefit": candidates,
        "answer_token_coverage": {
            "answers": answer_total, "covered": answer_covered,
            "coverage": answer_coverage, "max_answer_tokens": MAX_ANSWER_TOKENS,
        },
        "relation_label_counts": dict(sorted(relation_counts.items())),
        "mechanism_label_counts": dict(sorted(mechanism_counts.items())),
        "per_mechanism": per_mechanism,
        "safe_oracle_actions": dict(Counter(
            row["safe_oracle_action"] for row in rows
        )),
        "candidate_counts": dict(Counter(
            str(row["candidate_count"]) for row in rows
        )),
        "unrelated": unrelated,
        "label_time_seconds": {
            "mean": statistics.fmean(label_times),
            "median": statistics.median(label_times),
            "p90": percentile(label_times, .90),
            "p95": percentile(label_times, .95), "max": max(label_times),
        },
        "resource": {
            "two_gpu_critical_seconds": max(
                row["total_seconds"] for row in resources
            ),
            "peak_allocated_gib": max(
                row["overall_peak_allocated_gib"] for row in resources
            ),
            "peak_reserved_gib": peak_reserved,
        },
        "upstream_hashes": {
            key: resources[0][key] for key in (
                "data_manifest_sha256", "data_audit_sha256",
            )
        },
        "gate": gate,
        "next_stage": (
            "v11_development_feature_extraction"
            if gate["passed"] else "stop_v11_before_feature_extraction"
        ),
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "heldout_benchmarks_loaded": False,
        "internal_test_loaded": False,
        "external_test_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--data-audit", required=True)
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
        result = collect(args)
    else:
        result = summarize(args); write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
