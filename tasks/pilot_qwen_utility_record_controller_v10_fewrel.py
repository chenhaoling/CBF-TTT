"""Label and summarize the V10 28-group FewRel resource pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL, GROUPS_PER_RELATION,
    PILOT_RELATIONS, PROTOCOL as DATA_PROTOCOL,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    ALPHAS, OutcomeCache, VALUE_SCALE, action_grid, build_label,
    extract_embeddings, load_jsonl, percentile, retain_action_name,
    unique_relation_texts, unrelated_effects,
)


PROTOCOL = "qwen_utility_record_controller_v10_fewrel_resource_pilot"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v10_fewrel_resource_pilot_audit"
EXPECTED_GROUPS = PILOT_RELATIONS * GROUPS_PER_RELATION
EXPECTED_EPISODES = EXPECTED_GROUPS * 2
EXPECTED_LABELS = EXPECTED_EPISODES * 3
MAX_ANSWER_TOKENS = 32
FORMAL_GROUPS = 672


def _answer_coverage(episodes: list[dict], tokenizer) -> dict:
    answers = [query["answer"] for episode in episodes
               for boundary in episode["boundaries"]
               for query in boundary["queries"]]
    lengths = [len(tokenizer(value, add_special_tokens=False).input_ids)
               for value in answers]
    covered = [0 < value <= MAX_ANSWER_TOKENS for value in lengths]
    return {
        "answers": len(answers), "covered": sum(covered),
        "coverage": sum(covered) / len(covered),
        "token_min": min(lengths), "token_max": max(lengths),
        "max_answer_tokens": MAX_ANSWER_TOKENS,
    }


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, audit_path = Path(args.data_root), Path(args.data_audit)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unexpected V10 FewRel data protocol")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL or not audit.get("passed"):
        raise ValueError("V10 FewRel data audit did not pass")
    if audit.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("V10 FewRel manifest changed after audit")
    episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError("unexpected V10 FewRel pilot episode count")
    selected = [row for index, row in enumerate(episodes)
                if index % args.num_shards == args.shard]
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("V10 FewRel pilot shard output exists")

    torch.manual_seed(202610182)
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

    cache = OutcomeCache(model, tokenizer)
    rows, started = [], time.perf_counter()
    with output.open("x") as sink, torch.inference_mode():
        for episode_index, episode in enumerate(selected):
            for boundary_offset in range(len(episode["boundaries"])):
                row = build_label(
                    episode, boundary_offset, embeddings, cache, True,
                    alphas=ALPHAS,
                )
                row["protocol"] = PROTOCOL
                row["partition"] = "fewrel_train_resource_pilot"
                row["target_relation"] = episode["audit"]["target_relation"]
                row["shard"] = args.shard
                sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            print(json.dumps({"shard": args.shard,
                              "completed": episode_index + 1,
                              "total": len(selected)}), flush=True)
    label_seconds = time.perf_counter() - started
    resource = {
        "protocol": PROTOCOL, "shard": args.shard,
        "num_shards": args.num_shards, "episodes": len(selected),
        "labels": len(rows), "relation_texts": len(texts),
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
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path),
        "model": model_meta, "tokenizer": args.tokenizer,
        "value_scale": VALUE_SCALE, "action_alphas": list(ALPHAS),
        "gpu": torch.cuda.get_device_name(),
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "heldout_benchmarks_loaded": False,
    }
    write_json(resource_output, resource)
    return resource


def single_candidate_benefit(rows: list[dict]) -> dict:
    total = positive = risky = 0
    for row in rows:
        retain = row["actions"][row["retain_action"]]
        size = row["candidate_count"]
        for slot in range(size):
            alphas = [1.0] * size; alphas[slot] = 0.0
            name = next(action["name"] for action in action_grid(size, ALPHAS)
                        if action["kind"] == "grid"
                        and action["alphas"] == alphas)
            candidate = row["actions"][name]
            risk = candidate["unrelated_correct"] < retain["unrelated_correct"]
            gain = candidate["correct"] - retain["correct"]
            total += 1; risky += int(risk); positive += int(not risk and gain > 0)
    return {
        "candidates": total, "positive_safe_benefit": positive,
        "risky": risky,
        "positive_safe_benefit_fraction": positive / total if total else 0.0,
    }


def pilot_gate(*, finite_complete: bool, peak_reserved_gib: float,
               answer_coverage: float, positive_fraction: float,
               oracle_headroom: float, heldouts_sealed: bool) -> dict:
    checks = {
        "finite_complete_auditable_labels": finite_complete,
        "peak_reserved_at_most_28_gib": peak_reserved_gib <= 28.0,
        "answer_token_coverage_at_least_point_95": answer_coverage >= .95,
        "positive_safe_benefit_candidates_at_least_point_05": (
            positive_fraction >= .05
        ),
        "safe_oracle_headroom_at_least_point_015": oracle_headroom >= .015,
        "heldout_sources_sealed": heldouts_sealed,
    }
    return {"checks": checks, "passed": all(checks.values())}


def summarize(args) -> dict:
    root, data_root = Path(args.root), Path(args.data_root)
    expected_episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    expected = {(row["episode_id"], boundary["boundary_index"])
                for row in expected_episodes for boundary in row["boundaries"]}
    rows, resources = [], []
    for shard in range(args.num_shards):
        path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) \
                or resource["labels"] != len(part):
            raise ValueError("V10 FewRel pilot shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    total_queries = sum(row["actions"][row["retain_action"]]["queries"]
                        for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"]
                         for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"]
                         for row in rows)
    oracle_accuracy = oracle_correct / total_queries
    retain_accuracy = retain_correct / total_queries
    headroom = oracle_accuracy - retain_accuracy
    candidates = single_candidate_benefit(rows)
    answer_total = sum(row["answer_coverage"]["answers"] for row in resources)
    answer_covered = sum(row["answer_coverage"]["covered"] for row in resources)
    coverage = answer_covered / answer_total
    peak_reserved = max(row["overall_peak_reserved_gib"] for row in resources)
    label_times = [row["label_seconds"] for row in rows]
    finite = (
        len(rows) == EXPECTED_LABELS and identities == expected
        and all(math.isfinite(row["label_seconds"])
                and row["label_seconds"] > 0
                and row["retain_action"] == retain_action_name(row["candidate_count"])
                and set(row["actions"]) == {
                    action["name"] for action in action_grid(
                        row["candidate_count"], ALPHAS
                    )
                }
                for row in rows)
    )
    sealed = all(
        not resource["development_labels_loaded"]
        and not resource["development_model_scores_loaded"]
        and not resource["heldout_benchmarks_loaded"]
        for resource in resources
    )
    gate = pilot_gate(
        finite_complete=finite, peak_reserved_gib=peak_reserved,
        answer_coverage=coverage,
        positive_fraction=candidates["positive_safe_benefit_fraction"],
        oracle_headroom=headroom, heldouts_sealed=sealed,
    )
    critical_seconds = max(row["total_seconds"] for row in resources)
    unrelated = unrelated_effects(rows)
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
            "coverage": coverage, "max_answer_tokens": MAX_ANSWER_TOKENS,
        },
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
            "two_gpu_critical_seconds": critical_seconds,
            "peak_allocated_gib": max(
                row["overall_peak_allocated_gib"] for row in resources
            ),
            "peak_reserved_gib": peak_reserved,
            "projected_formal_two_gpu_seconds": (
                critical_seconds * FORMAL_GROUPS / EXPECTED_GROUPS
            ),
        },
        "gate": gate,
        "next_stage": (
            "v10_formal_fewrel_label_construction"
            if gate["passed"] else "stop_v10_before_formal_labeling"
        ),
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "heldout_benchmarks_loaded": False,
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
        result = summarize(args)
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
