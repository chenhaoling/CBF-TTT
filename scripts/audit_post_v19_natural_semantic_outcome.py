"""Independently audit post-V19 natural semantic Qwen outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tasks.evaluate_post_v19_natural_semantic_headroom import (
    MAX_ANSWER_TOKENS,
    MAX_NEW_TOKENS,
    PROTOCOL,
    QUERY_KINDS,
    aggregate,
    data_audit_path,
    read_json,
    render_prompt,
    state_arms,
    verify_data,
    verify_formal_release,
)
from tasks.evaluate_controlled_interference_v18_pilot import memory_from_ids
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl, normalize_generation
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def expected_state_counts(episodes, states, shards=2):
    index = {row["episode_id"]: row for row in states}
    unique, reused = {value: 0 for value in range(shards)}, {
        value: 0 for value in range(shards)}
    for position, episode in enumerate(episodes):
        memories = [tuple(row["memory_session_ids"])
                    for row in state_arms(index[episode["episode_id"]])]
        shard = position % shards
        unique[shard] += len(set(memories))
        reused[shard] += len(memories) - len(set(memories))
    return unique, reused


def valid_query(query, spec, prompt, max_position):
    scores, lengths = query.get("scores", {}), query.get("answer_token_lengths", {})
    prediction = min(spec["candidates"], key=lambda value: (scores[value], value)) \
        if set(scores) == set(spec["candidates"]) else None
    resources = query.get("resources", {})
    generation = query.get("generation", {})
    return all((
        query.get("kind") == spec["kind"],
        query.get("statement") == spec["statement"],
        query.get("answer") == spec["answer"],
        query.get("candidates") == spec["candidates"],
        set(lengths) == set(spec["candidates"]),
        all(isinstance(value, int) and 1 <= value <= MAX_ANSWER_TOKENS
            for value in lengths.values()),
        prediction == query.get("prediction"),
        query.get("correct") == int(prediction == spec["answer"]),
        math.isclose(query.get("answer_nll", math.nan),
                     scores.get(spec["answer"], math.inf), abs_tol=1e-12),
        all(math.isfinite(value) for value in scores.values()),
        query.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest(),
        0 < query.get("prompt_tokens", 0) < max_position,
        query.get("total_tokens_max")
            == query.get("prompt_tokens", 0) + max(lengths.values()),
        generation.get("max_new_tokens") == MAX_NEW_TOKENS,
        generation.get("exact") == int(generation.get("text") == spec["answer"]),
        generation.get("normalized_exact") == int(
            normalize_generation(generation.get("text", ""))
            == normalize_generation(spec["answer"])),
        math.isfinite(resources.get("seconds", math.nan))
            and resources.get("seconds", 0) >= 0,
        resources.get("candidate_scores") == len(spec["candidates"]),
        resources.get("gpu_peak_allocated_gib", -1)
            >= resources.get("gpu_start_allocated_gib", 0) >= 0,
        resources.get("gpu_peak_reserved_gib", -1)
            >= resources.get("gpu_start_reserved_gib", 0) >= 0,
    ))


def audit(data_root, split, inputs, summary_path, output, pilot_release_root=None):
    data_root, summary_path, output = Path(data_root), Path(summary_path), Path(output)
    dependencies = verify_data(data_root)
    if split == "formal":
        dependencies.update(verify_formal_release(pilot_release_root))
    episodes = load_jsonl(data_root / f"{split}.jsonl")
    states = load_jsonl(data_root / f"{split}_states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, shard_audits, hashes = [], [], {}
    shard_identity, shard_resources = True, True
    for value in inputs:
        path = Path(value)
        part, metadata = load_jsonl(path), read_json(str(path) + ".audit.json")
        rows.extend(part)
        shard_audits.append(metadata)
        hashes[path.name] = sha(path)
        shard_identity &= all((
            metadata.get("protocol") == PROTOCOL,
            metadata.get("split") == split,
            metadata.get("num_shards") == 2,
            metadata.get("rows") == len(part),
            metadata.get("rows_sha256") == sha(path),
            metadata.get("dependencies") == dependencies,
            metadata.get("model_scored") is True,
            metadata.get("controller_trained") is False,
            metadata.get("sealed_benchmarks_loaded") is False,
            metadata.get("readout_mode")
                == "plain_ttt_disabled_fresh_kv_contextual_evidence_memory",
            metadata.get("model_provenance", {}).get("ttt_mode") is False,
            bool(metadata.get("model_provenance", {}).get("excluded_adapters")),
            metadata.get("answer_token_lengths")
                == sorted(metadata.get("answer_token_lengths", [])),
            min(metadata.get("answer_token_lengths", [0])) >= 1,
            max(metadata.get("answer_token_lengths", [99])) <= MAX_ANSWER_TOKENS,
        ))
        shard_resources &= all((
            math.isfinite(metadata.get("seconds", math.nan)),
            metadata.get("seconds", 0) >= 0,
            metadata.get("gpu_peak_allocated_gib", -1) >= 0,
            metadata.get("gpu_peak_reserved_gib", -1) >= 0,
        ))
    ordered = sorted(rows, key=lambda row: row["row_index"])
    coverage = (
        len(ordered) == len(episodes)
        and len({row["episode_id"] for row in ordered}) == len(episodes)
        and [row["episode_id"] for row in ordered]
            == [row["episode_id"] for row in episodes]
    )
    state_ok, prompt_score_ok, dedup_ok = True, True, True
    query_measurements, latencies = [], []
    for row in ordered:
        episode, state = episode_index[row["episode_id"]], state_index[row["episode_id"]]
        expected = state_arms(state)
        arms = {value["arm_id"]: value for value in row["arms"]}
        state_ok &= (
            [value["arm_id"] for value in row["arms"]]
                == [value["arm_id"] for value in expected]
            and all(arms[value["arm_id"]]["memory_session_ids"]
                    == value["memory_session_ids"] for value in expected)
            and row["source_row_sha256"] == episode["source_row_sha256"]
            and row["relation_id"] == episode["relation_id"]
            and row["fold"] == episode["fold"]
        )
        source_by_memory = {}
        for expected_arm in expected:
            arm = arms[expected_arm["arm_id"]]
            memory_key = tuple(expected_arm["memory_session_ids"])
            source = source_by_memory.setdefault(memory_key, expected_arm["arm_id"])
            dedup_ok &= arm["score_source_arm_id"] == source
            memory = memory_from_ids(episode, expected_arm["memory_session_ids"])
            if len(arm["queries"]) != len(episode["queries"]):
                prompt_score_ok = False
                continue
            for query, spec in zip(arm["queries"], episode["queries"]):
                prompt = render_prompt(memory, spec["statement"])
                prompt_score_ok &= valid_query(query, spec, prompt, 40_960)
                if arm["score_source_arm_id"] == arm["arm_id"]:
                    query_measurements.append(query)
                    latencies.append(query["resources"]["seconds"])
    expected_unique, expected_reused = expected_state_counts(episodes, states)
    resource_counts = True
    for metadata in shard_audits:
        shard = metadata["shard"]
        resource_counts &= all((
            metadata["unique_memory_states_scored"] == expected_unique[shard],
            metadata["reused_arm_states"] == expected_reused[shard],
            metadata["query_measurements"] == expected_unique[shard] * len(QUERY_KINDS),
        ))
    summary = read_json(summary_path)
    recomputed = aggregate(ordered, split)
    resource_summary = {
        "unique_memory_states": sum(value["unique_memory_states_scored"]
                                     for value in shard_audits),
        "query_measurements": len(query_measurements),
        "wall_seconds_slower_shard": max(value["seconds"] for value in shard_audits),
        "mean_seconds_per_query": statistics.fmean(latencies),
        "median_seconds_per_query": statistics.median(latencies),
        "p95_seconds_per_query": sorted(latencies)[math.ceil(.95 * len(latencies)) - 1],
        "max_seconds_per_query": max(latencies),
        "max_gpu_peak_allocated_gib": max(
            row["resources"]["gpu_peak_allocated_gib"] for row in query_measurements),
        "max_gpu_peak_reserved_gib": max(
            row["resources"]["gpu_peak_reserved_gib"] for row in query_measurements),
    }
    summary_ok = all((
        summary.get("protocol") == PROTOCOL,
        summary.get("split") == split,
        summary.get("rows") == len(episodes),
        summary.get("dependencies") == dependencies,
        summary.get("raw_rows_sha256") == hashes,
        summary.get("fixed_selected_by_outer_fold")
            == recomputed["fixed_selected_by_outer_fold"],
        summary.get("arm_metrics") == recomputed["arm_metrics"],
        summary.get("semantic_oracle_vs_exact_surface")
            == recomputed["semantic_oracle_vs_exact_surface"],
        summary.get(f"{split}_gate") == recomputed[f"{split}_gate"],
        summary.get("resources") == resource_summary,
        summary.get("model_scored") is True,
        summary.get("controller_trained") is False,
        summary.get("sealed_benchmarks_loaded") is False,
        summary.get("memory_substrate")
            == "contextual_evidence_not_native_dense_ttt_writer",
    ))
    passed = recomputed[f"{split}_gate"]["passed"]
    expected_terminal = (f"{split}_semantic_headroom_supported" if passed
                         else f"{split}_semantic_headroom_not_supported")
    expected_next = ("score_sealed_natural_confirmation" if split == "pilot" and passed
                     else "train_natural_semantic_residual_controller"
                     if split == "formal" and passed
                     else "stop_post_v19_semantic_controller_branch")
    terminal_ok = (summary.get("terminal_status") == expected_terminal
                   and summary.get("next_stage") == expected_next)
    checks = {
        "audited_data_and_release_dependencies": shard_identity,
        "complete_ordered_two_shard_coverage": coverage,
        "episode_state_and_arm_identities": state_ok,
        "deduplicated_state_sources": dedup_ok,
        "prompt_scores_predictions_generation_and_resources": prompt_score_ok,
        "shard_counts_and_resources": shard_resources and resource_counts,
        "independent_metrics_intervals_and_gate_recomputation": summary_ok,
        "scientific_terminal_and_next_stage": terminal_ok,
    }
    result = {
        "protocol": "audit_post_v19_natural_semantic_outcome_v1",
        "split": split,
        "checks": checks,
        "audit_passed": all(checks.values()),
        "summary_sha256": sha(summary_path),
        "raw_rows_sha256": hashes,
        "recomputed_gate": recomputed[f"{split}_gate"],
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=("pilot", "formal"), required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-release-root")
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
