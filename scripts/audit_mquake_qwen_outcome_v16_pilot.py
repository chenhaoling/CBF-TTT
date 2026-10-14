"""Independently audit the causal frozen-Qwen V16 resource pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_outcome_v16_pilot"
POLICIES = ("no_memory", "oracle")
KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")
HEADER = (
    "The following are chronological session memories. When statements conflict, "
    "use the most recent statement. Answer the query with only the requested value."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render(memory, texts, statement):
    chunks = [HEADER, "", "Session memories:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.append(f"[Session {index + 1}]")
            chunks.extend(texts[identifier] for identifier in session["units"])
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def expected_specs(sample):
    primary = [sample["old"]["value"], sample["current"]["value"]]
    safety = []
    for name in ("old", "current", "hard_retain", "unrelated"):
        value = sample[name]["value"]
        if value not in safety:
            safety.append(value)
    return (
        ("primary_old_phrase", sample["old"]["statement"],
         sample["current"]["value"], primary),
        ("primary_current_phrase", sample["current"]["statement"],
         sample["current"]["value"], primary),
        ("hard_retain", sample["hard_retain"]["statement"],
         sample["hard_retain"]["value"], safety),
        ("unrelated", sample["unrelated"]["statement"],
         sample["unrelated"]["value"], safety),
    )


def expected_memory(episode, sample, policy):
    visible = episode["sessions"][:sample["query_step"] + 1]
    if not visible or visible[-1]["session_id"] != sample["primary_current_session_id"]:
        raise ValueError("noncausal query point")
    if policy == "no_memory":
        return []
    if policy == "oracle":
        return [row for row in visible
                if row["session_id"] != sample["primary_old_session_id"]]
    raise ValueError(policy)


def recompute(rows):
    policies = {}
    for policy in POLICIES:
        queries = [query for row in rows for query in row["policies"][policy]["queries"]]
        latencies = sorted(query["resources"]["seconds"] for query in queries)
        policies[policy] = {
            "queries": len(queries),
            "accuracy": statistics.fmean(query["correct"] for query in queries),
            "answer_nll": statistics.fmean(query["answer_nll"] for query in queries),
            "max_prompt_tokens": max(query["prompt_tokens"] for query in queries),
            "max_total_tokens": max(query["total_tokens_max"] for query in queries),
            "resources": {
                "mean_seconds_per_query": statistics.fmean(latencies),
                "median_seconds_per_query": statistics.median(latencies),
                "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
                "max_seconds_per_query": max(latencies),
                "max_gpu_peak_allocated_gib": max(
                    query["resources"]["gpu_peak_allocated_gib"] for query in queries),
                "max_gpu_peak_reserved_gib": max(
                    query["resources"]["gpu_peak_reserved_gib"] for query in queries),
            },
            "by_kind": {
                kind: {
                    "queries": sum(query["kind"] == kind for query in queries),
                    "accuracy": statistics.fmean(
                        query["correct"] for query in queries if query["kind"] == kind),
                    "answer_nll": statistics.fmean(
                        query["answer_nll"] for query in queries if query["kind"] == kind),
                } for kind in KINDS
            },
        }
    primary = {"primary_old_phrase", "primary_current_phrase"}
    oracle_primary = statistics.fmean(
        query["correct"] for row in rows
        for query in row["policies"]["oracle"]["queries"] if query["kind"] in primary)
    checks = {
        "oracle_primary_at_least_80": oracle_primary >= .80,
        "all_rows_present": len(rows) == 40,
        "all_scores_finite": all(
            math.isfinite(query["answer_nll"]) for row in rows for policy in POLICIES
            for query in row["policies"][policy]["queries"]),
        "all_within_context": max(value["max_total_tokens"]
                                  for value in policies.values()) <= 40960,
        "per_query_resources_complete": all(
            query["resources"]["seconds"] > 0
            and query["resources"]["candidate_scores"] == len(query["candidates"])
            and query["resources"]["gpu_peak_allocated_gib"]
                >= query["resources"]["gpu_start_allocated_gib"]
            and query["resources"]["gpu_peak_reserved_gib"]
                >= query["resources"]["gpu_start_reserved_gib"]
            for row in rows for policy in POLICIES
            for query in row["policies"][policy]["queries"]),
    }
    return {"policies": policies, "oracle_primary_accuracy": oracle_primary,
            "pilot_gate": {"checks": checks, "passed": all(checks.values())}}


def audit(data_root, sample_root, v15_root, model, tokenizer, inputs, summary_path, output):
    data_root, sample_root, v15_root = map(Path, (data_root, sample_root, v15_root))
    summary = read_json(summary_path)
    manifest = read_json(sample_root / "manifest.json")
    samples = read_jsonl(sample_root / "pilot.jsonl")
    sample_index = {row["episode_id"]: row for row in samples}
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    texts = read_json(data_root / "texts.json")
    rows, shard_audits, shard_hashes = [], [], {}
    for value in inputs:
        path = Path(value)
        part, resource = read_jsonl(path), read_json(str(path) + ".audit.json")
        rows.extend(part); shard_audits.append(resource); shard_hashes[path.name] = sha(path)

    identities_ok = True
    prompt_and_scores_ok = True
    resources_ok = True
    for row in rows:
        sample = sample_index.get(row.get("episode_id"))
        episode = episode_index.get(row.get("episode_id"))
        if sample is None or episode is None:
            identities_ok = False
            continue
        identities_ok &= all(row[key] == sample[key] for key in
                             ("subject", "fold", "length", "hard_relation_matched", "query_step"))
        for policy in POLICIES:
            memory = expected_memory(episode, sample, policy)
            arm = row["policies"][policy]
            identities_ok &= arm["memory_session_ids"] \
                == [value["session_id"] for value in memory]
            queries = arm["queries"]
            identities_ok &= len(queries) == len(KINDS)
            for query, (kind, statement, answer, candidates) in zip(
                    queries, expected_specs(sample)):
                prompt = render(memory, texts, statement)
                prompt_and_scores_ok &= (
                    query["kind"] == kind and query["statement"] == statement
                    and query["answer"] == answer and query["candidates"] == candidates
                    and query["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                    and set(query["scores"]) == set(candidates)
                    and set(query["answer_token_lengths"]) == set(candidates)
                    and all(math.isfinite(query["scores"][candidate]) for candidate in candidates)
                    and all(query["answer_token_lengths"][candidate] > 0
                            for candidate in candidates)
                    and query["prediction"] == min(
                        candidates, key=lambda candidate: (query["scores"][candidate], candidate))
                    and query["correct"] == int(query["prediction"] == answer)
                    and query["answer_nll"] == query["scores"][answer]
                    and query["total_tokens_max"] == query["prompt_tokens"]
                        + max(query["answer_token_lengths"].values())
                    and query["total_tokens_max"] <= 40960
                )
                resource = query.get("resources", {})
                resources_ok &= (
                    all(math.isfinite(resource.get(key, math.nan)) for key in (
                        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
                    and resource.get("seconds", 0) > 0
                    and resource.get("candidate_scores") == len(candidates)
                    and resource.get("gpu_peak_allocated_gib", -1)
                        >= resource.get("gpu_start_allocated_gib", 0) >= 0
                    and resource.get("gpu_peak_reserved_gib", -1)
                        >= resource.get("gpu_start_reserved_gib", 0) >= 0
                )

    shard_ids = {resource.get("shard") for resource in shard_audits}
    shard_resources_ok = (
        len(shard_audits) == 2 and shard_ids == {0, 1}
        and all(resource.get("protocol") == PROTOCOL and resource.get("num_shards") == 2
                and resource.get("rows") == 20 and resource.get("query_measurements") == 160
                and resource.get("rows_sha256") == shard_hashes[Path(path).name]
                and resource.get("sample_manifest_sha256") == sha(sample_root / "manifest.json")
                and resource.get("pilot_sha256") == manifest["pilot_sha256"]
                and not resource.get("external_sets_loaded")
                for resource, path in zip(shard_audits, inputs))
    )
    expected_by_shard = {
        shard: {sample["episode_id"] for index, sample in enumerate(samples) if index % 2 == shard}
        for shard in (0, 1)
    }
    observed_by_shard = {
        resource["shard"]: [row["episode_id"] for row in read_jsonl(path)]
        for resource, path in zip(shard_audits, inputs)
    }
    ordered_expected_by_shard = {
        shard: [sample["episode_id"] for index, sample in enumerate(samples) if index % 2 == shard]
        for shard in (0, 1)
    }
    coverage_ok = (
        len(rows) == len({row["episode_id"] for row in rows}) == 40
        and {row["episode_id"] for row in rows} == set(sample_index)
        and {key: set(value) for key, value in observed_by_shard.items()} == expected_by_shard
        and observed_by_shard == ordered_expected_by_shard
    )
    v15_summary, v15_audit = read_json(v15_root / "summary.json"), read_json(v15_root.parent / "audit.json")
    v15_ok = (
        v15_summary.get("protocol") == "mquake_jackknife_consensus_v15"
        and v15_summary.get("jackknife_consensus_supported") is True
        and v15_summary.get("terminal_status") == "completed_mquake_jackknife_consensus_v15"
        and v15_audit.get("audit_passed") is True
        and all(resource["v15_hashes"]["summary_sha256"] == sha(v15_root / "summary.json")
                and resource["v15_hashes"]["audit_sha256"] == sha(v15_root.parent / "audit.json")
                for resource in shard_audits)
    )
    recalculated = recompute(rows)
    tokenizer = Path(tokenizer)
    tokenizer_hashes = {
        path.name: sha(path) for path in sorted(tokenizer.iterdir())
        if path.is_file() and path.suffix in {".json", ".model", ".txt", ".tiktoken"}
    }
    model = Path(model)
    model_hashes = {path.name: sha(path) for path in sorted(model.glob("*.safetensors"))}
    model_provenance_ok = (
        bool(model_hashes) and bool(tokenizer_hashes)
        and all(resource.get("model") == str(model)
                and resource.get("tokenizer") == str(tokenizer)
                and resource.get("model_provenance", {}).get("path") == str(model)
                and resource.get("model_provenance", {}).get("config_sha256")
                    == sha(model / "config.json")
                and resource.get("model_provenance", {}).get("source_files_sha256")
                    == model_hashes
                and resource.get("readout_mode") == "plain_ttt_disabled_fresh_kv"
                and resource.get("candidate_scoring")
                    == "one_prompt_forward_with_independent_answer_cache_branches"
                and resource.get("model_provenance", {}).get("ttt_mode") is False
                and bool(resource.get("model_provenance", {}).get("excluded_adapters"))
                and resource.get("tokenizer_files_sha256") == tokenizer_hashes
                and all(not resource.get("loading_info", {}).get(key)
                        for key in ("missing_keys", "mismatched_keys", "error_msgs"))
                and set(resource.get("loading_info", {}).get("unexpected_keys", ()))
                    == set(resource.get("model_provenance", {}).get("excluded_adapters", ()))
                and all(".ttt_conv." in key or ".ttt_proj." in key
                        for key in resource.get("model_provenance", {}).get(
                            "excluded_adapters", ()))
                for resource in shard_audits)
        and len({json.dumps(resource["model_provenance"], sort_keys=True)
                 for resource in shard_audits}) == 1
    )
    checks = {
        "protocol_and_frozen_sample_hash": summary.get("protocol") == PROTOCOL
            and manifest["pilot_sha256"] == sha(sample_root / "pilot.jsonl"),
        "complete_disjoint_shard_coverage": coverage_ok,
        "episode_query_and_causal_memory_identities": identities_ok,
        "prompt_hash_candidate_scores_and_predictions": prompt_and_scores_ok,
        "per_query_latency_and_peak_memory_records": resources_ok,
        "shard_resource_audits_and_hashes": shard_resources_ok,
        "exact_model_tokenizer_and_loading_provenance": model_provenance_ok,
        "completed_supported_and_audited_v15_dependency": v15_ok,
        "independent_summary_recomputation": all(
            summary[key] == recalculated[key]
            for key in ("policies", "oracle_primary_accuracy", "pilot_gate")),
        "terminal_and_access_locks": summary.get("rows") == 40
            and summary.get("raw_rows_sha256") == shard_hashes
            and summary.get("resources") == shard_audits
            and summary.get("terminal_status") in {"pilot_passed", "stopped_by_pilot"}
            and summary.get("formal_scored") is False
            and summary.get("external_sets_loaded") is False,
    }
    result = {
        "protocol": "mquake_qwen_outcome_v16_pilot_audit",
        "audit_passed": all(checks.values()), "checks": checks,
        "summary_sha256": sha(summary_path), "raw_rows_sha256": shard_hashes,
        "recomputed_pilot_gate": recalculated["pilot_gate"],
        "model_scored": True, "formal_scored": False, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V16 pilot audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--v15-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
