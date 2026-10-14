"""Independently reconstruct and audit V18 frozen-Qwen pilot outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_qwen_pilot"
DATA_PROTOCOL = "controlled_memory_interference_v18_pilot"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")
UPDATE = frozenset(KINDS[:2]); RETAIN = frozenset(KINDS[2:4]); JOINT = UPDATE | RETAIN
HEADER = (
    "The following chronological session notes are available memory. Some notes may "
    "conflict or repeat. Answer the query with only the requested code."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render(memory, statement):
    chunks = [HEADER, "", "Session notes:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.extend((f"[Session {index + 1}]", session["statement"]))
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def expected_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]} for policy in FIXED]
    arms.extend({"arm_id": f"delete_one:{row['deleted_session_id']}",
                 "kind": "delete_one", "deletion_index": index,
                 "deleted_session_id": row["deleted_session_id"],
                 "memory_session_ids": row["memory_session_ids"]}
                for index, row in enumerate(state["delete_one"]))
    arms.extend({"arm_id": f"delete_key:{row['deleted_key_id']}",
                 "kind": "delete_key", "deletion_index": index,
                 "deleted_key_id": row["deleted_key_id"],
                 "deleted_session_ids": row["deleted_session_ids"],
                 "memory_session_ids": row["memory_session_ids"]}
                for index, row in enumerate(state["delete_key"]))
    return arms


def arm_index(row):
    return {arm["arm_id"]: arm for arm in row["arms"]}


def qmap(arm):
    return {query["kind"]: query for query in arm["queries"]}


def metrics(arm):
    queries = qmap(arm)
    return {
        "joint_success": int(all(queries[kind]["correct"] for kind in JOINT)),
        "update_accuracy": statistics.fmean(queries[kind]["correct"] for kind in UPDATE),
        "retained_key_accuracy": statistics.fmean(
            queries[kind]["correct"] for kind in RETAIN),
        "joint_answer_nll": statistics.fmean(
            queries[kind]["answer_nll"] for kind in JOINT),
        "locality_accuracy": queries["locality"]["correct"],
        "generation_joint_success": int(all(
            queries[kind]["generation"]["normalized_exact"] for kind in JOINT)),
    }


def choose_oracle(row):
    arms = arm_index(row)
    candidates = [arms["fixed:retain_all"]] + [
        arm for arm in row["arms"] if arm["kind"] in {"delete_one", "delete_key"}]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        metric = metrics(arm)
        ranked.append((metric["joint_success"], metric["update_accuracy"],
                       metric["retained_key_accuracy"], -metric["joint_answer_nll"],
                       -tie_index, arm, metric))
    winner = max(ranked, key=lambda value: value[:5])
    return winner[5], winner[6]


def recompute_table(rows):
    fixed_table = {}
    for policy in FIXED:
        values = [metrics(arm_index(row)[f"fixed:{policy}"]) for row in rows]
        fixed_table[policy] = {key: statistics.fmean(value[key] for value in values)
                               for key in values[0]}
    fixed_policy = max(FIXED, key=lambda policy: (
        fixed_table[policy]["joint_success"], fixed_table[policy]["update_accuracy"],
        fixed_table[policy]["retained_key_accuracy"],
        -fixed_table[policy]["joint_answer_nll"], -FIXED.index(policy)))
    oracle_values, fixed_values, selections = [], [], []
    for row in rows:
        oracle, value = choose_oracle(row)
        fixed_value = metrics(arm_index(row)[f"fixed:{fixed_policy}"])
        oracle_values.append(value); fixed_values.append(fixed_value)
        selections.append({"episode_id": row["episode_id"], "oracle_arm_id": oracle["arm_id"],
                           "oracle_kind": oracle["kind"], "metrics": value})
    oracle = {key: statistics.fmean(value[key] for value in oracle_values)
              for key in oracle_values[0]}
    fixed = {key: statistics.fmean(value[key] for value in fixed_values)
             for key in fixed_values[0]}
    by_mechanism = {}
    for name in sorted({row["mechanism"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["mechanism"] == name]
        by_mechanism[name] = {
            "rows": len(indices), "oracle_joint_success": statistics.fmean(
                oracle_values[index]["joint_success"] for index in indices),
            "fixed_joint_success": statistics.fmean(
                fixed_values[index]["joint_success"] for index in indices)}
    by_length = {}
    for length in sorted({row["length"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["length"] == length]
        by_length[str(length)] = {
            "rows": len(indices), "oracle_joint_success": statistics.fmean(
                oracle_values[index]["joint_success"] for index in indices),
            "fixed_joint_success": statistics.fmean(
                fixed_values[index]["joint_success"] for index in indices)}
    checks = {
        "no_memory_joint_success_at_most_70": fixed_table["no_memory"]["joint_success"] <= .70,
        "retain_all_joint_success_between_20_and_90":
            .20 <= fixed_table["retain_all"]["joint_success"] <= .90,
        "global_reset_retained_key_accuracy_below_80":
            fixed_table["global_reset"]["retained_key_accuracy"] < .80,
        "oracle_update_accuracy_at_least_90": oracle["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90": oracle["retained_key_accuracy"] >= .90,
        "oracle_joint_gain_over_best_fixed_at_least_10":
            oracle["joint_success"] - fixed["joint_success"] >= .10,
    }
    return {"fixed_arms": fixed_table, "best_fixed_policy": fixed_policy,
            "oracle": oracle, "best_fixed": fixed,
            "oracle_minus_fixed_joint_success": oracle["joint_success"] - fixed["joint_success"],
            "oracle_selection": selections, "by_mechanism": by_mechanism,
            "by_length": by_length,
            "corpus_admission_gate": {"checks": checks, "passed": all(checks.values())}}


def recompute_resources(rows, resources):
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    return {"unique_memory_states": sum(a["unique_memory_states_scored"] for a in resources),
            "query_measurements": len(queries),
            "wall_seconds_slower_shard": max(a["seconds"] for a in resources),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in queries),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in queries)}


def audit(data_root, model, tokenizer, inputs, summary_path, output):
    data_root, model, tokenizer = map(Path, (data_root, model, tokenizer))
    summary, manifest = read_json(summary_path), read_json(data_root / "manifest.json")
    source_audit_path = data_root.parent / "audit.json"
    if not source_audit_path.exists():
        source_audit_path = data_root / "audit.json"
    source_audit = read_json(source_audit_path)
    episodes, states = read_jsonl(data_root / "pilot.jsonl"), read_jsonl(
        data_root / "pilot_states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value); part = read_jsonl(path); resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = arms_ok = prompts_ok = measurements_ok = generation_ok = True
    for row in rows:
        episode, state = episode_index.get(row.get("episode_id")), state_index.get(row.get("episode_id"))
        if episode is None or state is None:
            identities_ok = False; continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "length", "mechanism", "target_key_id",
            "current_session_id"))
        expected = expected_arms(state)
        arms_ok &= len(row.get("arms", [])) == len(expected)
        sessions = {session["session_id"]: session for session in episode["sessions"]}
        first_source, first_queries = {}, {}
        for actual, wanted in zip(row.get("arms", []), expected):
            arms_ok &= all(actual.get(key) == value for key, value in wanted.items())
            memory_key = tuple(wanted["memory_session_ids"])
            source = first_source.setdefault(memory_key, wanted["arm_id"])
            arms_ok &= actual.get("score_source_arm_id") == source
            if memory_key in first_queries:
                arms_ok &= actual.get("queries") == first_queries[memory_key]
            else:
                first_queries[memory_key] = actual.get("queries")
            memory = [sessions[identifier] for identifier in wanted["memory_session_ids"]]
            prompts_ok &= len(actual.get("queries", [])) == 5
            for query, spec in zip(actual.get("queries", []), episode["queries"]):
                prompt = render(memory, spec["statement"])
                candidates = spec["candidates"]
                prompts_ok &= (
                    query.get("kind") == spec["kind"]
                    and query.get("statement") == spec["statement"]
                    and query.get("answer") == spec["answer"]
                    and query.get("candidates") == candidates
                    and query.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest()
                    and set(query.get("scores", {})) == set(candidates)
                    and query.get("answer_token_lengths") == {value: 5 for value in candidates}
                    and all(math.isfinite(query["scores"][value]) for value in candidates)
                    and query.get("prediction") == min(
                        candidates, key=lambda value: (query["scores"][value], value))
                    and query.get("correct") == int(query.get("prediction") == spec["answer"])
                    and query.get("answer_nll") == query["scores"][spec["answer"]]
                    and query.get("total_tokens_max") == query.get("prompt_tokens") + 5
                    and query.get("total_tokens_max", 40961) <= 40960)
                generation = query.get("generation", {})
                normalized = " ".join(generation.get("text", "").strip().casefold().split())
                truth = " ".join(spec["answer"].strip().casefold().split())
                generation_ok &= (
                    generation.get("exact") == int(generation.get("text") == spec["answer"])
                    and generation.get("normalized_exact") == int(normalized == truth)
                    and generation.get("max_new_tokens") == 13
                    and 0 <= generation.get("tokens", -1) <= 13
                    and generation.get("stop_reason") in {"eos", "newline", "max_new_tokens"})
                measurement = query.get("resources", {})
                measurements_ok &= (
                    all(math.isfinite(measurement.get(key, math.nan)) for key in (
                        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
                    and measurement.get("seconds", 0) > 0
                    and measurement.get("candidate_scores") == len(candidates)
                    and measurement.get("gpu_peak_allocated_gib", -1)
                        >= measurement.get("gpu_start_allocated_gib", 0) >= 0
                    and measurement.get("gpu_peak_reserved_gib", -1)
                        >= measurement.get("gpu_start_reserved_gib", 0) >= 0)

    expected_shards = {shard: [row["episode_id"] for index, row in enumerate(episodes)
                                if index % 2 == shard] for shard in (0, 1)}
    expected_unique, expected_reused = {}, {}
    for shard in (0, 1):
        unique = reused = 0
        for index, state in enumerate(states):
            if index % 2 != shard:
                continue
            arms = expected_arms(state); memory = [tuple(arm["memory_session_ids"]) for arm in arms]
            unique += len(set(memory)); reused += len(memory) - len(set(memory))
        expected_unique[shard], expected_reused[shard] = unique, reused
    dependencies = {"manifest_sha256": sha(data_root / "manifest.json"),
                    "data_audit_sha256": sha(source_audit_path),
                    "pilot_sha256": sha(data_root / "pilot.jsonl"),
                    "pilot_states_sha256": sha(data_root / "pilot_states.jsonl")}
    source_ok = (manifest.get("protocol") == DATA_PROTOCOL
                 and manifest.get("pilot_sha256") == dependencies["pilot_sha256"]
                 and manifest.get("pilot_states_sha256") == dependencies["pilot_states_sha256"]
                 and manifest.get("model_scored") is False
                 and manifest.get("formal_generated") is False
                 and manifest.get("external_sets_loaded") is False
                 and source_audit.get("audit_passed") is True
                 and source_audit.get("manifest_sha256") == dependencies["manifest_sha256"])
    coverage_ok = len(rows) == len({row["episode_id"] for row in rows}) == 40 \
        and observed == expected_shards
    shard_ok = len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1} \
        and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                and row.get("rows") == 20
                and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                and row.get("reused_arm_states") == expected_reused[row["shard"]]
                and row.get("query_measurements") == 5 * expected_unique[row["shard"]]
                and row.get("rows_sha256") == hashes[Path(path).name]
                and row.get("dependencies") == dependencies
                and row.get("opaque_answer_tokens") == 5
                and row.get("generation") == "greedy_from_independent_shared_prompt_cache_13_tokens"
                and row.get("formal_generated") is False
                and row.get("external_sets_loaded") is False
                for row, path in zip(resources, inputs))
    tokenizer_hashes = {path.name: sha(path) for path in sorted(tokenizer.iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    model_hashes = {path.name: sha(path) for path in sorted(model.glob("*.safetensors"))}
    provenance_ok = bool(model_hashes) and bool(tokenizer_hashes) and all(
        row.get("model") == str(model) and row.get("tokenizer") == str(tokenizer)
        and row.get("model_provenance", {}).get("path") == str(model)
        and row.get("model_provenance", {}).get("config_sha256") == sha(model / "config.json")
        and row.get("model_provenance", {}).get("source_files_sha256") == model_hashes
        and row.get("model_provenance", {}).get("ttt_mode") is False
        and bool(row.get("model_provenance", {}).get("excluded_adapters"))
        and all(".ttt_conv." in key or ".ttt_proj." in key
                for key in row["model_provenance"]["excluded_adapters"])
        and row.get("tokenizer_files_sha256") == tokenizer_hashes
        and all(not row.get("loading_info", {}).get(key)
                for key in ("missing_keys", "mismatched_keys", "error_msgs"))
        and set(row.get("loading_info", {}).get("unexpected_keys", ()))
            == set(row["model_provenance"]["excluded_adapters"])
        for row in resources)
    table, resource_table = recompute_table(rows), recompute_resources(rows, resources)
    checks = {
        "frozen_audited_pilot_dependency_and_access_locks": source_ok,
        "complete_disjoint_ordered_shard_coverage": coverage_ok,
        "episode_and_state_identities": identities_ok,
        "fixed_single_and_key_deletion_arms_with_dedup_sources": arms_ok,
        "prompt_hash_scores_predictions_and_five_token_answers": prompts_ok,
        "free_generation_records": generation_ok,
        "per_query_latency_and_peak_memory_records": measurements_ok,
        "shard_counts_dependencies_and_hashes": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_corpus_admission_recomputation": all(
            summary.get(key) == table[key] for key in table),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 40
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("formal_generated") is False
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") in {
                "pilot_corpus_admitted", "pilot_corpus_rejected"}),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes,
              "recomputed_corpus_admission_gate": table["corpus_admission_gate"],
              "model_scored": True, "formal_generated": False,
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 Qwen pilot audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
