"""Independently reconstruct and audit the frozen-Qwen V17 headroom pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_pilot"
STATE_PROTOCOL = "mquake_qwen_causal_headroom_v17_states"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")
PRIMARY = frozenset(("primary_old_phrase", "primary_current_phrase"))
HEADER = (
    "The following are chronological session memories. When statements conflict, "
    "use the most recent statement. Answer the query with only the requested value."
)
DRAWS = 20_000
SEED = 1717


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


def expected_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]} for policy in FIXED]
    arms.extend({
        "arm_id": f"delete_one:{candidate['deleted_session_id']}",
        "kind": "delete_one", "deletion_index": index,
        "deleted_session_id": candidate["deleted_session_id"],
        "memory_session_ids": candidate["memory_session_ids"],
    } for index, candidate in enumerate(state["delete_one"]))
    return arms


def qmap(arm):
    return {query["kind"]: query for query in arm["queries"]}


def primary_metrics(arm):
    queries = [query for query in arm["queries"] if query["kind"] in PRIMARY]
    return (statistics.fmean(query["correct"] for query in queries),
            statistics.fmean(query["answer_nll"] for query in queries))


def arm_index(row):
    return {arm["arm_id"]: arm for arm in row["arms"]}


def oracle(row):
    arms = arm_index(row)
    candidates = [arms["fixed:retain_all"]] + [
        arm for arm in row["arms"] if arm["kind"] == "delete_one"]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        accuracy, nll = primary_metrics(arm)
        ranked.append((accuracy, -nll, -tie_index, arm))
    winner = max(ranked, key=lambda value: value[:3])[3]
    return winner, [{"arm_id": value[3]["arm_id"], "primary_accuracy": value[0],
                     "primary_answer_nll": -value[1], "tie_index": -value[2]}
                    for value in ranked]


def fixed_by_fold(rows):
    selected, evidence = {}, {}
    for fold in range(5):
        development = [row for row in rows if row["fold"] != fold]
        table = {}
        for policy in FIXED:
            queries = [query for row in development
                       for query in arm_index(row)[f"fixed:{policy}"]["queries"]
                       if query["kind"] in PRIMARY]
            table[policy] = {
                "primary_accuracy": statistics.fmean(query["correct"] for query in queries),
                "primary_answer_nll": statistics.fmean(query["answer_nll"] for query in queries),
                "queries": len(queries),
            }
        selected[str(fold)] = max(FIXED, key=lambda policy: (
            table[policy]["primary_accuracy"], -table[policy]["primary_answer_nll"],
            -FIXED.index(policy)))
        evidence[str(fold)] = table
    return selected, evidence


def interval(records, field, seed):
    subjects = sorted({row["subject"] for row in records})
    values = {subject: statistics.fmean(row[field] for row in records
                                        if row["subject"] == subject)
              for subject in subjects}
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(values[rng.choice(subjects)] for _ in subjects)
                     for _ in range(DRAWS))
    return {"mean": statistics.fmean(values.values()),
            "lower_95": samples[int(.025 * DRAWS)],
            "upper_95": samples[math.ceil(.975 * DRAWS) - 1],
            "subjects": len(subjects), "draws": DRAWS, "seed": seed}


def recompute(rows, shard_audits, formal_unique_states):
    selected, evidence = fixed_by_fold(rows)
    selections, counts, comparisons, safety = [], {}, [], []
    for row in rows:
        chosen, ranking = oracle(row)
        fixed_policy = selected[str(row["fold"])]
        fixed_arm = arm_index(row)[f"fixed:{fixed_policy}"]
        oq, fq = qmap(chosen), qmap(fixed_arm)
        selections.append({"episode_id": row["episode_id"], "subject": row["subject"],
                           "fold": row["fold"], "oracle_arm_id": chosen["arm_id"],
                           "deleted_session_id": chosen.get("deleted_session_id"),
                           "fixed_policy": fixed_policy, "ranking": ranking})
        counts[chosen["arm_id"]] = counts.get(chosen["arm_id"], 0) + 1
        for kind in PRIMARY:
            comparisons.append({
                "subject": row["subject"], "oracle_correct": oq[kind]["correct"],
                "fixed_correct": fq[kind]["correct"],
                "oracle_answer_nll": oq[kind]["answer_nll"],
                "fixed_answer_nll": fq[kind]["answer_nll"],
                "accuracy_gain": oq[kind]["correct"] - fq[kind]["correct"],
                "nll_gain": fq[kind]["answer_nll"] - oq[kind]["answer_nll"],
            })
        for kind in ("hard_retain", "unrelated"):
            safety.append({"kind": kind, "oracle_correct": oq[kind]["correct"]})
    unique_queries = [query for row in rows for arm in row["arms"]
                      if arm["score_source_arm_id"] == arm["arm_id"]
                      for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in unique_queries)
    unique_states = sum(resource["unique_memory_states_scored"] for resource in shard_audits)
    projected = max(resource["seconds"] for resource in shard_audits) \
        * formal_unique_states / unique_states / 3600
    peak_reserved = max(query["resources"]["gpu_peak_reserved_gib"]
                        for query in unique_queries)
    checks = {
        "all_40_rows_present": len(rows) == 40,
        "all_scores_finite": all(math.isfinite(query["answer_nll"])
                                  for query in unique_queries),
        "all_within_context": max(query["total_tokens_max"]
                                   for query in unique_queries) <= 40960,
        "per_query_resources_complete": all(
            query["resources"]["seconds"] > 0
            and query["resources"]["candidate_scores"] == len(query["candidates"])
            and query["resources"]["gpu_peak_allocated_gib"]
                >= query["resources"]["gpu_start_allocated_gib"]
            and query["resources"]["gpu_peak_reserved_gib"]
                >= query["resources"]["gpu_start_reserved_gib"]
            for query in unique_queries),
        "peak_reserved_memory_at_most_31_5_gib": peak_reserved <= 31.5,
        "projected_formal_two_gpu_time_at_most_48_hours": projected <= 48.0,
    }
    return {
        "oracle_selection": selections, "oracle_arm_counts": counts,
        "fixed_selected_by_outer_fold": selected, "fixed_selection_evidence": evidence,
        "pilot_headroom": {
            "oracle_minus_fixed_primary_accuracy_subject_macro":
                interval(comparisons, "accuracy_gain", SEED),
            "fixed_minus_oracle_primary_answer_nll_subject_macro":
                interval(comparisons, "nll_gain", SEED + 1),
            "oracle_primary_accuracy": statistics.fmean(
                row["oracle_correct"] for row in comparisons),
            "selected_fixed_primary_accuracy": statistics.fmean(
                row["fixed_correct"] for row in comparisons),
            "oracle_primary_answer_nll": statistics.fmean(
                row["oracle_answer_nll"] for row in comparisons),
            "selected_fixed_primary_answer_nll": statistics.fmean(
                row["fixed_answer_nll"] for row in comparisons),
            "oracle_hard_retain_accuracy": statistics.fmean(
                row["oracle_correct"] for row in safety if row["kind"] == "hard_retain"),
            "oracle_unrelated_accuracy": statistics.fmean(
                row["oracle_correct"] for row in safety if row["kind"] == "unrelated"),
            "descriptive_only_not_a_formal_gate": True,
        },
        "resources": {
            "unique_memory_states": unique_states, "query_measurements": len(unique_queries),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in unique_queries),
            "max_gpu_peak_reserved_gib": peak_reserved,
            "formal_unique_memory_states": formal_unique_states,
            "projected_formal_two_gpu_hours": projected,
            "projection_method": "max pilot shard wall time scaled by formal/pilot unique states",
        },
        "pilot_gate": {"checks": checks, "passed": all(checks.values())},
    }


def audit(data_root, sample_root, state_root, model, tokenizer, inputs,
          summary_path, output):
    data_root, sample_root, state_root = map(Path, (data_root, sample_root, state_root))
    summary = read_json(summary_path)
    sample_manifest = read_json(sample_root / "manifest.json")
    state_manifest = read_json(state_root / "manifest.json")
    state_audit_path = state_root.parent / "audit.json"
    if not state_audit_path.exists():
        state_audit_path = state_root / "audit.json"
    state_audit = read_json(state_audit_path)
    samples = read_jsonl(sample_root / "pilot.jsonl")
    states = read_jsonl(state_root / "pilot_states.jsonl")
    sample_index = {row["episode_id"]: row for row in samples}
    state_index = {row["episode_id"]: row for row in states}
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    texts = read_json(data_root / "texts.json")
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value)
        part, resource = read_jsonl(path), read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = arms_ok = prompts_ok = per_query_resources_ok = True
    for row in rows:
        sample = sample_index.get(row.get("episode_id"))
        state = state_index.get(row.get("episode_id"))
        episode = episode_index.get(row.get("episode_id"))
        if sample is None or state is None or episode is None:
            identities_ok = False
            continue
        identities_ok &= all(row[key] == sample[key] == state[key] for key in (
            "subject", "fold", "length", "hard_relation_matched", "query_step"))
        expected = expected_arms(state)
        arms_ok &= len(row["arms"]) == len(expected)
        first_source, first_queries = {}, {}
        visible = episode["sessions"][:sample["query_step"] + 1]
        session_index = {session["session_id"]: session for session in visible}
        for actual, wanted in zip(row["arms"], expected):
            arms_ok &= all(actual.get(key) == value for key, value in wanted.items())
            memory_key = tuple(wanted["memory_session_ids"])
            source = first_source.setdefault(memory_key, wanted["arm_id"])
            arms_ok &= actual.get("score_source_arm_id") == source
            if memory_key in first_queries:
                arms_ok &= actual.get("queries") == first_queries[memory_key]
            else:
                first_queries[memory_key] = actual.get("queries")
            memory = [session_index[identifier] for identifier in wanted["memory_session_ids"]]
            queries = actual.get("queries", [])
            prompts_ok &= len(queries) == 4
            for query, (kind, statement, answer, candidates) in zip(
                    queries, expected_specs(sample)):
                prompt = render(memory, texts, statement)
                prompts_ok &= (
                    query.get("kind") == kind and query.get("statement") == statement
                    and query.get("answer") == answer and query.get("candidates") == candidates
                    and query.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest()
                    and set(query.get("scores", {})) == set(candidates)
                    and set(query.get("answer_token_lengths", {})) == set(candidates)
                    and all(math.isfinite(query["scores"][candidate]) for candidate in candidates)
                    and query.get("prediction") == min(
                        candidates, key=lambda candidate: (query["scores"][candidate], candidate))
                    and query.get("correct") == int(query.get("prediction") == answer)
                    and query.get("answer_nll") == query["scores"][answer]
                    and query.get("total_tokens_max") == query.get("prompt_tokens")
                        + max(query["answer_token_lengths"].values())
                    and query.get("total_tokens_max", 40961) <= 40960)
                measurement = query.get("resources", {})
                per_query_resources_ok &= (
                    all(math.isfinite(measurement.get(key, math.nan)) for key in (
                        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
                    and measurement.get("seconds", 0) > 0
                    and measurement.get("candidate_scores") == len(candidates)
                    and measurement.get("gpu_peak_allocated_gib", -1)
                        >= measurement.get("gpu_start_allocated_gib", 0) >= 0
                    and measurement.get("gpu_peak_reserved_gib", -1)
                        >= measurement.get("gpu_start_reserved_gib", 0) >= 0)

    expected_shards = {
        shard: [sample["episode_id"] for index, sample in enumerate(samples) if index % 2 == shard]
        for shard in (0, 1)}
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 40
                   and observed == expected_shards)
    state_dependency_ok = (
        state_manifest.get("protocol") == STATE_PROTOCOL
        and state_manifest.get("fixed_policies") == list(FIXED)
        and state_manifest.get("state_sha256", {}).get("pilot_states.jsonl")
            == sha(state_root / "pilot_states.jsonl")
        and state_manifest.get("dependencies", {}).get("pilot_sample_sha256")
            == sample_manifest.get("pilot_sha256") == sha(sample_root / "pilot.jsonl")
        and state_manifest.get("model_scored") is False
        and state_manifest.get("external_sets_loaded") is False
        and state_audit.get("audit_passed") is True
        and state_audit.get("manifest_sha256") == sha(state_root / "manifest.json")
        and state_audit.get("state_sha256", {}).get("pilot_states.jsonl")
            == sha(state_root / "pilot_states.jsonl"))
    shard_resources_ok = (
        len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
        and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                and row.get("rows") == 20 and row.get("unique_memory_states_scored") == 680
                and row.get("reused_arm_states") == 20
                and row.get("query_measurements") == 2720
                and row.get("rows_sha256") == hashes[Path(path).name]
                and row.get("dependencies", {}).get("state_manifest_sha256")
                    == sha(state_root / "manifest.json")
                and row.get("generation_scored") is False
                and row.get("external_sets_loaded") is False
                for row, path in zip(resources, inputs)))

    tokenizer_hashes = {path.name: sha(path) for path in sorted(Path(tokenizer).iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    model_hashes = {path.name: sha(path) for path in sorted(Path(model).glob("*.safetensors"))}
    provenance_ok = (bool(model_hashes) and bool(tokenizer_hashes) and all(
        row.get("model") == str(Path(model)) and row.get("tokenizer") == str(Path(tokenizer))
        and row.get("model_provenance", {}).get("path") == str(Path(model))
        and row.get("model_provenance", {}).get("config_sha256")
            == sha(Path(model) / "config.json")
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
        and row.get("readout_mode") == "plain_ttt_disabled_fresh_kv"
        and row.get("candidate_scoring")
            == "one_prompt_forward_with_independent_answer_cache_branches"
        for row in resources))

    formal_unique = state_manifest["counts"]["formal"]["single_deletion_states"] \
        + 4 * state_manifest["counts"]["formal"]["rows"]
    recalculated = recompute(rows, resources, formal_unique)
    checks = {
        "frozen_sample_and_independently_audited_state_dependencies": state_dependency_ok,
        "complete_disjoint_ordered_two_shard_coverage": coverage_ok,
        "episode_sample_and_state_identities": identities_ok,
        "fixed_and_exhaustive_single_deletion_arms": arms_ok,
        "prompt_hash_candidate_scores_and_predictions": prompts_ok,
        "per_query_latency_and_peak_memory_records": per_query_resources_ok,
        "shard_counts_resources_and_hashes": shard_resources_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_oracle_fixed_headroom_and_resource_recomputation": all(
            summary.get(key) == recalculated[key] for key in (
                "oracle_selection", "oracle_arm_counts", "fixed_selected_by_outer_fold",
                "fixed_selection_evidence", "pilot_headroom", "resources", "pilot_gate")),
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 40
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("formal_scored") is False
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") in {
                "pilot_resource_passed", "stopped_by_resource_pilot"}),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "summary_sha256": sha(summary_path),
        "raw_rows_sha256": hashes, "recomputed_pilot_gate": recalculated["pilot_gate"],
        "model_scored": True, "formal_scored": False, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V17 pilot audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
