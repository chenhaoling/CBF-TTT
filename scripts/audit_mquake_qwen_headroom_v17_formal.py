"""Independently reconstruct and audit the formal V17 causal-headroom run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from scripts.audit_mquake_qwen_headroom_v17_pilot import (
    FIXED, KINDS, PRIMARY, expected_arms, expected_specs,
    expected_unique_states_by_shard, read_json, read_jsonl, render,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_formal"
PILOT_PROTOCOL = "mquake_qwen_causal_headroom_v17_pilot"
STATE_PROTOCOL = "mquake_qwen_causal_headroom_v17_states"
DRAWS = 20_000
SEED = 1727


def arm_index(row):
    return {arm["arm_id"]: arm for arm in row["arms"]}


def qmap(arm):
    return {query["kind"]: query for query in arm["queries"]}


def primary_metrics(arm):
    queries = [query for query in arm["queries"] if query["kind"] in PRIMARY]
    return (statistics.fmean(query["correct"] for query in queries),
            statistics.fmean(query["answer_nll"] for query in queries))


def choose_oracle(row):
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


def choose_fixed(rows):
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


def subject_macro(rows, field):
    subjects = sorted({row["subject"] for row in rows})
    return statistics.fmean(statistics.fmean(row[field] for row in rows
                                             if row["subject"] == subject)
                            for subject in subjects)


def recompute_table(rows):
    selected, evidence = choose_fixed(rows)
    selections, comparisons, query_records, pairs = [], [], [], []
    counts = {"retain_all": 0, "delete_one": 0, "delete_primary_old": 0,
              "delete_other": 0}
    for row in rows:
        chosen, ranking = choose_oracle(row)
        fixed_policy = selected[str(row["fold"])]
        fixed = arm_index(row)[f"fixed:{fixed_policy}"]
        oq, fq = qmap(chosen), qmap(fixed)
        action = "retain_all" if chosen["kind"] == "fixed" else "delete_one"
        counts[action] += 1
        if action == "delete_one":
            counts["delete_primary_old" if chosen["deleted_session_id"]
                   == row["primary_old_session_id"] else "delete_other"] += 1
        selections.append({"episode_id": row["episode_id"], "subject": row["subject"],
                           "fold": row["fold"], "oracle_arm_id": chosen["arm_id"],
                           "deleted_session_id": chosen.get("deleted_session_id"),
                           "fixed_policy": fixed_policy, "ranking": ranking})
        op, fp = [], []
        for kind in KINDS:
            record = {"episode_id": row["episode_id"], "subject": row["subject"],
                      "fold": row["fold"], "kind": kind, "fixed_policy": fixed_policy,
                      "oracle_correct": oq[kind]["correct"],
                      "fixed_correct": fq[kind]["correct"],
                      "oracle_answer_nll": oq[kind]["answer_nll"],
                      "fixed_answer_nll": fq[kind]["answer_nll"]}
            query_records.append(record)
            if kind in PRIMARY:
                record["accuracy_gain"] = record["oracle_correct"] - record["fixed_correct"]
                record["nll_gain"] = record["fixed_answer_nll"] - record["oracle_answer_nll"]
                comparisons.append(record); op.append(record["oracle_correct"])
                fp.append(record["fixed_correct"])
        pairs.append({"oracle_pair": int(all(op)), "fixed_pair": int(all(fp))})
    accuracy = interval(comparisons, "accuracy_gain", SEED)
    nll = interval(comparisons, "nll_gain", SEED + 1)
    by_kind = {}
    for kind in KINDS:
        subset = [row for row in query_records if row["kind"] == kind]
        by_kind[kind] = {
            "queries": len(subset),
            "oracle_accuracy": statistics.fmean(row["oracle_correct"] for row in subset),
            "fixed_accuracy": statistics.fmean(row["fixed_correct"] for row in subset),
            "oracle_subject_macro_accuracy": subject_macro(subset, "oracle_correct"),
            "fixed_subject_macro_accuracy": subject_macro(subset, "fixed_correct"),
            "oracle_answer_nll": statistics.fmean(row["oracle_answer_nll"] for row in subset),
            "fixed_answer_nll": statistics.fmean(row["fixed_answer_nll"] for row in subset),
        }
    by_fold = {}
    for fold in range(5):
        subset = [row for row in comparisons if row["fold"] == fold]
        by_fold[str(fold)] = {
            "rows": sum(row["fold"] == fold for row in rows),
            "fixed_policy": selected[str(fold)],
            "oracle_minus_fixed_primary_accuracy":
                statistics.fmean(row["accuracy_gain"] for row in subset),
            "fixed_minus_oracle_primary_answer_nll":
                statistics.fmean(row["nll_gain"] for row in subset),
        }
    checks = {
        "subject_macro_primary_accuracy_gain_at_least_5_points": accuracy["mean"] >= .05,
        "subject_cluster_accuracy_gain_ci_lower_above_zero": accuracy["lower_95"] > 0,
    }
    supported = all(checks.values())
    return {
        "oracle_selection": selections, "oracle_action_counts": counts,
        "fixed_selected_by_outer_fold": selected, "fixed_selection_evidence": evidence,
        "primary_comparison": {
            "oracle_minus_fixed_accuracy_subject_macro": accuracy,
            "fixed_minus_oracle_answer_nll_subject_macro": nll,
            "oracle_accuracy": statistics.fmean(row["oracle_correct"] for row in comparisons),
            "fixed_accuracy": statistics.fmean(row["fixed_correct"] for row in comparisons),
            "oracle_pair_accuracy": statistics.fmean(row["oracle_pair"] for row in pairs),
            "fixed_pair_accuracy": statistics.fmean(row["fixed_pair"] for row in pairs),
        },
        "by_kind": by_kind, "by_fold": by_fold,
        "headroom_gate": {"checks": checks, "passed": supported},
        "causal_headroom_supported": supported,
        "next_stage": "train_readout_utility_policy" if supported
            else "stop_controller_training_no_causal_headroom",
    }


def recompute_resources(rows, shard_resources):
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"]
               for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    return {
        "unique_memory_states": sum(row["unique_memory_states_scored"]
                                     for row in shard_resources),
        "query_measurements": len(queries),
        "wall_seconds_slower_shard": max(row["seconds"] for row in shard_resources),
        "mean_seconds_per_query": statistics.fmean(latencies),
        "median_seconds_per_query": statistics.median(latencies),
        "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
        "max_seconds_per_query": max(latencies),
        "max_gpu_peak_allocated_gib": max(
            query["resources"]["gpu_peak_allocated_gib"] for query in queries),
        "max_gpu_peak_reserved_gib": max(
            query["resources"]["gpu_peak_reserved_gib"] for query in queries),
    }


def audit(data_root, sample_root, state_root, pilot_root, model, tokenizer,
          inputs, summary_path, output):
    data_root, sample_root, state_root, pilot_root = map(
        Path, (data_root, sample_root, state_root, pilot_root))
    summary = read_json(summary_path)
    sample_manifest = read_json(sample_root / "manifest.json")
    state_manifest = read_json(state_root / "manifest.json")
    state_audit_path = state_root.parent / "audit.json"
    if not state_audit_path.exists():
        state_audit_path = state_root / "audit.json"
    state_audit = read_json(state_audit_path)
    pilot_summary, pilot_audit = read_json(pilot_root / "summary.json"), read_json(pilot_root / "audit.json")
    samples = read_jsonl(sample_root / "formal.jsonl")
    states = read_jsonl(state_root / "formal_states.jsonl")
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

    identities_ok = arms_ok = prompts_ok = measurements_ok = True
    for row in rows:
        sample, state, episode = (sample_index.get(row.get("episode_id")),
                                  state_index.get(row.get("episode_id")),
                                  episode_index.get(row.get("episode_id")))
        if sample is None or state is None or episode is None:
            identities_ok = False; continue
        identities_ok &= all(row[key] == sample[key] for key in (
            "subject", "fold", "length", "hard_relation_matched", "query_step",
            "primary_old_session_id", "primary_current_session_id"))
        identities_ok &= all(row[key] == state[key] for key in (
            "subject", "fold", "length", "hard_relation_matched", "query_step"))
        expected = expected_arms(state)
        arms_ok &= len(row["arms"]) == len(expected)
        visible = episode["sessions"][:sample["query_step"] + 1]
        session_index = {session["session_id"]: session for session in visible}
        first_source, first_queries = {}, {}
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

    expected_shards = {shard: [sample["episode_id"] for index, sample in enumerate(samples)
                                if index % 2 == shard] for shard in (0, 1)}
    expected_unique = expected_unique_states_by_shard(states)
    expected_dependencies = {
        "state_manifest_sha256": sha(state_root / "manifest.json"),
        "state_audit_sha256": sha(state_audit_path),
        "pilot_states_sha256": sha(state_root / "pilot_states.jsonl"),
        "pilot_sample_sha256": sha(sample_root / "pilot.jsonl"),
        "pilot_summary_sha256": sha(pilot_root / "summary.json"),
        "pilot_audit_sha256": sha(pilot_root / "audit.json"),
        "formal_states_sha256": sha(state_root / "formal_states.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
    }
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 272
                   and observed == expected_shards)
    state_and_pilot_ok = (
        state_manifest.get("protocol") == STATE_PROTOCOL
        and state_manifest.get("state_sha256", {}).get("formal_states.jsonl")
            == sha(state_root / "formal_states.jsonl")
        and state_manifest.get("dependencies", {}).get("formal_sample_sha256")
            == sample_manifest.get("formal_sha256") == sha(sample_root / "formal.jsonl")
        and state_manifest.get("model_scored") is False
        and state_manifest.get("external_sets_loaded") is False
        and state_audit.get("audit_passed") is True
        and state_audit.get("manifest_sha256") == sha(state_root / "manifest.json")
        and state_audit.get("state_sha256", {}).get("formal_states.jsonl")
            == sha(state_root / "formal_states.jsonl")
        and state_audit.get("model_scored") is False
        and state_audit.get("external_sets_loaded") is False
        and pilot_summary.get("protocol") == PILOT_PROTOCOL
        and pilot_summary.get("pilot_gate", {}).get("passed") is True
        and pilot_summary.get("terminal_status") == "pilot_resource_passed"
        and pilot_summary.get("formal_scored") is False
        and pilot_summary.get("external_sets_loaded") is False
        and pilot_audit.get("audit_passed") is True
        and pilot_audit.get("summary_sha256") == sha(pilot_root / "summary.json")
        and pilot_audit.get("formal_scored") is False
        and pilot_audit.get("external_sets_loaded") is False
        and (pilot_root / "runner_exit_code.txt").read_text().strip() == "0"
        and (pilot_root / "terminal_status.txt").read_text().strip()
            == "completed_mquake_qwen_headroom_v17_pilot")
    shard_ok = (len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
                and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                        and row.get("rows") == 136
                        and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                        and row.get("reused_arm_states") == 136
                        and row.get("query_measurements") == 4 * expected_unique[row["shard"]]
                        and row.get("rows_sha256") == hashes[Path(path).name]
                        and row.get("dependencies") == expected_dependencies
                        and row.get("generation_scored") is False
                        and row.get("external_sets_loaded") is False
                        for row, path in zip(resources, inputs)))

    tokenizer_hashes = {path.name: sha(path) for path in sorted(Path(tokenizer).iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    model_hashes = {path.name: sha(path) for path in sorted(Path(model).glob("*.safetensors"))}
    provenance_ok = bool(model_hashes) and bool(tokenizer_hashes) and all(
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
        for row in resources)

    table, resource_table = recompute_table(rows), recompute_resources(rows, resources)
    checks = {
        "frozen_sample_state_and_passed_pilot_dependencies": state_and_pilot_ok,
        "complete_disjoint_ordered_two_shard_coverage": coverage_ok,
        "episode_sample_and_state_identities": identities_ok,
        "fixed_and_exhaustive_single_deletion_arms": arms_ok,
        "prompt_hash_candidate_scores_and_predictions": prompts_ok,
        "per_query_latency_and_peak_memory_records": measurements_ok,
        "shard_counts_resources_dependencies_and_hashes": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_oracle_fixed_headroom_recomputation": all(
            summary.get(key) == table[key] for key in table),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 272
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True and summary.get("formal_scored") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") in {
                "formal_headroom_supported", "formal_no_sufficient_headroom"}),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "summary_sha256": sha(summary_path),
        "raw_rows_sha256": hashes, "recomputed_headroom_gate": table["headroom_gate"],
        "model_scored": True, "formal_scored": True, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("formal V17 independent audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
