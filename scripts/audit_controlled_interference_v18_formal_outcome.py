"""Independently reconstruct and audit V18 formal Qwen outcomes."""

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


PROTOCOL = "controlled_memory_interference_v18_qwen_formal"
DATA_PROTOCOL = "controlled_memory_interference_v18_formal"
PILOT_PROTOCOL = "controlled_memory_interference_v18_qwen_pilot"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")
UPDATE = frozenset(KINDS[:2]); RETAIN = frozenset(KINDS[2:4]); JOINT = UPDATE | RETAIN
HEADER = (
    "The following chronological session notes are available memory. Some notes may "
    "conflict or repeat. Answer the query with only the requested code."
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1817


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
    result = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(result) != len(row["arms"]):
        raise ValueError("duplicate formal V18 arm")
    return result


def metrics(arm):
    queries = {query["kind"]: query for query in arm["queries"]}
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
        value = metrics(arm)
        ranked.append((value["joint_success"], value["update_accuracy"],
                       value["retained_key_accuracy"], -value["joint_answer_nll"],
                       -tie_index, arm, value))
    winner = max(ranked, key=lambda value: value[:5])
    return winner[5], winner[6]


def select_fixed(rows):
    table = {}
    for policy in FIXED:
        values = [metrics(arm_index(row)[f"fixed:{policy}"]) for row in rows]
        table[policy] = {key: statistics.fmean(value[key] for value in values)
                         for key in values[0]}
    winner = max(FIXED, key=lambda policy: (
        table[policy]["joint_success"], table[policy]["update_accuracy"],
        table[policy]["retained_key_accuracy"], -table[policy]["joint_answer_nll"],
        -FIXED.index(policy)))
    return winner, table


def interval(records, value_key, cluster_key, seed, draws=BOOTSTRAP_DRAWS):
    grouped = {}
    for row in records:
        grouped.setdefault(row[cluster_key], []).append(row[value_key])
    values = [statistics.fmean(group) for _, group in sorted(grouped.items())]
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choice(values) for _ in values)
                     for _ in range(draws))
    return {"mean": statistics.fmean(values),
            "lower_95": samples[math.floor(.025 * draws)],
            "upper_95": samples[min(draws - 1, math.ceil(.975 * draws) - 1)],
            "clusters": len(values), "draws": draws, "seed": seed}


def recompute_table(rows):
    selected, evidence = {}, {}
    for fold in range(5):
        selected[str(fold)], evidence[str(fold)] = select_fixed(
            [row for row in rows if row["fold"] != fold])
    records, selections = [], []
    counts = {"retain_all": 0, "delete_one": 0, "delete_key": 0}
    for row in rows:
        oracle_arm, oracle = choose_oracle(row)
        fixed_policy = selected[str(row["fold"])]
        fixed = metrics(arm_index(row)[f"fixed:{fixed_policy}"])
        action = "retain_all" if oracle_arm["arm_id"] == "fixed:retain_all" \
            else oracle_arm["kind"]
        counts[action] += 1
        selections.append({"episode_id": row["episode_id"], "subject": row["subject"],
                           "template_group": row["template_group"], "fold": row["fold"],
                           "oracle_arm_id": oracle_arm["arm_id"],
                           "oracle_kind": oracle_arm["kind"],
                           "fixed_policy": fixed_policy, "oracle_metrics": oracle,
                           "fixed_metrics": fixed})
        records.append({
            "subject": row["subject"], "template_group": row["template_group"],
            "fold": row["fold"], "length": row["length"], "mechanism": row["mechanism"],
            "joint_gain": oracle["joint_success"] - fixed["joint_success"],
            "generation_joint_gain": (oracle["generation_joint_success"]
                                      - fixed["generation_joint_success"]),
            "oracle_joint_success": oracle["joint_success"],
            "fixed_joint_success": fixed["joint_success"],
            "oracle_generation_joint_success": oracle["generation_joint_success"],
            "fixed_generation_joint_success": fixed["generation_joint_success"],
            "oracle_update_accuracy": oracle["update_accuracy"],
            "fixed_update_accuracy": fixed["update_accuracy"],
            "oracle_retained_key_accuracy": oracle["retained_key_accuracy"],
            "fixed_retained_key_accuracy": fixed["retained_key_accuracy"],
            "oracle_locality_accuracy": oracle["locality_accuracy"],
            "fixed_locality_accuracy": fixed["locality_accuracy"],
            "fixed_minus_oracle_joint_nll": (fixed["joint_answer_nll"]
                                             - oracle["joint_answer_nll"]),
        })

    def mean(key, subset=records):
        return statistics.fmean(row[key] for row in subset)

    subject_ci = interval(records, "joint_gain", "subject", BOOTSTRAP_SEED)
    template_ci = interval(records, "joint_gain", "template_group", BOOTSTRAP_SEED + 1)
    generation_subject_ci = interval(
        records, "generation_joint_gain", "subject", BOOTSTRAP_SEED + 2)
    generation_template_ci = interval(
        records, "generation_joint_gain", "template_group", BOOTSTRAP_SEED + 3)
    by_fold, by_mechanism, by_length, by_template = {}, {}, {}, {}
    for fold in range(5):
        subset = [row for row in records if row["fold"] == fold]
        by_fold[str(fold)] = {"rows": len(subset), "fixed_policy": selected[str(fold)],
                              "oracle_minus_fixed_joint_success": mean("joint_gain", subset),
                              "oracle_joint_success": mean("oracle_joint_success", subset),
                              "fixed_joint_success": mean("fixed_joint_success", subset)}
    for name in sorted({row["mechanism"] for row in records}):
        subset = [row for row in records if row["mechanism"] == name]
        by_mechanism[name] = {"rows": len(subset),
                              "oracle_minus_fixed_joint_success": mean("joint_gain", subset)}
    for length in sorted({row["length"] for row in records}):
        subset = [row for row in records if row["length"] == length]
        by_length[str(length)] = {"rows": len(subset),
                                  "oracle_minus_fixed_joint_success": mean("joint_gain", subset)}
    for name in sorted({row["template_group"] for row in records}):
        subset = [row for row in records if row["template_group"] == name]
        by_template[name] = {"rows": len(subset), "fold": subset[0]["fold"],
                             "oracle_minus_fixed_joint_success": mean("joint_gain", subset)}
    oracle = {"joint_success": mean("oracle_joint_success"),
              "generation_joint_success": mean("oracle_generation_joint_success"),
              "update_accuracy": mean("oracle_update_accuracy"),
              "retained_key_accuracy": mean("oracle_retained_key_accuracy"),
              "locality_accuracy": mean("oracle_locality_accuracy")}
    fixed = {"joint_success": mean("fixed_joint_success"),
             "generation_joint_success": mean("fixed_generation_joint_success"),
             "update_accuracy": mean("fixed_update_accuracy"),
             "retained_key_accuracy": mean("fixed_retained_key_accuracy"),
             "locality_accuracy": mean("fixed_locality_accuracy")}
    checks = {
        "subject_macro_joint_gain_at_least_5_points": subject_ci["mean"] >= .05,
        "subject_cluster_joint_gain_ci_lower_above_zero": subject_ci["lower_95"] > 0,
        "template_macro_joint_gain_at_least_5_points": template_ci["mean"] >= .05,
        "template_cluster_joint_gain_ci_lower_above_zero": template_ci["lower_95"] > 0,
        "oracle_update_accuracy_at_least_90": oracle["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90": oracle["retained_key_accuracy"] >= .90,
        "oracle_locality_accuracy_at_least_97": oracle["locality_accuracy"] >= .97,
        "no_outer_fold_has_negative_joint_gain":
            all(row["oracle_minus_fixed_joint_success"] >= 0 for row in by_fold.values()),
    }
    supported = all(checks.values())
    return {
        "fixed_selected_by_outer_fold": selected, "fixed_selection_evidence": evidence,
        "oracle_selection": selections, "oracle_action_counts": counts,
        "oracle": oracle, "outer_fold_fixed": fixed,
        "primary_comparison": {
            "oracle_minus_fixed_joint_success_micro": mean("joint_gain"),
            "oracle_minus_fixed_joint_success_subject_macro": subject_ci,
            "oracle_minus_fixed_joint_success_template_macro": template_ci,
            "oracle_minus_fixed_generation_joint_success_micro": mean(
                "generation_joint_gain"),
            "generation_subject_macro": generation_subject_ci,
            "generation_template_macro": generation_template_ci,
            "fixed_minus_oracle_joint_answer_nll": mean("fixed_minus_oracle_joint_nll")},
        "by_fold": by_fold, "by_mechanism": by_mechanism, "by_length": by_length,
        "by_template": by_template,
        "headroom_gate": {"checks": checks, "passed": supported},
        "causal_headroom_supported": supported,
        "next_stage": "train_causal_forgetting_policy" if supported
            else "stop_controller_training_no_causal_headroom",
    }


def recompute_resources(rows, resources):
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    return {"unique_memory_states": sum(row["unique_memory_states_scored"]
                                        for row in resources),
            "query_measurements": len(queries),
            "wall_seconds_slower_shard": max(row["seconds"] for row in resources),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in queries),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in queries)}


def audit(data_root, pilot_root, model, tokenizer, inputs, summary_path, output):
    data_root, pilot_root, model, tokenizer = map(
        Path, (data_root, pilot_root, model, tokenizer))
    summary, manifest = read_json(summary_path), read_json(data_root / "manifest.json")
    source_audit_path = data_root.parent / "audit.json"
    if not source_audit_path.exists():
        source_audit_path = data_root / "audit.json"
    source_audit = read_json(source_audit_path)
    pilot_summary, pilot_audit = (read_json(pilot_root / "summary.json"),
                                  read_json(pilot_root / "audit.json"))
    episodes, states = read_jsonl(data_root / "formal.jsonl"), read_jsonl(
        data_root / "formal_states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value); part = read_jsonl(path); resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = arms_ok = prompts_ok = measurements_ok = generation_ok = True
    for row in rows:
        episode, state = episode_index.get(row.get("episode_id")), state_index.get(
            row.get("episode_id"))
        if episode is None or state is None:
            identities_ok = False; continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "target_key_id",
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
                prompt, candidates = render(memory, spec["statement"]), spec["candidates"]
                prompts_ok &= (
                    query.get("kind") == spec["kind"] and query.get("statement") == spec["statement"]
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
            memory = [tuple(arm["memory_session_ids"]) for arm in expected_arms(state)]
            unique += len(set(memory)); reused += len(memory) - len(set(memory))
        expected_unique[shard], expected_reused[shard] = unique, reused
    dependencies = {"formal_manifest_sha256": sha(data_root / "manifest.json"),
                    "formal_data_audit_sha256": sha(source_audit_path),
                    "formal_sha256": sha(data_root / "formal.jsonl"),
                    "formal_states_sha256": sha(data_root / "formal_states.jsonl"),
                    "pilot_summary_sha256": sha(pilot_root / "summary.json"),
                    "pilot_audit_sha256": sha(pilot_root / "audit.json")}
    source_ok = (
        manifest.get("protocol") == DATA_PROTOCOL
        and manifest.get("formal_sha256") == dependencies["formal_sha256"]
        and manifest.get("formal_states_sha256") == dependencies["formal_states_sha256"]
        and manifest.get("model_scored") is False and manifest.get("formal_generated") is True
        and manifest.get("external_sets_loaded") is False
        and source_audit.get("audit_passed") is True
        and source_audit.get("manifest_sha256") == dependencies["formal_manifest_sha256"]
        and pilot_summary.get("protocol") == PILOT_PROTOCOL
        and pilot_summary.get("corpus_admission_gate", {}).get("passed") is True
        and pilot_audit.get("audit_passed") is True
        and pilot_audit.get("summary_sha256") == dependencies["pilot_summary_sha256"])
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 320
                   and observed == expected_shards)
    shard_ok = (len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
                and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                        and row.get("rows") == 160
                        and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                        and row.get("reused_arm_states") == expected_reused[row["shard"]]
                        and row.get("query_measurements") == 5 * expected_unique[row["shard"]]
                        and row.get("rows_sha256") == hashes[Path(path).name]
                        and row.get("dependencies") == dependencies
                        and row.get("opaque_answer_tokens") == 5
                        and row.get("formal_generated") is True
                        and row.get("formal_scored") is True
                        and row.get("external_sets_loaded") is False
                        for row, path in zip(resources, inputs)))
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
        "frozen_formal_and_admitted_pilot_dependencies": source_ok,
        "complete_disjoint_ordered_two_shard_coverage": coverage_ok,
        "episode_state_and_split_identities": identities_ok,
        "fixed_single_and_key_deletion_arms_with_dedup_sources": arms_ok,
        "prompt_hash_scores_predictions_and_five_token_answers": prompts_ok,
        "free_generation_records": generation_ok,
        "per_query_latency_and_peak_memory_records": measurements_ok,
        "shard_counts_dependencies_resources_and_hashes": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_outer_fold_oracle_and_cluster_recomputation": all(
            summary.get(key) == table[key] for key in table),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("formal_generated") is True
            and summary.get("formal_scored") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") in {
                "formal_headroom_supported", "formal_no_sufficient_headroom"}),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes, "recomputed_headroom_gate": table["headroom_gate"],
              "model_scored": True, "formal_generated": True, "formal_scored": True,
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 formal independent outcome audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
