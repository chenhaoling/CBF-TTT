"""Independently audit model-level V19 nested-OOF policy outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.audit_controlled_interference_v18_formal_outcome import (
    interval, metrics, read_json, read_jsonl, recompute_resources, render,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_v19_nested_forgetting_policy import SEEDS


PROTOCOL = "v19_nested_oof_forgetting_policy_qwen_eval_v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1937


def arm_map(row):
    result = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(result) != len(row["arms"]):
        raise ValueError("duplicate V19 policy evaluation arm")
    return result


def exact_arm(episode, raw):
    arms = arm_map(raw)
    current = episode["sessions"][-1]
    conflict = any(row["key_id"] == current["key_id"]
                   and row["value"] != current["value"]
                   for row in episode["sessions"][:-1])
    return arms[f"delete_key:{current['key_id']}"] if conflict else arms["fixed:retain_all"]


def recompute(episodes, policy_rows, raw_rows, outcome, policy_summary):
    episode_index = {row["episode_id"]: row for row in episodes}
    raw_index = {row["episode_id"]: row for row in raw_rows}
    oracle_ids = {row["episode_id"]: row["arm_id"]
                  for row in outcome["selections"]["0.03"]}
    fixed_by_fold = outcome["fixed_selected_by_outer_fold"]
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    records = []
    for row in policy_rows:
        episode, raw = episode_index[row["episode_id"]], raw_index[row["episode_id"]]
        policy_arms, raw_arms = arm_map(row), arm_map(raw)
        per_seed = {str(seed): metrics(policy_arms[f"policy:{seed}"]) for seed in SEEDS}
        policy = {key: statistics.fmean(value[key] for value in per_seed.values())
                  for key in keys}
        fixed = metrics(raw_arms[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
        exact = metrics(exact_arm(episode, raw))
        oracle = metrics(raw_arms[oracle_ids[row["episode_id"]]])
        records.append({"subject": row["subject"],
                        "template_group": row["template_group"], "fold": row["fold"],
                        "policy": policy, "fixed": fixed, "exact": exact,
                        "oracle": oracle, "per_seed": per_seed,
                        "fixed_gain": policy["joint_success"] - fixed["joint_success"],
                        "exact_gain": policy["joint_success"] - exact["joint_success"],
                        "fixed_generation_gain": policy["generation_joint_success"]
                            - fixed["generation_joint_success"]})
    tables = {name: {key: statistics.fmean(row[name][key] for row in records)
                     for key in keys} for name in ("policy", "fixed", "exact", "oracle")}
    fixed_subject = interval(records, "fixed_gain", "subject", BOOTSTRAP_SEED,
                             BOOTSTRAP_DRAWS)
    fixed_template = interval(records, "fixed_gain", "template_group",
                              BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    exact_subject = interval(records, "exact_gain", "subject", BOOTSTRAP_SEED + 2,
                             BOOTSTRAP_DRAWS)
    exact_template = interval(records, "exact_gain", "template_group",
                              BOOTSTRAP_SEED + 3, BOOTSTRAP_DRAWS)
    generation_subject = interval(records, "fixed_generation_gain", "subject",
                                  BOOTSTRAP_SEED + 4, BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["fixed_gain"] for row in records if row["fold"] == fold) for fold in range(5)}
    by_seed = {str(seed): statistics.fmean(
        row["per_seed"][str(seed)]["joint_success"] - row["fixed"]["joint_success"]
        for row in records) for seed in SEEDS}
    gap = tables["oracle"]["joint_success"] - tables["fixed"]["joint_success"]
    closure = ((tables["policy"]["joint_success"] - tables["fixed"]["joint_success"])
               / gap if gap > 0 else None)
    policy = tables["policy"]
    checks = {
        "subject_macro_joint_gain_over_fixed_at_least_5_points":
            fixed_subject["mean"] >= .05,
        "subject_cluster_fixed_gain_ci_lower_above_zero": fixed_subject["lower_95"] > 0,
        "template_macro_joint_gain_over_fixed_at_least_5_points":
            fixed_template["mean"] >= .05,
        "template_cluster_fixed_gain_ci_lower_above_zero": fixed_template["lower_95"] > 0,
        "no_outer_fold_has_negative_fixed_gain": all(value >= 0 for value in by_fold.values()),
        "no_seed_has_negative_fixed_gain": all(value >= 0 for value in by_seed.values()),
        "policy_update_accuracy_at_least_90": policy["update_accuracy"] >= .90,
        "policy_retained_key_accuracy_at_least_90":
            policy["retained_key_accuracy"] >= .90,
        "policy_locality_accuracy_at_least_97": policy["locality_accuracy"] >= .97,
        "closes_at_least_40_percent_of_positive_oracle_gap":
            closure is not None and closure >= .40,
        "subject_macro_joint_gain_over_exact_at_least_2_points":
            exact_subject["mean"] >= .02,
        "subject_cluster_exact_gain_ci_lower_above_zero": exact_subject["lower_95"] > 0,
        "all_inner_selection_constraints_satisfied": all(
            value["constraints_satisfied"] for value in policy_summary["selection"].values()),
    }
    passed = all(checks.values())
    return {
        "arm_metrics": {"policy_three_seed_mean": tables["policy"],
                        "outer_fold_fixed": tables["fixed"],
                        "whole_key_exact": tables["exact"],
                        "registered_v19_oracle": tables["oracle"]},
        "policy_vs_fixed": {
            "joint_gain_micro": tables["policy"]["joint_success"]
                - tables["fixed"]["joint_success"],
            "joint_gain_subject_macro": fixed_subject,
            "joint_gain_template_macro": fixed_template,
            "generation_joint_gain_subject_macro": generation_subject,
            "by_fold_gain": by_fold, "by_seed_gain": by_seed,
            "registered_oracle_gap": gap, "fraction_oracle_gap_closed": closure},
        "policy_vs_exact": {
            "joint_gain_micro": tables["policy"]["joint_success"]
                - tables["exact"]["joint_success"],
            "joint_gain_subject_macro": exact_subject,
            "joint_gain_template_macro": exact_template},
        "controlled_policy_gate": {"checks": checks, "passed": passed},
        "controlled_policy_supported": passed,
        "next_stage": "build_source_disjoint_semantic_policy_confirmation" if passed
            else "stop_v19_learned_policy_iteration",
    }


def expected_arms(state):
    return [{"arm_id": f"policy:{seed}", "kind": "nested_oof_policy", "seed": seed,
             "memory_session_ids": state["per_seed"][str(seed)]["retained_session_ids"],
             "deleted_session_ids": state["per_seed"][str(seed)]["deleted_session_ids"]}
            for seed in SEEDS]


def audit(data_root, outcome_root, rule_root, examples_root, policy_root, model,
          tokenizer, inputs, summary_path, output):
    roots = map(Path, (data_root, outcome_root, rule_root, examples_root, policy_root,
                       model, tokenizer, summary_path))
    data_root, outcome_root, rule_root, examples_root, policy_root, model, tokenizer, \
        summary_path = roots
    summary = read_json(summary_path)
    episodes = read_jsonl(data_root / "confirmation.jsonl")
    states = read_jsonl(policy_root / "result/oof_policy_states.jsonl")
    state_index = {row["episode_id"]: row for row in states}
    raw_rows = read_jsonl(outcome_root / "shard0.jsonl") + read_jsonl(
        outcome_root / "shard1.jsonl")
    outcome = read_json(outcome_root / "summary.json")
    policy_summary = read_json(policy_root / "result/summary.json")
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value); part = read_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]
    episode_index = {row["episode_id"]: row for row in episodes}
    identities_ok = arms_ok = prompts_ok = generation_ok = measurements_ok = True
    for row in rows:
        episode, state = episode_index.get(row.get("episode_id")), state_index.get(
            row.get("episode_id"))
        if episode is None or state is None:
            identities_ok = False
            continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "target_key_id",
            "current_session_id")) and row.get("selection") == state.get("selection")
        wanted_arms = expected_arms(state)
        arms_ok &= len(row.get("arms", [])) == len(wanted_arms)
        sessions = {item["session_id"]: item for item in episode["sessions"]}
        first_source, first_queries = {}, {}
        for actual, wanted in zip(row.get("arms", []), wanted_arms):
            arms_ok &= all(actual.get(key) == value for key, value in wanted.items())
            memory_key = tuple(wanted["memory_session_ids"])
            source = first_source.setdefault(memory_key, wanted["arm_id"])
            arms_ok &= actual.get("score_source_arm_id") == source
            if memory_key in first_queries:
                arms_ok &= actual.get("queries") == first_queries[memory_key]
            else:
                first_queries[memory_key] = actual.get("queries")
            memory = [sessions[identifier] for identifier in wanted["memory_session_ids"]]
            arms_ok &= episode["current_session_id"] in wanted["memory_session_ids"]
            for query, spec in zip(actual.get("queries", []), episode["queries"]):
                prompt, candidates = render(memory, spec["statement"]), spec["candidates"]
                prompts_ok &= (
                    query.get("kind") == spec["kind"] and query.get("answer") == spec["answer"]
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
                generated = query.get("generation", {})
                normalized = " ".join(generated.get("text", "").strip().casefold().split())
                truth = " ".join(spec["answer"].strip().casefold().split())
                generation_ok &= (
                    generated.get("exact") == int(generated.get("text") == spec["answer"])
                    and generated.get("normalized_exact") == int(normalized == truth)
                    and generated.get("max_new_tokens") == 13
                    and 0 <= generated.get("tokens", -1) <= 13
                    and generated.get("stop_reason") in {
                        "eos", "newline", "max_new_tokens"})
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
            memories = [tuple(row["memory_session_ids"]) for row in expected_arms(state)]
            unique += len(set(memories)); reused += len(memories) - len(set(memories))
        expected_unique[shard], expected_reused[shard] = unique, reused
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    outcome_summary_path, outcome_audit_path = (outcome_root / "summary.json",
                                                outcome_root / "audit.json")
    rule_summary_path, rule_audit_path = rule_root / "summary.json", rule_root / "audit.json"
    example_manifest_path, example_audit_path = (examples_root / "data/manifest.json",
                                                 examples_root / "audit.json")
    policy_summary_path, policy_audit_path = (policy_root / "result/summary.json",
                                              policy_root / "audit.json")
    dependencies = {
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_audit_path),
        "data_sha256": sha(data_root / "confirmation.jsonl"),
        "outcome_summary_sha256": sha(outcome_summary_path),
        "outcome_audit_sha256": sha(outcome_audit_path),
        "outcome_raw_sha256": outcome["raw_rows_sha256"],
        "rule_summary_sha256": sha(rule_summary_path),
        "rule_audit_sha256": sha(rule_audit_path),
        "example_manifest_sha256": sha(example_manifest_path),
        "example_audit_sha256": sha(example_audit_path),
        "policy_summary_sha256": sha(policy_summary_path),
        "policy_audit_sha256": sha(policy_audit_path),
        "policy_checkpoint_sha256": policy_summary["checkpoint_sha256"],
        "policy_states_sha256": policy_summary["oof_policy_states_sha256"],
    }
    source_ok = (
        read_json(data_audit_path).get("audit_passed") is True
        and read_json(outcome_audit_path).get("audit_passed") is True
        and read_json(outcome_audit_path).get("summary_sha256") == sha(outcome_summary_path)
        and read_json(rule_audit_path).get("audit_passed") is True
        and read_json(rule_summary_path).get("next_stage")
            == "train_v19_nested_oof_causal_policy"
        and read_json(example_audit_path).get("audit_passed") is True
        and read_json(policy_audit_path).get("audit_passed") is True
        and read_json(policy_audit_path).get("summary_sha256") == sha(policy_summary_path)
        and summary.get("dependencies") == dependencies)
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 320
                   and observed == expected_shards)
    shard_ok = (len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
                and all(row.get("protocol") == PROTOCOL and row.get("rows") == 160
                        and row.get("num_shards") == 2
                        and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                        and row.get("reused_arm_states") == expected_reused[row["shard"]]
                        and row.get("query_measurements") == 5 * expected_unique[row["shard"]]
                        and row.get("dependencies") == dependencies
                        for row in resources))
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
    table = recompute(episodes, rows, raw_rows, outcome, policy_summary)
    resource_table = recompute_resources(rows, resources)
    checks = {
        "audited_fallback_chain_and_hash_dependencies": source_ok,
        "complete_ordered_two_shard_coverage": coverage_ok,
        "episode_policy_state_and_split_identities": identities_ok,
        "three_seed_arms_current_retention_and_dedup_sources": arms_ok,
        "prompt_hash_scores_predictions_and_five_token_answers": prompts_ok,
        "free_generation_records": generation_ok,
        "latency_and_peak_memory_records": measurements_ok,
        "shard_counts_dependencies_and_resources": shard_ok,
        "exact_model_and_tokenizer_provenance": provenance_ok,
        "independent_comparators_intervals_gap_and_gate_recomputation": all(
            summary.get(key) == value for key, value in table.items()),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("memory_substrate")
                == "contextual_evidence_not_native_dense_ttt_writer"
            and summary.get("terminal_status") in {
                "v19_nested_policy_supported", "v19_nested_policy_not_supported"}),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes,
              "recomputed_controlled_policy_gate": table["controlled_policy_gate"],
              "model_scored": True, "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V19 policy outcome audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "outcome_root", "rule_root", "examples_root",
                 "policy_root", "model", "tokenizer", "summary_path", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
