"""Independently reconstruct and audit V20 semantic-alias Qwen outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.audit_controlled_interference_v18_formal_outcome import (
    interval, metrics, read_json, read_jsonl, recompute_resources, render, select_fixed,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v20_semantic_alias_qwen"
DATA_PROTOCOL = "controlled_memory_interference_v20_semantic_alias"
RULE_PROTOCOL = "v19_causal_rule_explanation_v1"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2017


def expected_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]} for policy in FIXED]
    for policy, removed in (("exact_conflict", "exact_removed"),
                            ("semantic_oracle", "semantic_removed")):
        arms.append({"arm_id": f"special:{policy}", "kind": "semantic_rule",
                     "policy": policy, "removed_session_ids": state[removed],
                     "memory_session_ids": state["special"][policy]})
    return arms


def recompute(rows):
    fixed_by_fold = {str(fold): select_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(5)}
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    records, exact_identity = [], True
    for row in rows:
        arms = {arm["arm_id"]: arm for arm in row["arms"]}
        if len(arms) != len(row["arms"]):
            raise ValueError("duplicate V20 outcome arm")
        semantic, exact = (metrics(arms["special:semantic_oracle"]),
                           metrics(arms["special:exact_conflict"]))
        retain = metrics(arms["fixed:retain_all"])
        fixed = metrics(arms[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
        exact_identity &= (arms["special:exact_conflict"]["queries"]
                           == arms["fixed:retain_all"]["queries"])
        records.append({
            "subject": row["subject"], "template_group": row["template_group"],
            "fold": row["fold"],
            "joint_gain": semantic["joint_success"] - exact["joint_success"],
            "generation_joint_gain": (semantic["generation_joint_success"]
                                      - exact["generation_joint_success"]),
            "semantic": semantic, "exact": exact, "retain_all": retain, "fixed": fixed,
        })
    table = {name: {key: statistics.fmean(row[name][key] for row in records)
                    for key in keys}
             for name in ("semantic", "exact", "retain_all", "fixed")}
    subject = interval(records, "joint_gain", "subject", BOOTSTRAP_SEED,
                       BOOTSTRAP_DRAWS)
    template = interval(records, "joint_gain", "template_group", BOOTSTRAP_SEED + 1,
                        BOOTSTRAP_DRAWS)
    generation_subject = interval(records, "generation_joint_gain", "subject",
                                  BOOTSTRAP_SEED + 2, BOOTSTRAP_DRAWS)
    generation_template = interval(records, "generation_joint_gain", "template_group",
                                   BOOTSTRAP_SEED + 3, BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["joint_gain"] for row in records if row["fold"] == fold)
               for fold in range(5)}
    semantic = table["semantic"]
    checks = {
        "semantic_minus_exact_joint_gain_at_least_10_points": subject["mean"] >= .10,
        "subject_cluster_joint_gain_ci_lower_above_zero": subject["lower_95"] > 0,
        "template_cluster_joint_gain_ci_lower_above_zero": template["lower_95"] > 0,
        "semantic_update_accuracy_at_least_90": semantic["update_accuracy"] >= .90,
        "semantic_retained_key_accuracy_at_least_90":
            semantic["retained_key_accuracy"] >= .90,
        "semantic_locality_accuracy_at_least_97": semantic["locality_accuracy"] >= .97,
        "no_outer_fold_has_negative_joint_gain": all(value >= 0 for value in by_fold.values()),
        "exact_surface_rule_identical_to_retain_all": (
            exact_identity and table["exact"] == table["retain_all"]),
    }
    passed = all(checks.values())
    return {
        "fixed_selected_by_outer_fold": fixed_by_fold,
        "arm_metrics": {"semantic_oracle": table["semantic"],
                        "exact_conflict": table["exact"],
                        "retain_all": table["retain_all"],
                        "outer_fold_fixed": table["fixed"]},
        "semantic_oracle_vs_exact_conflict": {
            "joint_gain_micro": table["semantic"]["joint_success"]
                - table["exact"]["joint_success"],
            "joint_gain_subject_macro": subject,
            "joint_gain_template_macro": template,
            "generation_joint_gain_subject_macro": generation_subject,
            "generation_joint_gain_template_macro": generation_template,
            "by_fold_gain": by_fold,
        },
        "semantic_headroom_gate": {"checks": checks, "passed": passed},
        "semantic_headroom_confirmed": passed,
        "next_stage": "train_nested_oof_semantic_forgetting_controller" if passed
            else "stop_synthetic_semantic_forgetting_claim",
    }


def audit(data_root, rule_root, model, tokenizer, inputs, summary_path, output):
    data_root, rule_root, model, tokenizer, summary_path = map(
        Path, (data_root, rule_root, model, tokenizer, summary_path))
    summary, manifest = read_json(summary_path), read_json(data_root / "manifest.json")
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    rule_summary, rule_audit = (read_json(rule_root / "summary.json"),
                                read_json(rule_root / "audit.json"))
    episodes, states = read_jsonl(data_root / "alias.jsonl"), read_jsonl(
        data_root / "states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value); part = read_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = arms_ok = prompts_ok = measurements_ok = generation_ok = True
    for row in rows:
        episode, state = episode_index.get(row.get("episode_id")), state_index.get(
            row.get("episode_id"))
        if episode is None or state is None:
            identities_ok = False
            continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "source_keys",
            "update_key_id", "query_key_id", "current_session_id"))
        expected = expected_arms(state)
        arms_ok &= len(row.get("arms", [])) == len(expected)
        sessions = {item["session_id"]: item for item in episode["sessions"]}
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
            memories = [tuple(row["memory_session_ids"]) for row in expected_arms(state)]
            unique += len(set(memories)); reused += len(memories) - len(set(memories))
        expected_unique[shard], expected_reused[shard] = unique, reused
    dependencies = {
        "manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_audit_path),
        "alias_sha256": sha(data_root / "alias.jsonl"),
        "states_sha256": sha(data_root / "states.jsonl"),
        "rule_summary_sha256": sha(rule_root / "summary.json"),
        "rule_audit_sha256": sha(rule_root / "audit.json"),
    }
    source_ok = (
        manifest.get("protocol") == DATA_PROTOCOL
        and manifest.get("alias_sha256") == dependencies["alias_sha256"]
        and manifest.get("states_sha256") == dependencies["states_sha256"]
        and manifest.get("counts", {}).get("exact_rule_deletions") == 0
        and manifest.get("model_scored") is False
        and manifest.get("controller_or_threshold_selected") is False
        and data_audit.get("audit_passed") is True
        and data_audit.get("manifest_sha256") == dependencies["manifest_sha256"]
        and rule_summary.get("protocol") == RULE_PROTOCOL
        and rule_summary.get("explicit_key_explanation_gate", {}).get("passed") is True
        and rule_audit.get("audit_passed") is True
        and rule_audit.get("summary_sha256") == dependencies["rule_summary_sha256"])
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
                        and row.get("model_scored") is True
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
    table, resource_table = recompute(rows), recompute_resources(rows, resources)
    checks = {
        "frozen_v20_and_passing_v19_rule_dependencies": source_ok,
        "complete_disjoint_ordered_two_shard_coverage": coverage_ok,
        "episode_state_and_split_identities": identities_ok,
        "fixed_exact_and_semantic_arms_with_dedup_sources": arms_ok,
        "prompt_hash_scores_predictions_and_five_token_answers": prompts_ok,
        "free_generation_records": generation_ok,
        "per_query_latency_and_peak_memory_records": measurements_ok,
        "shard_counts_dependencies_resources_and_hashes": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_primary_comparison_intervals_and_gate_recomputation": all(
            summary.get(key) == value for key, value in table.items()),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("controller_or_threshold_selected") is False
            and summary.get("external_sets_loaded") is False
            and summary.get("memory_substrate")
                == "contextual_evidence_not_native_dense_ttt_writer"
            and summary.get("terminal_status") in {
                "v20_semantic_headroom_confirmed",
                "v20_semantic_headroom_not_confirmed"}),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "summary_sha256": sha(summary_path),
        "raw_rows_sha256": hashes,
        "recomputed_semantic_headroom_gate": table["semantic_headroom_gate"],
        "model_scored": True, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V20 semantic-alias independent outcome audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--rule-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
