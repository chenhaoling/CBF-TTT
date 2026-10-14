"""Independently reconstruct and audit V19 confirmation outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.audit_controlled_interference_v18_formal_outcome import (
    expected_arms, interval, metrics, read_json, read_jsonl, recompute_resources, render,
    select_fixed,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v19_qwen_confirmation"
DATA_PROTOCOL = "controlled_memory_interference_v19_confirmation"
RELEASE_PROTOCOL = "v18_locality_constrained_oracle_posthoc_v1"
BUDGETS = (0.0, 0.01, 0.03, 0.05, 0.10)
PRIMARY_BUDGET = .03
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1917


def rank(total, failures):
    return (total["joint_success"], total["update_accuracy"],
            total["retained_key_accuracy"], total["locality_accuracy"],
            total["generation_joint_success"], -total["joint_answer_nll"], -failures)


def solve(rows, maximum_failures):
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    dp = {0: ({key: 0 for key in keys}, [])}
    for row in rows:
        next_dp = {}
        for failures, (total, chosen) in dp.items():
            for arm in row["arms"]:
                value = metrics(arm)
                new_failures = failures + 1 - value["locality_accuracy"]
                if new_failures > maximum_failures:
                    continue
                updated = {key: total[key] + value[key] for key in keys}
                candidate = (updated, chosen + [(arm["arm_id"], value)])
                previous = next_dp.get(new_failures)
                if previous is None or rank(updated, new_failures) > rank(
                        previous[0], new_failures):
                    next_dp[new_failures] = candidate
        if not next_dp:
            raise RuntimeError("independent V19 DP has no feasible state")
        dp = next_dp
    failures, (total, chosen) = max(
        dp.items(), key=lambda item: rank(item[1][0], item[0]))
    return failures, total, chosen


def unconstrained(rows):
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    total = {key: 0 for key in keys}
    failures, chosen = 0, []
    for row in rows:
        ranked = []
        for index, arm in enumerate(row["arms"]):
            value = metrics(arm)
            ranked.append(((value["joint_success"], value["update_accuracy"],
                            value["retained_key_accuracy"], value["locality_accuracy"],
                            value["generation_joint_success"], -value["joint_answer_nll"],
                            -index), arm["arm_id"], value))
        _, arm_id, value = max(ranked, key=lambda item: item[0])
        failures += 1 - value["locality_accuracy"]
        total = {key: total[key] + value[key] for key in keys}
        chosen.append((arm_id, value))
    return failures, total, chosen


def recompute_table(rows):
    fixed_by_fold = {str(fold): select_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(5)}
    fixed = [metrics({arm["arm_id"]: arm for arm in row["arms"]}[
        f"fixed:{fixed_by_fold[str(row['fold'])]}"]) for row in rows]
    metric_keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
                   "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    fixed_metrics = {key: statistics.fmean(value[key] for value in fixed)
                     for key in metric_keys}
    pareto, selections = {}, {}
    for budget in BUDGETS:
        maximum = math.floor(budget * len(rows) + 1e-12)
        failures, total, chosen = solve(rows, maximum)
        values = {key: value / len(rows) for key, value in total.items()}
        records = []
        for row, baseline, (arm_id, selected) in zip(rows, fixed, chosen):
            records.append({
                "episode_id": row["episode_id"], "subject": row["subject"],
                "template_group": row["template_group"], "fold": row["fold"],
                "mechanism": row["mechanism"], "length": row["length"],
                "arm_id": arm_id,
                "joint_gain": selected["joint_success"] - baseline["joint_success"],
                "generation_joint_gain": (selected["generation_joint_success"]
                                          - baseline["generation_joint_success"]),
                "selected": selected, "fixed": baseline,
            })
        offset = int(budget * 100)
        subject = interval(records, "joint_gain", "subject",
                           BOOTSTRAP_SEED + offset, BOOTSTRAP_DRAWS)
        template = interval(records, "joint_gain", "template_group",
                            BOOTSTRAP_SEED + 100 + offset, BOOTSTRAP_DRAWS)
        generation_subject = interval(
            records, "generation_joint_gain", "subject",
            BOOTSTRAP_SEED + 200 + offset, BOOTSTRAP_DRAWS)
        generation_template = interval(
            records, "generation_joint_gain", "template_group",
            BOOTSTRAP_SEED + 300 + offset, BOOTSTRAP_DRAWS)
        by_fold = {str(fold): statistics.fmean(
            row["joint_gain"] for row in records if row["fold"] == fold)
                   for fold in range(5)}
        key = str(budget)
        pareto[key] = {
            "budget_rate": budget,
            "maximum_locality_failures": maximum,
            "observed_locality_failures": failures,
            "metrics": values,
            "joint_gain_micro": values["joint_success"] - fixed_metrics["joint_success"],
            "joint_gain_subject_macro": subject,
            "joint_gain_template_macro": template,
            "generation_joint_gain_subject_macro": generation_subject,
            "generation_joint_gain_template_macro": generation_template,
            "by_fold_gain": by_fold,
            "action_counts": {kind: sum(
                (arm_id.startswith("fixed:retain_all") and kind == "retain_all")
                or (arm_id.startswith("fixed:") and kind == "other_fixed")
                or (arm_id.startswith("delete_one:") and kind == "delete_one")
                or (arm_id.startswith("delete_key:") and kind == "delete_key")
                for arm_id, _ in chosen)
                              for kind in ("retain_all", "other_fixed", "delete_one",
                                           "delete_key")},
        }
        selections[key] = [{"episode_id": row["episode_id"], "arm_id": arm_id}
                           for row, (arm_id, _) in zip(rows, chosen)]
    primary = pareto[str(PRIMARY_BUDGET)]
    checks = {
        "subject_macro_joint_gain_at_least_10_points":
            primary["joint_gain_subject_macro"]["mean"] >= .10,
        "subject_cluster_joint_gain_ci_lower_above_zero":
            primary["joint_gain_subject_macro"]["lower_95"] > 0,
        "template_macro_joint_gain_at_least_10_points":
            primary["joint_gain_template_macro"]["mean"] >= .10,
        "template_cluster_joint_gain_ci_lower_above_zero":
            primary["joint_gain_template_macro"]["lower_95"] > 0,
        "oracle_update_accuracy_at_least_90": primary["metrics"]["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90":
            primary["metrics"]["retained_key_accuracy"] >= .90,
        "oracle_locality_accuracy_at_least_97":
            primary["metrics"]["locality_accuracy"] >= .97,
        "no_outer_fold_has_negative_joint_gain":
            all(value >= 0 for value in primary["by_fold_gain"].values()),
    }
    unconstrained_failures, unconstrained_total, unconstrained_chosen = unconstrained(rows)
    return {
        "fixed_selected_by_outer_fold": fixed_by_fold,
        "outer_fold_fixed": fixed_metrics,
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "pareto": pareto,
        "selections": selections,
        "unconstrained_oracle": {
            "observed_locality_failures": unconstrained_failures,
            "metrics": {key: value / len(rows)
                        for key, value in unconstrained_total.items()},
            "action_counts": {kind: sum(
                (arm_id.startswith("fixed:retain_all") and kind == "retain_all")
                or (arm_id.startswith("fixed:") and kind == "other_fixed")
                or (arm_id.startswith("delete_one:") and kind == "delete_one")
                or (arm_id.startswith("delete_key:") and kind == "delete_key")
                for arm_id, _ in unconstrained_chosen)
                              for kind in ("retain_all", "other_fixed", "delete_one",
                                           "delete_key")},
        },
        "confirmation_gate": {"checks": checks, "passed": all(checks.values())},
        "causal_headroom_confirmed": all(checks.values()),
        "next_stage": "build_v19_causal_forgetting_examples"
            if all(checks.values()) else "stop_synthetic_selective_forgetting_claim",
    }


def audit(data_root, release_root, model, tokenizer, inputs, summary_path, output):
    data_root, release_root, model, tokenizer = map(
        Path, (data_root, release_root, model, tokenizer))
    summary, manifest = read_json(summary_path), read_json(data_root / "manifest.json")
    source_audit_path = data_root.parent / "audit.json"
    if not source_audit_path.exists():
        source_audit_path = data_root / "audit.json"
    source_audit = read_json(source_audit_path)
    release_summary, release_audit = (read_json(release_root / "summary.json"),
                                      read_json(release_root / "audit.json"))
    episodes = read_jsonl(data_root / "confirmation.jsonl")
    states = read_jsonl(data_root / "states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value)
        part, resource = read_jsonl(path), read_json(str(path) + ".audit.json")
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
            memories = [tuple(arm["memory_session_ids"]) for arm in expected_arms(state)]
            unique += len(set(memories)); reused += len(memories) - len(set(memories))
        expected_unique[shard], expected_reused[shard] = unique, reused
    dependencies = {
        "manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(source_audit_path),
        "confirmation_sha256": sha(data_root / "confirmation.jsonl"),
        "states_sha256": sha(data_root / "states.jsonl"),
        "release_summary_sha256": sha(release_root / "summary.json"),
        "release_audit_sha256": sha(release_root / "audit.json"),
    }
    source_ok = (
        manifest.get("protocol") == DATA_PROTOCOL
        and manifest.get("confirmation_sha256") == dependencies["confirmation_sha256"]
        and manifest.get("states_sha256") == dependencies["states_sha256"]
        and manifest.get("primary_locality_budget") == .03
        and manifest.get("maximum_locality_failures") == 9
        and manifest.get("model_scored") is False
        and manifest.get("external_sets_loaded") is False
        and source_audit.get("audit_passed") is True
        and source_audit.get("manifest_sha256") == dependencies["manifest_sha256"]
        and release_summary.get("protocol") == RELEASE_PROTOCOL
        and release_summary.get("diagnostic_gate", {}).get("passed") is True
        and release_summary.get("stage_b_released") is False
        and release_audit.get("audit_passed") is True
        and release_audit.get("summary_sha256") == dependencies["release_summary_sha256"])
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
    table, resource_table = recompute_table(rows), recompute_resources(rows, resources)
    checks = {
        "frozen_v19_and_audited_v18_lc_dependencies": source_ok,
        "complete_disjoint_ordered_two_shard_coverage": coverage_ok,
        "episode_state_and_split_identities": identities_ok,
        "fixed_single_and_key_deletion_arms_with_dedup_sources": arms_ok,
        "prompt_hash_scores_predictions_and_five_token_answers": prompts_ok,
        "free_generation_records": generation_ok,
        "per_query_latency_and_peak_memory_records": measurements_ok,
        "shard_counts_dependencies_resources_and_hashes": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_budget_dp_outer_fold_and_cluster_recomputation": all(
            summary.get(key) == value for key, value in table.items()),
        "independent_resource_recomputation": summary.get("resources") == resource_table,
        "terminal_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
            and summary.get("raw_rows_sha256") == hashes
            and summary.get("shard_resources") == resources
            and summary.get("model_scored") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") in {
                "v19_locality_safe_headroom_confirmed",
                "v19_locality_safe_headroom_not_confirmed"}),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(summary_path),
        "raw_rows_sha256": hashes,
        "recomputed_confirmation_gate": table["confirmation_gate"],
        "model_scored": True,
        "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V19 confirmation independent outcome audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
