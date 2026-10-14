"""Model-level Qwen evaluation of the V19 nested-OOF fallback policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index, arm_metrics, memory_from_ids, read_json, score_query,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_v19_nested_forgetting_policy import SEEDS


PROTOCOL = "v19_nested_oof_forgetting_policy_qwen_eval_v1"
POLICY_PROTOCOL = "v19_nested_oof_forgetting_policy_v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1937


def verify_release(data_root, outcome_root, rule_root, examples_root, policy_root):
    data_root, outcome_root, rule_root, examples_root, policy_root = map(
        Path, (data_root, outcome_root, rule_root, examples_root, policy_root))
    manifest = read_json(data_root / "manifest.json")
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    outcome_summary_path, outcome_audit_path = (outcome_root / "summary.json",
                                                outcome_root / "audit.json")
    outcome, outcome_audit = read_json(outcome_summary_path), read_json(outcome_audit_path)
    rule_summary_path, rule_audit_path = rule_root / "summary.json", rule_root / "audit.json"
    rule, rule_audit = read_json(rule_summary_path), read_json(rule_audit_path)
    example_manifest_path, example_audit_path = (examples_root / "data/manifest.json",
                                                 examples_root / "audit.json")
    examples, example_audit = read_json(example_manifest_path), read_json(example_audit_path)
    policy_summary_path, policy_audit_path = (policy_root / "result/summary.json",
                                              policy_root / "audit.json")
    policy, policy_audit = read_json(policy_summary_path), read_json(policy_audit_path)
    checks = (
        manifest.get("protocol") == "controlled_memory_interference_v19_confirmation",
        data_audit.get("audit_passed") is True,
        outcome.get("protocol") == "controlled_memory_interference_v19_qwen_confirmation",
        outcome.get("confirmation_gate", {}).get("passed") is True,
        outcome_audit.get("audit_passed") is True,
        outcome_audit.get("summary_sha256") == sha(outcome_summary_path),
        rule.get("protocol") == "v19_causal_rule_explanation_v1",
        rule.get("explicit_key_explanation_gate", {}).get("passed") is False,
        rule.get("next_stage") == "train_v19_nested_oof_causal_policy",
        rule_audit.get("audit_passed") is True,
        rule_audit.get("summary_sha256") == sha(rule_summary_path),
        examples.get("protocol") == "v19_nested_oof_policy_examples_v1",
        example_audit.get("audit_passed") is True,
        example_audit.get("manifest_sha256") == sha(example_manifest_path),
        policy.get("protocol") == POLICY_PROTOCOL,
        policy.get("episodes") == 320,
        policy.get("seeds") == list(SEEDS),
        policy.get("query_features_used") is False,
        policy.get("answer_features_used") is False,
        policy.get("current_update_always_retained") is True,
        policy_audit.get("audit_passed") is True,
        policy_audit.get("summary_sha256") == sha(policy_summary_path),
        (outcome_root / "runner_exit_code.txt").read_text().strip() == "0",
        (rule_root / "runner_exit_code.txt").read_text().strip() == "0",
        (examples_root / "runner_exit_code.txt").read_text().strip() == "0",
        (policy_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("V19 policy evaluation requires the audited fallback chain")
    return {
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
        "policy_checkpoint_sha256": policy["checkpoint_sha256"],
        "policy_states_sha256": policy["oof_policy_states_sha256"],
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, policy_root, output = map(Path, (args.data_root, args.policy_root,
                                                args.output))
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_release(args.data_root, args.outcome_root, args.rule_root,
                                  args.examples_root, args.policy_root)
    episodes = load_jsonl(data_root / "confirmation.jsonl")
    states = load_jsonl(policy_root / "result/oof_policy_states.jsonl")
    state_index = {row["episode_id"]: row for row in states}
    if len(states) != 320 or len(state_index) != 320:
        raise ValueError("invalid V19 policy-state coverage")
    selected = [episode for index, episode in enumerate(episodes)
                if index % args.num_shards == args.shard]
    torch.manual_seed(1943)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V19 policy evaluation requires final slow weights with TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    values = {value for episode in episodes for value in
              episode["initial_values"] + [episode["primary_current_value"],
                                             episode["locality_value"]]}
    if len(values) != 2560 or {len(tokenizer.encode(value, add_special_tokens=False))
                               for value in values} != {5}:
        raise ValueError("V19 policy tokenizer-wide five-token invariant failed")

    started = time.perf_counter(); records = []; unique = reused = 0
    with output.open("x") as sink:
        for index, episode in enumerate(selected):
            policy = state_index[episode["episode_id"]]
            scored, arms = {}, []
            for seed in SEEDS:
                seed_state = policy["per_seed"][str(seed)]
                identifiers = seed_state["retained_session_ids"]
                key = tuple(identifiers)
                arm_id = f"policy:{seed}"
                if key not in scored:
                    memory = memory_from_ids(episode, identifiers)
                    queries = [score_query(model, tokenizer, memory, spec)
                               for spec in episode["queries"]]
                    scored[key] = {"source_arm_id": arm_id, "queries": queries}
                    unique += 1
                else:
                    reused += 1
                arms.append({"arm_id": arm_id, "kind": "nested_oof_policy",
                             "seed": seed, "memory_session_ids": identifiers,
                             "deleted_session_ids": seed_state["deleted_session_ids"],
                             "score_source_arm_id": scored[key]["source_arm_id"],
                             "queries": copy.deepcopy(scored[key]["queries"])})
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "subject_index", "template_group",
                "template_group_index", "fold", "length", "mechanism", "target_key_id",
                "current_session_id")}
            row["selection"] = policy["selection"]
            row["arms"] = arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    queries = [query for row in records for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"]
               for query in arm["queries"]]
    audit = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
        "seconds": time.perf_counter() - started,
        "unique_memory_states_scored": unique, "reused_arm_states": reused,
        "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"]
                                      for q in queries),
        "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"]
                                     for q in queries),
        "model": args.model, "tokenizer": args.tokenizer,
        "gpu": torch.cuda.get_device_name(), "loading_info": model_meta["loading_info"],
        "readout_mode": "plain_ttt_disabled_fresh_kv_contextual_evidence_memory",
        "candidate_scoring": "one_prompt_forward_with_independent_answer_cache_branches",
        "generation": "greedy_from_independent_shared_prompt_cache_13_tokens",
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files, "opaque_answer_tokens": 5,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "model_scored": True, "external_sets_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def exact_arm(episode, raw_row):
    arms = arm_index(raw_row)
    current = episode["sessions"][-1]
    conflicts = [row for row in episode["sessions"][:-1]
                 if row["key_id"] == current["key_id"] and row["value"] != current["value"]]
    return arms[f"delete_key:{current['key_id']}"] if conflicts else arms["fixed:retain_all"]


def aggregate(episodes, policy_rows, raw_rows, outcome_summary, policy_summary):
    if not len(episodes) == len(policy_rows) == len(raw_rows) == 320:
        raise ValueError("invalid V19 policy evaluation coverage")
    episode_index = {row["episode_id"]: row for row in episodes}
    raw_index = {row["episode_id"]: row for row in raw_rows}
    selected_oracle = {row["episode_id"]: row["arm_id"] for row in
                       outcome_summary["selections"]["0.03"]}
    fixed_by_fold = outcome_summary["fixed_selected_by_outer_fold"]
    metric_keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
                   "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    records = []
    for row in policy_rows:
        episode, raw = episode_index[row["episode_id"]], raw_index[row["episode_id"]]
        raw_arms, policy_arms = arm_index(raw), arm_index(row)
        per_seed = {str(seed): arm_metrics(policy_arms[f"policy:{seed}"]) for seed in SEEDS}
        policy = {key: statistics.fmean(value[key] for value in per_seed.values())
                  for key in metric_keys}
        fixed = arm_metrics(raw_arms[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
        exact = arm_metrics(exact_arm(episode, raw))
        oracle = arm_metrics(raw_arms[selected_oracle[row["episode_id"]]])
        records.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "mechanism": row["mechanism"], "length": row["length"],
            "policy": policy, "fixed": fixed, "exact": exact, "oracle": oracle,
            "per_seed": per_seed,
            "fixed_gain": policy["joint_success"] - fixed["joint_success"],
            "exact_gain": policy["joint_success"] - exact["joint_success"],
            "fixed_generation_gain": (policy["generation_joint_success"]
                                      - fixed["generation_joint_success"]),
        })
    tables = {name: {key: statistics.fmean(row[name][key] for row in records)
                     for key in metric_keys}
              for name in ("policy", "fixed", "exact", "oracle")}
    fixed_subject = clustered_interval(records, "fixed_gain", "subject",
                                       BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    fixed_template = clustered_interval(records, "fixed_gain", "template_group",
                                        BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    exact_subject = clustered_interval(records, "exact_gain", "subject",
                                       BOOTSTRAP_SEED + 2, BOOTSTRAP_DRAWS)
    exact_template = clustered_interval(records, "exact_gain", "template_group",
                                        BOOTSTRAP_SEED + 3, BOOTSTRAP_DRAWS)
    generation_subject = clustered_interval(
        records, "fixed_generation_gain", "subject", BOOTSTRAP_SEED + 4,
        BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["fixed_gain"] for row in records if row["fold"] == fold) for fold in range(5)}
    by_seed = {str(seed): statistics.fmean(
        row["per_seed"][str(seed)]["joint_success"] - row["fixed"]["joint_success"]
        for row in records) for seed in SEEDS}
    oracle_gap = tables["oracle"]["joint_success"] - tables["fixed"]["joint_success"]
    closure = ((tables["policy"]["joint_success"] - tables["fixed"]["joint_success"])
               / oracle_gap if oracle_gap > 0 else None)
    selections_satisfied = all(value["constraints_satisfied"]
                               for value in policy_summary["selection"].values())
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
        "all_inner_selection_constraints_satisfied": selections_satisfied,
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
            "registered_oracle_gap": oracle_gap, "fraction_oracle_gap_closed": closure},
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


def summarize(args):
    dependencies = verify_release(args.data_root, args.outcome_root, args.rule_root,
                                  args.examples_root, args.policy_root)
    episodes = load_jsonl(Path(args.data_root) / "confirmation.jsonl")
    raw_rows = []
    for name in ("shard0.jsonl", "shard1.jsonl"):
        raw_rows.extend(load_jsonl(Path(args.outcome_root) / name))
    outcome_summary = read_json(Path(args.outcome_root) / "summary.json")
    policy_summary = read_json(Path(args.policy_root) / "result/summary.json")
    rows, audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V19 policy shard audit mismatch")
        rows.extend(part); audits.append(resource); hashes[path.name] = sha(path)
    if len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("invalid V19 policy shard coverage")
    table = aggregate(episodes, rows, raw_rows, outcome_summary, policy_summary)
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"]
               for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "shard_resources": audits, "dependencies": dependencies,
        "resources": {
            "unique_memory_states": sum(row["unique_memory_states_scored"] for row in audits),
            "query_measurements": len(queries),
            "wall_seconds_slower_shard": max(row["seconds"] for row in audits),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                row["resources"]["gpu_peak_allocated_gib"] for row in queries),
            "max_gpu_peak_reserved_gib": max(
                row["resources"]["gpu_peak_reserved_gib"] for row in queries),
        },
        "model_scored": True, "external_sets_loaded": False,
        "memory_substrate": "contextual_evidence_not_native_dense_ttt_writer",
        "terminal_status": "v19_nested_policy_supported"
            if table["controlled_policy_gate"]["passed"]
            else "v19_nested_policy_not_supported",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("data_root", "outcome_root", "rule_root", "examples_root",
                 "policy_root", "model", "tokenizer", "output"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    summary_parser = sub.add_parser("summarize")
    for name in ("data_root", "outcome_root", "rule_root", "examples_root",
                 "policy_root", "output"):
        summary_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
