"""Score and aggregate the preregistered V20 semantic-alias confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_controlled_interference_v18 import FIXED_POLICIES, QUERY_KINDS
from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index, arm_metrics, memory_from_ids, read_json, score_query, select_best_fixed,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v20_semantic_alias_qwen"
DATA_PROTOCOL = "controlled_memory_interference_v20_semantic_alias"
RULE_PROTOCOL = "v19_causal_rule_explanation_v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2017


def data_audit_path(data_root):
    candidates = (Path(data_root) / "audit.json", Path(data_root).parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("V20 data audit location is missing or ambiguous")
    return paths[0]


def verify_data(data_root, rule_root):
    data_root, rule_root = Path(data_root), Path(rule_root)
    manifest = read_json(data_root / "manifest.json")
    audit_path = data_audit_path(data_root)
    audit = read_json(audit_path)
    summary_path, rule_audit_path = rule_root / "summary.json", rule_root / "audit.json"
    summary, rule_audit = read_json(summary_path), read_json(rule_audit_path)
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("terminal_status") == "frozen_v20_alias_before_model_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("query_kinds") == list(QUERY_KINDS),
        manifest.get("counts", {}).get("rows") == 320,
        manifest.get("counts", {}).get("subjects") == 80,
        manifest.get("counts", {}).get("template_groups") == 20,
        manifest.get("counts", {}).get("exact_rule_deletions") == 0,
        manifest.get("alias_sha256") == sha(data_root / "alias.jsonl"),
        manifest.get("states_sha256") == sha(data_root / "states.jsonl"),
        manifest.get("model_scored") is False,
        manifest.get("controller_or_threshold_selected") is False,
        manifest.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        audit.get("alias_sha256") == sha(data_root / "alias.jsonl"),
        audit.get("states_sha256") == sha(data_root / "states.jsonl"),
        summary.get("protocol") == RULE_PROTOCOL,
        summary.get("explicit_key_explanation_gate", {}).get("passed") is True,
        summary.get("next_stage") == "build_v20_semantic_alias_confirmation",
        rule_audit.get("audit_passed") is True,
        rule_audit.get("summary_sha256") == sha(summary_path),
        (rule_root / "runner_exit_code.txt").read_text().strip() == "0",
        (rule_root / "terminal_status.txt").read_text().strip()
            == "completed_v19_causal_rule_diagnostic",
    )
    if not all(checks):
        raise ValueError("V20 scoring requires audited data and passing V19 rule release")
    return {
        "manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(audit_path),
        "alias_sha256": sha(data_root / "alias.jsonl"),
        "states_sha256": sha(data_root / "states.jsonl"),
        "rule_summary_sha256": sha(summary_path),
        "rule_audit_sha256": sha(rule_audit_path),
    }


def state_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]}
            for policy in FIXED_POLICIES]
    for policy, removed_key in (("exact_conflict", "exact_removed"),
                                ("semantic_oracle", "semantic_removed")):
        arms.append({"arm_id": f"special:{policy}", "kind": "semantic_rule",
                     "policy": policy,
                     "removed_session_ids": state[removed_key],
                     "memory_session_ids": state["special"][policy]})
    if len({row["arm_id"] for row in arms}) != len(arms):
        raise ValueError("duplicate V20 arm identity")
    return arms


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_data(data_root, args.rule_root)
    episodes, states = load_jsonl(data_root / "alias.jsonl"), load_jsonl(
        data_root / "states.jsonl")
    if [row["episode_id"] for row in episodes] != [row["episode_id"] for row in states]:
        raise ValueError("V20 data/state order mismatch")
    selected = [(episode, state) for index, (episode, state) in enumerate(
        zip(episodes, states)) if index % args.num_shards == args.shard]
    torch.manual_seed(2009)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V20 requires final slow weights with native TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    values = {value for episode in episodes for value in
              episode["initial_values"] + [episode["primary_current_value"],
                                             episode["locality_value"]]}
    if len(values) != 2560 or {len(tokenizer.encode(value, add_special_tokens=False))
                               for value in values} != {5}:
        raise ValueError("V20 tokenizer-wide five-token invariant failed")

    started = time.perf_counter()
    records, unique_states, reused = [], 0, 0
    with output.open("x") as sink:
        for index, (episode, state) in enumerate(selected):
            scored, arms = {}, []
            for arm in state_arms(state):
                key = tuple(arm["memory_session_ids"])
                if key not in scored:
                    memory = memory_from_ids(episode, arm["memory_session_ids"])
                    queries = [score_query(model, tokenizer, memory, spec)
                               for spec in episode["queries"]]
                    scored[key] = {"source_arm_id": arm["arm_id"], "queries": queries}
                    unique_states += 1
                else:
                    reused += 1
                result = copy.deepcopy(arm)
                result["score_source_arm_id"] = scored[key]["source_arm_id"]
                result["queries"] = copy.deepcopy(scored[key]["queries"])
                arms.append(result)
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "subject_index", "template_group",
                "template_group_index", "fold", "length", "mechanism", "source_keys",
                "update_key_id", "query_key_id", "current_session_id")}
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
        "unique_memory_states_scored": unique_states, "reused_arm_states": reused,
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


def aggregate(rows):
    if (len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320
            or {row["fold"] for row in rows} != set(range(5))):
        raise ValueError("V20 aggregation requires complete unique frozen coverage")
    fixed_by_fold = {str(fold): select_best_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(5)}
    metric_keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
                   "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    records = []
    exact_score_identity = True
    for row in rows:
        arms = arm_index(row)
        exact, semantic = (arm_metrics(arms["special:exact_conflict"]),
                           arm_metrics(arms["special:semantic_oracle"]))
        retain = arm_metrics(arms["fixed:retain_all"])
        fixed = arm_metrics(arms[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
        exact_score_identity &= (arms["special:exact_conflict"]["queries"]
                                 == arms["fixed:retain_all"]["queries"])
        records.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "mechanism": row["mechanism"], "length": row["length"],
            "joint_gain": semantic["joint_success"] - exact["joint_success"],
            "generation_joint_gain": (semantic["generation_joint_success"]
                                      - exact["generation_joint_success"]),
            "semantic": semantic, "exact": exact, "retain_all": retain, "fixed": fixed,
        })
    mean = lambda arm, key: statistics.fmean(row[arm][key] for row in records)
    tables = {arm: {key: mean(arm, key) for key in metric_keys}
              for arm in ("semantic", "exact", "retain_all", "fixed")}
    subject = clustered_interval(records, "joint_gain", "subject",
                                 BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    template = clustered_interval(records, "joint_gain", "template_group",
                                  BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    generation_subject = clustered_interval(
        records, "generation_joint_gain", "subject", BOOTSTRAP_SEED + 2,
        BOOTSTRAP_DRAWS)
    generation_template = clustered_interval(
        records, "generation_joint_gain", "template_group", BOOTSTRAP_SEED + 3,
        BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["joint_gain"] for row in records if row["fold"] == fold)
               for fold in range(5)}
    semantic = tables["semantic"]
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
            exact_score_identity and tables["exact"] == tables["retain_all"]),
    }
    return {
        "fixed_selected_by_outer_fold": fixed_by_fold,
        "arm_metrics": {"semantic_oracle": tables["semantic"],
                        "exact_conflict": tables["exact"],
                        "retain_all": tables["retain_all"],
                        "outer_fold_fixed": tables["fixed"]},
        "semantic_oracle_vs_exact_conflict": {
            "joint_gain_micro": (tables["semantic"]["joint_success"]
                                 - tables["exact"]["joint_success"]),
            "joint_gain_subject_macro": subject,
            "joint_gain_template_macro": template,
            "generation_joint_gain_subject_macro": generation_subject,
            "generation_joint_gain_template_macro": generation_template,
            "by_fold_gain": by_fold,
        },
        "semantic_headroom_gate": {"checks": checks, "passed": all(checks.values())},
        "semantic_headroom_confirmed": all(checks.values()),
        "next_stage": "train_nested_oof_semantic_forgetting_controller"
            if all(checks.values()) else "stop_synthetic_semantic_forgetting_claim",
    }


def summarize(args):
    data_root = Path(args.data_root)
    dependencies = verify_data(data_root, args.rule_root)
    rows, audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V20 shard audit mismatch")
        rows.extend(part); audits.append(resource); hashes[path.name] = sha(path)
    expected = {row["episode_id"] for row in load_jsonl(data_root / "alias.jsonl")}
    if (len(rows) != len({row["episode_id"] for row in rows}) or len(rows) != 320
            or {row["episode_id"] for row in rows} != expected):
        raise ValueError("V20 shard coverage mismatch")
    table = aggregate(rows)
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
        "model_scored": True, "controller_or_threshold_selected": False,
        "external_sets_loaded": False,
        "memory_substrate": "contextual_evidence_not_native_dense_ttt_writer",
        "terminal_status": "v20_semantic_headroom_confirmed"
            if table["semantic_headroom_gate"]["passed"]
            else "v20_semantic_headroom_not_confirmed",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--rule-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--rule-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
