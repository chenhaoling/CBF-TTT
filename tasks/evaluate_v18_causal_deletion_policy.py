"""Rescore nested-OOF V18 deletion-policy states in Qwen fast memory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.evaluate_v18_lowrank_fastmemory_bridge import (
    PRIMARY_SCALE, arm_metrics, build_factor_cache, score_query,
)
from tasks.evaluate_v18_physical_conflict_rules import read_json, verify_bridge_release
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_v18_causal_deletion_policy import SEEDS


PROTOCOL = "v18_causal_deletion_policy_qwen_eval_v1"
BOOTSTRAP_SEED = 1971
BOOTSTRAP_DRAWS = 20_000
METRIC_KEYS = ("joint_success", "update_accuracy", "retained_key_accuracy",
               "locality_accuracy", "joint_answer_nll", "generation_joint_success")


def verify_policy_release(data_root, formal_root, bridge_root, physical_root, policy_root):
    dependencies = verify_bridge_release(data_root, formal_root, bridge_root)
    physical_root, policy_root = Path(physical_root), Path(policy_root)
    physical_summary, physical_audit = (read_json(physical_root / "summary.json"),
                                        read_json(physical_root / "audit.json"))
    policy_summary, policy_audit = (read_json(policy_root / "result/summary.json"),
                                    read_json(policy_root / "audit.json"))
    checks = (
        physical_summary.get("protocol") == "v18_physical_conflict_rules_v1",
        physical_audit.get("audit_passed") is True,
        physical_audit.get("summary_sha256") == sha(physical_root / "summary.json"),
        policy_summary.get("protocol") == "v18_causal_deletion_policy_v1",
        policy_summary.get("terminal_status") == "trained_oof_deletion_policy",
        policy_audit.get("audit_passed") is True,
        policy_audit.get("summary_sha256") == sha(policy_root / "result/summary.json"),
        (policy_root / "runner_exit_code.txt").read_text().strip() == "0",
        (policy_root / "terminal_status.txt").read_text().strip()
            == "completed_v18_causal_deletion_policy",
    )
    if not all(checks):
        raise ValueError("Qwen policy evaluation requires audited OOF policy states")
    return dependencies | {
        "physical_summary_sha256": sha(physical_root / "summary.json"),
        "physical_audit_sha256": sha(physical_root / "audit.json"),
        "policy_summary_sha256": sha(policy_root / "result/summary.json"),
        "policy_audit_sha256": sha(policy_root / "audit.json"),
        "policy_states_sha256": sha(policy_root / "result/oof_policy_states.jsonl"),
        "policy_checkpoint_sha256": sha(policy_root / "result/controllers.pt"),
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_policy_release(
        args.data_root, args.formal_root, args.bridge_root, args.physical_root,
        args.policy_root)
    episodes = load_jsonl(Path(args.data_root) / "formal.jsonl")
    state_rows = load_jsonl(Path(args.policy_root) / "result/oof_policy_states.jsonl")
    state_index = {row["episode_id"]: row for row in state_rows}
    selected = [row for index, row in enumerate(episodes) if index % args.num_shards == args.shard]
    torch.manual_seed(1987); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_hashes = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    started = time.perf_counter(); records = []; unique_states = reused = 0
    with output.open("x") as sink, torch.no_grad():
        for index, episode in enumerate(selected):
            policy = state_index[episode["episode_id"]]
            factor_cache = build_factor_cache(model, tokenizer, episode)
            scored, arms = {}, []
            for seed in SEEDS:
                state = policy["per_seed"][str(seed)]
                identifiers = state["retained_session_ids"]
                key = tuple(identifiers)
                if key not in scored:
                    scored[key] = {"source_seed": seed,
                                   "queries": [score_query(
                                       model, tokenizer, episode, key, spec, factor_cache)
                                               for spec in episode["queries"]]}
                    unique_states += 1
                else:
                    reused += 1
                arms.append({"seed": seed, "memory_session_ids": identifiers,
                             "deleted_session_ids": state["deleted_session_ids"],
                             "component_scores": state["components"],
                             "score_source_seed": scored[key]["source_seed"],
                             "queries": scored[key]["queries"]})
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "subject_index", "template_group",
                "template_group_index", "fold", "length", "mechanism", "target_key_id",
                "current_session_id")}
            row.update({"selection": policy["selection"], "arms": arms})
            sink.write(json.dumps(row) + "\n"); sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    measurements = [query for row in records for arm in row["arms"]
                    if arm["seed"] == arm["score_source_seed"] for query in arm["queries"]]
    audit = {"protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
             "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
             "seconds": time.perf_counter() - started,
             "unique_memory_states_scored": unique_states, "reused_arm_states": reused,
             "query_measurements": len(measurements),
             "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"]
                                           for q in measurements),
             "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"]
                                          for q in measurements),
             "model": args.model, "tokenizer": args.tokenizer,
             "gpu": torch.cuda.get_device_name(), "loading_info": model_meta["loading_info"],
             "model_provenance": {key: model_meta[key] for key in (
                 "path", "class", "config_sha256", "attention", "parameter_count",
                 "source_files_sha256", "excluded_adapters")}
                 | {"ttt_mode": model_meta["config"]["ttt_mode"]},
             "tokenizer_files_sha256": tokenizer_hashes, "seeds": list(SEEDS),
             "primary_scale": PRIMARY_SCALE, "fresh_kv": True, "backbone_frozen": True,
             "query_features_used": False, "current_update_always_retained": True,
             "external_sets_loaded": False, "torch": torch.__version__,
             "transformers": __import__("transformers").__version__}
    write_json(str(output) + ".audit.json", audit); return audit


def _mean(records, field):
    return {key: statistics.fmean(row[field][key] for row in records) for key in METRIC_KEYS}


def _physical_rows(root):
    rows = []
    for path in sorted(Path(root).glob("shard[01].jsonl")):
        rows.extend(load_jsonl(path))
    return {row["episode_id"]: row for row in rows}


def aggregate(rows, physical_root):
    physical_summary = read_json(Path(physical_root) / "summary.json")
    physical_selection = {row["episode_id"]: row
                          for row in physical_summary["selection_records"]}
    physical_raw = _physical_rows(physical_root)
    records, seed_records = [], {str(seed): [] for seed in SEEDS}
    for row in rows:
        arms = {arm["seed"]: arm for arm in row["arms"]}
        reference = physical_selection[row["episode_id"]]
        exact_arm = next(arm for arm in physical_raw[row["episode_id"]]["arms"]
                         if arm["arm_name"] == "exact_conflict")
        per_seed = {str(seed): arm_metrics(arms[seed], PRIMARY_SCALE) for seed in SEEDS}
        learned = {key: statistics.fmean(per_seed[str(seed)][key] for seed in SEEDS)
                   for key in METRIC_KEYS}
        item = {"episode_id": row["episode_id"], "subject": row["subject"],
                "template_group": row["template_group"], "fold": row["fold"],
                "length": row["length"], "mechanism": row["mechanism"],
                "learned": learned, "per_seed": per_seed,
                "fixed": reference["fixed"], "physical": reference["physical"],
                "oracle": reference["oracle"],
                "exact_rule": arm_metrics(exact_arm, PRIMARY_SCALE)}
        item["gain_vs_fixed"] = learned["joint_success"] - item["fixed"]["joint_success"]
        item["gain_vs_physical"] = (learned["joint_success"]
                                    - item["physical"]["joint_success"])
        records.append(item)
        for seed in SEEDS:
            seed_records[str(seed)].append({**item, "learned": per_seed[str(seed)]})
    learned, fixed, physical, oracle, exact = (_mean(records, name) for name in (
        "learned", "fixed", "physical", "oracle", "exact_rule"))
    subject_rows = [{"subject": row["subject"], "template_group": row["template_group"],
                     "gain": row["gain_vs_fixed"]} for row in records]
    subject = clustered_interval(subject_rows, "gain", "subject", BOOTSTRAP_SEED,
                                 BOOTSTRAP_DRAWS)
    template = clustered_interval(subject_rows, "gain", "template_group",
                                  BOOTSTRAP_SEED + 100, BOOTSTRAP_DRAWS)
    denominator = oracle["joint_success"] - fixed["joint_success"]
    per_seed = {key: _mean(value, "learned") for key, value in seed_records.items()}
    checks = {
        "joint_gain_over_fixed_at_least_5_points":
            learned["joint_success"] - fixed["joint_success"] >= .05,
        "subject_cluster_ci_lower_above_zero": subject["lower_95"] > 0,
        "template_cluster_ci_lower_above_zero": template["lower_95"] > 0,
        "nonnegative_joint_gain_every_outer_fold": all(
            statistics.fmean(row["gain_vs_fixed"] for row in records if row["fold"] == fold)
            >= 0 for fold in range(5)),
        "update_accuracy_at_least_90": learned["update_accuracy"] >= .90,
        "retained_key_accuracy_at_least_90": learned["retained_key_accuracy"] >= .90,
        "locality_accuracy_at_least_97": learned["locality_accuracy"] >= .97,
        "fixed_to_oracle_gap_closed_at_least_40_percent": (
            denominator > 0 and (learned["joint_success"] - fixed["joint_success"])
            / denominator >= .40),
        "no_seed_below_best_fixed": all(
            value["joint_success"] >= fixed["joint_success"] for value in per_seed.values()),
    }
    controlled = all(checks.values())
    return {"learned_three_seed_mean": learned, "per_seed": per_seed,
            "outer_fold_fixed": fixed, "strongest_physical": physical,
            "exact_conflict_rule": exact, "exact_oracle": oracle,
            "joint_gain_over_fixed": learned["joint_success"] - fixed["joint_success"],
            "joint_gain_over_physical": learned["joint_success"] - physical["joint_success"],
            "joint_gain_over_exact_rule": learned["joint_success"] - exact["joint_success"],
            "joint_gain_subject_macro": subject, "joint_gain_template_macro": template,
            "by_fold_gain": {str(fold): statistics.fmean(
                row["gain_vs_fixed"] for row in records if row["fold"] == fold)
                             for fold in range(5)},
            "fixed_to_oracle_gap_closed": ((learned["joint_success"]
                                              - fixed["joint_success"]) / denominator
                                             if denominator > 0 else None),
            "controlled_policy_gate": {"checks": checks, "passed": controlled},
            "semantic_alias_gate": {"evaluated": False, "passed": False,
                                    "required_for_learned_policy_claim": True},
            "controlled_policy_supported": controlled,
            "learned_policy_claim_supported": False,
            "selection_records": records,
            "next_stage": "build_semantic_alias_confirmation" if controlled
                else "stop_learned_policy_claim"}


def summarize(args):
    dependencies = verify_policy_release(
        args.data_root, args.formal_root, args.bridge_root, args.physical_root,
        args.policy_root)
    rows, resources, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path); resource = read_json(str(path) + ".audit.json")
        if resource.get("dependencies") != dependencies or resource.get("rows_sha256") != sha(path):
            raise ValueError("policy-evaluation shard dependency mismatch")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
    if len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("policy-evaluation formal coverage mismatch")
    table = aggregate(rows, args.physical_root)
    measurements = [query for row in rows for arm in row["arms"]
                    if arm["seed"] == arm["score_source_seed"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in measurements)
    result = {"protocol": PROTOCOL, **table, "rows": len(rows),
              "raw_rows_sha256": hashes, "shard_resources": resources,
              "dependencies": dependencies,
              "resources": {"unique_memory_states": sum(
                  row["unique_memory_states_scored"] for row in resources),
                  "query_measurements": len(measurements),
                  "wall_seconds_slower_shard": max(row["seconds"] for row in resources),
                  "mean_seconds_per_query": statistics.fmean(latencies),
                  "median_seconds_per_query": statistics.median(latencies),
                  "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
                  "max_gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"]
                                                    for q in measurements),
                  "max_gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"]
                                                   for q in measurements)},
              "model_scored": True, "formal_scored": True, "external_sets_loaded": False,
              "terminal_status": "controlled_policy_supported" if table[
                  "controlled_policy_supported"] else "controlled_policy_unsupported"}
    write_json(args.output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for item in (collect_parser,):
        item.add_argument("--data-root", required=True); item.add_argument("--formal-root", required=True)
        item.add_argument("--bridge-root", required=True); item.add_argument("--physical-root", required=True)
        item.add_argument("--policy-root", required=True)
    collect_parser.add_argument("--model", required=True); collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True); summary_parser.add_argument("--formal-root", required=True)
    summary_parser.add_argument("--bridge-root", required=True); summary_parser.add_argument("--physical-root", required=True)
    summary_parser.add_argument("--policy-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
