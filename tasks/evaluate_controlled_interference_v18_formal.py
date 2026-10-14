"""Score and aggregate the frozen V18 controlled-interference formal study."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_controlled_interference_v18 import FIXED_POLICIES
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index, arm_metrics, memory_from_ids, read_json, score_query, select_best_fixed,
    select_oracle, state_arms,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_qwen_formal"
DATA_PROTOCOL = "controlled_memory_interference_v18_formal"
PILOT_PROTOCOL = "controlled_memory_interference_v18_qwen_pilot"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1817


def data_audit_path(data_root):
    data_root = Path(data_root)
    candidates = (data_root / "audit.json", data_root.parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("V18 formal data audit location is missing or ambiguous")
    return paths[0]


def verify_data(data_root, pilot_root):
    data_root, pilot_root = Path(data_root), Path(pilot_root)
    manifest = read_json(data_root / "manifest.json")
    audit_path = data_audit_path(data_root)
    audit = read_json(audit_path)
    pilot_summary_path, pilot_audit_path = (pilot_root / "summary.json",
                                            pilot_root / "audit.json")
    pilot_summary, pilot_audit = read_json(pilot_summary_path), read_json(pilot_audit_path)
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("terminal_status") == "frozen_v18_formal_before_model_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("counts", {}).get("rows") == 320,
        manifest.get("counts", {}).get("subjects") == 80,
        manifest.get("counts", {}).get("template_groups") == 20,
        manifest.get("formal_sha256") == sha(data_root / "formal.jsonl"),
        manifest.get("formal_states_sha256") == sha(data_root / "formal_states.jsonl"),
        manifest.get("model_scored") is False,
        manifest.get("formal_generated") is True,
        manifest.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        audit.get("formal_sha256") == sha(data_root / "formal.jsonl"),
        audit.get("formal_states_sha256") == sha(data_root / "formal_states.jsonl"),
        pilot_summary.get("protocol") == PILOT_PROTOCOL,
        pilot_summary.get("corpus_admission_gate", {}).get("passed") is True,
        pilot_summary.get("terminal_status") == "pilot_corpus_admitted",
        pilot_audit.get("audit_passed") is True,
        pilot_audit.get("summary_sha256") == sha(pilot_summary_path),
        (pilot_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("V18 formal scoring requires frozen audited data and admitted pilot")
    return {
        "formal_manifest_sha256": sha(data_root / "manifest.json"),
        "formal_data_audit_sha256": sha(audit_path),
        "formal_sha256": sha(data_root / "formal.jsonl"),
        "formal_states_sha256": sha(data_root / "formal_states.jsonl"),
        "pilot_summary_sha256": sha(pilot_summary_path),
        "pilot_audit_sha256": sha(pilot_audit_path),
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_data(data_root, args.pilot_root)
    episodes = load_jsonl(data_root / "formal.jsonl")
    states = load_jsonl(data_root / "formal_states.jsonl")
    if [row["episode_id"] for row in episodes] != [row["episode_id"] for row in states]:
        raise ValueError("V18 formal data/state order mismatch")
    selected = [(episode, state) for index, (episode, state) in enumerate(zip(episodes, states))
                if index % args.num_shards == args.shard]
    torch.manual_seed(1813)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V18 formal requires final slow weights with native TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    values = {value for episode in episodes for value in
              episode["initial_values"] + [episode["primary_current_value"],
                                             episode["locality_value"]]}
    if len(values) != 2560 or {len(tokenizer.encode(value, add_special_tokens=False))
                               for value in values} != {5}:
        raise ValueError("V18 formal tokenizer-wide five-token invariant failed")

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
                "template_group_index", "fold", "length", "mechanism", "target_key_id",
                "current_session_id")}
            row["arms"] = arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    queries = [query for row in records for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    audit = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(records),
        "rows_sha256": sha(output),
        "dependencies": dependencies,
        "seconds": time.perf_counter() - started,
        "unique_memory_states_scored": unique_states,
        "reused_arm_states": reused,
        "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"] for q in queries),
        "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"] for q in queries),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "gpu": torch.cuda.get_device_name(),
        "loading_info": model_meta["loading_info"],
        "readout_mode": "plain_ttt_disabled_fresh_kv",
        "candidate_scoring": "one_prompt_forward_with_independent_answer_cache_branches",
        "generation": "greedy_from_independent_shared_prompt_cache_13_tokens",
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files,
        "opaque_answer_tokens": 5,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "formal_generated": True,
        "formal_scored": True,
        "external_sets_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def clustered_interval(records, value_key, cluster_key, seed, draws=BOOTSTRAP_DRAWS):
    grouped = {}
    for row in records:
        grouped.setdefault(row[cluster_key], []).append(row[value_key])
    cluster_values = [statistics.fmean(values) for _, values in sorted(grouped.items())]
    if len(cluster_values) < 2:
        raise ValueError("cluster interval requires at least two clusters")
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choice(cluster_values)
                                      for _ in cluster_values) for _ in range(draws))
    return {
        "mean": statistics.fmean(cluster_values),
        "lower_95": samples[math.floor(.025 * draws)],
        "upper_95": samples[min(draws - 1, math.ceil(.975 * draws) - 1)],
        "clusters": len(cluster_values),
        "draws": draws,
        "seed": seed,
    }


def aggregate(rows):
    if len(rows) != 320 or {row["fold"] for row in rows} != set(range(5)):
        raise ValueError("V18 formal aggregation requires the complete frozen split")
    selected, evidence = {}, {}
    for fold in range(5):
        train = [row for row in rows if row["fold"] != fold]
        selected[str(fold)], evidence[str(fold)] = select_best_fixed(train)

    records, selections, action_counts = [], [], {"retain_all": 0, "delete_one": 0,
                                                   "delete_key": 0}
    for row in rows:
        oracle_arm, oracle = select_oracle(row)
        fixed_policy = selected[str(row["fold"])]
        fixed = arm_metrics(arm_index(row)[f"fixed:{fixed_policy}"])
        action = "retain_all" if oracle_arm["arm_id"] == "fixed:retain_all" \
            else oracle_arm["kind"]
        action_counts[action] += 1
        selections.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "oracle_arm_id": oracle_arm["arm_id"], "oracle_kind": oracle_arm["kind"],
            "fixed_policy": fixed_policy, "oracle_metrics": oracle,
            "fixed_metrics": fixed,
        })
        records.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "template_group": row["template_group"], "fold": row["fold"],
            "length": row["length"], "mechanism": row["mechanism"],
            "fixed_policy": fixed_policy,
            "oracle_joint_success": oracle["joint_success"],
            "fixed_joint_success": fixed["joint_success"],
            "joint_gain": oracle["joint_success"] - fixed["joint_success"],
            "oracle_generation_joint_success": oracle["generation_joint_success"],
            "fixed_generation_joint_success": fixed["generation_joint_success"],
            "generation_joint_gain": (oracle["generation_joint_success"]
                                      - fixed["generation_joint_success"]),
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

    subject_ci = clustered_interval(records, "joint_gain", "subject", BOOTSTRAP_SEED)
    template_ci = clustered_interval(
        records, "joint_gain", "template_group", BOOTSTRAP_SEED + 1)
    generation_subject_ci = clustered_interval(
        records, "generation_joint_gain", "subject", BOOTSTRAP_SEED + 2)
    generation_template_ci = clustered_interval(
        records, "generation_joint_gain", "template_group", BOOTSTRAP_SEED + 3)
    by_fold, by_mechanism, by_length, by_template = {}, {}, {}, {}
    for fold in range(5):
        subset = [row for row in records if row["fold"] == fold]
        by_fold[str(fold)] = {
            "rows": len(subset), "fixed_policy": selected[str(fold)],
            "oracle_minus_fixed_joint_success": mean("joint_gain", subset),
            "oracle_joint_success": mean("oracle_joint_success", subset),
            "fixed_joint_success": mean("fixed_joint_success", subset),
        }
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

    oracle_metrics = {
        "joint_success": mean("oracle_joint_success"),
        "generation_joint_success": mean("oracle_generation_joint_success"),
        "update_accuracy": mean("oracle_update_accuracy"),
        "retained_key_accuracy": mean("oracle_retained_key_accuracy"),
        "locality_accuracy": mean("oracle_locality_accuracy"),
    }
    fixed_metrics = {
        "joint_success": mean("fixed_joint_success"),
        "generation_joint_success": mean("fixed_generation_joint_success"),
        "update_accuracy": mean("fixed_update_accuracy"),
        "retained_key_accuracy": mean("fixed_retained_key_accuracy"),
        "locality_accuracy": mean("fixed_locality_accuracy"),
    }
    checks = {
        "subject_macro_joint_gain_at_least_5_points": subject_ci["mean"] >= .05,
        "subject_cluster_joint_gain_ci_lower_above_zero": subject_ci["lower_95"] > 0,
        "template_macro_joint_gain_at_least_5_points": template_ci["mean"] >= .05,
        "template_cluster_joint_gain_ci_lower_above_zero": template_ci["lower_95"] > 0,
        "oracle_update_accuracy_at_least_90": oracle_metrics["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90":
            oracle_metrics["retained_key_accuracy"] >= .90,
        "oracle_locality_accuracy_at_least_97": oracle_metrics["locality_accuracy"] >= .97,
        "no_outer_fold_has_negative_joint_gain":
            all(value["oracle_minus_fixed_joint_success"] >= 0 for value in by_fold.values()),
    }
    supported = all(checks.values())
    return {
        "fixed_selected_by_outer_fold": selected,
        "fixed_selection_evidence": evidence,
        "oracle_selection": selections,
        "oracle_action_counts": action_counts,
        "oracle": oracle_metrics,
        "outer_fold_fixed": fixed_metrics,
        "primary_comparison": {
            "oracle_minus_fixed_joint_success_micro": mean("joint_gain"),
            "oracle_minus_fixed_joint_success_subject_macro": subject_ci,
            "oracle_minus_fixed_joint_success_template_macro": template_ci,
            "oracle_minus_fixed_generation_joint_success_micro": mean(
                "generation_joint_gain"),
            "generation_subject_macro": generation_subject_ci,
            "generation_template_macro": generation_template_ci,
            "fixed_minus_oracle_joint_answer_nll": mean("fixed_minus_oracle_joint_nll"),
        },
        "by_fold": by_fold,
        "by_mechanism": by_mechanism,
        "by_length": by_length,
        "by_template": by_template,
        "headroom_gate": {"checks": checks, "passed": supported},
        "causal_headroom_supported": supported,
        "next_stage": "train_causal_forgetting_policy" if supported
            else "stop_controller_training_no_causal_headroom",
    }


def summarize(args):
    data_root = Path(args.data_root)
    dependencies = verify_data(data_root, args.pilot_root)
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value)
        part, resource = load_jsonl(path), read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V18 formal shard audit mismatch")
        rows.extend(part); shard_audits.append(resource); hashes[path.name] = sha(path)
    expected = {row["episode_id"] for row in load_jsonl(data_root / "formal.jsonl")}
    if len(rows) != len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("V18 formal duplicate coverage")
    if len(rows) != 320 or {row["episode_id"] for row in rows} != expected:
        raise ValueError("V18 formal coverage mismatch")
    table = aggregate(rows)
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "shard_resources": shard_audits, "dependencies": dependencies,
        "resources": {
            "unique_memory_states": sum(a["unique_memory_states_scored"]
                                        for a in shard_audits),
            "query_measurements": len(queries),
            "wall_seconds_slower_shard": max(a["seconds"] for a in shard_audits),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                q["resources"]["gpu_peak_allocated_gib"] for q in queries),
            "max_gpu_peak_reserved_gib": max(
                q["resources"]["gpu_peak_reserved_gib"] for q in queries),
        },
        "model_scored": True,
        "formal_generated": True,
        "formal_scored": True,
        "external_sets_loaded": False,
        "terminal_status": "formal_headroom_supported" if table["headroom_gate"]["passed"]
            else "formal_no_sufficient_headroom",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--pilot-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--pilot-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
