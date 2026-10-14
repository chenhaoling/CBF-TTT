"""Score and aggregate the preregistered V19 confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_controlled_interference_v18 import FIXED_POLICIES
from tasks.diagnose_v18_locality_constrained_oracle import constrained_selection
from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index, arm_metrics, memory_from_ids, read_json, score_query, select_best_fixed,
    state_arms,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v19_qwen_confirmation"
DATA_PROTOCOL = "controlled_memory_interference_v19_confirmation"
RELEASE_PROTOCOL = "v18_locality_constrained_oracle_posthoc_v1"
BUDGETS = (0.0, 0.01, 0.03, 0.05, 0.10)
PRIMARY_BUDGET = .03
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1917


def unconstrained_selection(rows):
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    total = {key: 0 for key in keys}
    failures, chosen = 0, []
    for row in rows:
        ranked = []
        for index, arm in enumerate(row["arms"]):
            value = arm_metrics(arm)
            ranked.append(((value["joint_success"], value["update_accuracy"],
                            value["retained_key_accuracy"], value["locality_accuracy"],
                            value["generation_joint_success"], -value["joint_answer_nll"],
                            -index), arm["arm_id"], value))
        _, arm_id, value = max(ranked, key=lambda item: item[0])
        failures += 1 - value["locality_accuracy"]
        total = {key: total[key] + value[key] for key in keys}
        chosen.append((arm_id, value))
    return failures, total, chosen


def data_audit_path(data_root):
    data_root = Path(data_root)
    paths = [path for path in (data_root / "audit.json", data_root.parent / "audit.json")
             if path.exists()]
    if len(paths) != 1:
        raise ValueError("V19 data audit location is missing or ambiguous")
    return paths[0]


def verify_data(data_root, release_root):
    data_root, release_root = Path(data_root), Path(release_root)
    manifest = read_json(data_root / "manifest.json")
    audit_path = data_audit_path(data_root)
    audit = read_json(audit_path)
    release_summary_path, release_audit_path = (release_root / "summary.json",
                                                release_root / "audit.json")
    release_summary, release_audit = (read_json(release_summary_path),
                                      read_json(release_audit_path))
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("terminal_status") == "frozen_v19_confirmation_before_model_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("counts", {}).get("rows") == 320,
        manifest.get("counts", {}).get("subjects") == 80,
        manifest.get("counts", {}).get("template_groups") == 20,
        manifest.get("primary_locality_budget") == PRIMARY_BUDGET,
        manifest.get("maximum_locality_failures") == 9,
        manifest.get("confirmation_sha256") == sha(data_root / "confirmation.jsonl"),
        manifest.get("states_sha256") == sha(data_root / "states.jsonl"),
        manifest.get("model_scored") is False,
        manifest.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        audit.get("confirmation_sha256") == sha(data_root / "confirmation.jsonl"),
        audit.get("states_sha256") == sha(data_root / "states.jsonl"),
        release_summary.get("protocol") == RELEASE_PROTOCOL,
        release_summary.get("diagnostic_gate", {}).get("passed") is True,
        release_summary.get("posthoc") is True,
        release_summary.get("stage_b_released") is False,
        release_audit.get("audit_passed") is True,
        release_audit.get("summary_sha256") == sha(release_summary_path),
        (release_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("V19 scoring requires frozen audited data and V18-LC release")
    return {
        "manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(audit_path),
        "confirmation_sha256": sha(data_root / "confirmation.jsonl"),
        "states_sha256": sha(data_root / "states.jsonl"),
        "release_summary_sha256": sha(release_summary_path),
        "release_audit_sha256": sha(release_audit_path),
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_data(data_root, args.release_root)
    episodes = load_jsonl(data_root / "confirmation.jsonl")
    states = load_jsonl(data_root / "states.jsonl")
    if [row["episode_id"] for row in episodes] != [row["episode_id"] for row in states]:
        raise ValueError("V19 data/state order mismatch")
    selected = [(episode, state) for index, (episode, state) in enumerate(zip(episodes, states))
                if index % args.num_shards == args.shard]
    torch.manual_seed(1913)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V19 requires final slow weights with native TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    values = {value for episode in episodes for value in
              episode["initial_values"] + [episode["primary_current_value"],
                                             episode["locality_value"]]}
    if len(values) != 2560 or {len(tokenizer.encode(value, add_special_tokens=False))
                               for value in values} != {5}:
        raise ValueError("V19 tokenizer-wide five-token invariant failed")

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
        "model_scored": True,
        "external_sets_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def aggregate(rows):
    if (len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320
            or {row["fold"] for row in rows} != set(range(5))):
        raise ValueError("V19 aggregation requires complete unique frozen coverage")
    fixed_by_fold = {str(fold): select_best_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(5)}
    fixed = [arm_metrics(arm_index(row)[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
             for row in rows]
    metric_keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
                   "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    fixed_metrics = {key: statistics.fmean(value[key] for value in fixed)
                     for key in metric_keys}
    pareto, selections = {}, {}
    for budget in BUDGETS:
        maximum = math.floor(budget * len(rows) + 1e-12)
        failures, total, chosen = constrained_selection(rows, maximum)
        metrics = {key: value / len(rows) for key, value in total.items()}
        records = []
        for row, fixed_value, (arm_id, selected) in zip(rows, fixed, chosen):
            records.append({
                "episode_id": row["episode_id"], "subject": row["subject"],
                "template_group": row["template_group"], "fold": row["fold"],
                "mechanism": row["mechanism"], "length": row["length"],
                "arm_id": arm_id,
                "joint_gain": selected["joint_success"] - fixed_value["joint_success"],
                "generation_joint_gain": (selected["generation_joint_success"]
                                          - fixed_value["generation_joint_success"]),
                "selected": selected, "fixed": fixed_value,
            })
        seed_offset = int(budget * 100)
        subject = clustered_interval(records, "joint_gain", "subject",
                                     BOOTSTRAP_SEED + seed_offset, BOOTSTRAP_DRAWS)
        template = clustered_interval(records, "joint_gain", "template_group",
                                      BOOTSTRAP_SEED + 100 + seed_offset,
                                      BOOTSTRAP_DRAWS)
        generation_subject = clustered_interval(
            records, "generation_joint_gain", "subject",
            BOOTSTRAP_SEED + 200 + seed_offset, BOOTSTRAP_DRAWS)
        generation_template = clustered_interval(
            records, "generation_joint_gain", "template_group",
            BOOTSTRAP_SEED + 300 + seed_offset, BOOTSTRAP_DRAWS)
        by_fold = {str(fold): statistics.fmean(
            row["joint_gain"] for row in records if row["fold"] == fold)
                   for fold in range(5)}
        key = str(budget)
        pareto[key] = {
            "budget_rate": budget,
            "maximum_locality_failures": maximum,
            "observed_locality_failures": failures,
            "metrics": metrics,
            "joint_gain_micro": metrics["joint_success"] - fixed_metrics["joint_success"],
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
    unconstrained_failures, unconstrained_total, unconstrained_chosen = \
        unconstrained_selection(rows)
    return {
        "fixed_selected_by_outer_fold": fixed_by_fold,
        "outer_fold_fixed": fixed_metrics,
        "budgets": list(BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "pareto": pareto,
        "selections": selections,
        "unconstrained_oracle": {
            "observed_locality_failures": unconstrained_failures,
            "metrics": {key: value / len(rows) for key, value in unconstrained_total.items()},
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


def summarize(args):
    data_root = Path(args.data_root)
    dependencies = verify_data(data_root, args.release_root)
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value)
        part, resource = load_jsonl(path), read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V19 shard audit mismatch")
        rows.extend(part); shard_audits.append(resource); hashes[path.name] = sha(path)
    expected = {row["episode_id"] for row in load_jsonl(data_root / "confirmation.jsonl")}
    if (len(rows) != len({row["episode_id"] for row in rows}) or len(rows) != 320
            or {row["episode_id"] for row in rows} != expected):
        raise ValueError("V19 shard coverage mismatch")
    table = aggregate(rows)
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    result = {
        "protocol": PROTOCOL,
        **table,
        "rows": len(rows),
        "raw_rows_sha256": hashes,
        "shard_resources": shard_audits,
        "dependencies": dependencies,
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
        "external_sets_loaded": False,
        "terminal_status": "v19_locality_safe_headroom_confirmed"
            if table["confirmation_gate"]["passed"]
            else "v19_locality_safe_headroom_not_confirmed",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--release-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--release-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
