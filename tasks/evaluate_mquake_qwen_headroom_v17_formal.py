"""Score the frozen 272-row V17 exact causal-headroom experiment."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_mquake_qwen_headroom_v17_states import FIXED_POLICIES
from tasks.evaluate_mquake_qwen_headroom_v17 import (
    PRIMARY_KINDS, QUERY_KINDS, arm_index, clustered_interval, load_jsonl,
    memory_from_state, primary_metrics, query_map, select_fixed_by_outer_fold,
    select_oracle, state_arms, verify_states,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import query_specs, score_query, source_index
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_formal"
PILOT_PROTOCOL = "mquake_qwen_causal_headroom_v17_pilot"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1727


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_release(pilot_root, state_root, sample_root):
    pilot_root = Path(pilot_root)
    dependencies = verify_states(state_root, sample_root)
    state_root, sample_root = Path(state_root), Path(sample_root)
    manifest = read_json(state_root / "manifest.json")
    audit = read_json(pilot_root / "audit.json")
    summary = read_json(pilot_root / "summary.json")
    checks = (
        (pilot_root / "runner_exit_code.txt").read_text().strip() == "0",
        (pilot_root / "terminal_status.txt").read_text().strip()
            == "completed_mquake_qwen_headroom_v17_pilot",
        summary.get("protocol") == PILOT_PROTOCOL,
        summary.get("terminal_status") == "pilot_resource_passed",
        summary.get("pilot_gate", {}).get("passed") is True,
        summary.get("formal_scored") is False,
        summary.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(pilot_root / "summary.json"),
        audit.get("formal_scored") is False,
        audit.get("external_sets_loaded") is False,
        manifest.get("state_sha256", {}).get("formal_states.jsonl")
            == sha(state_root / "formal_states.jsonl"),
        manifest.get("dependencies", {}).get("formal_sample_sha256")
            == sha(sample_root / "formal.jsonl"),
        manifest.get("counts", {}).get("formal", {}).get("rows") == 272,
    )
    if not all(checks):
        raise ValueError("formal V17 scoring requires a passed and audited resource pilot")
    return dependencies | {
        "pilot_summary_sha256": sha(pilot_root / "summary.json"),
        "pilot_audit_sha256": sha(pilot_root / "audit.json"),
        "formal_states_sha256": sha(state_root / "formal_states.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    sample_root, state_root, output = map(
        Path, (args.sample_root, args.state_root, args.output))
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_release(args.pilot_root, state_root, sample_root)
    samples = load_jsonl(sample_root / "formal.jsonl")
    states = load_jsonl(state_root / "formal_states.jsonl")
    if [row["episode_id"] for row in samples] != [row["episode_id"] for row in states]:
        raise ValueError("V17 formal state/sample ordering mismatch")
    selected = [(sample, state) for index, (sample, state) in enumerate(zip(samples, states))
                if index % args.num_shards == args.shard]
    episodes, texts = source_index(args.data_root)
    torch.manual_seed(1703)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("formal V17 requires final slow weights with native TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {
        path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
        if path.is_file() and path.suffix in {".json", ".model", ".txt", ".tiktoken"}
    }
    if not tokenizer_files:
        raise ValueError("no tokenizer provenance files")

    started = time.perf_counter()
    records, unique_states_scored, reused_states = [], 0, 0
    with output.open("x") as sink:
        for index, (sample, state) in enumerate(selected):
            if any(state[key] != sample[key] for key in (
                    "episode_id", "subject", "fold", "length", "query_step",
                    "hard_relation_matched")):
                raise ValueError("V17 formal state identity mismatch")
            episode = episodes[sample["episode_id"]]
            scored, output_arms = {}, []
            for arm in state_arms(state):
                identifiers = arm["memory_session_ids"]
                key = tuple(identifiers)
                if key not in scored:
                    memory = memory_from_state(episode, sample, identifiers)
                    queries = [score_query(model, tokenizer, memory, texts, spec)
                               for spec in query_specs(sample)]
                    scored[key] = {"source_arm_id": arm["arm_id"], "queries": queries}
                    unique_states_scored += 1
                else:
                    reused_states += 1
                source = scored[key]
                result = copy.deepcopy(arm)
                result["score_source_arm_id"] = source["source_arm_id"]
                result["queries"] = copy.deepcopy(source["queries"])
                output_arms.append(result)
            row = {key: sample[key] for key in (
                "episode_id", "subject", "fold", "length", "hard_relation_matched",
                "query_step", "primary_old_session_id", "primary_current_session_id")}
            row["arms"] = output_arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": sample["episode_id"],
                              "unique_states": len(scored)}), flush=True)

    queries = [query for row in records for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"]
               for query in arm["queries"]]
    audit = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
        "seconds": time.perf_counter() - started,
        "unique_memory_states_scored": unique_states_scored,
        "reused_arm_states": reused_states,
        "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(query["resources"]["gpu_peak_allocated_gib"]
                                        for query in queries),
        "gpu_peak_reserved_gib": max(query["resources"]["gpu_peak_reserved_gib"]
                                       for query in queries),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
        "readout_mode": "plain_ttt_disabled_fresh_kv",
        "candidate_scoring": "one_prompt_forward_with_independent_answer_cache_branches",
        "generation_scored": False,
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "fixed_policies": list(FIXED_POLICIES), "external_sets_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def subject_macro(rows, field):
    subjects = sorted({row["subject"] for row in rows})
    return statistics.fmean(statistics.fmean(row[field] for row in rows
                                             if row["subject"] == subject)
                            for subject in subjects)


def aggregate(rows):
    selected_fixed, fixed_evidence = select_fixed_by_outer_fold(rows)
    selections, comparisons, query_records, pair_records = [], [], [], []
    action_counts = {"retain_all": 0, "delete_one": 0, "delete_primary_old": 0,
                     "delete_other": 0}
    for row in rows:
        oracle, ranking = select_oracle(row)
        fixed_policy = selected_fixed[str(row["fold"])]
        fixed = arm_index(row)[f"fixed:{fixed_policy}"]
        oq, fq = query_map(oracle), query_map(fixed)
        action = "retain_all" if oracle["kind"] == "fixed" else "delete_one"
        action_counts[action] += 1
        if action == "delete_one":
            action_counts["delete_primary_old" if oracle["deleted_session_id"]
                          == row["primary_old_session_id"] else "delete_other"] += 1
        selections.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "fold": row["fold"], "oracle_arm_id": oracle["arm_id"],
            "deleted_session_id": oracle.get("deleted_session_id"),
            "fixed_policy": fixed_policy, "ranking": ranking,
        })
        oracle_primary, fixed_primary = [], []
        for kind in QUERY_KINDS:
            record = {
                "episode_id": row["episode_id"], "subject": row["subject"],
                "fold": row["fold"], "kind": kind, "fixed_policy": fixed_policy,
                "oracle_correct": oq[kind]["correct"],
                "fixed_correct": fq[kind]["correct"],
                "oracle_answer_nll": oq[kind]["answer_nll"],
                "fixed_answer_nll": fq[kind]["answer_nll"],
            }
            query_records.append(record)
            if kind in PRIMARY_KINDS:
                record["accuracy_gain"] = record["oracle_correct"] - record["fixed_correct"]
                record["nll_gain"] = record["fixed_answer_nll"] - record["oracle_answer_nll"]
                comparisons.append(record)
                oracle_primary.append(record["oracle_correct"])
                fixed_primary.append(record["fixed_correct"])
        pair_records.append({"subject": row["subject"], "fold": row["fold"],
                             "oracle_pair": int(all(oracle_primary)),
                             "fixed_pair": int(all(fixed_primary))})

    accuracy = clustered_interval(
        comparisons, "accuracy_gain", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    nll = clustered_interval(
        comparisons, "nll_gain", draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED + 1)
    by_kind = {}
    for kind in QUERY_KINDS:
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
            "fixed_policy": selected_fixed[str(fold)],
            "oracle_minus_fixed_primary_accuracy":
                statistics.fmean(row["accuracy_gain"] for row in subset),
            "fixed_minus_oracle_primary_answer_nll":
                statistics.fmean(row["nll_gain"] for row in subset),
        }
    gate_checks = {
        "subject_macro_primary_accuracy_gain_at_least_5_points": accuracy["mean"] >= .05,
        "subject_cluster_accuracy_gain_ci_lower_above_zero": accuracy["lower_95"] > 0,
    }
    return {
        "oracle_selection": selections, "oracle_action_counts": action_counts,
        "fixed_selected_by_outer_fold": selected_fixed,
        "fixed_selection_evidence": fixed_evidence,
        "primary_comparison": {
            "oracle_minus_fixed_accuracy_subject_macro": accuracy,
            "fixed_minus_oracle_answer_nll_subject_macro": nll,
            "oracle_accuracy": statistics.fmean(row["oracle_correct"] for row in comparisons),
            "fixed_accuracy": statistics.fmean(row["fixed_correct"] for row in comparisons),
            "oracle_pair_accuracy": statistics.fmean(row["oracle_pair"] for row in pair_records),
            "fixed_pair_accuracy": statistics.fmean(row["fixed_pair"] for row in pair_records),
        },
        "by_kind": by_kind, "by_fold": by_fold,
        "headroom_gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "causal_headroom_supported": all(gate_checks.values()),
        "next_stage": "train_readout_utility_policy" if all(gate_checks.values())
            else "stop_controller_training_no_causal_headroom",
    }


def summarize(args):
    sample_root, state_root = map(Path, (args.sample_root, args.state_root))
    dependencies = verify_release(args.pilot_root, state_root, sample_root)
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value)
        part, resource = load_jsonl(path), read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V17 formal shard audit mismatch")
        rows.extend(part); shard_audits.append(resource); hashes[path.name] = sha(path)
    samples = load_jsonl(sample_root / "formal.jsonl")
    if (len(rows) != 272 or len({row["episode_id"] for row in rows}) != 272
            or {row["episode_id"] for row in rows}
                != {row["episode_id"] for row in samples}):
        raise ValueError("V17 formal episode coverage mismatch")
    table = aggregate(rows)
    unique_queries = [query for row in rows for arm in row["arms"]
                      if arm["score_source_arm_id"] == arm["arm_id"]
                      for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in unique_queries)
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "shard_resources": shard_audits, "dependencies": dependencies,
        "resources": {
            "unique_memory_states": sum(row["unique_memory_states_scored"]
                                         for row in shard_audits),
            "query_measurements": len(unique_queries),
            "wall_seconds_slower_shard": max(row["seconds"] for row in shard_audits),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in unique_queries),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in unique_queries),
        },
        "model_scored": True, "formal_scored": True, "external_sets_loaded": False,
        "terminal_status": "formal_headroom_supported" if table["causal_headroom_supported"]
            else "formal_no_sufficient_headroom",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--pilot-root", required=True)
    collect_parser.add_argument("--sample-root", required=True)
    collect_parser.add_argument("--state-root", required=True)
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--pilot-root", required=True)
    summary_parser.add_argument("--sample-root", required=True)
    summary_parser.add_argument("--state-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
