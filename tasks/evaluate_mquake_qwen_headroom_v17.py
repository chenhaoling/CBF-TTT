"""Score the frozen V17 causal-memory headroom pilot with Qwen3-4B."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_mquake_qwen_headroom_v17_states import FIXED_POLICIES
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    load_jsonl, query_specs, score_query, source_index,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_pilot"
STATE_PROTOCOL = "mquake_qwen_causal_headroom_v17_states"
QUERY_KINDS = (
    "primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated",
)
PRIMARY_KINDS = frozenset(("primary_old_phrase", "primary_current_phrase"))
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1717
MAX_GPU_RESERVED_GIB = 31.5
MAX_PROJECTED_FORMAL_TWO_GPU_HOURS = 48.0


def read_json(path):
    return json.loads(Path(path).read_text())


def state_audit_path(state_root):
    state_root = Path(state_root)
    candidates = (state_root / "audit.json", state_root.parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("V17 state audit location is missing or ambiguous")
    return paths[0]


def verify_states(state_root, sample_root):
    state_root, sample_root = Path(state_root), Path(sample_root)
    manifest = read_json(state_root / "manifest.json")
    audit_path = state_audit_path(state_root)
    audit = read_json(audit_path)
    sample = read_json(sample_root / "manifest.json")
    checks = (
        manifest.get("protocol") == STATE_PROTOCOL,
        manifest.get("terminal_status")
            == "frozen_after_audited_unsupported_v15_before_headroom_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("model_scored") is False,
        manifest.get("external_sets_loaded") is False,
        manifest.get("state_sha256", {}).get("pilot_states.jsonl")
            == sha(state_root / "pilot_states.jsonl"),
        manifest.get("dependencies", {}).get("pilot_sample_sha256")
            == sample.get("pilot_sha256") == sha(sample_root / "pilot.jsonl"),
        manifest.get("counts", {}).get("pilot", {}).get("rows") == 40,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(state_root / "manifest.json"),
        audit.get("state_sha256", {}).get("pilot_states.jsonl")
            == sha(state_root / "pilot_states.jsonl"),
        audit.get("model_scored") is False,
        audit.get("external_sets_loaded") is False,
    )
    if not all(checks):
        raise ValueError("pilot scoring requires frozen, independently audited V17 states")
    return {
        "state_manifest_sha256": sha(state_root / "manifest.json"),
        "state_audit_sha256": sha(audit_path),
        "pilot_states_sha256": sha(state_root / "pilot_states.jsonl"),
        "pilot_sample_sha256": sha(sample_root / "pilot.jsonl"),
    }


def memory_from_state(episode, sample, identifiers):
    visible = episode["sessions"][:sample["query_step"] + 1]
    index = {row["session_id"]: row for row in visible}
    if (len(index) != len(visible) or len(identifiers) != len(set(identifiers))
            or any(identifier not in index for identifier in identifiers)):
        raise ValueError("V17 state contains duplicate, future, or unknown sessions")
    result = [index[identifier] for identifier in identifiers]
    if [row["session_id"] for row in result] != identifiers:
        raise RuntimeError("V17 state order changed")
    return result


def state_arms(state):
    arms = []
    for policy in FIXED_POLICIES:
        arms.append({"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
                     "memory_session_ids": state["fixed"][policy]})
    for index, candidate in enumerate(state["delete_one"]):
        arms.append({
            "arm_id": f"delete_one:{candidate['deleted_session_id']}",
            "kind": "delete_one", "deletion_index": index,
            "deleted_session_id": candidate["deleted_session_id"],
            "memory_session_ids": candidate["memory_session_ids"],
        })
    if len({arm["arm_id"] for arm in arms}) != len(arms):
        raise ValueError("duplicate V17 arm identity")
    return arms


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    sample_root, state_root, output = map(
        Path, (args.sample_root, args.state_root, args.output))
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_states(state_root, sample_root)
    samples = load_jsonl(sample_root / "pilot.jsonl")
    states = load_jsonl(state_root / "pilot_states.jsonl")
    if [row["episode_id"] for row in samples] != [row["episode_id"] for row in states]:
        raise ValueError("V17 pilot state/sample ordering mismatch")
    selected = [(sample, state) for index, (sample, state) in enumerate(zip(samples, states))
                if index % args.num_shards == args.shard]
    episodes, texts = source_index(args.data_root)
    torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V17 requires final slow weights with native TTT adapters disabled")
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
                raise ValueError("V17 pilot state identity mismatch")
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
                "query_step")}
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


def arm_index(row):
    result = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(result) != len(row["arms"]):
        raise ValueError("duplicate scored V17 arms")
    return result


def primary_metrics(arm):
    queries = [query for query in arm["queries"] if query["kind"] in PRIMARY_KINDS]
    if len(queries) != 2:
        raise ValueError("V17 arm must contain two primary queries")
    return (statistics.fmean(query["correct"] for query in queries),
            statistics.fmean(query["answer_nll"] for query in queries))


def select_oracle(row):
    arms = arm_index(row)
    candidates = [arms["fixed:retain_all"]] + [
        arm for arm in row["arms"] if arm["kind"] == "delete_one"]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        accuracy, nll = primary_metrics(arm)
        ranked.append((accuracy, -nll, -tie_index, arm))
    winner = max(ranked, key=lambda value: value[:3])[3]
    return winner, [{"arm_id": value[3]["arm_id"], "primary_accuracy": value[0],
                     "primary_answer_nll": -value[1], "tie_index": -value[2]}
                    for value in ranked]


def select_fixed_by_outer_fold(rows):
    selected, evidence = {}, {}
    for fold in range(5):
        development = [row for row in rows if row["fold"] != fold]
        table = {}
        for policy in FIXED_POLICIES:
            queries = [query for row in development
                       for query in arm_index(row)[f"fixed:{policy}"]["queries"]
                       if query["kind"] in PRIMARY_KINDS]
            table[policy] = {
                "primary_accuracy": statistics.fmean(query["correct"] for query in queries),
                "primary_answer_nll": statistics.fmean(query["answer_nll"] for query in queries),
                "queries": len(queries),
            }
        winner = max(FIXED_POLICIES, key=lambda policy: (
            table[policy]["primary_accuracy"], -table[policy]["primary_answer_nll"],
            -FIXED_POLICIES.index(policy)))
        selected[str(fold)] = winner
        evidence[str(fold)] = table
    return selected, evidence


def clustered_interval(records, field, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    subjects = sorted({row["subject"] for row in records})
    values = {subject: statistics.fmean(row[field] for row in records
                                        if row["subject"] == subject)
              for subject in subjects}
    point = statistics.fmean(values.values())
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(values[rng.choice(subjects)] for _ in subjects)
                     for _ in range(draws))
    return {"mean": point, "lower_95": samples[int(.025 * draws)],
            "upper_95": samples[math.ceil(.975 * draws) - 1],
            "subjects": len(subjects), "draws": draws, "seed": seed}


def query_map(arm):
    result = {query["kind"]: query for query in arm["queries"]}
    if set(result) != set(QUERY_KINDS):
        raise ValueError("incomplete V17 query kinds")
    return result


def summarize_rows(rows, shard_audits, formal_unique_states):
    selected_fixed, fixed_evidence = select_fixed_by_outer_fold(rows)
    selections, comparisons, safety = [], [], []
    deletion_counts = {}
    for row in rows:
        oracle, ranking = select_oracle(row)
        fixed_policy = selected_fixed[str(row["fold"])]
        fixed = arm_index(row)[f"fixed:{fixed_policy}"]
        oracle_queries, fixed_queries = query_map(oracle), query_map(fixed)
        selections.append({
            "episode_id": row["episode_id"], "subject": row["subject"],
            "fold": row["fold"], "oracle_arm_id": oracle["arm_id"],
            "deleted_session_id": oracle.get("deleted_session_id"),
            "fixed_policy": fixed_policy, "ranking": ranking,
        })
        deletion_counts[oracle["arm_id"]] = deletion_counts.get(oracle["arm_id"], 0) + 1
        for kind in PRIMARY_KINDS:
            comparisons.append({
                "episode_id": row["episode_id"], "subject": row["subject"],
                "fold": row["fold"], "kind": kind, "fixed_policy": fixed_policy,
                "oracle_correct": oracle_queries[kind]["correct"],
                "fixed_correct": fixed_queries[kind]["correct"],
                "oracle_answer_nll": oracle_queries[kind]["answer_nll"],
                "fixed_answer_nll": fixed_queries[kind]["answer_nll"],
                "accuracy_gain": oracle_queries[kind]["correct"]
                    - fixed_queries[kind]["correct"],
                "nll_gain": fixed_queries[kind]["answer_nll"]
                    - oracle_queries[kind]["answer_nll"],
            })
        for kind in ("hard_retain", "unrelated"):
            safety.append({"subject": row["subject"], "kind": kind,
                           "oracle_correct": oracle_queries[kind]["correct"],
                           "fixed_correct": fixed_queries[kind]["correct"]})

    accuracy = clustered_interval(comparisons, "accuracy_gain")
    nll = clustered_interval(comparisons, "nll_gain", seed=BOOTSTRAP_SEED + 1)
    unique_queries = [query for row in rows for arm in row["arms"]
                      if arm["score_source_arm_id"] == arm["arm_id"]
                      for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in unique_queries)
    pilot_unique_states = sum(audit["unique_memory_states_scored"] for audit in shard_audits)
    projected_hours = max(audit["seconds"] for audit in shard_audits) \
        * formal_unique_states / pilot_unique_states / 3600
    peak_reserved = max(query["resources"]["gpu_peak_reserved_gib"]
                        for query in unique_queries)
    checks = {
        "all_40_rows_present": len(rows) == 40,
        "all_scores_finite": all(math.isfinite(query["answer_nll"])
                                  for query in unique_queries),
        "all_within_context": max(query["total_tokens_max"]
                                   for query in unique_queries) <= 40960,
        "per_query_resources_complete": all(
            query["resources"]["seconds"] > 0
            and query["resources"]["candidate_scores"] == len(query["candidates"])
            and query["resources"]["gpu_peak_allocated_gib"]
                >= query["resources"]["gpu_start_allocated_gib"]
            and query["resources"]["gpu_peak_reserved_gib"]
                >= query["resources"]["gpu_start_reserved_gib"]
            for query in unique_queries),
        "peak_reserved_memory_at_most_31_5_gib": peak_reserved <= MAX_GPU_RESERVED_GIB,
        "projected_formal_two_gpu_time_at_most_48_hours":
            projected_hours <= MAX_PROJECTED_FORMAL_TWO_GPU_HOURS,
    }
    return {
        "oracle_selection": selections,
        "oracle_arm_counts": deletion_counts,
        "fixed_selected_by_outer_fold": selected_fixed,
        "fixed_selection_evidence": fixed_evidence,
        "pilot_headroom": {
            "oracle_minus_fixed_primary_accuracy_subject_macro": accuracy,
            "fixed_minus_oracle_primary_answer_nll_subject_macro": nll,
            "oracle_primary_accuracy": statistics.fmean(
                row["oracle_correct"] for row in comparisons),
            "selected_fixed_primary_accuracy": statistics.fmean(
                row["fixed_correct"] for row in comparisons),
            "oracle_primary_answer_nll": statistics.fmean(
                row["oracle_answer_nll"] for row in comparisons),
            "selected_fixed_primary_answer_nll": statistics.fmean(
                row["fixed_answer_nll"] for row in comparisons),
            "oracle_hard_retain_accuracy": statistics.fmean(
                row["oracle_correct"] for row in safety if row["kind"] == "hard_retain"),
            "oracle_unrelated_accuracy": statistics.fmean(
                row["oracle_correct"] for row in safety if row["kind"] == "unrelated"),
            "descriptive_only_not_a_formal_gate": True,
        },
        "resources": {
            "unique_memory_states": pilot_unique_states,
            "query_measurements": len(unique_queries),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in unique_queries),
            "max_gpu_peak_reserved_gib": peak_reserved,
            "formal_unique_memory_states": formal_unique_states,
            "projected_formal_two_gpu_hours": projected_hours,
            "projection_method": "max pilot shard wall time scaled by formal/pilot unique states",
        },
        "pilot_gate": {"checks": checks, "passed": all(checks.values())},
    }


def summarize(args):
    sample_root, state_root = map(Path, (args.sample_root, args.state_root))
    dependencies = verify_states(state_root, sample_root)
    state_manifest = read_json(state_root / "manifest.json")
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value)
        part = load_jsonl(path)
        audit = read_json(str(path) + ".audit.json")
        if (audit.get("protocol") != PROTOCOL or audit.get("rows") != len(part)
                or audit.get("rows_sha256") != sha(path)
                or audit.get("dependencies") != dependencies):
            raise ValueError("V17 pilot shard audit mismatch")
        rows.extend(part)
        shard_audits.append(audit)
        hashes[path.name] = sha(path)
    samples = load_jsonl(sample_root / "pilot.jsonl")
    order = {sample["episode_id"]: index for index, sample in enumerate(samples)}
    if ([row["episode_id"] for row in sorted(rows, key=lambda row: order[row["episode_id"]])]
            != [sample["episode_id"] for sample in samples]
            or len({row["episode_id"] for row in rows}) != 40):
        raise ValueError("V17 pilot episode coverage mismatch")
    table = summarize_rows(
        rows, shard_audits,
        state_manifest["counts"]["formal"]["single_deletion_states"]
            + 4 * state_manifest["counts"]["formal"]["rows"])
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "shard_resources": shard_audits, "dependencies": dependencies,
        "model_scored": True, "formal_scored": False, "external_sets_loaded": False,
        "terminal_status": "pilot_resource_passed"
            if table["pilot_gate"]["passed"] else "stopped_by_resource_pilot",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--sample-root", required=True)
    collect_parser.add_argument("--state-root", required=True)
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--sample-root", required=True)
    summary_parser.add_argument("--state-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
