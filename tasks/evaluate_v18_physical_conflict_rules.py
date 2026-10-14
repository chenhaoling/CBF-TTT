"""Evaluate preregistered physical forgetting rules on V18 fast memory.

This stage is released only by an independently audited positive V18 low-rank
bridge.  Rules may inspect past memory components, their frozen Qwen keys and
their stored values.  They never inspect query answers or downstream losses.
Thresholds and rule families are selected only on the training side of each
outer fold.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.evaluate_v18_lowrank_fastmemory_bridge import (
    PRIMARY_SCALE, arm_metrics, build_factor_cache, score_query, verify_release,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_physical_conflict_rules_v1"
COSINE_THRESHOLDS = (0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
RANDOM_SEEDS = (1901, 1907, 1913)
RULE_FAMILIES = ("exact_conflict", "qwen_cosine_conflict", "most_recent_same_key")
BOOTSTRAP_SEED = 1931
BOOTSTRAP_DRAWS = 20_000


def _active_after_conflicts(sessions, conflicts):
    """Apply a causal suppress-older graph while preserving useful agreement."""
    active = []
    for newer in sessions:
        active = [older for older in active if not (
            older["value"] != newer["value"] and conflicts(older, newer))]
        active.append(newer)
    return [row["session_id"] for row in active]


def exact_conflict_ids(episode):
    """Delete an older component only for an exact-key value conflict."""
    return _active_after_conflicts(
        episode["sessions"], lambda older, newer: older["key_id"] == newer["key_id"])


def cosine_conflict_ids(episode, similarities, threshold):
    """Delete older conflicting values whose frozen Qwen keys are sufficiently close."""
    def conflicts(older, newer):
        pair = tuple(sorted((older["key_id"], newer["key_id"])))
        return similarities[pair] >= threshold
    return _active_after_conflicts(episode["sessions"], conflicts)


def most_recent_same_key_ids(episode):
    """Keep only the most recent component for every literal key, including repeats."""
    newest = {}
    for row in episode["sessions"]:
        newest[row["key_id"]] = row["session_id"]
    return [row["session_id"] for row in episode["sessions"]
            if newest[row["key_id"]] == row["session_id"]]


def matched_random_ids(episode, reference_ids, seed, namespace):
    """Delete the same number of historical components with a stable random draw."""
    all_ids = [row["session_id"] for row in episode["sessions"]]
    current = episode["current_session_id"]
    deletion_count = len(all_ids) - len(reference_ids)
    eligible = [value for value in all_ids if value != current]
    if not 0 <= deletion_count <= len(eligible):
        raise ValueError("invalid matched-random deletion count")
    digest = hashlib.sha256(
        f"{PROTOCOL}:{episode['episode_id']}:{namespace}:{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    removed = set(rng.sample(eligible, deletion_count))
    return [value for value in all_ids if value not in removed]


def physical_states(episode, similarities):
    """Return all frozen deterministic and matched-random policy states."""
    all_ids = [row["session_id"] for row in episode["sessions"]]
    deterministic = {
        "retain_all": all_ids,
        "exact_conflict": exact_conflict_ids(episode),
        "most_recent_same_key": most_recent_same_key_ids(episode),
    }
    for threshold in COSINE_THRESHOLDS:
        deterministic[f"qwen_cosine_conflict:{threshold}"] = cosine_conflict_ids(
            episode, similarities, threshold)
    output = dict(deterministic)
    for name, identifiers in deterministic.items():
        if name == "retain_all":
            continue
        for seed in RANDOM_SEEDS:
            output[f"matched_random:{name}:{seed}"] = matched_random_ids(
                episode, identifiers, seed, name)
    if any(episode["current_session_id"] not in identifiers for identifiers in output.values()):
        raise RuntimeError("physical forgetting rule removed the current update")
    return output


def metric_tuple(value):
    """Registered selection order: joint, update, retention, locality, then NLL."""
    return (value["joint_success"], value["update_accuracy"],
            value["retained_key_accuracy"], value["locality_accuracy"],
            -value["joint_answer_nll"])


def mean_metrics(rows, arm_name):
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "joint_answer_nll")
    return {key: statistics.fmean(row["metrics"][arm_name][key] for row in rows)
            for key in keys}


def select_cosine_threshold(train_rows):
    """Select a cosine threshold inside an outer training fold only."""
    evidence = {str(value): mean_metrics(
        train_rows, f"qwen_cosine_conflict:{value}") for value in COSINE_THRESHOLDS}
    # Higher thresholds win exact ties, conserving more unrelated memory.
    selected = max(COSINE_THRESHOLDS,
                   key=lambda value: metric_tuple(evidence[str(value)]) + (value,))
    return selected, evidence


def select_physical_rule(train_rows):
    """Select the strongest rule family without looking at the held-out fold."""
    threshold, threshold_evidence = select_cosine_threshold(train_rows)
    names = {
        "exact_conflict": "exact_conflict",
        "qwen_cosine_conflict": f"qwen_cosine_conflict:{threshold}",
        "most_recent_same_key": "most_recent_same_key",
    }
    evidence = {family: mean_metrics(train_rows, name) for family, name in names.items()}
    family = max(RULE_FAMILIES, key=lambda name: (
        metric_tuple(evidence[name]) + (-RULE_FAMILIES.index(name),)))
    return {"family": family, "arm_name": names[family],
            "cosine_threshold": threshold if family == "qwen_cosine_conflict" else None,
            "family_evidence": evidence, "threshold_evidence": threshold_evidence}


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_bridge_release(data_root, formal_root, bridge_root):
    """Hard gate model loading on a positive, independently audited bridge."""
    dependencies = verify_release(data_root, formal_root)
    bridge_root = Path(bridge_root)
    summary, audit = read_json(bridge_root / "summary.json"), read_json(
        bridge_root / "audit.json")
    checks = (
        summary.get("protocol") == "v18_qwen_lowrank_fastmemory_bridge_v1",
        summary.get("fastmemory_bridge_supported") is True,
        summary.get("bridge_gate", {}).get("passed") is True,
        summary.get("terminal_status") == "fastmemory_bridge_supported",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(bridge_root / "summary.json"),
        audit.get("recomputed_bridge_gate", {}).get("passed") is True,
        (bridge_root / "runner_exit_code.txt").read_text().strip() == "0",
        (bridge_root / "terminal_status.txt").read_text().strip()
            == "completed_v18_lowrank_fastmemory_bridge",
    )
    if not all(checks):
        raise ValueError("physical rules require an audited positive V18 fast-memory bridge")
    shards = sorted(bridge_root.glob("shard[01].jsonl"))
    if len(shards) != 2:
        raise ValueError("physical rules require both frozen bridge shards")
    return dependencies | {
        "bridge_summary_sha256": sha(bridge_root / "summary.json"),
        "bridge_audit_sha256": sha(bridge_root / "audit.json"),
        "bridge_raw_sha256": {path.name: sha(path) for path in shards},
    }


def key_similarities(factor_cache):
    keys = factor_cache["keys"]
    output = {}
    for left in sorted(keys):
        for right in sorted(keys):
            if left > right:
                continue
            output[(left, right)] = float((keys[left] * keys[right]).sum())
    return output


def _arm_metadata(name):
    if name.startswith("qwen_cosine_conflict:"):
        return {"family": "qwen_cosine_conflict", "threshold": float(name.split(":")[1])}
    if name.startswith("matched_random:"):
        _, reference, *tail = name.split(":")
        seed = int(tail[-1]); reference = ":".join([reference] + tail[:-1])
        return {"family": "matched_random", "matched_rule": reference, "seed": seed}
    return {"family": name}


def bridge_rows(bridge_root):
    rows = []
    for path in sorted(Path(bridge_root).glob("shard[01].jsonl")):
        rows.extend(load_jsonl(path))
    index = {row["episode_id"]: row for row in rows}
    if len(rows) != len(index) or len(rows) != 320:
        raise ValueError("invalid bridge raw-row coverage")
    return index


def bridge_references(row, summary):
    scale = summary["scales"][str(PRIMARY_SCALE)]
    fixed_policy = scale["fixed_selected_by_outer_fold"][str(row["fold"])]
    fixed_arm = next(arm for arm in row["arms"]
                     if arm["arm_id"] == f"fixed:{fixed_policy}")
    selection = next(value for value in scale["oracle_selection"]
                     if value["episode_id"] == row["episode_id"])
    oracle_arm = next(arm for arm in row["arms"]
                      if arm["arm_id"] == selection["oracle_arm_id"])
    return {"fixed_policy": fixed_policy, "fixed": arm_metrics(fixed_arm, PRIMARY_SCALE),
            "oracle_arm_id": oracle_arm["arm_id"],
            "oracle": arm_metrics(oracle_arm, PRIMARY_SCALE)}


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_bridge_release(args.data_root, args.formal_root, args.bridge_root)
    episodes = load_jsonl(Path(args.data_root) / "formal.jsonl")
    selected = [row for index, row in enumerate(episodes) if index % args.num_shards == args.shard]
    source_rows = bridge_rows(args.bridge_root)
    bridge_summary = read_json(Path(args.bridge_root) / "summary.json")
    torch.manual_seed(1943); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_hashes = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    started = time.perf_counter(); records = []; unique_states = reused = 0
    with output.open("x") as sink, torch.no_grad():
        for index, episode in enumerate(selected):
            factor_cache = build_factor_cache(model, tokenizer, episode)
            similarities = key_similarities(factor_cache)
            states = physical_states(episode, similarities)
            scored, arms = {}, []
            for name, identifiers in states.items():
                key = tuple(identifiers)
                if key not in scored:
                    scored[key] = {"source_arm_name": name,
                                   "queries": [score_query(
                                       model, tokenizer, episode, key, spec, factor_cache)
                                               for spec in episode["queries"]]}
                    unique_states += 1
                else:
                    reused += 1
                arms.append({"arm_name": name, **_arm_metadata(name),
                             "memory_session_ids": identifiers,
                             "score_source_arm_name": scored[key]["source_arm_name"],
                             "queries": scored[key]["queries"]})
            bridge = source_rows[episode["episode_id"]]
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "subject_index", "template_group",
                "template_group_index", "fold", "length", "mechanism", "target_key_id",
                "current_session_id")}
            row.update({"arms": arms, "key_similarities": {
                "||".join(pair): value for pair, value in similarities.items()},
                        "bridge_reference": bridge_references(bridge, bridge_summary)})
            sink.write(json.dumps(row) + "\n"); sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    measurements = [query for row in records for arm in row["arms"]
                    if arm["arm_name"] == arm["score_source_arm_name"]
                    for query in arm["queries"]]
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
             "tokenizer_files_sha256": tokenizer_hashes,
             "cosine_thresholds": list(COSINE_THRESHOLDS),
             "random_seeds": list(RANDOM_SEEDS), "primary_scale": PRIMARY_SCALE,
             "fresh_kv": True, "backbone_frozen": True, "external_sets_loaded": False,
             "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    write_json(str(output) + ".audit.json", audit); return audit


def _row_metrics(row):
    row = dict(row)
    row["metrics"] = {arm["arm_name"]: arm_metrics(arm, PRIMARY_SCALE)
                      for arm in row["arms"]}
    return row


def _mean_record_metrics(records, field):
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "joint_answer_nll", "generation_joint_success")
    return {key: statistics.fmean(row[field][key] for row in records) for key in keys}


def _matched_random_metrics(row, reference):
    names = [f"matched_random:{reference}:{seed}" for seed in RANDOM_SEEDS]
    keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
            "locality_accuracy", "joint_answer_nll", "generation_joint_success")
    return {key: statistics.fmean(row["metrics"][name][key] for name in names) for key in keys}


def aggregate(rows):
    rows = [_row_metrics(row) for row in rows]
    selections = {}
    for fold in range(5):
        selections[str(fold)] = select_physical_rule(
            [row for row in rows if row["fold"] != fold])
    records = []
    for row in rows:
        selection = selections[str(row["fold"])]
        physical = row["metrics"][selection["arm_name"]]
        fixed, oracle = row["bridge_reference"]["fixed"], row["bridge_reference"]["oracle"]
        records.append({"episode_id": row["episode_id"], "subject": row["subject"],
                        "template_group": row["template_group"], "fold": row["fold"],
                        "length": row["length"], "mechanism": row["mechanism"],
                        "selected_family": selection["family"],
                        "selected_arm_name": selection["arm_name"],
                        "physical": physical, "matched_random": _matched_random_metrics(
                            row, selection["arm_name"]), "fixed": fixed, "oracle": oracle,
                        "joint_gain_vs_fixed": physical["joint_success"]
                            - fixed["joint_success"]})
    physical = _mean_record_metrics(records, "physical")
    fixed = _mean_record_metrics(records, "fixed")
    oracle = _mean_record_metrics(records, "oracle")
    matched = _mean_record_metrics(records, "matched_random")
    denominator = oracle["joint_success"] - fixed["joint_success"]
    gain_records = [{"subject": row["subject"], "template_group": row["template_group"],
                     "joint_gain": row["joint_gain_vs_fixed"]} for row in records]
    exact = mean_metrics(rows, "exact_conflict")
    recent = mean_metrics(rows, "most_recent_same_key")
    by_fold = {str(fold): statistics.fmean(row["joint_gain_vs_fixed"] for row in records
                                           if row["fold"] == fold) for fold in range(5)}
    return {"primary_scale": PRIMARY_SCALE, "outer_fold_selection": selections,
            "strongest_physical": physical, "matched_random": matched,
            "outer_fold_fixed": fixed, "exact_oracle": oracle,
            "exact_conflict_all_rows": exact, "most_recent_same_key_all_rows": recent,
            "joint_gain_micro": statistics.fmean(row["joint_gain_vs_fixed"] for row in records),
            "joint_gain_subject_macro": clustered_interval(
                gain_records, "joint_gain", "subject", BOOTSTRAP_SEED, BOOTSTRAP_DRAWS),
            "joint_gain_template_macro": clustered_interval(
                gain_records, "joint_gain", "template_group", BOOTSTRAP_SEED + 100,
                BOOTSTRAP_DRAWS),
            "by_fold_gain": by_fold,
            "fixed_to_oracle_gap_closed": ((physical["joint_success"]
                                              - fixed["joint_success"]) / denominator
                                             if denominator > 0 else None),
            "selection_records": records,
            "next_stage": "train_causal_deletion_utility_policy"}


def summarize(args):
    dependencies = verify_bridge_release(args.data_root, args.formal_root, args.bridge_root)
    rows, resources, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path); resource = read_json(str(path) + ".audit.json")
        if resource.get("dependencies") != dependencies or resource.get("rows_sha256") != sha(path):
            raise ValueError("physical-rule shard dependency mismatch")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
    if len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("physical-rule formal coverage mismatch")
    table = aggregate(rows)
    measurements = [query for row in rows for arm in row["arms"]
                    if arm["arm_name"] == arm["score_source_arm_name"]
                    for query in arm["queries"]]
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
              "terminal_status": "physical_conflict_rules_evaluated"}
    write_json(args.output, result); return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for item in (collect_parser,):
        item.add_argument("--data-root", required=True)
        item.add_argument("--formal-root", required=True)
        item.add_argument("--bridge-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--formal-root", required=True)
    summary_parser.add_argument("--bridge-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()

