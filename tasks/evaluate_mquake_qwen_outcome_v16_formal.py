"""Score the frozen nine-arm V16 MQuAKE outcome matrix with Qwen3-4B."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_mquake_qwen_outcome_v16_formal_states import POLICIES
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    load_jsonl, query_specs, score_query, source_index,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_outcome_v16_formal"
QUERY_KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")
PRIMARY_KINDS = frozenset(("primary_old_phrase", "primary_current_phrase"))
FIXED_POLICIES = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1611


def read_json(path):
    return json.loads(Path(path).read_text())


def state_audit_path(state_root):
    state_root = Path(state_root)
    candidates = (state_root / "audit.json", state_root.parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("formal-state audit location is missing or ambiguous")
    return paths[0]


def verify_formal_states(state_root, sample_root):
    state_root, sample_root = Path(state_root), Path(sample_root)
    manifest = read_json(state_root / "manifest.json")
    audit_path = state_audit_path(state_root)
    audit = read_json(audit_path)
    sample = read_json(sample_root / "manifest.json")
    checks = (
        manifest.get("protocol") == "mquake_qwen_outcome_v16_formal_states",
        manifest.get("terminal_status") == "frozen_after_passed_v16_pilot_before_formal_scoring",
        manifest.get("model_scored") is False,
        manifest.get("external_sets_loaded") is False,
        manifest.get("policies") == list(POLICIES),
        manifest.get("formal_states_sha256") == sha(state_root / "formal_states.jsonl"),
        manifest.get("dependencies", {}).get("formal_sample_sha256")
            == sample.get("formal_sha256") == sha(sample_root / "formal.jsonl"),
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(state_root / "manifest.json"),
        audit.get("formal_states_sha256") == sha(state_root / "formal_states.jsonl"),
        audit.get("model_scored") is False,
        audit.get("external_sets_loaded") is False,
    )
    if not all(checks):
        raise ValueError("formal scoring requires frozen, audited V16 states")
    return {
        "state_manifest_sha256": sha(state_root / "manifest.json"),
        "state_audit_sha256": sha(audit_path),
        "formal_states_sha256": sha(state_root / "formal_states.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
    }


def memory_from_state(episode, sample, identifiers):
    visible = episode["sessions"][:sample["query_step"] + 1]
    index = {row["session_id"]: row for row in visible}
    if len(index) != len(visible) or any(identifier not in index for identifier in identifiers):
        raise ValueError("formal state contains duplicate, future, or unknown sessions")
    result = [index[identifier] for identifier in identifiers]
    if [row["session_id"] for row in result] != identifiers:
        raise RuntimeError("formal state order changed")
    return result


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    sample_root, state_root, output = map(Path, (args.sample_root, args.state_root, args.output))
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_formal_states(state_root, sample_root)
    samples = load_jsonl(sample_root / "formal.jsonl")
    states = load_jsonl(state_root / "formal_states.jsonl")
    if [row["episode_id"] for row in samples] != [row["episode_id"] for row in states]:
        raise ValueError("formal state/sample ordering mismatch")
    selected = [(sample, state) for index, (sample, state) in enumerate(zip(samples, states))
                if index % args.num_shards == args.shard]
    episodes, texts = source_index(args.data_root)
    torch.manual_seed(1603)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("formal V16 requires final slow weights with TTT adapters disabled")
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
            if any(state[key] != sample[key] for key in
                   ("episode_id", "subject", "fold", "length", "query_step",
                    "hard_relation_matched")) or tuple(state["policies"]) != POLICIES:
                raise ValueError("formal state identity mismatch")
            episode = episodes[sample["episode_id"]]
            scored, arms = {}, {}
            for policy in POLICIES:
                identifiers = state["policies"][policy]
                key = tuple(identifiers)
                if key not in scored:
                    memory = memory_from_state(episode, sample, identifiers)
                    queries = [score_query(model, tokenizer, memory, texts, spec, generate=True)
                               for spec in query_specs(sample)]
                    scored[key] = {"source_policy": policy, "queries": queries}
                    unique_states_scored += 1
                else:
                    reused_states += 1
                source = scored[key]
                arms[policy] = {
                    "memory_session_ids": identifiers,
                    "score_source_policy": source["source_policy"],
                    "queries": copy.deepcopy(source["queries"]),
                }
            row = {key: sample[key] for key in
                   ("episode_id", "subject", "fold", "length", "hard_relation_matched",
                    "query_step")}
            row["policies"] = arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": sample["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    queries = [query for row in records for policy in POLICIES
               if row["policies"][policy]["score_source_policy"] == policy
               for query in row["policies"][policy]["queries"]]
    audit = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
        "seconds": time.perf_counter() - started,
        "unique_memory_states_scored": unique_states_scored,
        "reused_policy_states": reused_states,
        "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(query["resources"]["gpu_peak_allocated_gib"]
                                        for query in queries),
        "gpu_peak_reserved_gib": max(query["resources"]["gpu_peak_reserved_gib"]
                                       for query in queries),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
        "readout_mode": "plain_ttt_disabled_fresh_kv",
        "candidate_scoring": "one_prompt_forward_with_independent_answer_cache_branches",
        "generation": "greedy_from_independent_shared_prompt_cache",
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "policies": list(POLICIES), "external_sets_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def policy_queries(rows, policy, kinds=None):
    kinds = set(kinds) if kinds is not None else None
    return [(row, query) for row in rows for query in row["policies"][policy]["queries"]
            if kinds is None or query["kind"] in kinds]


def policy_table(rows, policy):
    pairs = policy_queries(rows, policy)
    queries = [query for _, query in pairs]
    subjects = sorted({row["subject"] for row in rows})
    primary = policy_queries(rows, policy, PRIMARY_KINDS)
    pair_correct = []
    for row in rows:
        values = [query["correct"] for query in row["policies"][policy]["queries"]
                  if query["kind"] in PRIMARY_KINDS]
        if len(values) != 2:
            raise ValueError("every row must contain two primary queries")
        pair_correct.append(int(all(values)))
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    memory_sizes = [len(row["policies"][policy]["memory_session_ids"]) for row in rows]
    def metrics(values):
        return {
            "queries": len(values),
            "accuracy": statistics.fmean(query["correct"] for _, query in values),
            "answer_nll": statistics.fmean(query["answer_nll"] for _, query in values),
            "generation_exact": statistics.fmean(query["generation"]["exact"]
                                                   for _, query in values),
            "generation_normalized_exact": statistics.fmean(
                query["generation"]["normalized_exact"] for _, query in values),
        }
    return {
        **metrics(pairs),
        "primary": metrics(primary),
        "primary_pair_accuracy": statistics.fmean(pair_correct),
        "by_kind": {kind: metrics(policy_queries(rows, policy, (kind,)))
                    for kind in QUERY_KINDS},
        "by_fold": {str(fold): metrics(policy_queries(
            [row for row in rows if row["fold"] == fold], policy)) for fold in range(5)},
        "subject_macro": statistics.fmean(statistics.fmean(
            query["correct"] for row, query in pairs if row["subject"] == subject)
            for subject in subjects),
        "memory_sessions": {"mean": statistics.fmean(memory_sizes),
                            "maximum": max(memory_sizes)},
        "max_prompt_tokens": max(query["prompt_tokens"] for query in queries),
        "max_total_tokens": max(query["total_tokens_max"] for query in queries),
        "resources": {
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in queries),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in queries),
        },
    }


def select_fixed_by_outer_fold(rows):
    selected, evidence = {}, {}
    for fold in range(5):
        development = [row for row in rows if row["fold"] != fold]
        table = {}
        for policy in FIXED_POLICIES:
            pairs = policy_queries(development, policy, PRIMARY_KINDS)
            table[policy] = {
                "primary_accuracy": statistics.fmean(query["correct"] for _, query in pairs),
                "primary_answer_nll": statistics.fmean(query["answer_nll"] for _, query in pairs),
                "queries": len(pairs),
            }
        winner = max(FIXED_POLICIES, key=lambda policy: (
            table[policy]["primary_accuracy"], -table[policy]["primary_answer_nll"],
            -FIXED_POLICIES.index(policy)))
        selected[str(fold)] = winner
        evidence[str(fold)] = table
    return selected, evidence


def comparison_records(rows, selected):
    records = []
    for row in rows:
        fixed = selected[str(row["fold"])]
        fixed_queries = {query["kind"]: query for query in row["policies"][fixed]["queries"]}
        v15_queries = {query["kind"]: query for query in row["policies"]["v15"]["queries"]}
        for kind in PRIMARY_KINDS:
            records.append({
                "subject": row["subject"], "fold": row["fold"], "episode_id": row["episode_id"],
                "kind": kind, "fixed_policy": fixed,
                "accuracy_gain": v15_queries[kind]["correct"] - fixed_queries[kind]["correct"],
                "nll_gain": fixed_queries[kind]["answer_nll"] - v15_queries[kind]["answer_nll"],
            })
    return records


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


def aggregate_formal(rows):
    if (len(rows) != 272 or len({row["episode_id"] for row in rows}) != 272
            or tuple(rows[0]["policies"]) != POLICIES):
        raise ValueError("formal row coverage or policy order mismatch")
    policies = {policy: policy_table(rows, policy) for policy in POLICIES}
    selected, evidence = select_fixed_by_outer_fold(rows)
    comparisons = comparison_records(rows, selected)
    accuracy = clustered_interval(comparisons, "accuracy_gain")
    nll = clustered_interval(comparisons, "nll_gain", seed=BOOTSTRAP_SEED + 1)
    by_fold = {}
    for fold in range(5):
        subset = [row for row in comparisons if row["fold"] == fold]
        by_fold[str(fold)] = {
            "fixed_policy": selected[str(fold)], "queries": len(subset),
            "v15_minus_fixed_accuracy": statistics.fmean(row["accuracy_gain"] for row in subset),
            "fixed_minus_v15_nll": statistics.fmean(row["nll_gain"] for row in subset),
        }
    v15 = policies["v15"]
    checks = {
        "v15_primary_accuracy_gain_ci_lower_above_zero": accuracy["mean"] > 0
            and accuracy["lower_95"] > 0,
        "v15_primary_nll_gain_ci_lower_above_zero": nll["mean"] > 0
            and nll["lower_95"] > 0,
        "v15_primary_current_accuracy_at_least_97":
            v15["by_kind"]["primary_current_phrase"]["accuracy"] >= .97,
        "v15_unrelated_accuracy_at_least_97":
            v15["by_kind"]["unrelated"]["accuracy"] >= .97,
        "v15_hard_retain_accuracy_at_least_97":
            v15["by_kind"]["hard_retain"]["accuracy"] >= .97,
        "no_outer_fold_has_negative_accuracy_gain": all(
            value["v15_minus_fixed_accuracy"] >= 0 for value in by_fold.values()),
        "all_scores_finite": all(math.isfinite(query["answer_nll"])
            for row in rows for policy in POLICIES for query in row["policies"][policy]["queries"]),
        "all_prompts_within_checkpoint_limit": max(
            value["max_total_tokens"] for value in policies.values()) <= 40960,
    }
    return {
        "policies": policies,
        "fixed_selection": {"rule": "leave-one-outer-fold-out primary accuracy, then NLL, then registered order",
                            "candidates": list(FIXED_POLICIES), "selected": selected,
                            "development_evidence": evidence},
        "primary_comparison": {"v15_minus_crossfit_fixed_accuracy": accuracy,
                               "crossfit_fixed_minus_v15_nll": nll,
                               "by_fold": by_fold},
        "outcome_gate": {"checks": checks, "passed": all(checks.values())},
    }


def summarize(args):
    sample_root, state_root = Path(args.sample_root), Path(args.state_root)
    dependencies = verify_formal_states(state_root, sample_root)
    bridge = read_json(args.generation_bridge)
    if (bridge.get("protocol") != "mquake_qwen_outcome_v16_generation_bridge"
            or bridge.get("passed") is not True or bridge.get("synthetic_only") is not True
            or bridge.get("experiment_rows_loaded") != 0
            or bridge.get("external_sets_loaded") is not False):
        raise ValueError("formal summary requires a passed no-data generation bridge")
    rows, hashes, resources = [], {}, []
    for value in args.inputs:
        path = Path(value)
        part, resource = load_jsonl(path), read_json(str(path) + ".audit.json")
        if (resource.get("rows") != len(part) or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("formal shard audit mismatch")
        rows.extend(part); hashes[path.name] = sha(path); resources.append(resource)
    expected = load_jsonl(sample_root / "formal.jsonl")
    if (len(rows) != 272 or len({row["episode_id"] for row in rows}) != 272
            or {row["episode_id"] for row in rows} != {row["episode_id"] for row in expected}):
        raise ValueError("formal episode coverage mismatch")
    table = aggregate_formal(rows)
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "resources": resources, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "formal_states_dependencies": dependencies, "external_sets_loaded": False,
        "generation_bridge_sha256": sha(args.generation_bridge),
        "terminal_status": "formal_outcome_passed" if table["outcome_gate"]["passed"]
                           else "formal_outcome_failed",
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("sample-root", "state-root", "data-root", "model", "tokenizer", "output"):
        collect_parser.add_argument(f"--{name}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--sample-root", required=True)
    summary_parser.add_argument("--state-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--generation-bridge", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
