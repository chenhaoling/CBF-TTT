"""Calibration-only counterfactual future-utility labels for session forgetting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    SCALE,
    SCOPE_SEED,
    active_queries,
    chain_map,
    checkpoint_label,
    primary_count,
    record_for,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.evaluate_qwen_natural_value_bridge_v1 import candidate_scores, prepare, value_rows
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_utility_pilot_v1"
PLAN = "QWEN_NATURAL_FORGETTING_UTILITY_PILOT_V1_PLAN.md"
PILOT_SESSIONS = 16
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610221
EXPECTED = {
    "data_manifest": "ceb2c85ae2a01fd6de7ef08d68004e58e43b8ae424708bea4fa313344d1a7ba7",
    "v5_checkpoint": "3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc",
    "transforms": "6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378",
    "scope": "0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619",
}
SCOPES = ("global", "last1", "last2", "scope_top1", "hybrid")
ALPHAS = (0.0, 0.5, 1.0)
UNIQUE_ACTIONS = (
    "retain_all",
    "global_a0", "global_a05",
    "last1_a0", "last1_a05",
    "last2_a0", "last2_a05",
    "scope_top1_a0", "scope_top1_a05",
    "hybrid_a0", "hybrid_a05",
)
ACTION_ORDER = {name: index for index, name in enumerate(UNIQUE_ACTIONS)}


def stable_rank(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def registered_grid():
    rows = []
    for scope in SCOPES:
        for alpha in ALPHAS:
            canonical = ("retain_all" if alpha == 1 else
                         f"{scope}_a{'0' if alpha == 0 else '05'}")
            rows.append({"scope": scope, "alpha": alpha, "canonical": canonical})
    return rows


def select_pilot_sessions(sessions):
    short = [row for row in sessions if primary_count(row) < 4]
    if len(short) > 1:
        raise ValueError("pilot expected at most one short calibration session")
    selected = list(short)
    remaining = [row for row in sessions if row not in selected]
    remaining.sort(key=lambda row: stable_rank(
        "natural-forgetting-utility-pilot-v1", row["id"]
    ))
    selected.extend(remaining[:PILOT_SESSIONS - len(selected)])
    if len(selected) != PILOT_SESSIONS:
        raise ValueError("not enough calibration sessions for pilot")
    return sorted(selected, key=lambda row: row["id"])


def selected_decision_steps(session):
    by_kind = {}
    for step, event in enumerate(session["events"]):
        by_kind.setdefault(event["kind"], []).append(step)
    selected = list(by_kind.get("primary_update", []))
    for kind in ("primary_initial", "hard_retain"):
        candidates = by_kind.get(kind, [])
        if not candidates:
            raise ValueError(f"session lacks {kind} anchor")
        selected.append(min(candidates, key=lambda step: stable_rank(
            PROTOCOL, session["id"], kind, step
        )))
    return sorted(set(selected))


def action_spec(name):
    if name == "retain_all":
        return None, 1.0
    scope, suffix = name.rsplit("_a", 1)
    return scope, 0.0 if suffix == "0" else 0.5


def intervention_indices(scope, memory, incoming_prompt, scorer):
    if not memory or scope is None:
        return []
    if scope == "global":
        return list(range(len(memory)))
    if scope == "last1":
        return [len(memory) - 1]
    if scope == "last2":
        return list(range(max(0, len(memory) - 2), len(memory)))
    encoded = scorer.encode(incoming_prompt, [row["prompt"] for row in memory], SCOPE_SEED)
    top = encoded[0]["top"]
    if scope == "scope_top1":
        return [top]
    if scope == "hybrid":
        return sorted({top, len(memory) - 1})
    raise ValueError(f"unknown intervention scope {scope}")


def apply_intervention(memory, name, incoming_prompt, scorer):
    scope, alpha = action_spec(name)
    old = [dict(row) for row in memory]
    indices = intervention_indices(scope, old, incoming_prompt, scorer)
    selected = set(indices)
    if alpha == 0:
        updated = [row for index, row in enumerate(old) if index not in selected]
    else:
        updated = old
        for index in indices:
            updated[index]["strength"] *= alpha
    return updated, {
        "scope": scope, "alpha": alpha, "indices": indices,
        "affected": [{"chain_id": old[index]["chain_id"],
                      "version": old[index]["version"]} for index in indices],
    }


def append_record(memory, chain, version):
    return [*memory, {**record_for(chain, version), "strength": 1.0}]


def state_before(session, step):
    chains = chain_map(session); memory, active = [], {}
    for index, event in enumerate(session["events"][:step]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        memory = append_record(memory, chain, version)
        active[event["chain_id"]] = event["version"]
    return memory, active


def future_trace(session, step, action, scorer):
    chains = chain_map(session); memory, active = state_before(session, step)
    event = session["events"][step]
    chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
    memory, intervention = apply_intervention(memory, action, version["prompt"], scorer)
    checkpoints = []
    for index in range(step, len(session["events"])):
        current = session["events"][index]
        current_chain = chains[current["chain_id"]]
        current_version = current_chain["versions"][current["version"]]
        memory = append_record(memory, current_chain, current_version)
        active[current["chain_id"]] = current["version"]
        phase = checkpoint_label(index, session)
        if phase is None:
            continue
        checkpoints.append({
            "phase": phase,
            "final": phase == f"update_{primary_count(session)}",
            "memory": [dict(row) for row in memory],
            "memory_mass": sum(row["strength"] for row in memory),
            "queries": active_queries(session, active),
        })
    return checkpoints, intervention


def scope_examples(sessions):
    output = []
    for session in sessions:
        keys = sorted({version["prompt"] for chain in session["chains"]
                       for version in chain["versions"]})
        queries = sorted({text for chain in session["chains"]
                          for version in chain["versions"]
                          for text in (version["prompt"], version["rephrase_prompt"])})
        output.extend({"query": query, "keys": keys} for query in queries)
    return output


def v5_select(query, memory, scorer, gate, threshold):
    import torch

    if not memory:
        return None, {"retrieve": False, "top": None, "probability": 0.0}
    frozen, feature = scorer.encode(query, [row["prompt"] for row in memory], SCOPE_SEED)
    with torch.no_grad():
        probability = float(gate(feature.unsqueeze(0)).sigmoid()[0])
    retrieve = probability >= threshold
    return (memory[frozen["top"]] if retrieve else None), {
        "retrieve": retrieve, "top": frozen["top"], "probability": probability,
    }


class QwenOutcomeCache:
    def __init__(self, model, tokenizer, candidates):
        self.model = model; self.tokenizer = tokenizer; self.candidates = candidates
        self.prepared = {}; self.results = {}; self.hits = 0; self.misses = 0

    def score(self, identity, query, answer, selected):
        import torch

        target = None if selected is None else selected["target"]
        strength = 0.0 if selected is None else float(selected["strength"])
        key = (identity, target, strength)
        if key in self.results:
            self.hits += 1
            return self.results[key]
        self.misses += 1
        if identity not in self.prepared:
            self.prepared[identity] = prepare(
                self.model, self.tokenizer, query, self.candidates
            )
        prepared = self.prepared[identity]
        length = max(len(ids) for ids in prepared["encoded"].values())
        if selected is None:
            residual = torch.zeros(
                (length, self.model.config.hidden_size), dtype=torch.float32,
                device=self.model.device,
            )
        else:
            residual = value_rows(self.model, prepared["encoded"][target])
        scores = candidate_scores(self.model, prepared, residual, SCALE * strength)
        prediction = min(self.candidates, key=lambda value: (scores[value], value))
        result = {"prediction": prediction, "correct": int(prediction == answer),
                  "answer_nll": scores[answer]}
        self.results[key] = result
        return result


def score_action(session, event_step, action, scorer, gate, threshold, cache):
    checkpoints, intervention = future_trace(session, event_step, action, scorer)
    incoming_chain = session["events"][event_step]["chain_id"]
    outcomes, masses = [], []
    for checkpoint in checkpoints:
        masses.append(checkpoint["memory_mass"])
        for query in checkpoint["queries"]:
            selected, selection = v5_select(
                query["query"], checkpoint["memory"], scorer, gate, threshold
            )
            identity = (checkpoint["phase"], query["chain_id"],
                        query["active_version"], query["surface"], query["query"])
            scored = cache.score(identity, query["query"], query["answer"], selected)
            outcomes.append({
                "phase": checkpoint["phase"], "final": checkpoint["final"],
                "chain_id": query["chain_id"], "active_version": query["active_version"],
                "surface": query["surface"], "answer": query["answer"],
                **scored,
                "related": int(query["chain_id"] == incoming_chain),
                "selected_chain_id": None if selected is None else selected["chain_id"],
                "selected_version": None if selected is None else selected["version"],
                "selected_strength": 0.0 if selected is None else selected["strength"],
                "stale": int(selected is not None
                             and selected["chain_id"] == query["chain_id"]
                             and selected["version"] != query["active_version"]),
                "wrong_chain": int(selected is not None
                                   and selected["chain_id"] != query["chain_id"]),
                "defer": int(selected is None),
                "retrieve_probability": selection["probability"],
            })
    related = [row for row in outcomes if row["related"]]
    unrelated = [row for row in outcomes if not row["related"]]
    total_nll = sum(row["answer_nll"] for row in outcomes)
    return {
        "intervention": intervention,
        "queries": len(outcomes),
        "correct": sum(row["correct"] for row in outcomes),
        "candidate_accuracy": statistics.fmean(row["correct"] for row in outcomes),
        "total_answer_nll": total_nll,
        "mean_answer_nll": total_nll / len(outcomes),
        "related_queries": len(related),
        "related_correct": sum(row["correct"] for row in related),
        "unrelated_queries": len(unrelated),
        "unrelated_correct": sum(row["correct"] for row in unrelated),
        "mean_memory_mass": statistics.fmean(masses),
        "stale_reads": sum(row["stale"] for row in outcomes),
        "wrong_chain_reads": sum(row["wrong_chain"] for row in outcomes),
        "defers": sum(row["defer"] for row in outcomes),
        "outcomes": outcomes,
    }


def choose_oracle(actions, constrained=True):
    baseline = actions["retain_all"]
    candidates = list(UNIQUE_ACTIONS)
    if constrained:
        candidates = [name for name in candidates
                      if actions[name]["unrelated_correct"] >= baseline["unrelated_correct"]]
    return max(candidates, key=lambda name: (
        actions[name]["correct"],
        -actions[name]["total_answer_nll"],
        -actions[name]["mean_memory_mass"],
        -ACTION_ORDER[name],
    ))


def build_label(session, event_step, scorer, gate, threshold, cache):
    import torch

    event = session["events"][event_step]
    chains = chain_map(session); version = chains[event["chain_id"]]["versions"][event["version"]]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); hits, misses = cache.hits, cache.misses
    actions = {name: score_action(
        session, event_step, name, scorer, gate, threshold, cache
    ) for name in UNIQUE_ACTIONS}
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    safe = choose_oracle(actions, True); unconstrained = choose_oracle(actions, False)
    return {
        "protocol": PROTOCOL, "session_id": session["id"], "split": "calibration",
        "event_step": event_step, "event_kind": event["kind"],
        "chain_id": event["chain_id"], "version": event["version"],
        "incoming_prompt": version["prompt"], "desired_source_action": event["action"],
        "memory_size_before": len(state_before(session, event_step)[0]),
        "actions": actions, "safe_oracle_action": safe,
        "unconstrained_oracle_action": unconstrained,
        "safe_oracle_nonretain": int(safe != "retain_all"),
        "safe_oracle_gain": (actions[safe]["candidate_accuracy"]
                             - actions["retain_all"]["candidate_accuracy"]),
        "label_seconds": elapsed, "qwen_cache_hits": cache.hits - hits,
        "qwen_cache_misses": cache.misses - misses,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def bootstrap_session(labels):
    by_session = {}
    for session_id in sorted({row["session_id"] for row in labels}):
        part = [row for row in labels if row["session_id"] == session_id]
        safe_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in part)
        retain_correct = sum(row["actions"]["retain_all"]["correct"] for row in part)
        queries = sum(row["actions"]["retain_all"]["queries"] for row in part)
        by_session[session_id] = (safe_correct - retain_correct) / queries
    groups = sorted(by_session); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(by_session[group] for group in sample))
    draws.sort()
    return [quantile(draws, .025), quantile(draws, .975)], by_session


def aggregate(labels, resources):
    safe_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in labels)
    retain_correct = sum(row["actions"]["retain_all"]["correct"] for row in labels)
    queries = sum(row["actions"]["retain_all"]["queries"] for row in labels)
    gain = (safe_correct - retain_correct) / queries
    ci, session_gains = bootstrap_session(labels)
    selected = Counter(row["safe_oracle_action"] for row in labels)
    unrelated_errors = sum(max(
        0, row["actions"]["retain_all"]["unrelated_correct"]
        - row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
    ) for row in labels)
    seconds = [row["label_seconds"] for row in labels]
    max_allocated = max(row["peak_allocated_gib"] for row in labels)
    checks = {
        "at_least_90_labels": len(labels) >= 90,
        "complete_unique_and_registered_grid": all(
            set(row["actions"]) == set(UNIQUE_ACTIONS) for row in labels
        ) and len(registered_grid()) == 15,
        "nonretain_rate_at_least_point_10":
            statistics.fmean(row["safe_oracle_nonretain"] for row in labels) >= .10,
        "safe_oracle_gain_at_least_point_01": gain >= .01,
        "zero_added_unrelated_errors": unrelated_errors == 0,
        "at_least_three_selected_actions": len(selected) >= 3,
        "median_label_seconds_at_most_120": statistics.median(seconds) <= 120,
        "peak_allocated_gib_at_most_28": max_allocated <= 28,
    }
    return {
        "protocol": PROTOCOL,
        "stage": "calibration_pilot",
        "sessions": len({row["session_id"] for row in labels}),
        "labels": len(labels),
        "registered_grid": registered_grid(),
        "unique_actions": list(UNIQUE_ACTIONS),
        "label_distribution": dict(sorted(selected.items())),
        "safe_oracle_nonretain_rate": statistics.fmean(
            row["safe_oracle_nonretain"] for row in labels),
        "retain_candidate_accuracy": retain_correct / queries,
        "safe_oracle_candidate_accuracy": safe_correct / queries,
        "safe_oracle_gain": gain,
        "safe_oracle_gain_session_bootstrap_95ci": ci,
        "session_gains": session_gains,
        "added_unrelated_errors": unrelated_errors,
        "resources": {
            "median_label_seconds": statistics.median(seconds),
            "p90_label_seconds": quantile(seconds, .90),
            "max_label_seconds": max(seconds),
            "peak_allocated_gib": max_allocated,
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in labels),
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "shards": resources,
        },
        "gate": {"checks": checks, "scale_to_formal": all(checks.values()),
                 "pilot_evidence_ready": ci[0] > 0},
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    outputs = [Path(args.output), Path(args.resource_output)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("pilot shard output exists")
    paths = {
        "data_manifest": Path(args.data_root) / "manifest.json",
        "v5_checkpoint": Path(args.v5_root) / "result" / "constrained_scope_gate.pt",
        "transforms": Path(args.context_root) / "transforms.pt",
        "scope": Path(args.scope_root) / "scope_gates.pt",
    }
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED:
        raise ValueError(f"sealed pilot input mismatch: {observed}")
    _, sessions = load(args.data_root, "calibration")
    pilot = select_pilot_sessions(sessions)
    selected = [row for index, row in enumerate(pilot)
                if index % args.num_shards == args.shard]
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    source_summary = json.loads((Path(args.context_root) / "summary.json").read_text())
    transforms = torch.load(paths["transforms"], map_location="cpu", weights_only=False)
    scope_artifact = torch.load(paths["scope"], map_location="cpu", weights_only=False)
    scorer, feature_meta = build_scorer(
        scope_examples(selected), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact,
        args.batch_size,
    )
    gate, threshold, reader_selected = load_v5_gate(
        paths["v5_checkpoint"], scope_artifact["heads"]["7301"]
    )
    if threshold != .30:
        raise ValueError("frozen V5 threshold changed")
    labels = []
    with Path(args.output).open("x") as sink, torch.no_grad():
        for session_index, session in enumerate(selected):
            cache = QwenOutcomeCache(model, tokenizer, session["candidates"])
            for step in selected_decision_steps(session):
                label = build_label(session, step, scorer, gate, threshold, cache)
                label["shard"] = args.shard
                sink.write(json.dumps(label, ensure_ascii=False) + "\n"); sink.flush()
                labels.append(label)
                print(json.dumps({"shard": args.shard, "session": session["id"],
                                  "session_index": session_index + 1,
                                  "sessions": len(selected), "labels": len(labels),
                                  "last_seconds": label["label_seconds"]}), flush=True)
    resource = {
        "protocol": PROTOCOL, "stage": "calibration_pilot", "shard": args.shard,
        "num_shards": args.num_shards, "sessions": len(selected), "labels": len(labels),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "labels_sha256": sha(args.output), "sealed_inputs": observed,
        "plan_sha256": sha(PLAN), "feature_meta": feature_meta,
        "reader_selected": reader_selected, "reader_threshold": threshold,
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(), "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(Path(args.resource_output), resource)
    return resource


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    labels, resources = [], []
    for shard in range(args.num_shards):
        label_path = root / f"labels_shard{shard}.jsonl"
        resource_path = root / f"resource_shard{shard}.json"
        resource = json.loads(resource_path.read_text())
        part = [json.loads(line) for line in label_path.read_text().splitlines() if line]
        if resource["labels_sha256"] != sha(label_path) or resource["labels"] != len(part):
            raise ValueError("pilot shard hash/count mismatch")
        labels.extend(part); resources.append(resource)
    if len({(row["session_id"], row["event_step"]) for row in labels}) != len(labels):
        raise ValueError("duplicate pilot labels")
    result = aggregate(labels, resources)
    result.update({
        "label_files_sha256": {f"labels_shard{index}.jsonl": resource["labels_sha256"]
                               for index, resource in enumerate(resources)},
        "plan_sha256": sha(PLAN), "sealed_inputs": EXPECTED,
    })
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("data_root", "v5_root", "model", "tokenizer", "context_root",
                 "scope_root", "output", "resource_output"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=128)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
