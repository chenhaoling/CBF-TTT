"""Compose learned forgetting, natural scope and variable-length Qwen values."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import EXPECTED, load
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    candidate_scores,
    generate_fixed,
    prepare,
    value_rows,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_causal_memory_v1"
ARMS = (
    "learned_auto",
    "cosine_auto",
    "retain_auto",
    "global_auto",
    "oracle_auto",
    "oracle_oracle",
    "oracle_dense",
    "zero",
)
AUTO_POLICIES = {
    "learned_auto": "learned",
    "cosine_auto": "cosine",
    "retain_auto": "retain",
    "global_auto": "global",
    "oracle_auto": "oracle",
}
STATE_POLICIES = ("learned", "cosine", "retain", "global", "oracle")
FIXED_CONTROLS = ("cosine_auto", "retain_auto", "global_auto", "oracle_dense", "zero")
SCALE = 8.0
SCOPE_SEED = 7301
FORGETTING_THRESHOLD = .675
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610171


def chain_map(session):
    return {chain["id"]: chain for chain in session["chains"]}


def primary_count(session):
    return sum(chain["kind"] == "primary" for chain in session["chains"])


def checkpoint_label(step, session):
    count = primary_count(session)
    initial_step = 2 * count - 1
    if step == initial_step:
        return "initial"
    if step > initial_step:
        return f"update_{step - initial_step}"
    return None


def record_for(chain, version):
    group = chain.get("parent_chain_id", chain["id"])
    return {
        "chain_id": chain["id"],
        "group_id": group,
        "chain_kind": chain["kind"],
        "version": version["version"],
        "prompt": version["prompt"],
        "target": version["target"],
    }


def action_correct(event, deleted):
    if event["action"] == "retain":
        return not deleted
    return deleted == [event["chain_id"]]


def advance(policy, memory, event, chain, version, scorer, controller):
    old = list(memory); delete_indices = []; probability = None
    if policy == "global":
        delete_indices = list(range(len(memory)))
    elif policy == "oracle":
        delete_indices = [index for index, row in enumerate(memory)
                          if row["chain_id"] == event["chain_id"]]
    elif memory and policy == "learned":
        result = controller.predict(version["prompt"], [row["prompt"] for row in memory])
        probability = result["probability"]
        if result["prediction"] < len(memory):
            delete_indices = [result["prediction"]]
    elif memory and policy == "cosine":
        result = scorer.score(version["prompt"], [row["prompt"] for row in memory], SCOPE_SEED)
        probability = result["top_score"]
        if result["cosine_retrieve"]:
            delete_indices = [result["top"]]
    elif policy != "retain":
        if policy not in ("learned", "cosine"):
            raise ValueError(f"unknown state policy {policy}")
    deleted = [old[index]["chain_id"] for index in delete_indices]
    removed = set(delete_indices)
    memory = [row for index, row in enumerate(memory) if index not in removed]
    memory.append(record_for(chain, version))
    decision = {
        "event_kind": event["kind"],
        "desired": event["action"],
        "chain_id": event["chain_id"],
        "version": event["version"],
        "before": [row["chain_id"] for row in old],
        "deleted": deleted,
        "after": [row["chain_id"] for row in memory],
        "correct": int(action_correct(event, deleted)),
        "probability": probability,
    }
    return memory, decision


def active_queries(session, active):
    chains = chain_map(session); output = []
    for chain_id in sorted(active):
        chain, version_index = chains[chain_id], active[chain_id]
        version = chain["versions"][version_index]
        group = chain.get("parent_chain_id", chain_id)
        for surface, query in (("exact", version["prompt"]),
                               ("rephrase", version["rephrase_prompt"])):
            output.append({
                "chain_id": chain_id,
                "group_id": group,
                "chain_kind": chain["kind"],
                "active_version": version_index,
                "surface": surface,
                "query": query,
                "answer": version["target"],
            })
    return output


def auto_select(query, memory, scorer):
    if not memory:
        return None, {"retrieve": False, "top": None, "retrieve_probability": 0.0}
    result = scorer.score(query, [row["prompt"] for row in memory], SCOPE_SEED)
    selected = memory[result["top"]] if result["retrieve"] else None
    return selected, {
        "retrieve": bool(result["retrieve"]),
        "top": int(result["top"]),
        "retrieve_probability": float(result["retrieve_probability"]),
    }


def dense_rows(model, prepared, memory):
    import torch

    length = max(len(ids) for ids in prepared["encoded"].values())
    output = torch.zeros((length, model.config.hidden_size), dtype=torch.float32,
                         device=model.device)
    for record in memory:
        ids = prepared["encoded"][record["target"]]
        rows = value_rows(model, ids)
        output[:len(ids)] += rows
    return output


def score_query(model, tokenizer, session, query, phase, states, scorer, final):
    import torch

    started = time.perf_counter()
    prepared = prepare(model, tokenizer, query["query"], session["candidates"])
    max_length = max(len(ids) for ids in prepared["encoded"].values())
    zero = torch.zeros((max_length, model.config.hidden_size), dtype=torch.float32,
                       device=model.device)
    arm_rows = {}
    for arm in ARMS:
        selected = None; scope = {"retrieve": False, "top": None,
                                  "retrieve_probability": 0.0}
        if arm in AUTO_POLICIES:
            memory = states[AUTO_POLICIES[arm]]
            selected, scope = auto_select(query["query"], memory, scorer)
            residual = zero if selected is None else value_rows(
                model, prepared["encoded"][selected["target"]]
            )
        elif arm == "oracle_oracle":
            memory = states["oracle"]
            matches = [row for row in memory if row["chain_id"] == query["chain_id"]
                       and row["version"] == query["active_version"]]
            if len(matches) != 1:
                raise ValueError("oracle scope cannot find one active record")
            selected = matches[0]
            residual = value_rows(model, prepared["encoded"][selected["target"]])
            scope = {"retrieve": True, "top": None, "retrieve_probability": 1.0}
        elif arm == "oracle_dense":
            memory = states["oracle"]
            residual = dense_rows(model, prepared, memory)
            scope = {"retrieve": True, "top": None, "retrieve_probability": 1.0}
        elif arm == "zero":
            memory = []
            residual = zero
        else:
            raise ValueError(f"unknown causal-memory arm {arm}")
        scores = candidate_scores(model, prepared, residual, SCALE)
        prediction = min(session["candidates"], key=lambda value: (scores[value], value))
        selected_target = None if selected is None else selected["target"]
        row = {
            "scores": scores,
            "prediction": prediction,
            "candidate_correct": int(prediction == query["answer"]),
            "answer_nll": scores[query["answer"]],
            "selected_chain_id": None if selected is None else selected["chain_id"],
            "selected_version": None if selected is None else selected["version"],
            "selected_target": selected_target,
            "selection_correct": int(selected_target == query["answer"]),
            "stale_selection": int(selected is not None
                                   and selected["chain_id"] == query["chain_id"]
                                   and selected["version"] != query["active_version"]),
            "wrong_chain_selection": int(selected is not None
                                         and selected["chain_id"] != query["chain_id"]),
            "defer": int(selected is None),
            "memory_size": len(memory),
            "scope": scope,
        }
        if final:
            generation = generate_fixed(
                model, tokenizer, prepared, residual, SCALE,
                len(prepared["encoded"][query["answer"]]),
            )
            generation["token_exact"] = int(
                generation["ids"] == prepared["encoded"][query["answer"]]
            )
            row["generation"] = generation
        arm_rows[arm] = row
    torch.cuda.synchronize()
    return {
        "protocol": PROTOCOL,
        "session_id": session["id"],
        "split": session["split"],
        "phase": phase,
        "final": final,
        **query,
        "candidates": session["candidates"],
        "candidate_token_ids": prepared["encoded"],
        "prompt_tokens": len(prepared["prompt_ids"]),
        "answer_tokens": len(prepared["encoded"][query["answer"]]),
        "arms": arm_rows,
        "seconds": time.perf_counter() - started,
    }


def replay(model, tokenizer, session, scorer, controller):
    chains = chain_map(session)
    states = {policy: [] for policy in STATE_POLICIES}
    active, decisions, snapshots, rows = {}, {policy: [] for policy in STATE_POLICIES}, [], []
    count = primary_count(session)
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        for policy in STATE_POLICIES:
            states[policy], decision = advance(
                policy, states[policy], event, chain, version, scorer, controller
            )
            decision["step"] = step; decisions[policy].append(decision)
        active[event["chain_id"]] = event["version"]
        phase = checkpoint_label(step, session)
        if phase is None:
            continue
        final = phase == f"update_{count}"
        snapshots.append({
            "phase": phase,
            "step": step,
            "active": dict(active),
            "states": {policy: list(memory) for policy, memory in states.items()},
        })
        rows.extend(score_query(
            model, tokenizer, session, query, phase, states, scorer, final
        ) for query in active_queries(session, active))
    return rows, {
        "protocol": PROTOCOL,
        "session_id": session["id"],
        "split": session["split"],
        "decisions": decisions,
        "snapshots": snapshots,
    }


def load_components(args, model, tokenizer, sessions):
    import torch
    from tasks.evaluate_zsre_counterfactual_forgetting_controller import build_scorer
    from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead, TopGateController

    source, scope, controller_root = map(
        Path, (args.context_root, args.scope_root, args.controller_root)
    )
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    controller_training = json.loads((controller_root / "training.json").read_text())
    checkpoint = controller_root / "counterfactual_forgetting_action_oof_gate.pt"
    if controller_training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("forgetting checkpoint mismatch")
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head = BinaryActionHead(artifact["feature_dim"], artifact["training_seed"])
    head.load_state_dict(artifact["head"])
    scorer, feature_meta = build_scorer(
        sessions, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size,
    )
    controller = TopGateController(
        scorer, head, artifact["mean"], artifact["std"], FORGETTING_THRESHOLD
    )
    return scorer, controller, feature_meta


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    paths = [Path(value) for value in (args.output, args.state_output, args.resource_output)]
    if any(path.exists() for path in paths):
        raise FileExistsError("natural causal-memory shard output exists")
    _, sessions = load(args.data_root, args.stage)
    selected = [row for index, row in enumerate(sessions)
                if index % args.num_shards == args.shard]
    torch.manual_seed(202610171)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    scorer, controller, feature_meta = load_components(args, model, tokenizer, selected)
    started = time.perf_counter(); rows, states = [], []
    with paths[0].open("x") as row_sink, paths[1].open("x") as state_sink, torch.no_grad():
        for index, session in enumerate(selected):
            part, state = replay(model, tokenizer, session, scorer, controller)
            for row in part:
                row["shard"] = args.shard
                row_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_sink.flush()
            state["shard"] = args.shard
            state_sink.write(json.dumps(state, ensure_ascii=False) + "\n"); state_sink.flush()
            rows.extend(part); states.append(state)
            print(json.dumps({"stage": args.stage, "shard": args.shard,
                              "completed": index + 1, "total": len(selected),
                              "rows": len(rows)}), flush=True)
    resource = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "sessions": len(selected),
        "rows": len(rows),
        "states": len(states),
        "seconds": time.perf_counter() - started,
        "mean_row_seconds": statistics.fmean(row["seconds"] for row in rows),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(paths[0]),
        "states_sha256": sha(paths[1]),
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "feature_meta": feature_meta,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "scale": SCALE,
        "scope_seed": SCOPE_SEED,
        "forgetting_threshold": FORGETTING_THRESHOLD,
        "calibration_scored": True,
        "development_scored": args.stage == "dev",
        "test_loaded": False,
        "test_scored": False,
        "external_benchmarks_loaded": False,
    }
    write_json(paths[2], resource)
    return resource


def bootstrap(values, seed):
    groups = sorted(values); rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    draws.sort()
    return [draws[max(0, math.ceil(.025 * len(draws)) - 1)],
            draws[min(len(draws) - 1, math.ceil(.975 * len(draws)) - 1)]]


def arm_metric(rows, arm):
    final = [row for row in rows if row["final"]]
    def accuracy(part):
        return statistics.fmean(row["arms"][arm]["candidate_correct"] for row in part)
    return {
        "candidate_accuracy": accuracy(rows),
        "answer_nll": statistics.fmean(row["arms"][arm]["answer_nll"] for row in rows),
        "final_candidate_accuracy": accuracy(final),
        "final_generation_exact": statistics.fmean(
            row["arms"][arm]["generation"]["token_exact"] for row in final
        ),
        "selection_accuracy": statistics.fmean(
            row["arms"][arm]["selection_correct"] for row in rows
        ),
        "stale_selection_rate": statistics.fmean(
            row["arms"][arm]["stale_selection"] for row in rows
        ),
        "wrong_chain_selection_rate": statistics.fmean(
            row["arms"][arm]["wrong_chain_selection"] for row in rows
        ),
        "defer_rate": statistics.fmean(row["arms"][arm]["defer"] for row in rows),
        "mean_memory_size": statistics.fmean(
            row["arms"][arm]["memory_size"] for row in rows
        ),
        "by_surface": {surface: accuracy([row for row in rows if row["surface"] == surface])
                       for surface in ("exact", "rephrase")},
        "by_chain_kind": {kind: accuracy([row for row in rows if row["chain_kind"] == kind])
                          for kind in ("primary", "auxiliary")},
        "by_phase": {phase: accuracy([row for row in rows if row["phase"] == phase])
                     for phase in sorted({row["phase"] for row in rows})},
    }


def comparison(rows, left, right, seed):
    groups = sorted({row["group_id"] for row in rows})
    values = {
        group: statistics.fmean(
            row["arms"][left]["candidate_correct"]
            - row["arms"][right]["candidate_correct"]
            for row in rows if row["group_id"] == group
        ) for group in groups
    }
    return {"candidate_accuracy_gain": statistics.fmean(values.values()),
            "bootstrap_95ci": bootstrap(values, seed), "subject_groups": len(groups)}


def action_metrics(states):
    output = {}
    for policy in STATE_POLICIES:
        decisions = [decision for state in states for decision in state["decisions"][policy]]
        output[policy] = {
            "accuracy": statistics.fmean(row["correct"] for row in decisions),
            "events": len(decisions),
            "by_event_kind": {
                kind: statistics.fmean(row["correct"] for row in decisions
                                       if row["event_kind"] == kind)
                for kind in ("primary_initial", "hard_retain", "primary_update")
            },
        }
    return output


def aggregate(rows, states):
    metrics = {arm: arm_metric(rows, arm) for arm in ARMS}
    strongest = max(FIXED_CONTROLS, key=lambda arm: metrics[arm]["candidate_accuracy"])
    learned = comparison(rows, "learned_auto", strongest, BOOTSTRAP_SEED)
    oracle_controls = {
        arm: comparison(rows, "oracle_auto", arm, BOOTSTRAP_SEED + index + 1)
        for index, arm in enumerate(("retain_auto", "global_auto", "oracle_dense", "zero"))
    }
    checks = {
        "oracle_oracle_overall_at_least_point_95":
            metrics["oracle_oracle"]["candidate_accuracy"] >= .95,
        "oracle_oracle_final_at_least_point_95":
            metrics["oracle_oracle"]["final_candidate_accuracy"] >= .95,
        "oracle_auto_overall_at_least_point_88":
            metrics["oracle_auto"]["candidate_accuracy"] >= .88,
        "oracle_auto_gap_at_most_point_08":
            metrics["oracle_oracle"]["candidate_accuracy"]
            - metrics["oracle_auto"]["candidate_accuracy"] <= .08,
        "learned_overall_at_least_point_68":
            metrics["learned_auto"]["candidate_accuracy"] >= .68,
        "learned_primary_auxiliary_at_least_point_65":
            min(metrics["learned_auto"]["by_chain_kind"].values()) >= .65,
        "learned_final_generation_at_least_point_55":
            metrics["learned_auto"]["final_generation_exact"] >= .55,
        "learned_beats_strongest_fixed_by_point_02":
            learned["candidate_accuracy_gain"] >= .02,
        "learned_gain_ci_lower_above_zero": learned["bootstrap_95ci"][0] > 0,
        "oracle_auto_beats_all_controls": all(
            value["candidate_accuracy_gain"] > 0 for value in oracle_controls.values()
        ),
        "oracle_control_ci_lowers_above_zero": all(
            value["bootstrap_95ci"][0] > 0 for value in oracle_controls.values()
        ),
    }
    return {
        "metrics": metrics,
        "actions": action_metrics(states),
        "strongest_fixed_control": strongest,
        "learned_comparison": learned,
        "oracle_control_comparisons": oracle_controls,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def read_stage(root, stage):
    root = Path(root); rows, states, resources = [], [], []
    expected_sessions = EXPECTED[stage]["sessions"]
    for shard in (0, 1):
        row_path = root / f"{stage}_shard{shard}.jsonl"
        state_path = root / f"{stage}_states_shard{shard}.jsonl"
        resource_path = root / f"{stage}_resource_shard{shard}.json"
        part = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        state_part = [json.loads(line) for line in state_path.read_text().splitlines() if line]
        resource = json.loads(resource_path.read_text())
        expected_part = sum(index % 2 == shard for index in range(expected_sessions))
        if any((resource["rows_sha256"] != sha(row_path),
                resource["states_sha256"] != sha(state_path),
                resource["sessions"] != expected_part,
                resource["rows"] != len(part), resource["states"] != len(state_part),
                len(state_part) != expected_part)):
            raise ValueError("natural causal-memory stage shard mismatch")
        rows.extend(part); states.extend(state_part); resources.append(resource)
    if len(states) != expected_sessions or len({row["session_id"] for row in states}) != expected_sessions:
        raise ValueError("natural causal-memory state coverage mismatch")
    return rows, states, resources


def summarize(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, states, resources = read_stage(args.root, args.stage)
    result = aggregate(rows, states)
    result.update({
        "protocol": PROTOCOL,
        "stage": args.stage,
        "rows": len(rows),
        "sessions": len(states),
        "scale": SCALE,
        "scope_seed": SCOPE_SEED,
        "forgetting_threshold": FORGETTING_THRESHOLD,
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "calibration_scored": True,
        "development_scored": args.stage == "dev",
        "test_loaded": False,
        "test_scored": False,
        "external_benchmarks_loaded": False,
    })
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("data_root", "model", "tokenizer", "context_root", "scope_root",
                 "controller_root", "output", "state_output", "resource_output"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--stage", choices=("calibration", "dev"), required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=128)
    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("--root", required=True)
    summarize_parser.add_argument("--stage", choices=("calibration", "dev"), required=True)
    summarize_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
