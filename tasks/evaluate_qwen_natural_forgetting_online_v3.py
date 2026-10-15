"""Frozen on-policy sequential development evaluation for Factor Gate V2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import EXPECTED as DATA_COUNTS, load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    QwenOutcomeCache,
    SCOPE_SEED,
    scope_examples,
)
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    SCALE,
    active_queries,
    chain_map,
    checkpoint_label,
    primary_count,
    record_for,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.evaluate_qwen_natural_value_bridge_v1 import generate_fixed, value_rows
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import lexical_features
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    BinaryHead,
    ENGINEERED,
    FEATURE_SETS,
    PROTOCOL as FACTOR_PROTOCOL,
    SEEDS,
)
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_online_v3"
PLAN = "QWEN_NATURAL_FORGETTING_ONLINE_V3_PLAN.md"
ARMS = (
    "retain_all", "always_scope_top1", "source_action_rule",
    "factor_ensemble", "factor_seed_9401", "factor_seed_9402",
    "factor_seed_9403", "benefit_only_ensemble", "no_schedule_ensemble",
)
GENERATIVE_ARMS = ("retain_all", "always_scope_top1", "factor_ensemble")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEEDS = {"retain_all": 202610251, "always_scope_top1": 202610252}
EXPECTED_V5 = "3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc"


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def release(args):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    factor_root, repo = Path(args.factor_root), Path(args.repo)
    summary_path = factor_root / "result" / "summary.json"
    audit_path = factor_root / "execution_audit.json"
    checkpoint = factor_root / "result" / "factor_gate.pt"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (summary.get("protocol") != FACTOR_PROTOCOL
            or not summary["gate"]["development_authorized"]
            or not audit.get("passed")
            or (factor_root / "terminal_status.txt").read_text().strip()
            != "completed_qwen_natural_forgetting_factor_gate_v2_development"
            or (factor_root / "runner_exit_code.txt").read_text().strip() != "0"):
        raise ValueError("Factor Gate V2 is not released for development")
    paths = {
        "plan": repo / PLAN,
        "factor_summary": summary_path,
        "factor_audit": audit_path,
        "factor_checkpoint": checkpoint,
        "data_manifest": Path(args.data_root) / "manifest.json",
        "v5_checkpoint": Path(args.v5_root) / "result" / "constrained_scope_gate.pt",
        "context_transforms": Path(args.context_root) / "transforms.pt",
        "scope_artifact": Path(args.scope_root) / "scope_gates.pt",
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes["factor_checkpoint"] != summary["hashes"]["checkpoint"]:
        raise ValueError("Factor Gate V2 checkpoint hash mismatch")
    if hashes["v5_checkpoint"] != EXPECTED_V5:
        raise ValueError("V5 reader hash mismatch")
    root.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL, "stage": "development_release",
        "hashes": hashes, "factor_protocol": summary["protocol"],
        "factor_config": summary["final_configs"], "factor_seeds": summary["seeds"],
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(root / "release_manifest.json", manifest)
    return manifest


class FactorController:
    def __init__(self, artifact):
        if artifact.get("protocol") != FACTOR_PROTOCOL:
            raise ValueError("unexpected factor checkpoint")
        self.configs = artifact["configs"]
        self.models = {}
        for feature_set in FEATURE_SETS:
            config = self.configs[feature_set]
            self.models[feature_set] = {}
            for seed in SEEDS:
                row = artifact["checkpoints"][feature_set][str(seed)]
                benefit = BinaryHead.create(
                    len(FEATURE_SETS[feature_set]), config["hidden"], seed)
                risk = BinaryHead.create(
                    len(FEATURE_SETS[feature_set]), config["hidden"], seed + 100_000)
                benefit.load_state_dict(row["benefit_state"])
                risk.load_state_dict(row["risk_state"])
                benefit.eval(); risk.eval()
                self.models[feature_set][seed] = {
                    "benefit": benefit, "risk": risk,
                    "benefit_mean": row["benefit_mean"],
                    "benefit_std": row["benefit_std"],
                    "risk_mean": row["risk_mean"], "risk_std": row["risk_std"],
                }

    def probabilities(self, engineered, feature_set="engineered"):
        import torch
        names = ENGINEERED
        x = torch.tensor([engineered[names.index(name)]
                          for name in FEATURE_SETS[feature_set]], dtype=torch.float32)
        output = {}
        with torch.no_grad():
            for seed, row in self.models[feature_set].items():
                pb = row["benefit"](
                    ((x - row["benefit_mean"]) / row["benefit_std"]).unsqueeze(0)
                ).squeeze().sigmoid()
                pr = row["risk"](
                    ((x - row["risk_mean"]) / row["risk_std"]).unsqueeze(0)
                ).squeeze().sigmoid()
                output[seed] = (float(pb), float(pr))
        return output


def engineered_features(query, memory, step, event_count, scorer, v5_gate):
    import torch
    position = step / max(1, event_count - 1)
    if not memory:
        values = [0.] * len(ENGINEERED); values[2] = position
        return values, None, 0.0
    frozen, base = scorer.encode(query, [row["prompt"] for row in memory], SCOPE_SEED)
    with torch.no_grad():
        probability = float(v5_gate(base.unsqueeze(0)).sigmoid()[0])
    scores = frozen["mean_scores"].float(); top = int(frozen["top"])
    if len(scores) > 1:
        top_two = torch.topk(scores, 2).values; margin = float(top_two[0] - top_two[1])
    else:
        margin = float(scores[0] + 1.)
    distribution = torch.softmax(scores / .07, dim=0)
    entropy = (0. if len(scores) == 1 else float(
        -(distribution * distribution.clamp_min(1e-12).log()).sum() / math.log(len(scores))))
    lexical = lexical_features(query, memory[top]["prompt"])
    values = [
        1., len(memory) / 12., position,
        (len(memory) - 1 - top) / max(1, len(memory) - 1),
        float(scores[top]), margin, float(scores.mean()),
        float(scores.std(unbiased=False)), entropy, probability, *lexical,
    ]
    return values, top, probability


def decide(arm, memory, event, version, step, event_count, scorer, v5_gate, factor):
    old = [dict(row) for row in memory]
    features, top, v5_probability = engineered_features(
        version["prompt"], old, step, event_count, scorer, v5_gate)
    probabilities = {}; delete = False; feature_set = "engineered"; use_risk = True
    if arm == "always_scope_top1":
        delete = bool(old)
    elif arm == "source_action_rule":
        delete = bool(old) and event["action"] == "local"
    elif arm.startswith("factor_seed_"):
        probabilities = factor.probabilities(features)
        seed = int(arm.rsplit("_", 1)[1]); config = factor.configs[feature_set]
        pb, pr = probabilities[seed]
        delete = bool(old) and pb >= config["benefit_threshold"] \
            and pr <= config["risk_threshold"]
    elif arm in ("factor_ensemble", "benefit_only_ensemble", "no_schedule_ensemble"):
        feature_set = "no_schedule" if arm == "no_schedule_ensemble" else "engineered"
        use_risk = arm != "benefit_only_ensemble"
        probabilities = factor.probabilities(features, feature_set)
        pb = statistics.fmean(row[0] for row in probabilities.values())
        pr = statistics.fmean(row[1] for row in probabilities.values())
        config = factor.configs[feature_set]
        delete = bool(old) and pb >= config["benefit_threshold"] \
            and (not use_risk or pr <= config["risk_threshold"])
    elif arm != "retain_all":
        raise ValueError(f"unknown arm {arm}")
    deleted = [] if not delete else [old[top]]
    updated = [row for index, row in enumerate(old) if not delete or index != top]
    return updated, {
        "arm": arm, "step": step, "event_kind": event["kind"],
        "desired_source_action": event["action"], "incoming_chain_id": event["chain_id"],
        "before_size": len(old), "after_delete_size": len(updated),
        "deleted": deleted, "top": top, "features": features,
        "v5_probability": v5_probability,
        "probabilities": {str(seed): list(value) for seed, value in probabilities.items()},
        "feature_set": feature_set, "use_risk": use_risk,
        "action": "scope_top1_a0" if delete else "retain_all",
    }


def score_checkpoint(model, tokenizer, session, phase, final, recent_chain, queries,
                     states, scorer, v5_gate, v5_threshold, cache):
    import torch
    rows = []
    for query in queries:
        arms = {}
        for arm in ARMS:
            memory = states[arm]
            if memory:
                frozen, base = scorer.encode(
                    query["query"], [row["prompt"] for row in memory], SCOPE_SEED)
                with torch.no_grad():
                    probability = float(v5_gate(base.unsqueeze(0)).sigmoid()[0])
                selected = memory[int(frozen["top"])] if probability >= v5_threshold else None
                top = int(frozen["top"])
            else:
                selected = None; probability = 0.; top = None
            identity = (phase, query["chain_id"], query["active_version"],
                        query["surface"], query["query"])
            scored = cache.score(identity, query["query"], query["answer"], selected)
            value = {
                **scored, "selected_chain_id": None if selected is None else selected["chain_id"],
                "selected_version": None if selected is None else selected["version"],
                "selection_correct": int(selected is not None
                    and selected["chain_id"] == query["chain_id"]
                    and selected["version"] == query["active_version"]),
                "stale_selection": int(selected is not None
                    and selected["chain_id"] == query["chain_id"]
                    and selected["version"] != query["active_version"]),
                "wrong_chain_selection": int(selected is not None
                    and selected["chain_id"] != query["chain_id"]),
                "defer": int(selected is None), "memory_size": len(memory),
                "scope_top": top, "retrieve_probability": probability,
            }
            if final and arm in GENERATIVE_ARMS:
                prepared = cache.prepared[identity]
                length = max(len(ids) for ids in prepared["encoded"].values())
                residual = (torch.zeros((length, model.config.hidden_size),
                    dtype=torch.float32, device=model.device) if selected is None else
                    value_rows(model, prepared["encoded"][selected["target"]]))
                generation = generate_fixed(model, tokenizer, prepared, residual, SCALE,
                    len(prepared["encoded"][query["answer"]]))
                generation["token_exact"] = int(
                    generation["ids"] == prepared["encoded"][query["answer"]])
                value["generation"] = generation
            arms[arm] = value
        rows.append({"protocol": PROTOCOL, "session_id": session["id"],
                     "split": "dev", "phase": phase, "final": final,
                     "recent_chain_id": recent_chain, **query, "arms": arms})
    return rows


def replay(model, tokenizer, session, scorer, v5_gate, v5_threshold, factor):
    chains = chain_map(session); states = {arm: [] for arm in ARMS}
    decisions = {arm: [] for arm in ARMS}; active = {}; rows = []
    cache = QwenOutcomeCache(model, tokenizer, session["candidates"])
    count = primary_count(session)
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        for arm in ARMS:
            memory, decision = decide(
                arm, states[arm], event, version, step, len(session["events"]),
                scorer, v5_gate, factor)
            states[arm] = [*memory, {**record_for(chain, version), "strength": 1.0}]
            decision["after_write_size"] = len(states[arm]); decisions[arm].append(decision)
        active[event["chain_id"]] = event["version"]
        phase = checkpoint_label(step, session)
        if phase is None:
            continue
        rows.extend(score_checkpoint(
            model, tokenizer, session, phase, phase == f"update_{count}",
            event["chain_id"], active_queries(session, active), states,
            scorer, v5_gate, v5_threshold, cache))
    return rows, {"protocol": PROTOCOL, "session_id": session["id"],
                  "split": "dev", "decisions": decisions,
                  "cache_hits": cache.hits, "cache_misses": cache.misses}


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.root); release_manifest = json.loads(
        (root / "release_manifest.json").read_text())
    if release_manifest["protocol"] != PROTOCOL or release_manifest["development_loaded"]:
        raise ValueError("invalid online release manifest")
    outputs = [root / f"rows_shard{args.shard}.jsonl",
               root / f"decisions_shard{args.shard}.jsonl",
               root / f"resource_shard{args.shard}.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("online shard exists")
    _, sessions = load(args.data_root, "dev")
    selected = [row for index, row in enumerate(sessions)
                if index % args.num_shards == args.shard]
    factor_path = Path(args.factor_root) / "result" / "factor_gate.pt"
    factor = FactorController(torch.load(factor_path, map_location="cpu", weights_only=False))
    torch.manual_seed(202610251); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    context_root, scope_root = Path(args.context_root), Path(args.scope_root)
    source_summary = json.loads((context_root / "summary.json").read_text())
    transforms = torch.load(context_root / "transforms.pt", map_location="cpu", weights_only=False)
    scope_artifact = torch.load(scope_root / "scope_gates.pt", map_location="cpu", weights_only=False)
    scorer, feature_meta = build_scorer(
        scope_examples(selected), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact, args.batch_size)
    v5_path = Path(args.v5_root) / "result" / "constrained_scope_gate.pt"
    v5_gate, v5_threshold, v5_selected = load_v5_gate(
        v5_path, scope_artifact["heads"]["7301"])
    rows = []; states = []
    with outputs[0].open("x") as row_sink, outputs[1].open("x") as state_sink, torch.no_grad():
        for index, session in enumerate(selected):
            part, state = replay(
                model, tokenizer, session, scorer, v5_gate, v5_threshold, factor)
            for row in part: row_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_sink.flush(); state_sink.write(json.dumps(state, ensure_ascii=False) + "\n")
            state_sink.flush(); rows.extend(part); states.append(state)
            print(json.dumps({"shard": args.shard, "completed": index + 1,
                              "sessions": len(selected), "rows": len(rows)}), flush=True)
    resource = {
        "protocol": PROTOCOL, "stage": "development", "shard": args.shard,
        "num_shards": args.num_shards, "sessions": len(selected), "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(outputs[0]), "decisions_sha256": sha(outputs[1]),
        "feature_meta": feature_meta, "v5_selected": v5_selected,
        "v5_threshold": v5_threshold, "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
        "development_scored": True, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(outputs[2], resource); return resource


def arm_metric(rows, decisions, arm):
    final = [row for row in rows if row["final"]]
    generated = [row for row in final if "generation" in row["arms"][arm]]
    arm_decisions = [row for state in decisions for row in state["decisions"][arm]]
    return {
        "candidate_accuracy": statistics.fmean(row["arms"][arm]["correct"] for row in rows),
        "answer_nll": statistics.fmean(row["arms"][arm]["answer_nll"] for row in rows),
        "final_candidate_accuracy": statistics.fmean(
            row["arms"][arm]["correct"] for row in final),
        "final_generation_exact": (statistics.fmean(
            row["arms"][arm]["generation"]["token_exact"] for row in generated)
            if generated else None),
        "by_surface": {surface: statistics.fmean(
            row["arms"][arm]["correct"] for row in rows if row["surface"] == surface)
            for surface in ("exact", "rephrase")},
        "stale_reads": sum(row["arms"][arm]["stale_selection"] for row in rows),
        "wrong_chain_reads": sum(row["arms"][arm]["wrong_chain_selection"] for row in rows),
        "defers": sum(row["arms"][arm]["defer"] for row in rows),
        "mean_memory_size": statistics.fmean(row["arms"][arm]["memory_size"] for row in rows),
        "max_memory_size": max(row["arms"][arm]["memory_size"] for row in rows),
        "delete_rate": statistics.fmean(row["action"] == "scope_top1_a0"
                                         for row in arm_decisions),
        "action_distribution": dict(sorted(Counter(row["action"]
                                                     for row in arm_decisions).items())),
        "queries": len(rows), "correct": sum(row["arms"][arm]["correct"] for row in rows),
    }


def paired_accounting(rows, arm):
    added = recovered = collateral = 0
    for row in rows:
        retain = row["arms"]["retain_all"]["correct"]
        value = row["arms"][arm]["correct"]
        added += int(retain and not value); recovered += int(value and not retain)
        collateral += int(retain and not value and row["chain_id"] != row["recent_chain_id"])
    return {"added_errors": added, "recovered_errors": recovered,
            "collateral_added_errors": collateral}


def paired_bootstrap(rows, arm, baseline, seed):
    sessions = sorted({row["session_id"] for row in rows}); by_session = {}
    for session in sessions:
        part = [row for row in rows if row["session_id"] == session]
        by_session[session] = statistics.fmean(
            row["arms"][arm]["correct"] - row["arms"][baseline]["correct"] for row in part)
    gain = statistics.fmean(by_session.values()); rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [sessions[rng.randrange(len(sessions))] for _ in sessions]
        draws.append(statistics.fmean(by_session[value] for value in sample))
    return {"gain": gain, "bootstrap_95ci": [quantile(draws, .025), quantile(draws, .975)],
            "sessions": len(sessions)}


def aggregate(rows, decisions):
    metrics = {arm: arm_metric(rows, decisions, arm) for arm in ARMS}
    paired = {arm: paired_accounting(rows, arm) for arm in ARMS if arm != "retain_all"}
    comparisons = {baseline: paired_bootstrap(
        rows, "factor_ensemble", baseline, BOOTSTRAP_SEEDS[baseline])
        for baseline in BOOTSTRAP_SEEDS}
    primary = metrics["factor_ensemble"]; retain = metrics["retain_all"]
    fixed = metrics["always_scope_top1"]; benefit = metrics["benefit_only_ensemble"]
    checks = {
        "gain_over_retain_at_least_point_0075":
            primary["candidate_accuracy"] - retain["candidate_accuracy"] >= .0075,
        "retain_gain_ci_lower_above_zero":
            comparisons["retain_all"]["bootstrap_95ci"][0] > 0,
        "gain_over_fixed_at_least_point_0025":
            primary["candidate_accuracy"] - fixed["candidate_accuracy"] >= .0025,
        "fixed_gain_ci_lower_above_zero":
            comparisons["always_scope_top1"]["bootstrap_95ci"][0] > 0,
        "collateral_errors_no_more_than_source_rule":
            paired["factor_ensemble"]["collateral_added_errors"]
            <= paired["source_action_rule"]["collateral_added_errors"],
        "risk_gate_reduces_benefit_only_collateral":
            paired["factor_ensemble"]["collateral_added_errors"]
            < paired["benefit_only_ensemble"]["collateral_added_errors"],
        "risk_gate_within_point_005_of_benefit_only":
            primary["candidate_accuracy"] >= benefit["candidate_accuracy"] - .005,
        "generation_not_below_retain_by_point_005":
            primary["final_generation_exact"] >= retain["final_generation_exact"] - .005,
        "all_seed_members_improve_retain": all(
            metrics[f"factor_seed_{seed}"]["candidate_accuracy"]
            > retain["candidate_accuracy"] for seed in SEEDS),
        "delete_rate_between_point_05_and_point_60": .05 <= primary["delete_rate"] <= .60,
    }
    return {"metrics": metrics, "paired_accounting": paired,
            "comparisons": comparisons, "gate": {"checks": checks, "passed": all(checks.values())}}


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    rows = []; decisions = []; resources = []
    for shard in range(args.num_shards):
        row_path = root / f"rows_shard{shard}.jsonl"
        decision_path = root / f"decisions_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        if resource["rows_sha256"] != sha(row_path) \
                or resource["decisions_sha256"] != sha(decision_path):
            raise ValueError("online shard hash mismatch")
        rows.extend(json.loads(line) for line in row_path.read_text().splitlines() if line)
        decisions.extend(json.loads(line) for line in decision_path.read_text().splitlines() if line)
        resources.append(resource)
    rows.sort(key=lambda row: (row["session_id"], row["phase"], row["chain_id"], row["surface"]))
    decisions.sort(key=lambda row: row["session_id"])
    if len(decisions) != DATA_COUNTS["dev"]["sessions"] or len(rows) != 5120:
        raise ValueError("unexpected online development coverage")
    result = aggregate(rows, decisions)
    result.update({
        "protocol": PROTOCOL, "stage": "development", "sessions": len(decisions),
        "rows": len(rows), "arms": ARMS, "generative_arms": GENERATIVE_ARMS,
        "release_manifest_sha256": sha(root / "release_manifest.json"),
        "rows_sha256": {f"rows_shard{i}.jsonl": sha(root / f"rows_shard{i}.jsonl")
                        for i in range(args.num_shards)},
        "decisions_sha256": {f"decisions_shard{i}.jsonl":
                             sha(root / f"decisions_shard{i}.jsonl")
                             for i in range(args.num_shards)},
        "resources": resources, "development_scored": True,
        "test_loaded": False, "external_benchmarks_loaded": False,
    })
    write_json(output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    release_parser = sub.add_parser("release")
    for name in ("output", "repo", "factor_root", "data_root", "v5_root",
                 "context_root", "scope_root"):
        release_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("root", "factor_root", "data_root", "v5_root", "model", "tokenizer",
                 "context_root", "scope_root"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=128)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    result = release(args) if args.command == "release" else (
        collect(args) if args.command == "collect" else summarize(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
