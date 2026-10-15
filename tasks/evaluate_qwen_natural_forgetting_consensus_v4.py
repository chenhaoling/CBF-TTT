"""Cross-fit online calibration of registered V4 consensus policies."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_crossfit_v4 import PROTOCOL as CROSS_FIT_PROTOCOL
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    QwenOutcomeCache, SCOPE_SEED, scope_examples,
)
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    SCALE, active_queries, chain_map, checkpoint_label, primary_count, record_for,
)
from tasks.evaluate_qwen_natural_forgetting_online_v3 import (
    EXPECTED_V5, engineered_features,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.evaluate_qwen_natural_value_bridge_v1 import generate_fixed, value_rows
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS, fold_for
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead, FEATURE_SETS
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_consensus_v4"
PLAN = "QWEN_NATURAL_FORGETTING_ONPOLICY_CONSENSUS_V4_PLAN.md"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610261
CONTROLS = ("retain_all", "always_scope_top1", "source_action_rule")
N_GRID = tuple(f"n_b{int(b*100):02d}_r{int(r*1000):03d}"
               for b in (.5, .7, .8) for r in (.05, .1, .2))
INTERSECTIONS = (
    "x_n_b50_r050", "x_n_b50_r100", "x_n_b50_r200",
    "x_n_b70_r100", "x_n_b80_r100",
)
CANDIDATES = ("engineered_base", *N_GRID, *INTERSECTIONS,
              "n_majority", "n_unanimity")
ARMS = (*CONTROLS, *CANDIDATES)


def parse_thresholds(name):
    parts = name.split("_")
    benefit = int(next(value[1:] for value in parts if value.startswith("b"))) / 100
    risk = int(next(value[1:] for value in parts if value.startswith("r"))) / 1000
    return benefit, risk


class FoldController:
    def __init__(self, artifact, outer):
        self.outer = outer; self.models = {}; self.configs = {}
        for feature_set, names in FEATURE_SETS.items():
            fold = artifact["folds"][feature_set][str(outer)]
            self.configs[feature_set] = fold["config"]
            self.models[feature_set] = {}
            for seed in SEEDS:
                row = fold["checkpoints"][str(seed)]
                benefit = BinaryHead.create(len(names), fold["config"]["hidden"], seed)
                risk = BinaryHead.create(len(names), fold["config"]["hidden"], seed + 100_000)
                benefit.load_state_dict(row["benefit_state"]); risk.load_state_dict(row["risk_state"])
                benefit.eval(); risk.eval()
                self.models[feature_set][seed] = {
                    "benefit": benefit, "risk": risk,
                    "benefit_mean": row["benefit_mean"], "benefit_std": row["benefit_std"],
                    "risk_mean": row["risk_mean"], "risk_std": row["risk_std"],
                }

    def probabilities(self, engineered, feature_set):
        import torch
        from tasks.train_qwen_natural_forgetting_factor_gate_v2 import ENGINEERED
        x = torch.tensor([engineered[ENGINEERED.index(name)]
                          for name in FEATURE_SETS[feature_set]], dtype=torch.float32)
        output = {}
        with torch.no_grad():
            for seed, row in self.models[feature_set].items():
                pb = row["benefit"](((x - row["benefit_mean"]) /
                                     row["benefit_std"]).unsqueeze(0)).squeeze().sigmoid()
                pr = row["risk"](((x - row["risk_mean"]) /
                                  row["risk_std"]).unsqueeze(0)).squeeze().sigmoid()
                output[seed] = (float(pb), float(pr))
        return output


def candidate_action(name, engineered, no_schedule):
    e_b = statistics.fmean(value[0] for value in engineered.values())
    e_r = statistics.fmean(value[1] for value in engineered.values())
    n_b = statistics.fmean(value[0] for value in no_schedule.values())
    n_r = statistics.fmean(value[1] for value in no_schedule.values())
    e_pass = e_b >= .8 and e_r <= .1
    if name == "engineered_base": return e_pass
    if name == "n_majority":
        return sum(pb >= .5 and pr <= .2 for pb, pr in no_schedule.values()) >= 2
    if name == "n_unanimity":
        return all(pb >= .5 and pr <= .2 for pb, pr in no_schedule.values())
    benefit, risk = parse_thresholds(name)
    n_pass = n_b >= benefit and n_r <= risk
    return e_pass and n_pass if name.startswith("x_") else n_pass


def decide(arm, memory, event, version, step, event_count, scorer, v5_gate, controller):
    old = [dict(row) for row in memory]
    features, top, v5_probability = engineered_features(
        version["prompt"], old, step, event_count, scorer, v5_gate)
    e = n = {}; delete = False
    if arm == "always_scope_top1": delete = bool(old)
    elif arm == "source_action_rule": delete = bool(old) and event["action"] == "local"
    elif arm in CANDIDATES:
        e = controller.probabilities(features, "engineered")
        n = controller.probabilities(features, "no_schedule")
        delete = bool(old) and candidate_action(arm, e, n)
    elif arm != "retain_all": raise ValueError(arm)
    deleted = [] if not delete else [old[top]]
    updated = [row for index, row in enumerate(old) if not delete or index != top]
    return updated, {
        "arm": arm, "step": step, "event_kind": event["kind"],
        "desired_source_action": event["action"], "incoming_chain_id": event["chain_id"],
        "before_size": len(old), "after_delete_size": len(updated),
        "after_write_size": len(updated) + 1, "deleted": deleted, "top": top,
        "features": features, "v5_probability": v5_probability,
        "engineered_probabilities": {str(k): list(v) for k, v in e.items()},
        "no_schedule_probabilities": {str(k): list(v) for k, v in n.items()},
        "action": "scope_top1_a0" if delete else "retain_all",
    }


def score_checkpoint(model, tokenizer, session, phase, final, recent_chain, queries,
                     states, scorer, v5_gate, v5_threshold, cache, generation_cache):
    import torch
    rows = []
    for query in queries:
        arms = {}
        identity = (phase, query["chain_id"], query["active_version"],
                    query["surface"], query["query"])
        for arm in ARMS:
            memory = states[arm]
            if memory:
                frozen, base = scorer.encode(query["query"],
                    [row["prompt"] for row in memory], SCOPE_SEED)
                with torch.no_grad(): probability = float(v5_gate(base.unsqueeze(0)).sigmoid()[0])
                selected = memory[int(frozen["top"])] if probability >= v5_threshold else None
                top = int(frozen["top"])
            else:
                selected = None; probability = 0.; top = None
            scored = cache.score(identity, query["query"], query["answer"], selected)
            value = {
                **scored, "selected_chain_id": None if selected is None else selected["chain_id"],
                "selected_version": None if selected is None else selected["version"],
                "stale_selection": int(selected is not None and
                    selected["chain_id"] == query["chain_id"] and
                    selected["version"] != query["active_version"]),
                "wrong_chain_selection": int(selected is not None and
                    selected["chain_id"] != query["chain_id"]),
                "defer": int(selected is None), "memory_size": len(memory),
                "scope_top": top, "retrieve_probability": probability,
            }
            if final:
                target = None if selected is None else selected["target"]
                key = (identity, target)
                if key not in generation_cache:
                    prepared = cache.prepared[identity]
                    length = max(len(ids) for ids in prepared["encoded"].values())
                    residual = (torch.zeros((length, model.config.hidden_size),
                        dtype=torch.float32, device=model.device) if selected is None else
                        value_rows(model, prepared["encoded"][target]))
                    generated = generate_fixed(model, tokenizer, prepared, residual, SCALE,
                        len(prepared["encoded"][query["answer"]]))
                    generated["token_exact"] = int(
                        generated["ids"] == prepared["encoded"][query["answer"]])
                    generation_cache[key] = generated
                value["generation"] = generation_cache[key]
            arms[arm] = value
        rows.append({"protocol": PROTOCOL, "session_id": session["id"],
                     "split": "calibration", "outer_fold": fold_for(session["id"]),
                     "phase": phase, "final": final, "recent_chain_id": recent_chain,
                     **query, "arms": arms})
    return rows


def replay(model, tokenizer, session, scorer, v5_gate, v5_threshold, controller):
    chains = chain_map(session); states = {arm: [] for arm in ARMS}
    decisions = {arm: [] for arm in ARMS}; active = {}; rows = []
    cache = QwenOutcomeCache(model, tokenizer, session["candidates"]); generation_cache = {}
    count = primary_count(session)
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        for arm in ARMS:
            memory, decision = decide(arm, states[arm], event, version, step,
                                      len(session["events"]), scorer, v5_gate, controller)
            states[arm] = [*memory, {**record_for(chain, version), "strength": 1.0}]
            decisions[arm].append(decision)
        active[event["chain_id"]] = event["version"]
        phase = checkpoint_label(step, session)
        if phase is not None:
            rows.extend(score_checkpoint(model, tokenizer, session, phase,
                phase == f"update_{count}", event["chain_id"], active_queries(session, active),
                states, scorer, v5_gate, v5_threshold, cache, generation_cache))
    return rows, {"protocol": PROTOCOL, "session_id": session["id"],
                  "outer_fold": fold_for(session["id"]), "decisions": decisions,
                  "cache_hits": cache.hits, "cache_misses": cache.misses}


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.root); outputs = [root / f"rows_shard{args.shard}.jsonl",
        root / f"decisions_shard{args.shard}.jsonl", root / f"resource_shard{args.shard}.json"]
    if any(path.exists() for path in outputs): raise FileExistsError("V4 shard exists")
    crossfit_root = Path(args.crossfit_root)
    crossfit_summary = json.loads((crossfit_root / "result" / "summary.json").read_text())
    crossfit_audit = json.loads((crossfit_root / "execution_audit.json").read_text())
    if crossfit_summary["protocol"] != CROSS_FIT_PROTOCOL \
            or not crossfit_summary["gate"]["online_calibration_authorized"] \
            or not crossfit_audit["passed"]:
        raise ValueError("cross-fit source is not released")
    artifact = torch.load(crossfit_root / "result" / "crossfit_factor_gate.pt",
                          map_location="cpu", weights_only=False)
    controllers = {outer: FoldController(artifact, outer) for outer in range(5)}
    _, sessions = load(args.data_root, "calibration")
    selected = [row for index, row in enumerate(sessions)
                if index % args.num_shards == args.shard]
    torch.manual_seed(BOOTSTRAP_SEED); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    context, scope = Path(args.context_root), Path(args.scope_root)
    source_summary = json.loads((context / "summary.json").read_text())
    transforms = torch.load(context / "transforms.pt", map_location="cpu", weights_only=False)
    scope_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    scorer, feature_meta = build_scorer(scope_examples(selected), model, tokenizer,
        transforms["transforms"], source_summary["selected_config"]["regularizer"],
        scope_artifact, args.batch_size)
    v5_path = Path(args.v5_root) / "result" / "constrained_scope_gate.pt"
    if sha(v5_path) != EXPECTED_V5: raise ValueError("V5 reader changed")
    v5_gate, threshold, selected_reader = load_v5_gate(v5_path, scope_artifact["heads"]["7301"])
    rows = []; traces = []
    with outputs[0].open("x") as row_sink, outputs[1].open("x") as trace_sink, torch.no_grad():
        for index, session in enumerate(selected):
            part, trace = replay(model, tokenizer, session, scorer, v5_gate, threshold,
                                 controllers[fold_for(session["id"])])
            for row in part: row_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_sink.flush(); trace_sink.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_sink.flush(); rows.extend(part); traces.append(trace)
            print(json.dumps({"shard": args.shard, "completed": index + 1,
                              "sessions": len(selected), "rows": len(rows)}), flush=True)
    resource = {"protocol": PROTOCOL, "stage": "crossfit_online_calibration",
        "shard": args.shard, "num_shards": args.num_shards, "sessions": len(selected),
        "rows": len(rows), "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(outputs[0]), "decisions_sha256": sha(outputs[1]),
        "frozen_inputs": {
            "crossfit_checkpoint": sha(crossfit_root / "result" / "crossfit_factor_gate.pt"),
            "crossfit_summary": sha(crossfit_root / "result" / "summary.json"),
            "crossfit_audit": sha(crossfit_root / "execution_audit.json"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
            "v5_checkpoint": sha(v5_path),
            "context_transforms": sha(context / "transforms.pt"),
            "scope_artifact": sha(scope / "scope_gates.pt"),
        },
        "feature_meta": feature_meta, "reader_selected": selected_reader,
        "reader_threshold": threshold, "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False}
    write_json(outputs[2], resource); return resource


def metric(rows, traces, arm):
    final = [row for row in rows if row["final"]]
    actions = [value for trace in traces for value in trace["decisions"][arm]]
    by_fold = {}
    for fold in range(5):
        part = [row for row in rows if row["outer_fold"] == fold]
        by_fold[str(fold)] = statistics.fmean(row["arms"][arm]["correct"] for row in part)
    return {"candidate_accuracy": statistics.fmean(row["arms"][arm]["correct"] for row in rows),
        "answer_nll": statistics.fmean(row["arms"][arm]["answer_nll"] for row in rows),
        "final_candidate_accuracy": statistics.fmean(row["arms"][arm]["correct"] for row in final),
        "final_generation_exact": statistics.fmean(
            row["arms"][arm]["generation"]["token_exact"] for row in final),
        "delete_rate": statistics.fmean(row["action"] == "scope_top1_a0" for row in actions),
        "by_fold_accuracy": by_fold, "queries": len(rows),
        "correct": sum(row["arms"][arm]["correct"] for row in rows),
        "action_distribution": dict(sorted(Counter(row["action"] for row in actions).items()))}


def paired(rows, arm):
    added = recovered = collateral = 0
    for row in rows:
        retain = row["arms"]["retain_all"]["correct"]; value = row["arms"][arm]["correct"]
        added += int(retain and not value); recovered += int(value and not retain)
        collateral += int(retain and not value and row["chain_id"] != row["recent_chain_id"])
    return {"added_errors": added, "recovered_errors": recovered,
            "collateral_added_errors": collateral,
            "recovered_to_added_ratio": recovered / max(1, added)}


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, max(0, math.ceil(q*len(values))-1))]


def bootstrap(rows, arm):
    sessions = sorted({row["session_id"] for row in rows}); effects = {}
    for session in sessions:
        part = [row for row in rows if row["session_id"] == session]
        effects[session] = statistics.fmean(row["arms"][arm]["correct"] -
            row["arms"]["retain_all"]["correct"] for row in part)
    rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [sessions[rng.randrange(len(sessions))] for _ in sessions]
        draws.append(statistics.fmean(effects[value] for value in sample))
    return {"gain": statistics.fmean(effects.values()),
            "bootstrap_95ci": [quantile(draws,.025), quantile(draws,.975)],
            "sessions": len(sessions)}


def aggregate(rows, traces):
    metrics = {arm: metric(rows, traces, arm) for arm in ARMS}
    paired_values = {arm: paired(rows, arm) for arm in ARMS if arm != "retain_all"}
    comparisons = {arm: bootstrap(rows, arm) for arm in CANDIDATES}
    retain = metrics["retain_all"]; search = []
    for arm in CANDIDATES:
        value = metrics[arm]; comparison = comparisons[arm]; account = paired_values[arm]
        fold_gains = {fold: value["by_fold_accuracy"][fold] - retain["by_fold_accuracy"][fold]
                      for fold in value["by_fold_accuracy"]}
        checks = {"gain_at_least_point_05": value["candidate_accuracy"] -
                    retain["candidate_accuracy"] >= .05,
                  "gain_ci_lower_above_zero": comparison["bootstrap_95ci"][0] > 0,
                  "collateral_errors_at_most_four": account["collateral_added_errors"] <= 4,
                  "all_outer_folds_nonnegative": min(fold_gains.values()) >= 0,
                  "delete_rate_between_point_05_and_point_50": .05 <= value["delete_rate"] <= .50,
                  "generation_not_below_retain_by_point_005":
                    value["final_generation_exact"] >= retain["final_generation_exact"] - .005}
        search.append({"candidate": arm, "metrics": value, "paired": account,
                       "comparison": comparison, "fold_gains": fold_gains,
                       "checks": checks, "feasible": all(checks.values())})
    feasible = [row for row in search if row["feasible"]]
    selected = (min(feasible, key=lambda row: (
        -row["metrics"]["candidate_accuracy"], row["paired"]["collateral_added_errors"],
        -row["paired"]["recovered_to_added_ratio"], row["metrics"]["delete_rate"],
        row["candidate"])) if feasible else None)
    return {"metrics": metrics, "paired_accounting": paired_values,
            "comparisons": comparisons, "search": search, "selected": selected,
            "gate": {"online_calibration_passed": selected is not None,
                     "test_authorized": selected is not None}}


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists(): raise FileExistsError(output)
    rows=[]; traces=[]; resources=[]
    for shard in range(args.num_shards):
        rp=root/f"rows_shard{shard}.jsonl"; dp=root/f"decisions_shard{shard}.jsonl"
        resource=json.loads((root/f"resource_shard{shard}.json").read_text())
        if resource["rows_sha256"] != sha(rp) or resource["decisions_sha256"] != sha(dp):
            raise ValueError("V4 shard hash mismatch")
        rows.extend(json.loads(line) for line in rp.read_text().splitlines() if line)
        traces.extend(json.loads(line) for line in dp.read_text().splitlines() if line)
        resources.append(resource)
    rows.sort(key=lambda row:(row["session_id"],row["phase"],row["chain_id"],row["surface"]))
    traces.sort(key=lambda row:row["session_id"])
    if len(rows)!=5144 or len(traces)!=65: raise ValueError("incomplete V4 calibration")
    result=aggregate(rows,traces); result.update({"protocol":PROTOCOL,
        "stage":"crossfit_online_calibration","sessions":65,"rows":len(rows),
        "arms":ARMS,"candidates":CANDIDATES,"resources":resources,
        "rows_sha256":{f"rows_shard{i}.jsonl":sha(root/f"rows_shard{i}.jsonl")
                       for i in range(args.num_shards)},
        "decisions_sha256":{f"decisions_shard{i}.jsonl":sha(root/f"decisions_shard{i}.jsonl")
                       for i in range(args.num_shards)},
        "plan_sha256":sha(PLAN),"calibration_scored":True,
        "development_loaded":False,"test_loaded":False,"external_benchmarks_loaded":False})
    write_json(output,result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    collect_parser=sub.add_parser("collect")
    for name in ("root","crossfit_root","data_root","v5_root","model","tokenizer","context_root","scope_root"):
        collect_parser.add_argument(f"--{name.replace('_','-')}",required=True)
    collect_parser.add_argument("--shard",type=int,required=True); collect_parser.add_argument("--num-shards",type=int,default=2); collect_parser.add_argument("--batch-size",type=int,default=128)
    summary_parser=sub.add_parser("summarize"); summary_parser.add_argument("--root",required=True); summary_parser.add_argument("--output",required=True); summary_parser.add_argument("--num-shards",type=int,default=2)
    args=parser.parse_args(); result=collect(args) if args.command=="collect" else summarize(args)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
