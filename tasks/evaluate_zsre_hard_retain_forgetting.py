"""Calibration-only replay of same-subject hard-retain ZsRE forgetting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import (
    POLICIES, FrozenScopeScorer, apply_update, query_memory, table_hash,
)


PROTOCOL = "zsre_hard_retain_forgetting_pilot_v1"
QUERY_TYPES = ("current_exact", "current_rephrase", "historical")


def load_calibration(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != "zsre_hard_retain_forgetting_v1": raise ValueError("protocol mismatch")
    path = root / "calibration.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]: raise ValueError("split hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def all_names(sessions):
    names, pairs = set(), set()
    for session in sessions:
        keys = [version["prompt"] for chain in session["chains"] for version in chain["versions"]]
        queries = [text for chain in session["chains"] for version in chain["versions"]
                   for text in (version["prompt"], version["rephrase_prompt"])]
        names.update(keys); names.update(queries)
        pairs.update(PAIR_TEMPLATE.format(key=key, query=query) for key in keys for query in queries)
    return sorted(names), sorted(pairs)


def active_queries(session, active):
    chains = {chain["id"]: chain for chain in session["chains"]}; output = []
    for chain_id, current in active.items():
        chain = chains[chain_id]
        group_id = chain.get("parent_chain_id", chain_id)
        for index in range(current + 1):
            version = chain["versions"][index]
            output.append({"chain_id": chain_id, "active_version": current,
                           "group_id": group_id, "chain_kind": chain["kind"],
                           "type": "current_exact" if index == current else "historical",
                           "query": version["prompt"]})
            output.append({"chain_id": chain_id, "active_version": current,
                           "group_id": group_id, "chain_kind": chain["kind"],
                           "type": "current_rephrase" if index == current else "historical",
                           "query": version["rephrase_prompt"]})
    return output


def replay_session(session, policy, scorer, gate_seed):
    chains = {chain["id"]: chain for chain in session["chains"]}
    memory, active, decisions, queries, sizes = [], {}, [], [], []
    for step, event in enumerate(session["events"]):
        version = chains[event["chain_id"]]["versions"][event["version"]]
        memory, decision = apply_update(policy, memory, event, version, scorer, gate_seed)
        decision.update({"step": step, "chain_id": event["chain_id"], "version": event["version"],
                         "group_id": chains[event["chain_id"]].get("parent_chain_id", event["chain_id"]),
                         "event_kind": event["kind"]})
        decisions.append(decision); active[event["chain_id"]] = event["version"]; sizes.append(len(memory))
        queries.extend(query_memory(query, memory, scorer, gate_seed)
                       for query in active_queries(session, active))
    return {"session_id": session["id"], "split": "calibration", "policy": policy,
            "seed": gate_seed, "decisions": decisions, "queries": queries, "memory_sizes": sizes}


def aggregate(records):
    queries = [query for record in records for query in record["queries"]]
    decisions = [row for record in records for row in record["decisions"]]
    sizes = [size for record in records for size in record["memory_sizes"]]
    def metric(rows):
        return {"queries": len(rows), "accuracy": statistics.fmean(row["correct"] for row in rows),
                "nll": statistics.fmean(row["nll"] for row in rows),
                "stale_read_rate": statistics.fmean(row["stale_read"] for row in rows)}
    result = metric(queries); result["by_type"] = {
        kind: metric([row for row in queries if row["type"] == kind]) for kind in QUERY_TYPES}
    result["by_chain_kind"] = {
        kind: metric([row for row in queries if row["chain_kind"] == kind])
        for kind in ("primary", "auxiliary")}
    result["action_accuracy"] = statistics.fmean(row["correct"] for row in decisions)
    result["by_action"] = {kind: {"events": sum(row["desired"] == kind for row in decisions),
                                  "accuracy": statistics.fmean(row["correct"] for row in decisions
                                                               if row["desired"] == kind)}
                           for kind in ("retain", "local")}
    result["by_event_kind"] = {
        kind: {"events": sum(row["event_kind"] == kind for row in decisions),
               "accuracy": statistics.fmean(row["correct"] for row in decisions
                                              if row["event_kind"] == kind)}
        for kind in ("primary_initial", "hard_retain", "primary_update")}
    result["mean_memory_size"] = statistics.fmean(sizes); result["max_memory_size"] = max(sizes)
    result["actions"] = len(decisions); result["sessions"] = len(records)
    groups = sorted({row["group_id"] for row in queries})
    result["groups"] = {group: metric([row for row in queries if row["group_id"] == group])
                        for group in groups}
    return result


def multi_seed(per_seed):
    return {policy: {"mean_accuracy": statistics.fmean(per_seed[str(s)][policy]["accuracy"] for s in GATE_SEEDS),
                     "min_seed_accuracy": min(per_seed[str(s)][policy]["accuracy"] for s in GATE_SEEDS),
                     "mean_nll": statistics.fmean(per_seed[str(s)][policy]["nll"] for s in GATE_SEEDS),
                     "mean_action_accuracy": statistics.fmean(
                         per_seed[str(s)][policy]["action_accuracy"] for s in GATE_SEEDS),
                     "by_action": {kind: {"accuracy": statistics.fmean(
                         per_seed[str(s)][policy]["by_action"][kind]["accuracy"] for s in GATE_SEEDS)}
                         for kind in ("retain", "local")},
                     "by_event_kind": {kind: {"accuracy": statistics.fmean(
                         per_seed[str(s)][policy]["by_event_kind"][kind]["accuracy"] for s in GATE_SEEDS)}
                         for kind in ("primary_initial", "hard_retain", "primary_update")},
                     "mean_memory_size": statistics.fmean(
                         per_seed[str(s)][policy]["mean_memory_size"] for s in GATE_SEEDS),
                     "max_memory_size": max(per_seed[str(s)][policy]["max_memory_size"] for s in GATE_SEEDS),
                     "by_type": {kind: {"accuracy": statistics.fmean(
                         per_seed[str(s)][policy]["by_type"][kind]["accuracy"] for s in GATE_SEEDS),
                         "nll": statistics.fmean(
                         per_seed[str(s)][policy]["by_type"][kind]["nll"] for s in GATE_SEEDS)}
                         for kind in QUERY_TYPES},
                     "by_chain_kind": {kind: {"accuracy": statistics.fmean(
                         per_seed[str(s)][policy]["by_chain_kind"][kind]["accuracy"] for s in GATE_SEEDS),
                         "nll": statistics.fmean(
                         per_seed[str(s)][policy]["by_chain_kind"][kind]["nll"] for s in GATE_SEEDS)}
                         for kind in ("primary", "auxiliary")},
                     "per_seed": {str(s): per_seed[str(s)][policy] for s in GATE_SEEDS}}
            for policy in POLICIES}


def comparisons(summary, seed=8401):
    fixed = ("fixed_retain", "fixed_global", "fixed_top", "cosine_local")
    best = max(fixed, key=lambda name: summary[name]["mean_accuracy"])
    groups = sorted(summary["learned_local"]["per_seed"][str(GATE_SEEDS[0])]["groups"])
    differences = []
    for group in groups:
        learned = statistics.fmean(summary["learned_local"]["per_seed"][str(s)]["groups"][group]["accuracy"]
                                   for s in GATE_SEEDS)
        baseline = statistics.fmean(summary[best]["per_seed"][str(s)]["groups"][group]["accuracy"]
                                    for s in GATE_SEEDS)
        differences.append(learned - baseline)
    gain = statistics.fmean(differences); interval = _bootstrap(differences, seed)
    oracle_gap = summary["oracle_local"]["mean_accuracy"] - summary["cosine_local"]["mean_accuracy"]
    return {"best_fixed": best, "accuracy_gain": gain,
            "bootstrap_95ci": interval, "paired_subject_units": len(groups),
            "cosine_oracle_gap": oracle_gap,
            "stage_a2_gate": {
                "nontrivial_cosine_oracle_gap": oracle_gap >= .03,
                "learned_beats_best_fixed": gain > 0 and interval[0] > 0,
                "proceed_to_development": oracle_gap >= .03 and gain > 0 and interval[0] > 0}}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, data_manifest = load_calibration(args.data_root)
    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    names, pairs = all_names(sessions); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]},
                                         transforms_artifact["transforms"],
                                         source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    per_seed, raw_hashes = {}, {}
    for seed in GATE_SEEDS:
        per_seed[str(seed)] = {}
        for policy in POLICIES:
            records = [replay_session(session, policy, scorer, seed) for session in sessions]
            path = root / f"calibration_{policy}_seed{seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path); per_seed[str(seed)][policy] = aggregate(records)
    summary = multi_seed(per_seed); comparison = comparisons(summary)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "unique_texts": len(names), "unique_pairs": len(pairs),
                "embedding_hashes": embedding_hashes, "pair_features_sha256": table_hash(pair_features),
                "raw_rows_sha256": raw_hashes,
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": PROTOCOL, "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_only", "policies": list(POLICIES), "gate_seeds": list(GATE_SEEDS),
                "dev_loaded": False, "test_loaded": False, "test_scored": False}
    result = {"protocol": PROTOCOL, "calibration": summary, "comparison": comparison,
              "dev_loaded": False, "test_loaded": False, "test_scored": False,
              "training_seconds": training["seconds"], "terminal_status": "completed_hard_retain_pilot"}
    write_json(root / "training.json", training); write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128); args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
