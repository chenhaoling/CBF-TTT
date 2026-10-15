"""Calibration-only causal replay of natural ZsRE sequential forgetting policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.train_zsre_decomposed_scope_gate import BinaryGate, GATE_SEEDS
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features


PROTOCOL = "zsre_natural_forgetting_pilot_v1"
POLICIES = ("fixed_retain", "fixed_global", "fixed_top", "cosine_local",
            "learned_local", "oracle_local")
QUERY_TYPES = ("current_exact", "current_rephrase", "historical", "locality")


def load_calibration(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != "zsre_natural_forgetting_v1": raise ValueError("protocol mismatch")
    path = root / "calibration.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]: raise ValueError("calibration hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def all_names(sessions):
    names = set(); pairs = set()
    for session in sessions:
        keys = [version["prompt"] for chain in session["chains"] for version in chain["versions"]]
        queries = []
        for chain in session["chains"]:
            for version in chain["versions"]:
                queries.extend((version["prompt"], version["rephrase_prompt"], *version["locality_prompts"]))
        names.update(keys); names.update(queries)
        pairs.update(PAIR_TEMPLATE.format(key=key, query=query) for key in keys for query in queries)
    return sorted(names), sorted(pairs)


def table_hash(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode()); h.update(value.contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


class FrozenScopeScorer:
    def __init__(self, embeddings, pair_features, gate_artifact):
        import torch

        self.embeddings = embeddings; self.pair_features = pair_features
        self.geometry_mean = gate_artifact["geometry_mean"]
        self.geometry_std = gate_artifact["geometry_std"]
        self.gate_thresholds = gate_artifact["gate_thresholds"]
        self.control_threshold = gate_artifact["control_threshold"]
        pair_dim = next(iter(pair_features.values())).numel()
        self.heads = {}
        for seed in GATE_SEEDS:
            head = BinaryGate(pair_dim + 4, seed); head.load_state_dict(gate_artifact["heads"][str(seed)])
            self.heads[str(seed)] = head

    def encode(self, query, keys, gate_seed):
        """Return the frozen reader result and its pre-logit feature vector."""
        import torch

        if not keys: return None
        per_seed = []
        for seed in BI_SEEDS:
            table = self.embeddings[str(seed)]; q = table.get("entity", query)
            per_seed.append(torch.stack([table.get("entity", key) for key in keys]) @ q)
        per_seed = torch.stack(per_seed); mean = per_seed.mean(0); top = int(mean.argmax())
        top_values = per_seed[:, top]; own_top = per_seed.argmax(-1)
        if len(keys) > 1:
            top_two = torch.topk(mean, 2).values; margin = top_two[0] - top_two[1]
        else:
            margin = mean[0] + 1.
        geometry = torch.stack((top_values.mean(), top_values.std(unbiased=False), margin,
                                own_top.eq(top).float().mean()))
        standardized = (geometry - self.geometry_mean) / self.geometry_std
        pair = self.pair_features[PAIR_TEMPLATE.format(key=keys[top], query=query)]
        feature = torch.cat((pair, standardized)).unsqueeze(0)
        with torch.no_grad(): probability = float(self.heads[str(gate_seed)](feature).sigmoid()[0])
        result = {"top": top, "mean_scores": mean, "top_score": float(mean[top]),
                  "retrieve_probability": probability,
                  "retrieve": probability >= self.gate_thresholds[str(gate_seed)],
                  "cosine_retrieve": float(mean[top]) >= self.control_threshold}
        return result, feature.squeeze(0)

    def score(self, query, keys, gate_seed):
        encoded = self.encode(query, keys, gate_seed)
        if encoded is None:
            return None
        result, _ = encoded
        return result


def apply_update(policy, memory, event, version, scorer, gate_seed):
    desired = event["action"]; predicted = "retain"; deleted = []; old_memory = list(memory)
    if policy == "fixed_global":
        deleted = list(range(len(memory))); memory = []; predicted = "global"
    elif memory and policy in ("fixed_top", "cosine_local", "learned_local"):
        result = scorer.score(version["prompt"], [row["prompt"] for row in memory], gate_seed)
        should_delete = policy == "fixed_top" or (policy == "cosine_local" and result["cosine_retrieve"]) \
                        or (policy == "learned_local" and result["retrieve"])
        if should_delete:
            deleted = [result["top"]]; memory = [row for index, row in enumerate(memory) if index != result["top"]]
            predicted = "local"
    elif policy == "oracle_local":
        deleted = [index for index, row in enumerate(memory) if row["chain_id"] == event["chain_id"]]
        if deleted:
            memory = [row for index, row in enumerate(memory) if index not in set(deleted)]; predicted = "local"
    deleted_chains = [old_memory[index]["chain_id"] for index in deleted]
    action_correct = (desired == "retain" and not deleted) or \
                     (desired == "local" and predicted == "local" and
                      deleted_chains == [event["chain_id"]])
    memory = [*memory, {"chain_id": event["chain_id"], "version": event["version"],
                        "prompt": version["prompt"], "target": version["target"]}]
    return memory, {"desired": desired, "predicted": predicted,
                    "correct": int(action_correct), "deleted": len(deleted),
                    "deleted_chain_ids": deleted_chains}


def active_queries(session, active):
    output = []
    chains = {chain["id"]: chain for chain in session["chains"]}
    for chain_id, current in active.items():
        chain = chains[chain_id]
        for version_index in range(current + 1):
            version = chain["versions"][version_index]
            kind = "current_exact" if version_index == current else "historical"
            output.append({"chain_id": chain_id, "active_version": current,
                           "type": kind, "query": version["prompt"]})
            kind = "current_rephrase" if version_index == current else "historical"
            output.append({"chain_id": chain_id, "active_version": current,
                           "type": kind, "query": version["rephrase_prompt"]})
        current_row = chain["versions"][current]
        output.extend({"chain_id": chain_id, "active_version": current,
                       "type": "locality", "query": query}
                      for query in current_row["locality_prompts"])
    return output


def query_memory(query, memory, scorer, gate_seed):
    import torch

    result = scorer.score(query["query"], [row["prompt"] for row in memory], gate_seed)
    selected = None if not result["retrieve"] else memory[result["top"]]
    if query["type"] == "locality": correct = selected is None
    else:
        correct = selected is not None and selected["chain_id"] == query["chain_id"] \
                  and selected["version"] == query["active_version"]
    retrieve = torch.tensor(result["retrieve_probability"]).clamp(1e-7, 1 - 1e-7)
    key_probability = torch.softmax(result["mean_scores"] / .07, dim=0)
    if query["type"] == "locality": probability = 1 - retrieve
    else:
        indices = [index for index, row in enumerate(memory)
                   if row["chain_id"] == query["chain_id"] and row["version"] == query["active_version"]]
        probability = retrieve * key_probability[indices].sum() if indices else torch.tensor(1e-7)
    stale = selected is not None and selected["chain_id"] == query["chain_id"] \
            and selected["version"] != query["active_version"]
    return {**query, "prediction": "defer" if selected is None else selected["target"],
            "selected_chain_id": None if selected is None else selected["chain_id"],
            "selected_version": None if selected is None else selected["version"],
            "correct": int(correct), "stale_read": int(stale),
            "nll": float(-probability.clamp_min(1e-12).log()),
            "retrieve_probability": result["retrieve_probability"]}


def replay_session(session, policy, scorer, gate_seed):
    chains = {chain["id"]: chain for chain in session["chains"]}
    memory, active, decisions, queries, sizes = [], {}, [], [], []
    for step, event in enumerate(session["events"]):
        version = chains[event["chain_id"]]["versions"][event["version"]]
        memory, decision = apply_update(policy, memory, event, version, scorer, gate_seed)
        decision.update({"step": step, "chain_id": event["chain_id"], "version": event["version"]})
        decisions.append(decision); active[event["chain_id"]] = event["version"]; sizes.append(len(memory))
        queries.extend(query_memory(query, memory, scorer, gate_seed)
                       for query in active_queries(session, active))
    return {"session_id": session["id"], "split": "calibration", "policy": policy,
            "seed": gate_seed, "decisions": decisions, "queries": queries,
            "memory_sizes": sizes}


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
    result["action_accuracy"] = statistics.fmean(row["correct"] for row in decisions)
    result["actions"] = len(decisions); result["mean_memory_size"] = statistics.fmean(sizes)
    result["max_memory_size"] = max(sizes); result["sessions"] = len(records)
    chain_ids = sorted({row["chain_id"] for row in queries})
    result["groups"] = {chain_id: metric([row for row in queries if row["chain_id"] == chain_id])
                        for chain_id in chain_ids}
    return result


def multi_seed(per_seed):
    return {policy: {"mean_accuracy": statistics.fmean(
                per_seed[str(seed)][policy]["accuracy"] for seed in GATE_SEEDS),
                     "min_seed_accuracy": min(per_seed[str(seed)][policy]["accuracy"] for seed in GATE_SEEDS),
                     "mean_nll": statistics.fmean(per_seed[str(seed)][policy]["nll"] for seed in GATE_SEEDS),
                     "mean_action_accuracy": statistics.fmean(
                         per_seed[str(seed)][policy]["action_accuracy"] for seed in GATE_SEEDS),
                     "mean_memory_size": statistics.fmean(
                         per_seed[str(seed)][policy]["mean_memory_size"] for seed in GATE_SEEDS),
                     "max_memory_size": max(per_seed[str(seed)][policy]["max_memory_size"] for seed in GATE_SEEDS),
                     "by_type": {kind: {"accuracy": statistics.fmean(
                         per_seed[str(seed)][policy]["by_type"][kind]["accuracy"] for seed in GATE_SEEDS),
                         "nll": statistics.fmean(
                         per_seed[str(seed)][policy]["by_type"][kind]["nll"] for seed in GATE_SEEDS)}
                         for kind in QUERY_TYPES},
                     "per_seed": {str(seed): per_seed[str(seed)][policy] for seed in GATE_SEEDS}}
            for policy in POLICIES}


def comparisons(summary, seed=8201):
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
    return {"best_fixed": best, "accuracy_gain": statistics.fmean(differences),
            "bootstrap_95ci": _bootstrap(differences, seed), "chains": len(groups)}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, data_manifest = load_calibration(args.data_root)
    source = Path(args.source_root); scope = Path(args.scope_root)
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
    for gate_seed in GATE_SEEDS:
        per_seed[str(gate_seed)] = {}
        for policy in POLICIES:
            records = [replay_session(session, policy, scorer, gate_seed) for session in sessions]
            path = root / f"calibration_{policy}_seed{gate_seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path); per_seed[str(gate_seed)][policy] = aggregate(records)
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
                "source_protocol": source_summary["protocol"], "scope_protocol": "zsre_decomposed_scope_gate_v1",
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_only", "policies": list(POLICIES), "gate_seeds": list(GATE_SEEDS),
                "test_loaded": False, "test_scored": False}
    result = {"protocol": PROTOCOL, "calibration": summary, "comparison": comparison,
              "dev_loaded": False, "test_loaded": False, "test_scored": False,
              "training_seconds": training["seconds"], "terminal_status": "completed_calibration_pilot"}
    write_json(root / "training.json", training); write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
