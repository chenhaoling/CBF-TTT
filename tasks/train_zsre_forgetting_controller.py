"""Calibration-only nested-LOSO controller for ZsRE retain/local decisions."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_scope_resolver import MAX_KEYS, PAIR_TEMPLATE, ScopeHead, extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, query_memory, table_hash
from tasks.evaluate_zsre_hard_retain_forgetting import (
    QUERY_TYPES, active_queries, aggregate, all_names, load_calibration, replay_session,
)


PROTOCOL = "zsre_forgetting_controller_loso_v1"
CONTROLLER_SEEDS = (8601, 8602, 8603)
CHECKPOINT_EPOCHS = (10, 25, 50, 100, 200)
STOPWORDS = frozenset("a an and are as at be by did do does for from how in is it of on or the to was were what when where which who whom whose with".split())


def oracle_event_groups(sessions):
    """Build update decisions from oracle memory states; no chain id enters features."""
    groups = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}
        memory = []
        for step, event in enumerate(session["events"]):
            version = chains[event["chain_id"]]["versions"][event["version"]]
            matches = [index for index, row in enumerate(memory)
                       if row["chain_id"] == event["chain_id"]]
            if memory:
                if event["action"] == "local" and len(matches) != 1:
                    raise ValueError("local update must have one oracle predecessor")
                if event["action"] == "retain" and matches:
                    raise ValueError("retain update unexpectedly matches an active chain")
                groups.append({
                    "session_id": session["id"],
                    "group_id": chains[event["chain_id"]].get("parent_chain_id", event["chain_id"]),
                    "step": step,
                    "event_kind": event["kind"],
                    "query": version["prompt"],
                    "keys": [row["prompt"] for row in memory],
                    "target": matches[0] if matches else MAX_KEYS,
                })
            if event["action"] == "local":
                memory = [row for row in memory if row["chain_id"] != event["chain_id"]]
            memory.append({"chain_id": event["chain_id"], "prompt": version["prompt"]})
            if len(memory) > MAX_KEYS:
                raise ValueError("oracle memory exceeds controller candidate budget")
    return groups


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 0.0


def _overlap(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(1, min(len(left), len(right)))


def lexical_features(key, query):
    left, right = _tokens(key), _tokens(query)
    left_content = [token for token in left if token not in STOPWORDS]
    right_content = [token for token in right if token not in STOPWORDS]
    left_bigrams = list(zip(left, left[1:])); right_bigrams = list(zip(right, right[1:]))
    return (
        _jaccard(left, right),
        _overlap(left, right),
        _jaccard(left_content, right_content),
        _overlap(left_content, right_content),
        _jaccard(left_bigrams, right_bigrams),
        SequenceMatcher(None, " ".join(left), " ".join(right)).ratio(),
    )


def pair_feature(key, query, scorer):
    import torch

    hidden = scorer.pair_features[PAIR_TEMPLATE.format(key=key, query=query)]
    cosines = []
    for seed in BI_SEEDS:
        table = scorer.embeddings[str(seed)]
        cosines.append(float(table.get("entity", key) @ table.get("entity", query)))
    numeric = torch.tensor([*cosines, statistics.fmean(cosines),
                            statistics.pstdev(cosines), *lexical_features(key, query)])
    return torch.cat((hidden, numeric))


def feature_tensor(groups, scorer):
    import torch

    rows = []
    for group in groups:
        values = [pair_feature(key, group["query"], scorer) for key in group["keys"]]
        values.extend(torch.zeros_like(values[0]) for _ in range(MAX_KEYS - len(values)))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def candidate_counts(groups):
    return __import__("torch").tensor([len(group["keys"]) for group in groups])


def targets(groups):
    return __import__("torch").tensor([group["target"] for group in groups])


def fit_normalizer(groups, values):
    import torch

    valid = torch.cat([values[index, :len(group["keys"])] for index, group in enumerate(groups)])
    return valid.mean(0), valid.std(0, unbiased=False).clamp_min(1e-4)


def normalize(values, mean, std):
    return (values - mean) / std


def train_epochs(groups, values, seed, epochs, weight_decay=.1):
    import torch

    head = ScopeHead(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=weight_decay)
    labels, counts = targets(groups), candidate_counts(groups)
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(values, counts)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            with torch.no_grad():
                logits = head.logits(values, counts)
                curve.append({"epoch": epoch, "accuracy": float(logits.argmax(-1).eq(labels).float().mean()),
                              "nll": float(torch.nn.functional.cross_entropy(logits, labels))})
    return head, curve


def score_head(groups, values, head):
    import torch

    with torch.no_grad(): logits = head.logits(values, candidate_counts(groups))
    labels = targets(groups); probability = torch.softmax(logits, -1)
    return {"accuracy": float(logits.argmax(-1).eq(labels).float().mean()),
            "nll": float(-probability[torch.arange(len(groups)), labels].clamp_min(1e-12).log().mean())}


def choose_inner_tune(session_ids, outer_id):
    candidates = sorted(session_ids)
    return min(candidates, key=lambda value: hashlib.sha256(
        f"{PROTOCOL}|inner|{outer_id}|{value}".encode()).hexdigest())


def select_epoch(train_groups, tune_groups, train_raw, tune_raw, seed):
    mean, std = fit_normalizer(train_groups, train_raw)
    head = ScopeHead(train_raw.shape[-1], seed)
    import torch

    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    labels, counts = targets(train_groups), candidate_counts(train_groups)
    candidates, curve = [], []
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(normalize(train_raw, mean, std), counts)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            train_metric = score_head(train_groups, normalize(train_raw, mean, std), head)
            tune_metric = score_head(tune_groups, normalize(tune_raw, mean, std), head)
            curve.append({"epoch": epoch, "train": train_metric, "tune": tune_metric})
            candidates.append((tune_metric["accuracy"], -tune_metric["nll"], -epoch))
    selected = max(candidates)
    return -selected[2], curve


class Controller:
    def __init__(self, scorer, head, mean, std):
        self.scorer, self.head, self.mean, self.std = scorer, head, mean, std

    def predict(self, query, keys):
        import torch

        if not keys:
            return {"prediction": MAX_KEYS, "probability": 1.0}
        values = [pair_feature(key, query, self.scorer) for key in keys]
        values.extend(torch.zeros_like(values[0]) for _ in range(MAX_KEYS - len(values)))
        tensor = normalize(torch.stack(values).unsqueeze(0), self.mean, self.std)
        with torch.no_grad():
            logits = self.head.logits(tensor, torch.tensor([len(keys)]))[0]
            probabilities = torch.softmax(logits, -1); prediction = int(logits.argmax())
        return {"prediction": prediction, "probability": float(probabilities[prediction])}


def apply_controller(memory, event, version, controller):
    old_memory = list(memory); result = controller.predict(
        version["prompt"], [row["prompt"] for row in memory])
    prediction = result["prediction"]; deleted = []
    if prediction < len(memory):
        deleted = [prediction]
        memory = [row for index, row in enumerate(memory) if index != prediction]
    deleted_chains = [old_memory[index]["chain_id"] for index in deleted]
    desired = event["action"]
    correct = (desired == "retain" and not deleted) or (
        desired == "local" and deleted_chains == [event["chain_id"]])
    memory = [*memory, {"chain_id": event["chain_id"], "version": event["version"],
                        "prompt": version["prompt"], "target": version["target"]}]
    return memory, {"desired": desired, "predicted": "local" if deleted else "retain",
                    "correct": int(correct), "deleted": len(deleted),
                    "deleted_chain_ids": deleted_chains,
                    "controller_probability": result["probability"]}


def replay_controller(session, scorer, readout_seed, controller_seed, controller,
                      split="calibration_loso"):
    chains = {chain["id"]: chain for chain in session["chains"]}
    memory, active, decisions, queries, sizes = [], {}, [], [], []
    for step, event in enumerate(session["events"]):
        version = chains[event["chain_id"]]["versions"][event["version"]]
        memory, decision = apply_controller(memory, event, version, controller)
        decision.update({"step": step, "chain_id": event["chain_id"], "version": event["version"],
                         "group_id": chains[event["chain_id"]].get("parent_chain_id", event["chain_id"]),
                         "event_kind": event["kind"]})
        decisions.append(decision); active[event["chain_id"]] = event["version"]; sizes.append(len(memory))
        queries.extend(query_memory(query, memory, scorer, readout_seed)
                       for query in active_queries(session, active))
    return {"session_id": session["id"], "split": split,
            "policy": "trained_controller", "seed": controller_seed,
            "readout_seed": readout_seed, "decisions": decisions,
            "queries": queries, "memory_sizes": sizes}


def summarize_seed_metrics(per_seed):
    output = {"mean_accuracy": statistics.fmean(row["accuracy"] for row in per_seed.values()),
              "mean_nll": statistics.fmean(row["nll"] for row in per_seed.values()),
              "mean_action_accuracy": statistics.fmean(
                  row["action_accuracy"] for row in per_seed.values()),
              "mean_memory_size": statistics.fmean(
                  row["mean_memory_size"] for row in per_seed.values())}
    output["min_seed_accuracy"] = min(row["accuracy"] for row in per_seed.values())
    output["by_event_kind"] = {kind: {"accuracy": statistics.fmean(
        row["by_event_kind"][kind]["accuracy"] for row in per_seed.values())}
        for kind in ("primary_initial", "hard_retain", "primary_update")}
    output["by_chain_kind"] = {kind: {"accuracy": statistics.fmean(
        row["by_chain_kind"][kind]["accuracy"] for row in per_seed.values())}
        for kind in ("primary", "auxiliary")}
    output["by_type"] = {kind: {"accuracy": statistics.fmean(
        row["by_type"][kind]["accuracy"] for row in per_seed.values())}
        for kind in QUERY_TYPES}
    output["per_seed"] = per_seed
    return output


def paired_comparison(controller_records, baseline_records, seed=8701):
    group_ids = sorted({row["group_id"] for records in controller_records.values()
                        for record in records for row in record["queries"]})
    query_differences, action_differences = [], []
    for group_id in group_ids:
        learned_query = statistics.fmean(statistics.fmean(
            row["correct"] for record in records for row in record["queries"]
            if row["group_id"] == group_id) for records in controller_records.values())
        baseline_query = statistics.fmean(statistics.fmean(
            row["correct"] for record in records for row in record["queries"]
            if row["group_id"] == group_id) for records in baseline_records.values())
        learned_action_rows = [row for records in controller_records.values()
                               for record in records for row in record["decisions"]
                               if row["group_id"] == group_id]
        baseline_action_rows = [row for records in baseline_records.values()
                                for record in records for row in record["decisions"]
                                if row["group_id"] == group_id]
        query_differences.append(learned_query - baseline_query)
        action_differences.append(statistics.fmean(row["correct"] for row in learned_action_rows)
                                  - statistics.fmean(row["correct"] for row in baseline_action_rows))
    query_ci = _bootstrap(query_differences, seed); action_ci = _bootstrap(action_differences, seed + 1)
    return {"paired_subject_units": len(group_ids),
            "query_accuracy_gain": statistics.fmean(query_differences),
            "query_bootstrap_95ci": query_ci,
            "action_accuracy_gain": statistics.fmean(action_differences),
            "action_bootstrap_95ci": action_ci,
            "development_gate": {"query_gain_positive": query_ci[0] > 0,
                                 "action_gain_positive": action_ci[0] > 0,
                                 "passed": query_ci[0] > 0 and action_ci[0] > 0}}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, _ = load_calibration(args.data_root); groups = oracle_event_groups(sessions)
    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    names, pairs = all_names(sessions); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]},
                                         transforms_artifact["transforms"],
                                         source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    values = feature_tensor(groups, scorer); session_ids = sorted({group["session_id"] for group in groups})
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    controller_records = {str(seed): [] for seed in CONTROLLER_SEEDS}
    folds, selected_epochs = {}, {str(seed): [] for seed in CONTROLLER_SEEDS}
    for outer_id in session_ids:
        outer = [index for index, group in enumerate(groups) if group["session_id"] == outer_id]
        remaining_ids = [value for value in session_ids if value != outer_id]
        tune_id = choose_inner_tune(remaining_ids, outer_id)
        inner_train = [index for index, group in enumerate(groups)
                       if group["session_id"] not in (outer_id, tune_id)]
        inner_tune = [index for index, group in enumerate(groups) if group["session_id"] == tune_id]
        outer_train = [index for index, group in enumerate(groups) if group["session_id"] != outer_id]
        folds[outer_id] = {"inner_train_sessions": sorted(set(remaining_ids) - {tune_id}),
                           "inner_tune_session": tune_id,
                           "outer_train_sessions": sorted(remaining_ids), "seeds": {}}
        train_groups = [groups[index] for index in outer_train]
        mean, std = fit_normalizer(train_groups, values[outer_train])
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            epoch, curve = select_epoch(
                [groups[index] for index in inner_train], [groups[index] for index in inner_tune],
                values[inner_train], values[inner_tune], seed)
            selected_epochs[str(seed)].append(epoch)
            head, final_curve = train_epochs(train_groups, normalize(values[outer_train], mean, std),
                                             seed, epoch)
            controller = Controller(scorer, head, mean, std)
            session = next(row for row in sessions if row["id"] == outer_id)
            readout_seed = GATE_SEEDS[replicate]
            controller_records[str(seed)].append(
                replay_controller(session, scorer, readout_seed, seed, controller))
            folds[outer_id]["seeds"][str(seed)] = {
                "selected_epoch": epoch, "inner_curve": curve, "refit_curve": final_curve,
                "heldout_event_metric": score_head(
                    [groups[index] for index in outer], normalize(values[outer], mean, std), head)}

    controller_metrics, raw_hashes = {}, {}
    for seed, records in controller_records.items():
        path = root / f"calibration_loso_controller_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path); controller_metrics[seed] = aggregate(records)

    baseline_records = {}
    baseline_metrics = {}
    for replicate, readout_seed in enumerate(GATE_SEEDS):
        records = [replay_session(session, "cosine_local", scorer, readout_seed) for session in sessions]
        for record in records: record["split"] = "calibration_loso"
        baseline_records[str(readout_seed)] = records; baseline_metrics[str(readout_seed)] = aggregate(records)
        path = root / f"calibration_cosine_local_seed{readout_seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)

    final_mean, final_std = fit_normalizer(groups, values)
    final_heads, final_epoch = {}, {}
    for seed in CONTROLLER_SEEDS:
        epoch = int(statistics.median(selected_epochs[str(seed)])); final_epoch[str(seed)] = epoch
        head, _ = train_epochs(groups, normalize(values, final_mean, final_std), seed, epoch)
        final_heads[str(seed)] = head.state_dict()
    checkpoint = root / "forgetting_controller.pt"
    torch.save({"heads": final_heads, "mean": final_mean, "std": final_std,
                "selected_epochs": final_epoch, "controller_seeds": CONTROLLER_SEEDS,
                "feature_dim": values.shape[-1], "max_keys": MAX_KEYS}, checkpoint)

    controller_summary = summarize_seed_metrics(controller_metrics)
    baseline_summary = summarize_seed_metrics(baseline_metrics)
    comparison = paired_comparison(controller_records, baseline_records)
    summary = {"protocol": PROTOCOL, "controller": controller_summary,
               "cosine_local": baseline_summary, "comparison": comparison,
               "dev_loaded": False, "test_loaded": False, "test_scored": False,
               "terminal_status": "completed_calibration_nested_loso"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "sessions": len(session_ids), "events_with_candidates": len(groups),
                "unique_texts": len(names), "unique_pairs": len(pairs), "feature_dim": values.shape[-1],
                "folds": folds, "selected_epochs": selected_epochs, "final_epochs": final_epoch,
                "embedding_hashes": embedding_hashes, "pair_features_sha256": table_hash(pair_features),
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": PROTOCOL, "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_nested_loso", "controller_seeds": list(CONTROLLER_SEEDS),
                "readout_seeds": list(GATE_SEEDS), "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "outer_folds": session_ids, "feature_sources": ["frozen_pair_hidden", "bi_cosine", "lexical"],
                "chain_identity_in_features": False, "dev_loaded": False,
                "test_loaded": False, "test_scored": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128); args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
