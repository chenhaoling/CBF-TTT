"""Nested-LOSO low-dimensional gate over the frozen cosine update candidate."""

from __future__ import annotations

import argparse
import copy
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
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names, load_calibration, replay_session
from tasks.train_zsre_forgetting_controller import (
    CHECKPOINT_EPOCHS, CONTROLLER_SEEDS, MAX_KEYS, STOPWORDS, choose_inner_tune,
    lexical_features, oracle_event_groups, paired_comparison, replay_controller,
    summarize_seed_metrics,
)


PROTOCOL = "zsre_forgetting_controller_loso_v2"
THRESHOLDS = tuple(value / 20 for value in range(2, 19))


def relation_tokens(text):
    words = re.findall(r"[A-Za-z0-9]+", text)
    return [word.lower() for word in words
            if word.lower() not in STOPWORDS | {"s"} and not word[0].isupper()]


def set_similarity(left, right):
    left, right = set(left), set(right)
    union = len(left | right); smaller = min(len(left), len(right))
    return (len(left & right) / union if union else 0.0,
            len(left & right) / smaller if smaller else 0.0)


def numeric_feature(key, query, scorer):
    import torch

    cosines = []
    for seed in BI_SEEDS:
        table = scorer.embeddings[str(seed)]
        cosines.append(float(table.get("entity", key) @ table.get("entity", query)))
    relation = set_similarity(relation_tokens(key), relation_tokens(query))
    return torch.tensor([*cosines, statistics.fmean(cosines), statistics.pstdev(cosines),
                         *lexical_features(key, query), *relation])


def top_examples(groups, scorer):
    import torch

    values, tops = [], []
    for group in groups:
        features = [numeric_feature(key, group["query"], scorer) for key in group["keys"]]
        scores = torch.tensor([float(feature[:3].mean()) for feature in features])
        top = int(scores.argmax()); tops.append(top); values.append(features[top])
    return torch.stack(values), torch.tensor(tops)


def fit_normalizer(values):
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-4)


class BinaryActionHead:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed); self.linear = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.linear.weight); torch.nn.init.zeros_(self.linear.bias)

    def parameters(self): return self.linear.parameters()
    def __call__(self, values): return self.linear(values).squeeze(-1)
    def state_dict(self): return self.linear.state_dict()
    def load_state_dict(self, state): self.linear.load_state_dict(state)


def labels(groups):
    return __import__("torch").tensor([float(group["target"] != MAX_KEYS) for group in groups])


def decision_metrics(groups, tops, logits, threshold):
    import torch

    probabilities = logits.sigmoid(); delete = probabilities >= threshold
    correct = []
    for group, top, should_delete in zip(groups, tops.tolist(), delete.tolist()):
        correct.append((not should_delete and group["target"] == MAX_KEYS)
                       or (should_delete and group["target"] == top))
    target = labels(groups)
    return {"accuracy": statistics.fmean(correct),
            "binary_nll": float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target))}


def select_model(train_groups, tune_groups, train_raw, tune_raw, train_top, tune_top, seed):
    import torch

    mean, std = fit_normalizer(train_raw); train = (train_raw - mean) / std; tune = (tune_raw - mean) / std
    head = BinaryActionHead(train.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    target = labels(train_groups); candidates, curve = [], []
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True); logits = head(train)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward(); optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            with torch.no_grad(): train_logits, tune_logits = head(train), head(tune)
            for threshold in THRESHOLDS:
                tune_metric = decision_metrics(tune_groups, tune_top, tune_logits, threshold)
                candidates.append((tune_metric["accuracy"], -tune_metric["binary_nll"],
                                   -abs(threshold - .5), -epoch, threshold,
                                   copy.deepcopy(head.state_dict())))
            best_at_epoch = max((row for row in candidates if -row[3] == epoch),
                                key=lambda row: row[:4])
            curve.append({"epoch": epoch, "train": decision_metrics(
                train_groups, train_top, train_logits, best_at_epoch[4]),
                "tune": decision_metrics(tune_groups, tune_top, tune_logits, best_at_epoch[4]),
                "threshold": best_at_epoch[4]})
    selected = max(candidates, key=lambda row: row[:4]); head.load_state_dict(selected[5])
    return head, -selected[3], selected[4], curve


def train_fixed(groups, raw, seed, epochs):
    import torch

    mean, std = fit_normalizer(raw); values = (raw - mean) / std
    head = BinaryActionHead(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    target = labels(groups)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(values), target); loss.backward(); optimizer.step()
    return head, mean, std


class TopGateController:
    def __init__(self, scorer, head, mean, std, threshold):
        self.scorer, self.head, self.mean, self.std, self.threshold = scorer, head, mean, std, threshold

    def predict(self, query, keys):
        import torch

        if not keys: return {"prediction": MAX_KEYS, "probability": 0.0}
        features = [numeric_feature(key, query, self.scorer) for key in keys]
        top = int(torch.tensor([float(feature[:3].mean()) for feature in features]).argmax())
        value = (features[top] - self.mean) / self.std
        with torch.no_grad(): probability = float(self.head(value.unsqueeze(0)).sigmoid()[0])
        return {"prediction": top if probability >= self.threshold else MAX_KEYS,
                "probability": probability}


def run(args, protocol=PROTOCOL, data_protocol="zsre_hard_retain_forgetting_v1",
        output_stem="top_gate", checkpoint_filename="forgetting_top_gate.pt",
        comparison_seed=8801):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, _ = load_calibration(args.data_root, data_protocol); groups = oracle_event_groups(sessions)
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
    values, tops = top_examples(groups, scorer); session_ids = sorted({g["session_id"] for g in groups})
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()

    controller_records = {str(seed): [] for seed in CONTROLLER_SEEDS}
    folds, selected_epochs, selected_thresholds = {}, {str(s): [] for s in CONTROLLER_SEEDS}, {
        str(s): [] for s in CONTROLLER_SEEDS}
    for outer_id in session_ids:
        outer = [i for i, group in enumerate(groups) if group["session_id"] == outer_id]
        remaining_ids = [value for value in session_ids if value != outer_id]
        tune_id = choose_inner_tune(remaining_ids, outer_id)
        inner_train = [i for i, group in enumerate(groups)
                       if group["session_id"] not in (outer_id, tune_id)]
        inner_tune = [i for i, group in enumerate(groups) if group["session_id"] == tune_id]
        outer_train = [i for i, group in enumerate(groups) if group["session_id"] != outer_id]
        folds[outer_id] = {"inner_train_sessions": sorted(set(remaining_ids) - {tune_id}),
                           "inner_tune_session": tune_id,
                           "outer_train_sessions": sorted(remaining_ids), "seeds": {}}
        outer_groups = [groups[i] for i in outer_train]
        for replicate, seed in enumerate(CONTROLLER_SEEDS):
            _, epoch, threshold, curve = select_model(
                [groups[i] for i in inner_train], [groups[i] for i in inner_tune],
                values[inner_train], values[inner_tune], tops[inner_train], tops[inner_tune], seed)
            selected_epochs[str(seed)].append(epoch); selected_thresholds[str(seed)].append(threshold)
            head, mean, std = train_fixed(outer_groups, values[outer_train], seed, epoch)
            controller = TopGateController(scorer, head, mean, std, threshold)
            session = next(row for row in sessions if row["id"] == outer_id)
            controller_records[str(seed)].append(replay_controller(
                session, scorer, GATE_SEEDS[replicate], seed, controller))
            with torch.no_grad(): held_logits = head((values[outer] - mean) / std)
            folds[outer_id]["seeds"][str(seed)] = {"selected_epoch": epoch,
                "selected_threshold": threshold, "inner_curve": curve,
                "heldout_oracle_state": decision_metrics(
                    [groups[i] for i in outer], tops[outer], held_logits, threshold)}

    controller_metrics, raw_hashes = {}, {}
    for seed, records in controller_records.items():
        path = root / f"calibration_loso_{output_stem}_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path); controller_metrics[seed] = aggregate(records)
    baseline_records, baseline_metrics = {}, {}
    for readout_seed in GATE_SEEDS:
        records = [replay_session(session, "cosine_local", scorer, readout_seed) for session in sessions]
        for record in records: record["split"] = "calibration_loso"
        baseline_records[str(readout_seed)] = records; baseline_metrics[str(readout_seed)] = aggregate(records)
        path = root / f"calibration_cosine_local_seed{readout_seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)

    final_heads, final_epochs, final_thresholds = {}, {}, {}
    for seed in CONTROLLER_SEEDS:
        epoch = int(statistics.median(selected_epochs[str(seed)]))
        threshold = statistics.median(selected_thresholds[str(seed)])
        head, mean, std = train_fixed(groups, values, seed, epoch)
        final_heads[str(seed)] = head.state_dict(); final_epochs[str(seed)] = epoch
        final_thresholds[str(seed)] = threshold
    checkpoint = root / checkpoint_filename
    torch.save({"heads": final_heads, "mean": mean, "std": std,
                "selected_epochs": final_epochs, "thresholds": final_thresholds,
                "controller_seeds": CONTROLLER_SEEDS, "feature_dim": values.shape[-1]}, checkpoint)

    summary = {"protocol": protocol, "controller": summarize_seed_metrics(controller_metrics),
               "cosine_local": summarize_seed_metrics(baseline_metrics),
               "comparison": paired_comparison(controller_records, baseline_records, comparison_seed),
               "dev_loaded": False, "test_loaded": False, "test_scored": False,
               "terminal_status": "completed_calibration_nested_loso_v2"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "sessions": len(session_ids), "events_with_candidates": len(groups),
                "unique_texts": len(names), "unique_pairs": len(pairs), "feature_dim": values.shape[-1],
                "folds": folds, "selected_epochs": selected_epochs,
                "selected_thresholds": selected_thresholds, "final_epochs": final_epochs,
                "final_thresholds": final_thresholds, "embedding_hashes": embedding_hashes,
                "pair_features_sha256": table_hash(pair_features), "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "scope_training_sha256": sha(scope / "training.json"),
                "scope_checkpoint_sha256": sha(scope / "scope_gates.pt")}
    manifest = {"protocol": protocol, "data_protocol": data_protocol,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer, "loading_info": model_meta["loading_info"],
                "split": "calibration_nested_loso", "controller_seeds": list(CONTROLLER_SEEDS),
                "readout_seeds": list(GATE_SEEDS), "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "thresholds": list(THRESHOLDS), "outer_folds": session_ids,
                "candidate_selector": "frozen_bi_cosine", "gate_features": ["bi_cosine", "lexical"],
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


if __name__ == "__main__": main()
