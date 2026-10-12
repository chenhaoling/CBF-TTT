"""Train a decomposed retrieve/defer gate over frozen ZsRE candidate retrieval."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract, load_split, texts
from tasks.train_zsre_scope_resolver import (
    CHECKPOINT_EPOCHS, MAX_KEYS, PAIR_TEMPLATE, QUERY_TYPES, THRESHOLDS,
    extract_pair_features, feature_tensor, pair_texts, query_groups, split_calibration,
    tensor_hash,
)


GATE_SEEDS = (7301, 7302, 7303)
ARMS = ("ensemble_threshold", "decomposed_gate")


def ensemble_geometry(groups, embeddings):
    import torch

    all_scores = []
    for seed in BI_SEEDS:
        rows = []
        table = embeddings[str(seed)]
        for group in groups:
            query = table.get("entity", group["query"])
            valid = torch.stack([table.get("entity", key) for key in group["keys"]]) @ query
            rows.append(torch.cat((valid, torch.full((MAX_KEYS - len(valid),), float("-inf")))))
        all_scores.append(torch.stack(rows))
    per_seed = torch.stack(all_scores)
    mean_scores = per_seed.mean(0)
    top = mean_scores.argmax(-1)
    top_by_seed = per_seed.argmax(-1)
    row = torch.arange(len(groups)); top_seed_scores = per_seed[:, row, top]
    finite = mean_scores.clone(); finite[~torch.isfinite(finite)] = -1e9
    top_two = torch.topk(finite, k=2, dim=1).values
    geometry = torch.stack((top_seed_scores.mean(0), top_seed_scores.std(0, unbiased=False),
                            top_two[:, 0] - top_two[:, 1],
                            top_by_seed.eq(top[None, :]).float().mean(0)), dim=1)
    return mean_scores, top, geometry


def compose_features(groups, pair_values, top, geometry, geometry_mean=None, geometry_std=None):
    import torch

    selected = pair_values[torch.arange(len(groups)), top]
    if geometry_mean is None:
        geometry_mean = geometry.mean(0)
        geometry_std = geometry.std(0, unbiased=False).clamp_min(1e-6)
    standardized = (geometry - geometry_mean) / geometry_std
    return torch.cat((selected, standardized), dim=1), geometry_mean, geometry_std


class BinaryGate:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(dim, 1)
        torch.nn.init.normal_(self.linear.weight, std=.001); torch.nn.init.zeros_(self.linear.bias)

    def parameters(self): return self.linear.parameters()
    def __call__(self, values): return self.linear(values).squeeze(-1)
    def state_dict(self): return self.linear.state_dict()
    def load_state_dict(self, state): self.linear.load_state_dict(state)


def retrieval_labels(groups):
    return __import__("torch").tensor([float(group["target"] != MAX_KEYS) for group in groups])


def final_accuracy(groups, top, retrieve_probability, threshold):
    import torch

    decisions = retrieve_probability >= threshold
    predictions = torch.where(decisions, top, torch.full_like(top, MAX_KEYS))
    targets = torch.tensor([group["target"] for group in groups])
    return float(predictions.eq(targets).float().mean())


def select_probability_threshold(groups, top, probabilities):
    values = {str(threshold): final_accuracy(groups, top, probabilities, threshold)
              for threshold in THRESHOLDS}
    best = max(values.values())
    selected = min((float(key) for key, value in values.items() if value == best),
                   key=lambda value: (abs(value - .5), value))
    return selected, values


def train_gate(train_groups, tune_groups, train_values, tune_values,
               train_top, tune_top, seed, epochs=200):
    import torch

    head = BinaryGate(train_values.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=1e-4)
    train_labels, tune_labels = retrieval_labels(train_groups), retrieval_labels(tune_groups)
    curve, candidates = [], []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = head(train_values)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(head.parameters()), 1.); optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            with torch.no_grad():
                train_logits, tune_logits = head(train_values), head(tune_values)
                train_prob, tune_prob = train_logits.sigmoid(), tune_logits.sigmoid()
                threshold, grid = select_probability_threshold(tune_groups, tune_top, tune_prob)
                train_accuracy = final_accuracy(train_groups, train_top, train_prob, threshold)
                tune_accuracy = final_accuracy(tune_groups, tune_top, tune_prob, threshold)
                train_nll = float(torch.nn.functional.binary_cross_entropy_with_logits(
                    train_logits, train_labels)); tune_nll = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(tune_logits, tune_labels))
            curve.append({"epoch": epoch, "threshold": threshold,
                          "train_accuracy": train_accuracy, "train_binary_nll": train_nll,
                          "tune_accuracy": tune_accuracy, "tune_binary_nll": tune_nll,
                          "tune_threshold_grid": grid})
            candidates.append((tune_accuracy, -tune_nll, -epoch, threshold,
                               copy.deepcopy(head.state_dict())))
    selected = max(candidates, key=lambda row: row[:3]); head.load_state_dict(selected[4])
    return head, curve, -selected[2], selected[3]


def select_ensemble_threshold(tune_groups, mean_scores, top):
    import torch

    maximum = mean_scores[torch.arange(len(tune_groups)), top]
    values = {str(threshold): final_accuracy(tune_groups, top,
                                             (maximum >= threshold).float(), .5)
              for threshold in THRESHOLDS}
    best = max(values.values())
    selected = min((float(key) for key, value in values.items() if value == best),
                   key=lambda value: (abs(value - .5), value))
    return selected, values


def records_from_gate(groups, mean_scores, top, retrieve_probability, threshold, seed, arm):
    import torch

    valid_scores = mean_scores / .07
    key_log_probability = torch.log_softmax(valid_scores, dim=1)
    retrieve_probability = retrieve_probability.clamp(1e-7, 1 - 1e-7)
    log_probability = torch.cat((key_log_probability + retrieve_probability.log()[:, None],
                                 (1 - retrieve_probability).log()[:, None]), dim=1)
    decisions = retrieve_probability >= threshold
    predictions = torch.where(decisions, top, torch.full_like(top, MAX_KEYS))
    records = {}
    for index, group in enumerate(groups):
        target = group["target"]; scores = log_probability[index]
        order = torch.argsort(scores, descending=True).tolist()
        alternatives = torch.cat((scores[:target], scores[target + 1:]))
        query = {"subject": group["subject"], "type": group["type"], "query": group["query"],
                 "target": target, "candidate": int(top[index]), "prediction": int(predictions[index]),
                 "retrieve_probability": float(retrieve_probability[index]),
                 "correct": int(predictions[index] == target), "nll": float(-scores[target]),
                 "mrr": 1. / (order.index(target) + 1),
                 "margin": float(scores[target] - alternatives.max())}
        records.setdefault(group["session_id"], []).append(query)
    return [{"session_id": session_id, "split": groups[0]["split"], "seed": seed,
             "arm": arm, "queries": records[session_id]} for session_id in sorted(records)]


def aggregate(records):
    pairs = [(record, query) for record in records for query in record["queries"]]
    def metric(rows):
        return {"queries": len(rows), "accuracy": statistics.fmean(q["correct"] for _, q in rows),
                "nll": statistics.fmean(q["nll"] for _, q in rows),
                "mrr": statistics.fmean(q["mrr"] for _, q in rows),
                "margin": statistics.fmean(q["margin"] for _, q in rows)}
    result = metric(pairs); result["sessions"] = len(records)
    result["by_type"] = {kind: metric([(r, q) for r, q in pairs if q["type"] == kind])
                         for kind in QUERY_TYPES}
    result["groups"] = {record["session_id"]: metric(
        [(r, q) for r, q in pairs if r["session_id"] == record["session_id"]]) for record in records}
    return result


def summarize(per_seed):
    return {"mean_accuracy": statistics.fmean(row["accuracy"] for row in per_seed.values()),
            "min_seed_accuracy": min(row["accuracy"] for row in per_seed.values()),
            "mean_nll": statistics.fmean(row["nll"] for row in per_seed.values()),
            "by_type": {kind: {"accuracy": statistics.fmean(
                row["by_type"][kind]["accuracy"] for row in per_seed.values()),
                "nll": statistics.fmean(row["by_type"][kind]["nll"] for row in per_seed.values())}
                for kind in QUERY_TYPES}, "per_seed": per_seed}


def compare(records, seed=7401):
    categories = {"overall": lambda q: True,
                  **{kind: lambda q, value=kind: q["type"] == value for kind in QUERY_TYPES}}
    sessions = sorted({row["session_id"] for rows in records["decomposed_gate"].values() for row in rows})
    output = {}
    for name, predicate in categories.items():
        differences = []
        for session_id in sessions:
            primary = statistics.fmean(statistics.fmean(
                q["correct"] for q in next(row for row in records["decomposed_gate"][str(rep_seed)]
                                             if row["session_id"] == session_id)["queries"] if predicate(q))
                for rep_seed in GATE_SEEDS)
            control_row = next(row for row in records["ensemble_threshold"]["ensemble"]
                               if row["session_id"] == session_id)
            control = statistics.fmean(q["correct"] for q in control_row["queries"] if predicate(q))
            differences.append(primary - control)
        output[name] = {"accuracy_gain": statistics.fmean(differences),
                        "bootstrap_95ci": _bootstrap(differences, seed), "sessions": len(differences)}
        seed += 1
    return output


def development_gate(primary, comparison):
    checks = {"exact": primary["by_type"]["exact"]["accuracy"] >= .95,
              "rephrase": primary["by_type"]["rephrase"]["accuracy"] >= .90,
              "locality": primary["by_type"]["locality"]["accuracy"] >= .92,
              "min_seed_overall": primary["min_seed_accuracy"] >= .93,
              "control_margin": comparison["overall"]["accuracy_gain"] >= .01,
              "control_confidence": comparison["overall"]["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def build_split_inputs(sessions, split, model, tokenizer, transforms, regularizer,
                       batch_size, geometry_mean=None, geometry_std=None):
    groups = query_groups(sessions, split)
    pair_names = pair_texts(groups); pair_raw = extract_pair_features(model, tokenizer, pair_names, batch_size)
    pair_values = feature_tensor(groups, pair_raw)
    names = texts(sessions); raw = extract(model, tokenizer, names, batch_size)
    embeddings, embedding_hashes = apply(raw, names, transforms, regularizer)
    mean_scores, top, geometry = ensemble_geometry(groups, embeddings)
    gate_values, geometry_mean, geometry_std = compose_features(
        groups, pair_values, top, geometry, geometry_mean, geometry_std)
    return {"groups": groups, "mean_scores": mean_scores, "top": top,
            "gate_values": gate_values, "geometry_mean": geometry_mean,
            "geometry_std": geometry_std, "pair_sha256": tensor_hash(pair_values),
            "gate_sha256": tensor_hash(gate_values), "embedding_hashes": embedding_hashes}


def evaluate_split(root, split, values, heads, gate_thresholds, control_threshold):
    import torch

    groups, mean_scores, top = values["groups"], values["mean_scores"], values["top"]
    maximum = mean_scores[torch.arange(len(groups)), top]
    control_probability = torch.sigmoid((maximum - control_threshold) / .05)
    control_rows = records_from_gate(groups, mean_scores, top, control_probability, .5,
                                     "ensemble", "ensemble_threshold")
    records = {"ensemble_threshold": {"ensemble": control_rows}, "decomposed_gate": {}}
    per_seed = {"ensemble": aggregate(control_rows)}; raw_hashes = {}
    path = root / f"{split}_ensemble_threshold.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in control_rows)); raw_hashes[path.name] = sha(path)
    summaries = {"ensemble_threshold": summarize(per_seed)}; per_seed = {}
    for seed in GATE_SEEDS:
        with torch.no_grad(): probability = heads[str(seed)](values["gate_values"]).sigmoid()
        rows = records_from_gate(groups, mean_scores, top, probability, gate_thresholds[str(seed)],
                                 seed, "decomposed_gate")
        records["decomposed_gate"][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
        path = root / f"{split}_decomposed_gate_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)); raw_hashes[path.name] = sha(path)
    summaries["decomposed_gate"] = summarize(per_seed)
    candidate_upper_bound = statistics.fmean(int(group["target"] == MAX_KEYS or int(top[index]) == group["target"])
                                             for index, group in enumerate(groups))
    return summaries, records, raw_hashes, candidate_upper_bound


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False); source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    if source_summary["protocol"] != "zsre_contextual_readout_v1" or source_summary["test_scored"]:
        raise ValueError("requires sealed ZsRE v1 source")
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("source transform mismatch")
    transforms = artifact["transforms"]; regularizer = source_summary["selected_config"]["regularizer"]
    calibration = load_split(args.data_root, "calibration"); train_sessions, tune_sessions = split_calibration(calibration)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train = build_split_inputs(train_sessions, "gate_train", model, tokenizer, transforms,
                               regularizer, args.batch_size)
    tune = build_split_inputs(tune_sessions, "gate_tune", model, tokenizer, transforms,
                              regularizer, args.batch_size, train["geometry_mean"], train["geometry_std"])
    control_threshold, control_grid = select_ensemble_threshold(
        tune["groups"], tune["mean_scores"], tune["top"])
    heads, curves, selected_epochs, gate_thresholds = {}, {}, {}, {}
    for seed in GATE_SEEDS:
        head, curve, epoch, threshold = train_gate(
            train["groups"], tune["groups"], train["gate_values"], tune["gate_values"],
            train["top"], tune["top"], seed, args.epochs)
        heads[str(seed)] = head; curves[str(seed)] = curve
        selected_epochs[str(seed)] = epoch; gate_thresholds[str(seed)] = threshold

    dev_sessions = load_split(args.data_root, "dev")
    dev_values = build_split_inputs(dev_sessions, "dev", model, tokenizer, transforms,
                                    regularizer, args.batch_size,
                                    train["geometry_mean"], train["geometry_std"])
    dev, dev_records, raw_hashes, dev_upper = evaluate_split(
        root, "dev", dev_values, heads, gate_thresholds, control_threshold)
    dev_comparison = compare(dev_records); dev_gate = development_gate(dev["decomposed_gate"], dev_comparison)

    test = test_comparison = test_upper = test_values = None; test_scored = False
    if dev_gate["passed"]:
        test_sessions = load_split(args.data_root, "test")
        test_values = build_split_inputs(test_sessions, "test", model, tokenizer, transforms,
                                         regularizer, args.batch_size,
                                         train["geometry_mean"], train["geometry_std"])
        test, test_records, test_hashes, test_upper = evaluate_split(
            root, "test", test_values, heads, gate_thresholds, control_threshold)
        raw_hashes.update(test_hashes); test_comparison = compare(test_records); test_scored = True

    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache(); checkpoint = root / "scope_gates.pt"
    torch.save({"heads": {seed: head.state_dict() for seed, head in heads.items()},
                "selected_epochs": selected_epochs, "gate_thresholds": gate_thresholds,
                "control_threshold": control_threshold,
                "geometry_mean": train["geometry_mean"], "geometry_std": train["geometry_std"]}, checkpoint)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "curves": curves, "selected_epochs": selected_epochs,
                "gate_thresholds": gate_thresholds, "control_threshold": control_threshold,
                "control_threshold_grid": control_grid,
                "train_session_ids": [row["id"] for row in train_sessions],
                "tune_session_ids": [row["id"] for row in tune_sessions],
                "feature_hashes": {"train_pair": train["pair_sha256"], "train_gate": train["gate_sha256"],
                                   "tune_pair": tune["pair_sha256"], "tune_gate": tune["gate_sha256"],
                                   "dev_pair": dev_values["pair_sha256"], "dev_gate": dev_values["gate_sha256"],
                                   "test_pair": None if test_values is None else test_values["pair_sha256"],
                                   "test_gate": None if test_values is None else test_values["gate_sha256"]},
                "embedding_hashes": {"train": train["embedding_hashes"], "tune": tune["embedding_hashes"],
                                     "dev": dev_values["embedding_hashes"],
                                     "test": None if test_values is None else test_values["embedding_hashes"]},
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt")}
    manifest = {"protocol": "zsre_decomposed_scope_gate_v1",
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "arms": list(ARMS),
                "gate_seeds": list(GATE_SEEDS), "bi_seeds": list(BI_SEEDS),
                "pair_template": PAIR_TEMPLATE, "geometry_features": 4,
                "epochs": args.epochs, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "thresholds": list(THRESHOLDS), "learning_rate": .01, "weight_decay": 1e-4,
                "fit_split": "calibration_fit", "selection_split": "calibration_tune",
                "gate_split": "dev", "test_release_rule": "all decomposed-scope dev checks",
                "test_scored": test_scored}
    result = {"protocol": "zsre_decomposed_scope_gate_v1", "dev": dev,
              "dev_candidate_upper_bound": dev_upper, "dev_comparison": dev_comparison,
              "dev_gate": dev_gate, "test": test, "test_candidate_upper_bound": test_upper,
              "test_comparison": test_comparison, "test_scored": test_scored,
              "stage_passed": dev_gate["passed"], "training_seconds": training["seconds"],
              "terminal_status": "completed_zsre_decomposed_scope_gate"}
    write_json(root / "training.json", training); write_json(root / "training_manifest.json", manifest)
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__": main()
