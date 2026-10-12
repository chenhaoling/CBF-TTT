"""Train a frozen-Qwen pairwise scope resolver for structured ZsRE memory."""

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
from tasks.evaluate_zsre_contextual_readout import (
    SEEDS as BI_SEEDS, apply, extract, load_split, texts,
)


HEAD_SEEDS = (7101, 7102, 7103)
ARMS = ("bi_cosine", "cross_linear")
QUERY_TYPES = ("exact", "rephrase", "locality")
THRESHOLDS = tuple(value / 10 for value in range(1, 10))
CHECKPOINT_EPOCHS = (10, 25, 50, 100, 200)
PAIR_TEMPLATE = "Memory update scope: {key}\nCandidate query: {query}\nSame factual scope:"


def split_calibration(sessions):
    ranked = sorted(sessions, key=lambda row: hashlib.sha256(
        ("scope-resolver-v1|" + row["id"]).encode()).hexdigest())
    if len(ranked) != 34:
        raise ValueError("expected 34 calibration sessions")
    return ranked[:27], ranked[27:]


def query_groups(sessions, split):
    output = []
    for session in sessions:
        keys = [fact["prompt"] for fact in session["facts"]]
        defer = len(keys)
        for index, fact in enumerate(session["facts"]):
            rows = [("exact", fact["prompt"], index),
                    ("rephrase", fact["rephrase_prompt"], index)]
            rows.extend(("locality", query, defer) for query in fact["locality_prompts"])
            for kind, query, target in rows:
                output.append({"session_id": session["id"], "split": split,
                               "subject": fact["subject"], "type": kind,
                               "query": query, "keys": keys, "target": target})
    return output


def pair_texts(groups):
    return sorted({PAIR_TEMPLATE.format(key=key, query=group["query"])
                   for group in groups for key in group["keys"]})


def extract_pair_features(model, tokenizer, names, batch_size):
    import torch

    output = {}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for offset in range(0, len(names), batch_size):
        batch = names[offset:offset + batch_size]
        rows = [tokenizer.encode(text, add_special_tokens=False) for text in batch]
        if any(not row for row in rows):
            raise ValueError("empty pair tokenization")
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, device="cuda"); mask[index, :len(row)] = 1
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 return_dict=True).last_hidden_state.float()
            rows_out = hidden[torch.arange(len(rows), device="cuda"),
                              torch.tensor([len(row) - 1 for row in rows], device="cuda")]
            rows_out = torch.nn.functional.normalize(rows_out, dim=-1).cpu()
        output.update(zip(batch, rows_out))
    return output


def feature_tensor(groups, features):
    import torch

    return torch.stack([torch.stack([features[PAIR_TEMPLATE.format(key=key, query=group["query"])]
                                     for key in group["keys"]]) for group in groups])


def tensor_hash(values):
    return hashlib.sha256(values.contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


class ScopeHead:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(dim, 1)
        torch.nn.init.normal_(self.linear.weight, std=.001)
        torch.nn.init.zeros_(self.linear.bias)
        self.defer = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        return [*self.linear.parameters(), self.defer]

    def logits(self, values):
        import torch

        scores = self.linear(values).squeeze(-1)
        return torch.cat((scores, self.defer.expand(scores.shape[0], 1)), dim=1)

    def state_dict(self):
        return {"linear": self.linear.state_dict(), "defer": self.defer.detach().clone()}

    def load_state_dict(self, state):
        self.linear.load_state_dict(state["linear"])
        self.defer.data.copy_(state["defer"])


def metrics_from_logits(logits, targets):
    import torch

    loss = torch.nn.functional.cross_entropy(logits, targets)
    accuracy = logits.argmax(-1).eq(targets).float().mean()
    return float(accuracy), float(loss)


def train_head(train_groups, tune_groups, train_features, tune_features, seed, epochs=200):
    import copy
    import torch

    head = ScopeHead(train_features.shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=1e-4)
    train_targets = torch.tensor([group["target"] for group in train_groups])
    tune_targets = torch.tensor([group["target"] for group in tune_groups])
    curve, candidates = [], []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(train_features)
        loss = torch.nn.functional.cross_entropy(logits, train_targets)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if epoch in CHECKPOINT_EPOCHS:
            with torch.no_grad():
                train_acc, train_nll = metrics_from_logits(head.logits(train_features), train_targets)
                tune_acc, tune_nll = metrics_from_logits(head.logits(tune_features), tune_targets)
            row = {"epoch": epoch, "train_accuracy": train_acc, "train_nll": train_nll,
                   "tune_accuracy": tune_acc, "tune_nll": tune_nll}
            curve.append(row); candidates.append((tune_acc, -tune_nll, -epoch,
                                                   copy.deepcopy(head.state_dict())))
    selected = max(candidates, key=lambda row: row[:3])
    head.load_state_dict(selected[3])
    return head, curve, -selected[2]


def rows_from_logits(groups, logits, seed, arm):
    import torch

    probabilities = torch.softmax(logits, dim=-1)
    records = {}
    for index, (group, scores) in enumerate(zip(groups, logits)):
        target = group["target"]; prediction = int(scores.argmax())
        order = torch.argsort(scores, descending=True).tolist()
        alternatives = torch.cat((scores[:target], scores[target + 1:]))
        query = {"subject": group["subject"], "type": group["type"],
                 "query": group["query"], "target": target, "prediction": prediction,
                 "correct": int(prediction == target),
                 "nll": float(-probabilities[index, target].clamp_min(1e-12).log()),
                 "mrr": 1. / (order.index(target) + 1),
                 "margin": float(scores[target] - alternatives.max())}
        records.setdefault(group["session_id"], []).append(query)
    return [{"session_id": session_id, "split": groups[0]["split"], "seed": seed,
             "arm": arm, "queries": records[session_id]} for session_id in sorted(records)]


def head_records(groups, feature_values, head, seed, arm="cross_linear"):
    import torch

    with torch.no_grad(): logits = head.logits(feature_values)
    return rows_from_logits(groups, logits, seed, arm)


def cosine_logits(groups, embeddings, defer_logit):
    import torch

    rows = []
    for group in groups:
        query = embeddings.get("entity", group["query"])
        keys = torch.stack([embeddings.get("entity", key) for key in group["keys"]])
        scores = keys @ query / .07
        rows.append(torch.cat((scores, torch.tensor([defer_logit / .07]))))
    return torch.stack(rows)


def cosine_records(groups, embeddings, defer_logit):
    return rows_from_logits(groups, cosine_logits(groups, embeddings, defer_logit),
                            embeddings.seed, "bi_cosine")


def aggregate(records):
    pairs = [(record, query) for record in records for query in record["queries"]]
    def metric(rows):
        return {"queries": len(rows),
                "accuracy": statistics.fmean(query["correct"] for _, query in rows),
                "nll": statistics.fmean(query["nll"] for _, query in rows),
                "mrr": statistics.fmean(query["mrr"] for _, query in rows),
                "margin": statistics.fmean(query["margin"] for _, query in rows)}
    result = metric(pairs); result["sessions"] = len(records)
    result["by_type"] = {kind: metric([(r, q) for r, q in pairs if q["type"] == kind])
                         for kind in QUERY_TYPES}
    result["groups"] = {record["session_id"]: metric(
        [(r, q) for r, q in pairs if r["session_id"] == record["session_id"]])
        for record in records}
    return result


def summarize(per_seed):
    return {"mean_accuracy": statistics.fmean(row["accuracy"] for row in per_seed.values()),
            "min_seed_accuracy": min(row["accuracy"] for row in per_seed.values()),
            "mean_nll": statistics.fmean(row["nll"] for row in per_seed.values()),
            "by_type": {kind: {"accuracy": statistics.fmean(
                row["by_type"][kind]["accuracy"] for row in per_seed.values()),
                "nll": statistics.fmean(row["by_type"][kind]["nll"] for row in per_seed.values())}
                for kind in QUERY_TYPES}, "per_seed": per_seed}


def select_cosine_threshold(tune_groups, embeddings):
    grid, selected = {}, {}
    for seed in BI_SEEDS:
        values = {}
        for threshold in THRESHOLDS:
            values[str(threshold)] = aggregate(cosine_records(
                tune_groups, embeddings[str(seed)], threshold))["accuracy"]
        best = max(values.values())
        selected[str(seed)] = min((float(key) for key, value in values.items() if value == best),
                                  key=lambda value: (abs(value - .5), value))
        grid[str(seed)] = values
    return grid, selected


def compare(records, seed=7201):
    output = {}
    categories = {"overall": lambda row: True,
                  **{kind: lambda row, value=kind: row["type"] == value for kind in QUERY_TYPES}}
    sessions = sorted({row["session_id"] for rows in records["cross_linear"].values() for row in rows})
    for name, predicate in categories.items():
        differences = []
        for session_id in sessions:
            scores = {}
            for arm, seeds in (("cross_linear", HEAD_SEEDS), ("bi_cosine", BI_SEEDS)):
                values = []
                for rep_seed in seeds:
                    record = next(row for row in records[arm][str(rep_seed)]
                                  if row["session_id"] == session_id)
                    queries = [query["correct"] for query in record["queries"] if predicate(query)]
                    values.append(statistics.fmean(queries))
                scores[arm] = statistics.fmean(values)
            differences.append(scores["cross_linear"] - scores["bi_cosine"])
        output[name] = {"accuracy_gain": statistics.fmean(differences),
                        "bootstrap_95ci": _bootstrap(differences, seed),
                        "sessions": len(differences)}
        seed += 1
    return output


def development_gate(primary, comparison):
    checks = {"exact": primary["by_type"]["exact"]["accuracy"] >= .90,
              "rephrase": primary["by_type"]["rephrase"]["accuracy"] >= .70,
              "locality": primary["by_type"]["locality"]["accuracy"] >= .90,
              "min_seed_overall": primary["min_seed_accuracy"] >= .80,
              "control_margin": comparison["overall"]["accuracy_gain"] >= .05,
              "control_confidence": comparison["overall"]["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_split(root, split, sessions, model, tokenizer, transforms, regularizer,
                   heads, cosine_thresholds, batch_size):
    groups = query_groups(sessions, split)
    names = pair_texts(groups); raw = extract_pair_features(model, tokenizer, names, batch_size)
    values = feature_tensor(groups, raw); feature_digest = tensor_hash(values)
    entity_names = texts(sessions); entity_raw = extract(model, tokenizer, entity_names, batch_size)
    base, embedding_hashes = apply(entity_raw, entity_names, transforms, regularizer)
    records, summaries, hashes = {arm: {} for arm in ARMS}, {}, {}
    per_seed = {}
    for seed in BI_SEEDS:
        rows = cosine_records(groups, base[str(seed)], cosine_thresholds[str(seed)])
        records["bi_cosine"][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
        path = root / f"{split}_bi_cosine_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)); hashes[path.name] = sha(path)
    summaries["bi_cosine"] = summarize(per_seed); per_seed = {}
    for seed in HEAD_SEEDS:
        rows = head_records(groups, values, heads[str(seed)], seed)
        records["cross_linear"][str(seed)] = rows; per_seed[str(seed)] = aggregate(rows)
        path = root / f"{split}_cross_linear_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)); hashes[path.name] = sha(path)
    summaries["cross_linear"] = summarize(per_seed)
    return summaries, records, hashes, feature_digest, embedding_hashes


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_summary["protocol"] != "zsre_contextual_readout_v1" or source_summary["test_scored"]:
        raise ValueError("requires sealed audited ZsRE v1 source")
    artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("source transform mismatch")
    transforms = artifact["transforms"]; regularizer = source_summary["selected_config"]["regularizer"]
    calibration = load_split(args.data_root, "calibration")
    train_sessions, tune_sessions = split_calibration(calibration)
    train_groups = query_groups(train_sessions, "resolver_train")
    tune_groups = query_groups(tune_sessions, "resolver_tune")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    calibration_pair_names = pair_texts(train_groups + tune_groups)
    calibration_pair_raw = extract_pair_features(model, tokenizer, calibration_pair_names, args.batch_size)
    train_values = feature_tensor(train_groups, calibration_pair_raw)
    tune_values = feature_tensor(tune_groups, calibration_pair_raw)
    heads, curves, selected_epochs = {}, {}, {}
    for seed in HEAD_SEEDS:
        head, curve, selected_epoch = train_head(
            train_groups, tune_groups, train_values, tune_values, seed, args.epochs)
        heads[str(seed)] = head; curves[str(seed)] = curve; selected_epochs[str(seed)] = selected_epoch

    calibration_names = texts(calibration)
    calibration_raw = extract(model, tokenizer, calibration_names, args.batch_size)
    calibration_base, calibration_embedding_hashes = apply(
        calibration_raw, calibration_names, transforms, regularizer)
    threshold_grid, cosine_thresholds = select_cosine_threshold(tune_groups, calibration_base)

    dev_sessions = load_split(args.data_root, "dev")
    dev, dev_records, raw_hashes, dev_feature_hash, dev_embedding_hashes = evaluate_split(
        root, "dev", dev_sessions, model, tokenizer, transforms, regularizer,
        heads, cosine_thresholds, args.batch_size)
    dev_comparison = compare(dev_records); dev_gate = development_gate(dev["cross_linear"], dev_comparison)

    test = test_comparison = test_feature_hash = test_embedding_hashes = None
    test_scored = False
    if dev_gate["passed"]:
        test_sessions = load_split(args.data_root, "test")
        test, test_records, test_raw_hashes, test_feature_hash, test_embedding_hashes = evaluate_split(
            root, "test", test_sessions, model, tokenizer, transforms, regularizer,
            heads, cosine_thresholds, args.batch_size)
        raw_hashes.update(test_raw_hashes); test_comparison = compare(test_records); test_scored = True

    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model; torch.cuda.empty_cache()
    checkpoint = root / "scope_heads.pt"
    torch.save({"heads": {seed: head.state_dict() for seed, head in heads.items()},
                "selected_epochs": selected_epochs, "cosine_thresholds": cosine_thresholds,
                "pair_template": PAIR_TEMPLATE, "head_seeds": HEAD_SEEDS}, checkpoint)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
                "curves": curves, "selected_epochs": selected_epochs,
                "cosine_threshold_grid": threshold_grid, "cosine_thresholds": cosine_thresholds,
                "train_session_ids": [row["id"] for row in train_sessions],
                "tune_session_ids": [row["id"] for row in tune_sessions],
                "train_feature_sha256": tensor_hash(train_values),
                "tune_feature_sha256": tensor_hash(tune_values),
                "dev_feature_sha256": dev_feature_hash, "test_feature_sha256": test_feature_hash,
                "calibration_embedding_hashes": calibration_embedding_hashes,
                "dev_embedding_hashes": dev_embedding_hashes,
                "test_embedding_hashes": test_embedding_hashes,
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt")}
    manifest = {"protocol": "zsre_scope_resolver_v1",
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "arms": list(ARMS),
                "head_seeds": list(HEAD_SEEDS), "bi_seeds": list(BI_SEEDS),
                "pair_template": PAIR_TEMPLATE, "pair_pooling": "normalized_context_last",
                "calibration_sessions": {"fit": 27, "tune": 7},
                "epochs": args.epochs, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
                "learning_rate": .01, "weight_decay": 1e-4,
                "cosine_thresholds": list(THRESHOLDS), "fit_split": "calibration_fit",
                "selection_split": "calibration_tune", "gate_split": "dev",
                "test_release_rule": "score once only after every scope-resolver dev check",
                "test_scored": test_scored}
    result = {"protocol": "zsre_scope_resolver_v1", "dev": dev,
              "dev_comparison": dev_comparison, "dev_gate": dev_gate,
              "test": test, "test_comparison": test_comparison, "test_scored": test_scored,
              "stage_passed": dev_gate["passed"], "training_seconds": training["seconds"],
              "terminal_status": "completed_zsre_scope_resolver"}
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
