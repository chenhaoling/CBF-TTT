"""Train a nested-OOF low-dimensional text-only scope calibrator."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import (
    build_fold_views, load_v5, relation_features, source_documents,
)
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_relation_supervised_scope_v5 import ensemble_logits, load_v4
from tasks.train_scoped_delta_ranker_v4 import load_corpus, valid_mask
from tasks.train_zsre_forgetting_controller import lexical_features


PROTOCOL = "text_only_scope_calibrator_v8"
SEEDS = (16101, 16102, 16103)
EPOCHS = (10, 25, 50)


def feature_rows(fold_rows, features, artifacts):
    import torch

    output = []
    for row in fold_rows:
        values, bases = [], []
        group = row["group"]
        for candidate, original in zip(row["candidates"], group["candidates"]):
            logits = ensemble_logits([candidate["query_relation"], candidate["key_relation"]],
                                     features, artifacts)
            relation = float(logits[0].softmax(-1) @ logits[1].softmax(-1))
            matched = len(candidate["matched_tokens"])
            q_len = len(group["query"].split()); k_len = len(original["prompt"].split())
            length_ratio = min(q_len, k_len) / max(1, max(q_len, k_len))
            lexical = lexical_features(group["query"], original["prompt"])
            values.append(torch.tensor([
                relation, math.log(max(1e-12, relation)),
                candidate["entity_score"], candidate["weighted_overlap"],
                candidate["block_mass"], matched / max(1, min(q_len, k_len)),
                length_ratio, *lexical,
            ]))
            bases.append(math.log(max(1e-12, relation))
                         + 8 * candidate["entity_score"])
        output.append({"group": group, "values": torch.stack(values),
                       "base": torch.tensor(bases)})
    return output


def tensorize(rows):
    import torch
    dim = rows[0]["values"].shape[-1]; values, bases = [], []
    for row in rows:
        count = len(row["values"])
        values.append(torch.cat((row["values"], torch.zeros(MAX_KEYS - count, dim))))
        bases.append(torch.cat((row["base"], torch.full((MAX_KEYS - count,), float("-inf")))))
    return torch.stack(values), torch.stack(bases), \
        torch.tensor([len(row["values"]) for row in rows]), \
        torch.tensor([row["group"]["target"] for row in rows])


class Calibrator:
    def __init__(self, dim, seed, rank=16):
        import torch
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(dim, rank), torch.nn.Tanh(),
                                         torch.nn.Linear(rank, 1))
        torch.nn.init.normal_(self.model[0].weight, std=.01)
        torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.zeros_(self.model[2].weight); torch.nn.init.zeros_(self.model[2].bias)
    def parameters(self): return self.model.parameters()
    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, state): self.model.load_state_dict(state)
    def logits(self, values, base, counts):
        return (base + self.model(values).squeeze(-1)).masked_fill(
            ~valid_mask(counts, values.shape[1]), float("-inf"))


def fit_normalizer(values, counts, indices):
    selected = values[indices]; flat = selected[valid_mask(counts[indices])]
    return flat.mean(0), flat.std(0, unbiased=False).clamp_min(1e-4)


def train(values, base, counts, target, indices, mean, std, seed, epochs):
    import torch
    device = torch.device("cuda")
    x = ((values[indices] - mean) / std).to(device); b = base[indices].to(device)
    c = counts[indices].to(device); y = target[indices].to(device)
    head = Calibrator(values.shape[-1], seed); head.model.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.05)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True); logits = head.logits(x, b, c)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
    head.model.cpu(); return head


def metric(logits, targets):
    order = logits.argsort(-1, descending=True)
    ranks = order.eq(targets[:, None]).nonzero()[:, 1].float() + 1
    return {"top1_accuracy": float(logits.argmax(-1).eq(targets).float().mean()),
            "mrr": float((1 / ranks).mean())}


def choose_epoch(values, base, counts, target, train_indices, tune_indices, seed):
    mean, std = fit_normalizer(values, counts, train_indices); candidates, curve = [], []
    for epoch in EPOCHS:
        head = train(values, base, counts, target, train_indices, mean, std, seed, epoch)
        with __import__("torch").no_grad():
            logits = head.logits((values[tune_indices] - mean) / std,
                                 base[tune_indices], counts[tune_indices])
        result = metric(logits, target[tune_indices]); curve.append({"epoch": epoch, **result})
        candidates.append((result["top1_accuracy"], result["mrr"], -epoch))
    return -max(candidates)[-1], curve


def fit_outer(values, base, counts, target, indices, seed, epoch):
    mean, std = fit_normalizer(values, counts, indices)
    head = train(values, base, counts, target, indices, mean, std, seed, epoch)
    return {"state": copy.deepcopy(head.state_dict()), "mean": mean, "std": std,
            "dim": values.shape[-1], "seed": seed, "rank": 16, "epoch": epoch}


def predict(values, base, counts, indices, artifacts):
    import torch
    rows = []
    for artifact in artifacts:
        head = Calibrator(artifact["dim"], artifact["seed"], artifact["rank"])
        head.load_state_dict(artifact["state"])
        with torch.no_grad(): rows.append(head.logits(
            (values[indices] - artifact["mean"]) / artifact["std"],
            base[indices], counts[indices]))
    return torch.stack(rows).mean(0)


def records(groups, logits, fold):
    output = []
    for group, scores in zip(groups, logits):
        target = group["target"]; prediction = int(scores.argmax())
        order = scores.argsort(descending=True).tolist()
        output.append({"session_id": group["session_id"], "group_id": group["group_id"],
                       "target": target, "prediction": prediction,
                       "correct": int(prediction == target),
                       "reciprocal_rank": 1 / (order.index(target) + 1), "fold": fold})
    return output


def aggregate(rows):
    return {"events": len(rows), "top1_accuracy": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows)}


def load_trace(root, stem):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    path = root / stem
    if training["raw_rows_sha256"][path.name] != sha(path): raise ValueError("trace hash mismatch")
    return {(row["session_id"], row["group_id"]): row
            for row in map(json.loads, path.read_text().splitlines())}


def compare(rows, baseline, seed):
    delta = [row["correct"] - baseline[(row["session_id"], row["group_id"])]["correct"]
             for row in rows]
    return {"top1_gain": statistics.fmean(delta), "bootstrap_95ci": _bootstrap(delta, seed),
            "units": len(delta)}


def development_gate(result, vs_v4, vs_v7):
    checks = {"top1_at_least_95": result["top1_accuracy"] >= .95,
              "mrr_at_least_97": result["mrr"] >= .97,
              "gain_v4_at_least_10pp": vs_v4["top1_gain"] >= .10,
              "gain_v4_ci_positive": vs_v4["bootstrap_95ci"][0] > 0,
              "gain_v7_at_least_1pp": vs_v7["top1_gain"] >= .01,
              "gain_v7_ci_positive": vs_v7["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus, corpus_manifest = load_corpus(args.corpus_root)
    groups = [row for row in corpus if row["source"] == "counterfact" and row["target"] < MAX_KEYS]
    fold_map = deterministic_folds({row["session_id"] for row in groups})
    views, idf_sizes = build_fold_views(groups, fold_map, source_documents(args.counterfact_source))
    relation_texts = sorted({candidate[field] for rows in views.values() for row in rows
                             for candidate in row["candidates"]
                             for field in ("query_relation", "key_relation")})
    v5_artifact, v5_summary = load_v5(args.v5_root)
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": relation_texts, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(raw, {"entity": relation_texts, "value": ["unknown"]},
                                         transforms["transforms"],
                                         source_summary["selected_config"]["regularizer"])
    features = relation_features(relation_texts, embeddings); del raw, model; torch.cuda.empty_cache()
    rows = []
    for fold in range(FOLDS): rows.extend(feature_rows(
        views[str(fold)], features, v5_artifact["artifacts"][str(fold)]))
    rows.sort(key=lambda row: (row["group"]["session_id"], row["group"]["group_id"]))
    values, base, counts, target = tensorize(rows)
    fold_for_index = [fold_map[row["group"]["session_id"]] for row in rows]
    final, selection, checkpoint = [], {}, {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        train_inner = [i for i,f in enumerate(fold_for_index) if f not in (outer,inner)]
        tune = [i for i,f in enumerate(fold_for_index) if f == inner]
        train_outer = [i for i,f in enumerate(fold_for_index) if f != outer]
        test = [i for i,f in enumerate(fold_for_index) if f == outer]
        artifacts, curves, epochs = [], {}, {}
        for seed in SEEDS:
            epoch, curve = choose_epoch(values, base, counts, target, train_inner, tune,
                                         seed + outer * 100)
            artifacts.append(fit_outer(values, base, counts, target, train_outer,
                                       seed + outer * 100, epoch))
            curves[str(seed)] = curve; epochs[str(seed)] = epoch
        logits = predict(values, base, counts, test, artifacts)
        final.extend(records([rows[i]["group"] for i in test], logits, outer))
        selection[str(outer)] = {"inner_fold": inner, "epochs": epochs, "curves": curves}
        checkpoint[str(outer)] = artifacts
    final.sort(key=lambda row: (row["session_id"], row["group_id"]))
    path = root / "counterfact_oof_text_calibrator.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in final))
    result = aggregate(final); v4 = load_v4(args.v4_root)
    v7 = load_trace(args.v7_root, "counterfact_oof_automatic_factorized.jsonl")
    vs_v4 = compare(final, v4, 16201); vs_v7 = compare(final, v7, 16211)
    gate = development_gate(result, vs_v4, vs_v7)
    checkpoint_path = root / "calibrators.pt"; torch.save(checkpoint, checkpoint_path)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "counterfact_oof": result, "v8_vs_v4": vs_v4, "v8_vs_v7": vs_v7,
               "development_gate": gate,
               "next_stage": "causal_automatic_scope_forgetting" if gate["passed"]
               else "learned_entity_encoder",
               "terminal_status": "completed_text_only_scope_calibrator_v8"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "unique_relation_texts": len(relation_texts), "feature_dim": values.shape[-1],
                "selection": selection, "idf_training_documents": idf_sizes,
                "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {path.name: sha(path)},
                "checkpoint_sha256": sha(checkpoint_path)}
    manifest = {"protocol": PROTOCOL,
                "corpus_manifest_sha256": sha(Path(args.corpus_root)/"manifest.json"),
                "v5_summary_sha256": sha(Path(args.v5_root)/"summary.json"),
                "v7_summary_sha256": sha(Path(args.v7_root)/"summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
                "epochs": list(EPOCHS), "rank": 16,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False}
    write_json(root/"summary.json",summary); write_json(root/"training.json",training)
    write_json(root/"training_manifest.json",manifest); return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root","counterfact-source","v4-root","v5-root","v7-root",
                 "source-root","output","model","tokenizer"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--batch-size",type=int,default=256)
    print(json.dumps(run(parser.parse_args()),indent=2))


if __name__ == "__main__": main()
