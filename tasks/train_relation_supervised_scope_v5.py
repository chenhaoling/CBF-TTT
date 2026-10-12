"""Diagnose relation-supervised candidate scope under subject-disjoint OOF."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256, row_hash
from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_scoped_delta_ranker_v4 import load_corpus


PROTOCOL = "relation_supervised_scope_v5"
SEEDS = (14101, 14102, 14103)
EPOCHS = 10
ARMS = ("posterior_dot", "logit_cosine", "query_class_probability", "embedding_cosine")
WINDOW = 10


def focus_relation(text, subject, window=WINDOW):
    tokens = lambda value: re.findall(r"[a-z0-9]+", value.casefold())
    position = text.casefold().find(subject.casefold())
    if position < 0:
        values = tokens(text)
        return " ".join(values[-2 * window:]), False
    left = tokens(text[:position])[-window:]
    right = tokens(text[position + len(subject):])[:window]
    return " ".join([*left, "<ENTITY>", *right]), True


def load_source(path):
    path = Path(path)
    if sha(path) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    rows = json.loads(path.read_text())
    relations = sorted({row["requested_rewrite"]["relation_id"] for row in rows})
    relation_to_index = {value: index for index, value in enumerate(relations)}
    examples, by_hash = [], {}
    for row in rows:
        rewrite = row["requested_rewrite"]; subject = rewrite["subject"]
        relation = rewrite["relation_id"]; digest = row_hash(row)
        by_hash[digest] = relation
        texts = [rewrite["prompt"].format(subject), *row.get("paraphrase_prompts", [])]
        for text in texts:
            focused, found = focus_relation(text, subject)
            examples.append({"subject": subject, "relation": relation,
                             "label": relation_to_index[relation], "text": focused,
                             "subject_found": found})
    return examples, by_hash, relations


def relation_feature(text, embeddings):
    import torch
    return torch.cat([embeddings[str(seed)].get("entity", text) for seed in BI_SEEDS])


def feature_table(texts, embeddings):
    return {text: relation_feature(text, embeddings) for text in texts}


class RelationClassifier:
    def __init__(self, dim, classes, seed):
        import torch
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(dim, classes)
        torch.nn.init.normal_(self.linear.weight, std=.001)
        torch.nn.init.zeros_(self.linear.bias)

    def parameters(self): return self.linear.parameters()
    def state_dict(self): return self.linear.state_dict()
    def load_state_dict(self, state): self.linear.load_state_dict(state)
    def __call__(self, values): return self.linear(values)


def training_tensor(examples, features, subjects_excluded):
    import torch
    rows = [row for row in examples if row["subject"] not in subjects_excluded]
    x = torch.stack([features[row["text"]] for row in rows])
    y = torch.tensor([row["label"] for row in rows])
    return x, y


def fit_models(examples, features, relations, excluded_subjects, seed_offset=0):
    import torch

    x, y = training_tensor(examples, features, excluded_subjects)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-4)
    counts = torch.bincount(y, minlength=len(relations)).float().clamp_min(1)
    weights = (counts.mean() / counts).sqrt(); weights /= weights.mean()
    device = torch.device("cuda"); train_x = ((x - mean) / std).to(device)
    train_y = y.to(device); weights = weights.to(device)
    artifacts = []
    for seed in SEEDS:
        head = RelationClassifier(x.shape[-1], len(relations), seed + seed_offset)
        head.linear.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.05)
        for _ in range(EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_x)
            loss = torch.nn.functional.cross_entropy(logits, train_y, weight=weights)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        head.linear.cpu()
        artifacts.append({"state": copy.deepcopy(head.state_dict()), "mean": mean,
                          "std": std, "dim": x.shape[-1], "classes": len(relations),
                          "seed": seed + seed_offset})
    return artifacts


def ensemble_logits(texts, features, artifacts):
    import torch
    values = torch.stack([features[text] for text in texts])
    rows = []
    for artifact in artifacts:
        head = RelationClassifier(artifact["dim"], artifact["classes"], artifact["seed"])
        head.load_state_dict(artifact["state"])
        with torch.no_grad(): rows.append(head((values - artifact["mean"]) / artifact["std"]))
    return torch.stack(rows).mean(0)


def score_group(group, features, artifacts, arm):
    import torch

    query_text, _ = focus_relation(group["query"], group["query_subject"])
    candidates = [focus_relation(row["prompt"], row["subject"])[0]
                  for row in group["candidates"]]
    logits = ensemble_logits([query_text, *candidates], features, artifacts)
    query, candidate = logits[0], logits[1:]
    if arm == "posterior_dot":
        scores = candidate.softmax(-1) @ query.softmax(-1)
    elif arm == "logit_cosine":
        scores = torch.nn.functional.normalize(candidate, dim=-1) \
            @ torch.nn.functional.normalize(query, dim=-1)
    elif arm == "query_class_probability":
        scores = candidate.softmax(-1)[:, int(query.argmax())]
    elif arm == "embedding_cosine":
        query_feature = torch.nn.functional.normalize(features[query_text], dim=0)
        scores = torch.stack([torch.nn.functional.normalize(features[text], dim=0)
                              @ query_feature for text in candidates])
    else:
        raise ValueError(arm)
    same_subject = torch.tensor([row["subject"].casefold() == group["query_subject"].casefold()
                                 for row in group["candidates"]])
    if not same_subject[group["target"]]:
        raise ValueError("target violates entity constraint")
    scores = scores.masked_fill(~same_subject, float("-inf"))
    prediction = int(scores.argmax()); order = scores.argsort(descending=True).tolist()
    return {"prediction": prediction, "correct": int(prediction == group["target"]),
            "reciprocal_rank": 1 / (order.index(group["target"]) + 1),
            "query_relation_prediction": int(query.argmax())}


def evaluate(groups, features, artifacts, relation_by_hash, relation_to_index, arm, fold):
    rows = []
    for group in groups:
        result = score_group(group, features, artifacts, arm)
        digest = group["query_source_row_sha256"]
        label = relation_to_index[relation_by_hash[digest]]
        rows.append({"session_id": group["session_id"], "group_id": group["group_id"],
                     "target": group["target"], "prediction": result["prediction"],
                     "correct": result["correct"],
                     "reciprocal_rank": result["reciprocal_rank"],
                     "query_relation_correct": int(result["query_relation_prediction"] == label),
                     "fold": fold, "arm": arm})
    return rows


def metrics(rows):
    return {"events": len(rows),
            "top1_accuracy": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows),
            "query_relation_accuracy": statistics.fmean(
                row["query_relation_correct"] for row in rows)}


def select_arm(groups, features, artifacts, relation_by_hash, relation_to_index, fold):
    scored = {arm: metrics(evaluate(groups, features, artifacts, relation_by_hash,
                                    relation_to_index, arm, fold)) for arm in ARMS}
    arm = max(ARMS, key=lambda value: (scored[value]["top1_accuracy"],
                                       scored[value]["mrr"], -ARMS.index(value)))
    return arm, scored


def load_v4(root):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    path = root / "counterfact_oof_structured_full.jsonl"
    if training["raw_rows_sha256"][path.name] != sha(path):
        raise ValueError("V4 prediction hash mismatch")
    return {(row["session_id"], row["group_id"]): row
            for row in map(json.loads, path.read_text().splitlines())}


def comparison(rows, old, seed=14201):
    differences = [row["correct"] - old[(row["session_id"], row["group_id"])]["correct"]
                   for row in rows]
    return {"top1_gain": statistics.fmean(differences),
            "bootstrap_95ci": _bootstrap(differences, seed), "units": len(rows)}


def development_gate(metric, gain):
    checks = {"top1_at_least_90": metric["top1_accuracy"] >= .90,
              "mrr_at_least_95": metric["mrr"] >= .95,
              "query_relation_at_least_90": metric["query_relation_accuracy"] >= .90,
              "gain_at_least_5pp": gain["top1_gain"] >= .05,
              "gain_ci_positive": gain["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus, corpus_manifest = load_corpus(args.corpus_root)
    groups = [row for row in corpus if row["source"] == "counterfact"
              and row["target"] < MAX_KEYS]
    examples, relation_by_hash, relations = load_source(args.counterfact_source)
    relation_to_index = {value: index for index, value in enumerate(relations)}
    if any(group["query_source_row_sha256"] not in relation_by_hash for group in groups):
        raise ValueError("evaluation relation label missing")
    texts = {row["text"] for row in examples}
    for group in groups:
        texts.add(focus_relation(group["query"], group["query_subject"])[0])
        texts.update(focus_relation(row["prompt"], row["subject"])[0]
                     for row in group["candidates"])
    texts = sorted(texts)

    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": texts, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": texts, "value": ["unknown"]}, transforms["transforms"],
        source_summary["selected_config"]["regularizer"])
    features = feature_table(texts, embeddings)
    del raw, model; torch.cuda.empty_cache()

    fold_map = deterministic_folds({group["session_id"] for group in groups})
    final_rows, selection, artifacts_by_fold = [], {}, {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        outer_groups = [group for group in groups if fold_map[group["session_id"]] == outer]
        inner_groups = [group for group in groups if fold_map[group["session_id"]] == inner]
        excluded_inner = {group["query_subject"] for group in groups
                          if fold_map[group["session_id"]] in (outer, inner)}
        inner_artifacts = fit_models(examples, features, relations, excluded_inner,
                                     seed_offset=outer * 100)
        arm, inner_scores = select_arm(inner_groups, features, inner_artifacts,
                                       relation_by_hash, relation_to_index, inner)
        excluded_outer = {group["query_subject"] for group in outer_groups}
        artifacts = fit_models(examples, features, relations, excluded_outer,
                               seed_offset=outer * 1000)
        rows = evaluate(outer_groups, features, artifacts, relation_by_hash,
                        relation_to_index, arm, outer)
        final_rows.extend(rows); artifacts_by_fold[str(outer)] = artifacts
        selection[str(outer)] = {"inner_fold": inner, "selected_arm": arm,
                                  "inner_scores": inner_scores,
                                  "outer_subjects": sorted(excluded_outer)}

    final_rows.sort(key=lambda row: (row["session_id"], row["group_id"]))
    path = root / "counterfact_oof_relation_supervised.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in final_rows))
    metric = metrics(final_rows); old = load_v4(args.v4_root)
    gain = comparison(final_rows, old); gate = development_gate(metric, gain)
    checkpoint = root / "relation_classifiers.pt"
    torch.save({"artifacts": artifacts_by_fold, "relations": relations,
                "selection": selection, "window": WINDOW}, checkpoint)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "counterfact_oof": metric, "v5_vs_v4": gain,
               "development_gate": gate,
               "next_stage": "automatic_open_scope_distillation" if gate["passed"]
               else "revise_relation_representation",
               "terminal_status": "completed_relation_supervised_scope_v5"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "source_examples": len(examples), "unique_relation_texts": len(texts),
                "relations": len(relations), "selection": selection,
                "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {path.name: sha(path)},
                "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "corpus_manifest_sha256": sha(Path(args.corpus_root) / "manifest.json"),
                "counterfact_source_sha256": sha(args.counterfact_source),
                "v4_summary_sha256": sha(Path(args.v4_root) / "summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "seeds": list(SEEDS), "epochs": EPOCHS, "arms": list(ARMS),
                "window": WINDOW, "relation_ids_are_targets_only": True,
                "subject_metadata_role": "oracle_segmentation_diagnostic",
                "chain_identity_in_features": False, "candidate_age_used": False,
                "longmemeval_scored": False, "locomo_scored": False}
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--counterfact-source", required=True)
    parser.add_argument("--v4-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()

