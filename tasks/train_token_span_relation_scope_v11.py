"""Learn subject spans and use them to build text-only relation scope views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256
from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import load_rows, load_v5, relation_features
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_relation_supervised_scope_v5 import ensemble_logits
from tasks.train_scoped_delta_ranker_v4 import load_corpus
from tasks.train_text_only_scope_calibrator_v8 import load_trace


PROTOCOL = "token_span_relation_scope_v11"
PREFIX = "Identify the factual subject span: "
WINDOW = 10
PROJECTION_DIM = 256
HEAD_RANK = 128
HEAD_SEEDS = (19101, 19102, 19103)
EPOCHS = (3, 8, 15)
THRESHOLDS = (.3, .5, .7)
ARMS = ("hard_equal", "overlap8")
MAX_SPAN_TOKENS = 12


def source_rows(path):
    path = Path(path)
    if sha(path) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    return json.loads(path.read_text())


def character_spans(text, subject):
    """Return all non-overlapping case-insensitive exact subject spans."""
    spans, start = [], 0
    lower, needle = text.casefold(), subject.casefold()
    if not needle:
        return spans
    while True:
        position = lower.find(needle, start)
        if position < 0:
            return spans
        spans.append((position, position + len(subject)))
        start = position + len(subject)


def token_labels(offsets, spans):
    return [int(end > begin and any(begin < right and end > left
                                    for left, right in spans))
            for begin, end in offsets]


def span_examples(raw_rows, groups):
    """Deduplicate raw and constructed text while retaining its supervised subject."""
    examples = {}
    for row in raw_rows:
        rewrite = row["requested_rewrite"]; subject = rewrite["subject"]
        texts = [rewrite["prompt"].format(subject), *row.get("paraphrase_prompts", [])]
        for text in texts:
            if character_spans(text, subject):
                examples[text, subject] = {"text": text, "subject": subject,
                                            "sources": {"raw"}}
    for group in groups:
        rows = [(group["query"], group["query_subject"])]
        rows.extend((candidate["prompt"], candidate["subject"])
                    for candidate in group["candidates"])
        for text, subject in rows:
            if not character_spans(text, subject):
                raise ValueError("constructed scope text lacks its subject span")
            if (text, subject) in examples:
                examples[text, subject]["sources"].add("constructed")
            else:
                examples[text, subject] = {"text": text, "subject": subject,
                                            "sources": {"constructed"}}
    output = sorted(examples.values(), key=lambda row: (row["subject"], row["text"]))
    for row in output:
        row["sources"] = sorted(row["sources"])
    return output


def surface_features(text, offsets):
    values = []
    length = max(1, len(text))
    for begin, end in offsets:
        piece = text[begin:end]
        stripped = piece.strip()
        values.append([
            begin / length,
            (end - begin) / max(1, min(32, length)),
            float(bool(stripped) and stripped[0].isupper()),
            float(bool(stripped) and stripped.isupper()),
        ])
    return values


def extract_projected_features(model, tokenizer, examples, batch_size, max_length,
                               projection_dim=PROJECTION_DIM):
    import torch

    hidden_dim = model.config.hidden_size
    generator = torch.Generator().manual_seed(19001)
    projection = torch.randn(hidden_dim, projection_dim, generator=generator) \
        / math.sqrt(hidden_dim)
    projection = projection.to("cuda")
    tokenizer.padding_side = "right"
    output, skipped = [], 0
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        prompts = [PREFIX + row["text"] for row in batch]
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True,
                            truncation=True, max_length=max_length,
                            return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        mask = encoded["attention_mask"]
        inputs = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False, return_dict=True) \
                .last_hidden_state.float()
            last = hidden[torch.arange(len(batch), device="cuda"), mask.sum(-1).to("cuda") - 1]
            sentence = torch.nn.functional.layer_norm(last @ projection, (projection_dim,))
        for index, row in enumerate(batch):
            count = int(mask[index].sum())
            prompt_offsets = offsets[index, :count].tolist()
            selected = [(token_index, begin - len(PREFIX), end - len(PREFIX))
                        for token_index, (begin, end) in enumerate(prompt_offsets)
                        if end > len(PREFIX) and begin < len(PREFIX) + len(row["text"])]
            if not selected:
                skipped += 1; continue
            token_indices = torch.tensor([item[0] for item in selected], device="cuda")
            text_offsets = [(max(0, item[1]), min(len(row["text"]), item[2]))
                            for item in selected]
            projected = torch.nn.functional.layer_norm(
                hidden[index, token_indices] @ projection, (projection_dim,))
            global_value = sentence[index].expand_as(projected)
            surface = torch.tensor(surface_features(row["text"], text_offsets),
                                   device="cuda")
            features = torch.cat((projected, global_value, projected * global_value,
                                  surface), dim=-1).half().cpu()
            labels = token_labels(text_offsets, character_spans(row["text"], row["subject"]))
            if not any(labels):
                skipped += 1; continue
            output.append({**row, "features": features,
                           "labels": torch.tensor(labels, dtype=torch.float32),
                           "offsets": text_offsets})
    return output, {"projection_seed": 19001, "projection_dim": projection_dim,
                    "feature_dim": 3 * projection_dim + 4,
                    "skipped_after_truncation": skipped}


class SpanHead:
    def __init__(self, dim, seed, rank=HEAD_RANK):
        import torch
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(dim, rank), torch.nn.GELU(),
                                         torch.nn.Linear(rank, 1))
        torch.nn.init.normal_(self.model[0].weight, std=.01)
        torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.normal_(self.model[2].weight, std=.01)
        torch.nn.init.zeros_(self.model[2].bias)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def __call__(self, values):
        return self.model(values).squeeze(-1)


def flatten_rows(rows, excluded_subjects):
    import torch
    selected = [row for row in rows if row["subject"] not in excluded_subjects]
    if not selected:
        raise ValueError("empty span training set")
    return torch.cat([row["features"] for row in selected]), \
        torch.cat([row["labels"] for row in selected]), len(selected)


def train_snapshots(rows, excluded_subjects, seed, batch_size):
    import torch

    x, y, example_count = flatten_rows(rows, excluded_subjects)
    x_float = x.float(); mean = x_float.mean(0); std = x_float.std(0, unbiased=False).clamp_min(1e-4)
    positives = y.sum().clamp_min(1); negatives = len(y) - positives
    pos_weight = (negatives / positives).sqrt().to("cuda")
    head = SpanHead(x.shape[-1], seed); head.model.to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=.05)
    generator = torch.Generator().manual_seed(seed + 700000)
    snapshots = {}
    for epoch in range(1, max(EPOCHS) + 1):
        order = torch.randperm(len(y), generator=generator)
        for begin in range(0, len(y), batch_size):
            index = order[begin:begin + batch_size]
            values = ((x[index].float() - mean) / std).to("cuda", non_blocking=True)
            labels = y[index].to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(values)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        if epoch in EPOCHS:
            snapshots[epoch] = {"state": copy.deepcopy(head.state_dict()),
                                "mean": mean, "std": std, "dim": x.shape[-1],
                                "rank": HEAD_RANK, "seed": seed, "epoch": epoch}
    head.model.cpu(); del x_float
    return snapshots, {"examples": example_count, "tokens": len(y),
                       "positive_tokens": int(positives)}


def artifact_logits(row, artifact):
    import torch
    head = SpanHead(artifact["dim"], artifact["seed"], artifact["rank"])
    head.load_state_dict(artifact["state"])
    with torch.no_grad():
        return head((row["features"].float() - artifact["mean"]) / artifact["std"])


def ensemble_token_logits(row, artifacts):
    return __import__("torch").stack([artifact_logits(row, artifact)
                                      for artifact in artifacts]).mean(0)


def select_contiguous_span(logits, threshold, max_tokens=MAX_SPAN_TOKENS):
    """Select the best above-threshold contiguous run; always return one token."""
    boundary = math.log(threshold / (1 - threshold))
    active = [float(value) >= boundary for value in logits]
    runs, begin = [], None
    for index, value in enumerate([*active, False]):
        if value and begin is None:
            begin = index
        elif not value and begin is not None:
            runs.append((begin, index)); begin = None
    if not runs:
        position = max(range(len(logits)), key=lambda index: float(logits[index]))
        return position, position + 1
    candidates = []
    for left, right in runs:
        if right - left <= max_tokens:
            candidates.append((left, right))
        else:
            for start in range(left, right - max_tokens + 1):
                candidates.append((start, start + max_tokens))
    return max(candidates, key=lambda span: (
        sum(float(logits[index]) - boundary for index in range(*span)),
        span[1] - span[0], -span[0]))


def normalized_span_tokens(text, offsets, span):
    begin, end = offsets[span[0]][0], offsets[span[1] - 1][1]
    return tuple(re.findall(r"[a-z0-9]+", text[begin:end].casefold()))


def masked_relation_view(text, offsets, span, window=WINDOW):
    begin, end = offsets[span[0]][0], offsets[span[1] - 1][1]
    left = re.findall(r"[a-z0-9]+", text[:begin].casefold())[-window:]
    right = re.findall(r"[a-z0-9]+", text[end:].casefold())[:window]
    return " ".join([*left, "<ENTITY>", *right])


def span_metrics(predictions):
    exact, f1 = [], []
    for row in predictions.values():
        gold = {index for index, label in enumerate(row["labels"]) if label}
        predicted = set(range(*row["span"]))
        overlap = len(gold & predicted)
        exact.append(int(gold == predicted))
        f1.append(2 * overlap / max(1, len(gold) + len(predicted)))
    return {"instances": len(predictions), "exact_match": statistics.fmean(exact),
            "token_f1": statistics.fmean(f1)}


def predict_spans(rows, artifacts, threshold):
    output = {}
    for row in rows:
        logits = ensemble_token_logits(row, artifacts)
        span = select_contiguous_span(logits.tolist(), threshold)
        output[row["text"], row["subject"]] = {
            "span": span, "labels": row["labels"].tolist(),
            "relation_view": masked_relation_view(row["text"], row["offsets"], span),
            "span_tokens": normalized_span_tokens(row["text"], row["offsets"], span),
        }
    return output


def choose_span_config(rows, snapshots_by_seed):
    candidates, curves = [], []
    for epoch in EPOCHS:
        artifacts = [snapshots_by_seed[str(seed)][epoch] for seed in HEAD_SEEDS]
        for threshold in THRESHOLDS:
            metric = span_metrics(predict_spans(rows, artifacts, threshold))
            curves.append({"epoch": epoch, "threshold": threshold, **metric})
            candidates.append((metric["exact_match"], metric["token_f1"],
                               -epoch, -abs(threshold - .5), epoch, threshold))
    selected = max(candidates)
    return selected[-2], selected[-1], curves


def evaluation_rows(groups, feature_by_key):
    keys = {(group["query"], group["query_subject"]) for group in groups}
    keys.update((candidate["prompt"], candidate["subject"])
                for group in groups for candidate in group["candidates"])
    missing = keys - feature_by_key.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} evaluation span features")
    return [feature_by_key[key] for key in sorted(keys)]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(1, len(left | right))


def group_scores(group, predictions, relation_table, relation_artifacts, arm):
    import torch
    query = predictions[group["query"], group["query_subject"]]
    candidates = [predictions[row["prompt"], row["subject"]]
                  for row in group["candidates"]]
    texts = [query["relation_view"], *[row["relation_view"] for row in candidates]]
    logits = ensemble_logits(texts, relation_table, relation_artifacts)
    relation = logits[1:].softmax(-1) @ logits[0].softmax(-1)
    scores = relation.clamp_min(1e-12).log()
    if arm == "hard_equal":
        matches = torch.tensor([row["span_tokens"] == query["span_tokens"]
                                and bool(row["span_tokens"]) for row in candidates])
        if matches.any():
            scores = scores.masked_fill(~matches, float("-inf"))
    elif arm == "overlap8":
        overlap = torch.tensor([jaccard(query["span_tokens"], row["span_tokens"])
                                for row in candidates])
        scores = scores + 8 * overlap
    elif arm != "relation_only":
        raise ValueError(arm)
    return scores


def evaluate(groups, predictions, relation_table, relation_artifacts, arm, fold):
    output = []
    for group in groups:
        scores = group_scores(group, predictions, relation_table, relation_artifacts, arm)
        target = group["target"]; prediction = int(scores.argmax())
        order = scores.argsort(descending=True).tolist()
        output.append({"session_id": group["session_id"], "group_id": group["group_id"],
                       "target": target, "prediction": prediction,
                       "correct": int(prediction == target),
                       "reciprocal_rank": 1 / (order.index(target) + 1),
                       "fold": fold, "arm": arm})
    return output


def rank_metrics(rows):
    return {"events": len(rows),
            "top1_accuracy": statistics.fmean(row["correct"] for row in rows),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows)}


def compare(rows, baseline, seed):
    differences = [row["correct"] - baseline[(row["session_id"], row["group_id"])]["correct"]
                   for row in rows]
    return {"top1_gain": statistics.fmean(differences),
            "bootstrap_95ci": _bootstrap(differences, seed), "units": len(rows)}


def development_gate(result, vs_v10, v5_top1):
    checks = {"top1_at_least_96": result["top1_accuracy"] >= .96,
              "mrr_at_least_98": result["mrr"] >= .98,
              "gain_v10_at_least_1pp": vs_v10["top1_gain"] >= .01,
              "gain_v10_ci_positive": vs_v10["bootstrap_95ci"][0] > 0,
              "within_2pp_of_v5": result["top1_accuracy"] >= v5_top1 - .02}
    return {"checks": checks, "passed": all(checks.values())}


def tensor_hash(values):
    import torch
    digest = hashlib.sha256()
    for row in values:
        digest.update(row["text"].encode()); digest.update(b"\0")
        digest.update(row["subject"].encode()); digest.update(b"\0")
        digest.update(row["features"].contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus, _ = load_corpus(args.corpus_root)
    groups = [row for row in corpus if row["source"] == "counterfact"
              and row["target"] < MAX_KEYS]
    raw = source_rows(args.counterfact_source)
    examples = span_examples(raw, groups)
    fold_map = deterministic_folds({group["session_id"] for group in groups})
    fold_groups = {fold: [group for group in groups
                          if fold_map[group["session_id"]] == fold]
                   for fold in range(FOLDS)}
    fold_subjects = {fold: {group["query_subject"] for group in fold_groups[fold]}
                     for fold in range(FOLDS)}
    v5_artifact, v5_summary = load_v5(args.v5_root)
    context = Path(args.source_root)
    context_summary = json.loads((context / "summary.json").read_text())
    transforms = torch.load(context / "transforms.pt", map_location="cpu", weights_only=False)

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    feature_rows, projection_meta = extract_projected_features(
        model, tokenizer, examples, args.extract_batch_size, args.max_length)
    feature_by_key = {(row["text"], row["subject"]): row for row in feature_rows}
    if len(feature_by_key) != len(feature_rows):
        raise ValueError("duplicate token feature key")

    selection, checkpoints = {}, {}
    tune_predictions, test_predictions = {}, {}
    train_sizes = {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        inner_excluded = fold_subjects[outer] | fold_subjects[inner]
        snapshots, counts = {}, {}
        for seed in HEAD_SEEDS:
            snapshots[str(seed)], counts[str(seed)] = train_snapshots(
                feature_rows, inner_excluded, seed + outer * 100, args.token_batch_size)
        inner_rows = evaluation_rows(fold_groups[inner], feature_by_key)
        epoch, threshold, curves = choose_span_config(inner_rows, snapshots)
        selected_inner = [snapshots[str(seed)][epoch] for seed in HEAD_SEEDS]
        tune_predictions[outer] = predict_spans(inner_rows, selected_inner, threshold)

        outer_snapshots, outer_counts = {}, {}
        for seed in HEAD_SEEDS:
            values, outer_counts[str(seed)] = train_snapshots(
                feature_rows, fold_subjects[outer], seed + outer * 1000,
                args.token_batch_size)
            outer_snapshots[str(seed)] = values[epoch]
        outer_rows = evaluation_rows(fold_groups[outer], feature_by_key)
        test_predictions[outer] = predict_spans(
            outer_rows, list(outer_snapshots.values()), threshold)
        selection[str(outer)] = {"inner_fold": inner, "epoch": epoch,
                                 "threshold": threshold, "span_curves": curves}
        checkpoints[str(outer)] = outer_snapshots
        train_sizes[str(outer)] = {"inner": counts, "outer": outer_counts}

    relation_texts = set()
    for table in (*tune_predictions.values(), *test_predictions.values()):
        relation_texts.update(row["relation_view"] for row in table.values())
    relation_texts = sorted(relation_texts)
    relation_raw = extract(model, tokenizer, {"entity": relation_texts,
                                              "value": ["unknown"]}, args.relation_batch_size)
    embeddings, embedding_hashes = apply(
        relation_raw, {"entity": relation_texts, "value": ["unknown"]},
        transforms["transforms"], context_summary["selected_config"]["regularizer"])
    relation_table = relation_features(relation_texts, embeddings)
    del relation_raw, model; torch.cuda.empty_cache()

    final, relation_only, arm_rows, span_by_fold = [], [], {arm: [] for arm in ARMS}, {}
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        tune_scores = {arm: rank_metrics(evaluate(
            fold_groups[inner], tune_predictions[outer], relation_table,
            v5_artifact["artifacts"][str(inner)], arm, inner)) for arm in ARMS}
        arm = max(ARMS, key=lambda value: (
            tune_scores[value]["top1_accuracy"], tune_scores[value]["mrr"],
            -ARMS.index(value)))
        selection[str(outer)]["selected_arm"] = arm
        selection[str(outer)]["inner_arm_scores"] = tune_scores
        for candidate_arm in ARMS:
            arm_rows[candidate_arm].extend(evaluate(
                fold_groups[outer], test_predictions[outer], relation_table,
                v5_artifact["artifacts"][str(outer)], candidate_arm, outer))
        selected_rows = evaluate(fold_groups[outer], test_predictions[outer], relation_table,
                                 v5_artifact["artifacts"][str(outer)], arm, outer)
        final.extend(selected_rows)
        relation_only.extend(evaluate(
            fold_groups[outer], test_predictions[outer], relation_table,
            v5_artifact["artifacts"][str(outer)], "relation_only", outer))
        span_by_fold[str(outer)] = span_metrics(test_predictions[outer])

    key = lambda row: (row["session_id"], row["group_id"])
    final.sort(key=key); relation_only.sort(key=key)
    for rows in arm_rows.values(): rows.sort(key=key)
    primary_path = root / "counterfact_oof_token_span_relation.jsonl"
    primary_path.write_text("".join(json.dumps(row) + "\n" for row in final))
    relation_path = root / "counterfact_oof_relation_only.jsonl"
    relation_path.write_text("".join(json.dumps(row) + "\n" for row in relation_only))
    arm_paths = {}
    for arm, rows in arm_rows.items():
        path = root / f"counterfact_oof_{arm}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows)); arm_paths[arm] = path

    result = rank_metrics(final); v10 = load_trace(
        args.v10_root, "counterfact_oof_entity_fusion.jsonl")
    v5 = load_rows(args.v5_root, "counterfact_oof_relation_supervised.jsonl")
    vs_v10 = compare(final, v10, 19201); vs_v5 = compare(final, v5, 19211)
    v5_top1 = v5_summary["counterfact_oof"]["top1_accuracy"]
    gate = development_gate(result, vs_v10, v5_top1)
    all_span = [row for table in test_predictions.values() for row in table.values()]
    span_overall = span_metrics({str(index): row for index, row in enumerate(all_span)})
    checkpoint = root / "span_heads.pt"; torch.save(checkpoints, checkpoint)
    seconds = time.perf_counter() - started
    traces = [primary_path, relation_path, *arm_paths.values()]
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "counterfact_oof": result, "span_oof": span_overall,
               "span_by_fold": span_by_fold,
               "relation_only_oof": rank_metrics(relation_only),
               "fixed_arm_oof": {arm: rank_metrics(rows) for arm, rows in arm_rows.items()},
               "v11_vs_v10": vs_v10, "v11_vs_oracle_segmented_v5": vs_v5,
               "oracle_segmented_v5_top1": v5_top1, "development_gate": gate,
               "next_stage": "causal_automatic_scope_forgetting" if gate["passed"]
               else "audit_token_span_relation_errors",
               "terminal_status": "completed_token_span_relation_scope_v11"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "span_examples": len(feature_rows), "projection": projection_meta,
                "feature_cache_sha256": tensor_hash(feature_rows),
                "unique_relation_texts": len(relation_texts), "selection": selection,
                "training_sizes": train_sizes, "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {path.name: sha(path) for path in traces},
                "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "counterfact_source_sha256": sha(args.counterfact_source),
                "corpus_manifest_sha256": sha(Path(args.corpus_root) / "manifest.json"),
                "v5_summary_sha256": sha(Path(args.v5_root) / "summary.json"),
                "v10_summary_sha256": sha(Path(args.v10_root) / "summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "head_seeds": list(HEAD_SEEDS), "epochs": list(EPOCHS),
                "thresholds": list(THRESHOLDS), "arms": list(ARMS),
                "projection_dim": PROJECTION_DIM, "head_rank": HEAD_RANK,
                "max_length": args.max_length,
                "training_subject_labels_used": True,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root", "counterfact-source", "v5-root", "v10-root",
                 "source-root", "output", "model", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--relation-batch-size", type=int, default=256)
    parser.add_argument("--token-batch-size", type=int, default=65536)
    parser.add_argument("--max-length", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
