"""Train a subject-cross-fit frozen-Qwen semantic fact-key barrier."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from tasks.build_counterfact_zero_shot_forgetting import SOURCE_SHA256
from tasks.cbf_checkpoint_native import load_model
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import lexical_features
from tasks.train_qwen_natural_forgetting_entity_key_v7 import entity_features
from tasks.train_qwen_natural_forgetting_fact_key_v8 import (
    build_examples,
    oof_metrics,
    pair_metrics,
    v7_prompt_pairs,
)
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    SEEDS,
    THRESHOLDS,
    configs,
    fit_head,
    predict,
)
from tasks.train_token_span_relation_scope_v11 import (
    HEAD_SEEDS,
    PREFIX as SPAN_PREFIX,
    SpanHead,
    character_spans,
    masked_relation_view,
    normalized_span_tokens,
    select_contiguous_span,
    span_examples,
    surface_features,
    token_labels,
    train_snapshots,
)


PROTOCOL = "qwen_natural_forgetting_semantic_fact_key_v9"
PLAN = "QWEN_NATURAL_FORGETTING_SEMANTIC_FACT_KEY_V9_PLAN.md"
SEMANTIC_PREFIX = "Fact key: "
SEMANTIC_DIM = 256
SEMANTIC_PROJECTION_SEED = 29001
SPAN_EPOCH = 15
SPAN_THRESHOLD = .3
ARMS = ("scalar", "relation_vector")

SCALAR_FEATURE_NAMES = (
    "anchor_both_nonempty", "anchor_jaccard", "anchor_containment",
    "anchor_exact", "anchor_count_ratio",
    "span_both_nonempty", "span_jaccard", "span_containment",
    "span_exact", "span_count_ratio",
    "full_lexical_jaccard", "full_lexical_containment", "full_length_ratio",
    "relation_lexical_jaccard", "relation_lexical_containment",
    "relation_length_ratio",
    "full_semantic_cosine", "full_semantic_l1", "full_semantic_rms",
    "full_semantic_max_abs", "relation_semantic_cosine",
    "relation_semantic_l1", "relation_semantic_rms",
    "relation_semantic_max_abs",
)
VECTOR_FEATURE_NAMES = SCALAR_FEATURE_NAMES + tuple(
    f"relation_absdiff_{index}" for index in range(SEMANTIC_DIM)
) + tuple(f"relation_product_{index}" for index in range(SEMANTIC_DIM))
FEATURE_NAMES = {"scalar": SCALAR_FEATURE_NAMES,
                 "relation_vector": VECTOR_FEATURE_NAMES}


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(1, len(left | right))


def _containment(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(1, min(len(left), len(right)))


def span_compatibility(left, right):
    left, right = tuple(left), tuple(right)
    return (
        float(bool(left) and bool(right)),
        _jaccard(left, right),
        _containment(left, right),
        float(bool(left) and left == right),
        min(len(left), len(right)) / max(1, max(len(left), len(right))),
    )


def semantic_scalars(left, right):
    import torch

    left = left.float(); right = right.float()
    difference = (left - right).abs()
    return (
        float(torch.dot(left, right)),
        float(difference.mean()),
        float(difference.square().mean().sqrt()),
        float(difference.max()),
    )


def pair_features(left, right, left_view, right_view, full_embeddings,
                  relation_embeddings, arm):
    import torch

    anchors, _, _, _ = entity_features(left, right)
    spans = span_compatibility(left_view["span_tokens"], right_view["span_tokens"])
    full_lexical = lexical_features(left, right)
    relation_lexical = lexical_features(
        left_view["relation_view"], right_view["relation_view"])
    left_full, right_full = full_embeddings[left], full_embeddings[right]
    left_relation = relation_embeddings[left_view["relation_view"]]
    right_relation = relation_embeddings[right_view["relation_view"]]
    scalar = (*anchors, *spans, *full_lexical, *relation_lexical,
              *semantic_scalars(left_full, right_full),
              *semantic_scalars(left_relation, right_relation))
    if arm == "scalar":
        return scalar
    if arm != "relation_vector":
        raise ValueError(arm)
    difference = (left_relation.float() - right_relation.float()).abs()
    product = left_relation.float() * right_relation.float()
    values = (*scalar, *difference.tolist(), *product.tolist())
    if len(values) != len(VECTOR_FEATURE_NAMES):
        raise ValueError("semantic vector feature width mismatch")
    return values


def extract_span_cache(model, tokenizer, texts, batch_size, max_length):
    """Extract V11-compatible token features without using a gold subject."""
    import torch

    texts = sorted(set(texts))
    hidden_dim = model.config.hidden_size
    generator = torch.Generator().manual_seed(19001)
    projection = (torch.randn(hidden_dim, 256, generator=generator)
                  / math.sqrt(hidden_dim)).to("cuda")
    tokenizer.padding_side = "right"
    output = {}
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        prompts = [SPAN_PREFIX + text for text in batch]
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True,
                            truncation=True, max_length=max_length,
                            return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        mask = encoded["attention_mask"]
        inputs = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False, return_dict=True) \
                .last_hidden_state.float()
            last = hidden[torch.arange(len(batch), device="cuda"),
                          mask.sum(-1).to("cuda") - 1]
            sentence = torch.nn.functional.layer_norm(last @ projection, (256,))
        for index, text in enumerate(batch):
            count = int(mask[index].sum())
            prompt_offsets = offsets[index, :count].tolist()
            selected = [(token_index, begin - len(SPAN_PREFIX),
                         end - len(SPAN_PREFIX))
                        for token_index, (begin, end) in enumerate(prompt_offsets)
                        if end > len(SPAN_PREFIX)
                        and begin < len(SPAN_PREFIX) + len(text)]
            if not selected:
                raise ValueError("empty token-span feature row")
            indices = torch.tensor([item[0] for item in selected], device="cuda")
            text_offsets = [(max(0, item[1]), min(len(text), item[2]))
                            for item in selected]
            projected = torch.nn.functional.layer_norm(
                hidden[index, indices] @ projection, (256,))
            global_value = sentence[index].expand_as(projected)
            surface = torch.tensor(surface_features(text, text_offsets), device="cuda")
            features = torch.cat((projected, global_value,
                                  projected * global_value, surface), dim=-1)
            output[text] = {"text": text, "features": features.half().cpu(),
                            "offsets": text_offsets}
    return output


def supervised_span_rows(raw_rows, cache):
    output = []
    for row in span_examples(raw_rows, []):
        cached = cache.get(row["text"])
        if cached is None:
            continue
        labels = token_labels(cached["offsets"],
                              character_spans(row["text"], row["subject"])[:1])
        if not any(labels):
            continue
        output.append({**row, **cached,
                       "labels": __import__("torch").tensor(labels,
                                                              dtype=__import__("torch").float32)})
    return output


def predict_text_views(rows, artifacts, threshold=SPAN_THRESHOLD):
    """Apply a span ensemble with one construction per head, not per text."""
    import torch

    rows = list(rows)
    lengths = [len(row["features"]) for row in rows]
    flat = torch.cat([row["features"].float() for row in rows])
    ensemble = torch.zeros(len(flat), dtype=torch.float32)
    for artifact in artifacts:
        head = SpanHead(artifact["dim"], artifact["seed"], artifact["rank"])
        head.load_state_dict(artifact["state"]); head.model.eval()
        values = (flat - artifact["mean"]) / artifact["std"]
        logits = []
        with torch.no_grad():
            for begin in range(0, len(values), 65536):
                logits.append(head(values[begin:begin + 65536]).float())
        ensemble += torch.cat(logits)
    ensemble /= len(artifacts)
    output, begin = {}, 0
    for row, length in zip(rows, lengths):
        logits = ensemble[begin:begin + length]; begin += length
        span = select_contiguous_span(logits.tolist(), threshold)
        output[row["text"]] = {
            "relation_view": masked_relation_view(
                row["text"], row["offsets"], span),
            "span_tokens": normalized_span_tokens(
                row["text"], row["offsets"], span),
            "span": span,
        }
    return output


def semantic_embeddings(model, tokenizer, texts, batch_size, max_length):
    import torch

    texts = sorted(set(texts))
    generator = torch.Generator().manual_seed(SEMANTIC_PROJECTION_SEED)
    projection = (torch.randn(model.config.hidden_size, SEMANTIC_DIM,
                              generator=generator)
                  / math.sqrt(model.config.hidden_size)).to("cuda")
    prefix = tokenizer.encode(SEMANTIC_PREFIX, add_special_tokens=False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    output = {}
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        target = [tokenizer.encode(text, add_special_tokens=False)[:
                    max(1, max_length - len(prefix))] for text in batch]
        rows = [prefix + row for row in target]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, device="cuda")
            mask[index, :len(row)] = 1
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask,
                                 use_cache=False, return_dict=True) \
                .last_hidden_state.float()
            last = hidden[torch.arange(len(rows), device="cuda"),
                          mask.sum(-1) - 1]
            values = last @ projection
            values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for text, value in zip(batch, values.cpu()):
            output[text] = value
    return output


def _raw_counterfact(path):
    path = Path(path)
    if sha(path) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    return json.loads(path.read_text())


def build_crossfit_views(model, tokenizer, raw, examples, span_batch_size,
                         max_length, token_batch_size):
    all_span_rows = span_examples(raw, [])
    all_texts = {row["text"] for row in all_span_rows}
    cache = extract_span_cache(model, tokenizer, all_texts,
                               span_batch_size, max_length)
    supervised = supervised_span_rows(raw, cache)
    fold_subjects = {
        fold: {str(value).casefold() for row in examples if row["fold"] == fold
               for value in (row["left_subject"], row["right_subject"])}
        for fold in range(5)
    }
    heads, views, training = {}, {}, {}
    pair_texts = {fold: sorted({value for row in examples if row["fold"] == fold
                                for value in (row["left"], row["right"])})
                  for fold in range(5)}
    for fold in range(5):
        artifacts, counts = [], {}
        for seed in HEAD_SEEDS:
            snapshots, count = train_snapshots(
                supervised, fold_subjects[fold], seed + fold * 1000,
                token_batch_size)
            artifacts.append(snapshots[SPAN_EPOCH]); counts[str(seed)] = count
        heads[str(fold)] = artifacts
        views[fold] = predict_text_views([cache[text] for text in pair_texts[fold]],
                                         artifacts)
        training[str(fold)] = {
            "excluded_subjects": len(fold_subjects[fold]), "counts": counts,
        }
    return cache, heads, views, training


def feature_tables(examples, views, full_embeddings, relation_embeddings):
    import torch

    tables = {}
    for arm in ARMS:
        tables[arm] = torch.tensor([
            pair_features(row["left"], row["right"],
                          views[row["fold"]][row["left"]],
                          views[row["fold"]][row["right"]],
                          full_embeddings, relation_embeddings, arm)
            for row in examples
        ], dtype=torch.float32)
        if tables[arm].shape[1] != len(FEATURE_NAMES[arm]):
            raise ValueError(f"{arm} feature width mismatch")
    return tables


def search_pair_head(tables, targets, examples):
    import torch

    train = [index for index, row in enumerate(examples) if row["fold"] <= 2]
    tune = [index for index, row in enumerate(examples) if row["fold"] == 3]
    tune_rows = [examples[index] for index in tune]
    search = []
    for arm in ARMS:
        features = tables[arm]
        for config_index, config in enumerate(configs()):
            values = []
            for seed in SEEDS:
                model, mean, std = fit_head(features, targets, train, config, seed)
                values.append(predict(model, mean, std, features[tune]))
            probabilities = torch.stack(values).mean(0).tolist()
            for threshold in THRESHOLDS:
                metrics = pair_metrics(tune_rows, probabilities, threshold)
                feasible = (metrics["precision"] >= .99
                            and metrics["recall"] >= .90
                            and all(value["specificity"] is None
                                    or value["specificity"] >= .98
                                    for value in metrics["by_type"].values()))
                search.append({"arm": arm, "config_index": config_index,
                               "config": config, "threshold": threshold,
                               "metrics": metrics, "feasible": feasible})
    feasible = [row for row in search if row["feasible"]]
    pool = feasible or search
    selected = max(pool, key=lambda row: (
        row["feasible"], row["metrics"]["recall"], row["metrics"]["precision"],
        min(value["specificity"] for value in row["metrics"]["by_type"].values()
            if value["specificity"] is not None),
        -ARMS.index(row["arm"]), -row["config"]["hidden"],
        row["threshold"], -row["config_index"],
    ))
    return selected, search


class SemanticFactKeyBarrier:
    def __init__(self, checkpoint):
        self.arm = checkpoint["arm"]
        self.threshold = float(checkpoint["threshold"])
        self.feature_names = tuple(checkpoint["feature_names"])
        self.heads = []
        for seed in SEEDS:
            row = checkpoint["heads"][str(seed)]
            model = BinaryHead.create(len(self.feature_names),
                                      checkpoint["config"]["hidden"], seed)
            model.load_state_dict(row["state"]); model.eval()
            self.heads.append((model, row["mean"], row["std"]))

    def probabilities(self, features):
        import torch

        with torch.no_grad():
            return torch.stack([
                model((features - mean) / std).squeeze(-1).sigmoid()
                for model, mean, std in self.heads
            ]).mean(0)


def train_independent(args):
    import torch
    from transformers import AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    raw = _raw_counterfact(args.counterfact_source)
    examples, exclusions = build_examples(args.counterfact_source)
    corpus_path = output / "counterfact_semantic_fact_key_pairs.jsonl"
    with corpus_path.open("x") as sink:
        for row in examples:
            sink.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    model, model_meta = load_model(args.model, "step81381_native")
    model.requires_grad_(False); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    _, span_heads, views, span_training = build_crossfit_views(
        model, tokenizer, raw, examples, args.extract_batch_size,
        args.max_length, args.token_batch_size)
    full_texts = sorted({value for row in examples
                         for value in (row["left"], row["right"])})
    relation_texts = sorted({item["relation_view"] for table in views.values()
                             for item in table.values()})
    full_embeddings = semantic_embeddings(
        model, tokenizer, full_texts, args.semantic_batch_size, args.max_length)
    relation_embeddings = semantic_embeddings(
        model, tokenizer, relation_texts, args.semantic_batch_size, args.max_length)
    tables = feature_tables(examples, views, full_embeddings, relation_embeddings)
    targets = torch.tensor([row["label"] for row in examples], dtype=torch.float32)
    selected, search = search_pair_head(tables, targets, examples)
    final_train = [index for index, row in enumerate(examples) if row["fold"] <= 3]
    validation = [index for index, row in enumerate(examples) if row["fold"] == 4]
    selected_features = tables[selected["arm"]]
    checkpoint = {
        "protocol": PROTOCOL, "arm": selected["arm"],
        "feature_names": FEATURE_NAMES[selected["arm"]],
        "config": selected["config"], "threshold": selected["threshold"],
        "seeds": SEEDS, "heads": {},
    }
    probabilities = []
    for seed in SEEDS:
        head, mean, std = fit_head(selected_features, targets, final_train,
                                   selected["config"], seed)
        probabilities.append(predict(head, mean, std,
                                     selected_features[validation]))
        checkpoint["heads"][str(seed)] = {
            "state": head.state_dict(), "mean": mean, "std": std,
        }
    probabilities = torch.stack(probabilities).mean(0).tolist()
    validation_rows = [examples[index] for index in validation]
    metrics = pair_metrics(validation_rows, probabilities, selected["threshold"])

    features_path = output / "semantic_fact_key_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "arms": tables, "targets": targets}, features_path)
    checkpoint_path = output / "semantic_fact_key_barrier.pt"
    torch.save(checkpoint, checkpoint_path)
    span_path = output / "crossfit_span_heads.pt"
    torch.save({"protocol": PROTOCOL, "epoch": SPAN_EPOCH,
                "threshold": SPAN_THRESHOLD, "folds": span_heads}, span_path)
    search_path = output / "tune_search.json"; write_json(search_path, search)
    predictions_path = output / "validation_predictions.jsonl"
    with predictions_path.open("x") as sink:
        for row, probability in zip(validation_rows, probabilities):
            sink.write(json.dumps({"group_id": row["group_id"], "fold": 4,
                "pair_type": row["pair_type"], "label": row["label"],
                "probability": probability,
                "prediction": int(probability >= selected["threshold"])},
                sort_keys=True) + "\n")
    counts_ok = all(metrics["by_type"].get(name, {}).get("examples", 0) >= 100
                    for name in ("positive_same_fact_key",
                                 "negative_same_relation_different_subject",
                                 "negative_same_subject_different_relation"))
    fold_subjects = {fold: {row["left_subject"].casefold() for row in examples
                            if row["fold"] == fold} for fold in range(5)}
    disjoint = all(not fold_subjects[left] & fold_subjects[right]
                   for left in range(5) for right in range(left + 1, 5))
    checks = {
        "validation_precision_at_least_point_99": metrics["precision"] >= .99,
        "validation_recall_at_least_point_90": metrics["recall"] >= .90,
        "both_negative_specificities_at_least_point_98": all(
            value["specificity"] is None or value["specificity"] >= .98
            for value in metrics["by_type"].values()),
        "at_least_100_validation_examples_per_type": counts_ok,
        "subject_folds_disjoint": disjoint,
        "span_heads_excluded_target_fold_subjects": True,
        "qwen_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
        "tune_selected_without_validation": True,
        "zsre_sources_not_loaded": True,
    }
    summary = {
        "protocol": PROTOCOL, "stage": "counterfact_semantic_validation",
        "pairs": len(examples), "excluded": exclusions,
        "pair_distribution": dict(sorted(Counter(
            row["pair_type"] for row in examples).items())),
        "fold_distribution": dict(sorted(Counter(
            row["fold"] for row in examples).items())),
        "selected": selected, "validation": metrics,
        "span_training": span_training,
        "gate": {"checks": checks,
                 "counterfact_validation_passed": all(checks.values()),
                 "zsre_transfer_authorized": all(checks.values())},
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "semantic_projection_seed": SEMANTIC_PROJECTION_SEED,
        "semantic_dim": SEMANTIC_DIM,
        "hashes": {
            "plan": sha(PLAN), "counterfact_source": sha(args.counterfact_source),
            "corpus": sha(corpus_path), "features": sha(features_path),
            "checkpoint": sha(checkpoint_path), "span_heads": sha(span_path),
            "tune_search": sha(search_path),
            "validation_predictions": sha(predictions_path),
        },
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "counterfact_loaded": True, "zsre_calibration_loaded": False,
        "zsre_test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def _external_views(model, tokenizer, texts, span_checkpoint, batch_size,
                    max_length):
    cache = extract_span_cache(model, tokenizer, texts, batch_size, max_length)
    artifacts = [artifact for fold in sorted(span_checkpoint["folds"])
                 for artifact in span_checkpoint["folds"][fold]]
    return predict_text_views([cache[text] for text in sorted(cache)], artifacts,
                              span_checkpoint["threshold"])


def transfer(args):
    import torch
    from transformers import AutoTokenizer

    independent = Path(args.independent_root)
    summary = json.loads((independent / "summary.json").read_text())
    audit = json.loads((independent / "execution_audit.json").read_text())
    if not summary["gate"]["zsre_transfer_authorized"] or not audit.get("passed"):
        raise ValueError("semantic fact-key model not released")
    checkpoint_path = independent / "semantic_fact_key_barrier.pt"
    span_path = independent / "crossfit_span_heads.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    span_checkpoint = torch.load(span_path, map_location="cpu", weights_only=False)
    barrier = SemanticFactKeyBarrier(checkpoint)
    source = Path(args.v7_root)
    source_summary = json.loads((source / "result" / "summary.json").read_text())
    source_audit = json.loads((source / "execution_audit.json").read_text())
    if source_summary.get("protocol") != "qwen_natural_forgetting_entity_key_v7" \
            or not source_audit.get("passed"):
        raise ValueError("invalid V7 source")
    source_oof = [json.loads(line) for line in
                  (source / "result" / "oof_predictions.jsonl").read_text().splitlines()
                  if line]
    pairs = v7_prompt_pairs(source, args.data_root)
    if [row[0]["state_id"] for row in pairs] != [row["state_id"] for row in source_oof]:
        raise ValueError("V7 identity mismatch")

    model, model_meta = load_model(args.model, "step81381_native")
    model.requires_grad_(False); model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    texts = sorted({value for _, incoming, top in pairs
                    for value in (incoming, top) if value is not None})
    views = _external_views(model, tokenizer, texts, span_checkpoint,
                            args.extract_batch_size, args.max_length)
    relation_texts = sorted({row["relation_view"] for row in views.values()})
    full_embeddings = semantic_embeddings(
        model, tokenizer, texts, args.semantic_batch_size, args.max_length)
    relation_embeddings = semantic_embeddings(
        model, tokenizer, relation_texts, args.semantic_batch_size, args.max_length)
    rows, feature_rows = [], []
    for (label, incoming, top), source_row in zip(pairs, source_oof):
        if top is None:
            values = [0.0] * len(barrier.feature_names)
            probability = 0.0
        else:
            values = pair_features(incoming, top, views[incoming], views[top],
                                   full_embeddings, relation_embeddings, barrier.arm)
            probability = None
        feature_rows.append(values)
        rows.append((label, source_row, probability))
    tensor = torch.tensor(feature_rows, dtype=torch.float32)
    computed = barrier.probabilities(tensor).tolist()
    transferred = []
    for index, (label, source_row, fixed) in enumerate(rows):
        probability = fixed if fixed is not None else computed[index]
        delete = (source_row["action"] == "scope_top1_a0"
                  and probability >= barrier.threshold)
        transferred.append({
            "protocol": PROTOCOL, "state_id": label["state_id"],
            "session_id": label["session_id"], "outer_fold": label["outer_fold"],
            "safe_delete": label["safe_delete"], "v7_action": source_row["action"],
            "fact_key_probability": probability,
            "fact_key_threshold": barrier.threshold,
            "action": "scope_top1_a0" if delete else "retain_all",
        })
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    predictions_path = output / "transfer_predictions.jsonl"
    with predictions_path.open("x") as sink:
        for row in transferred:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    features_path = output / "transfer_features.pt"
    torch.save({"protocol": PROTOCOL, "arm": barrier.arm,
                "feature_names": barrier.feature_names, "features": tensor},
               features_path)
    metrics = oof_metrics([row for row, _, _ in pairs], transferred)
    active = [value for value in metrics["by_outer_fold"].values()
              if value["true_positives"] + value["false_positives"]]
    checks = {
        "precision_at_least_point_98": metrics["precision"] >= .98,
        "recall_at_least_point_50": metrics["recall"] >= .50,
        "false_positive_deletes_at_most_four": metrics["false_positives"] <= 4,
        "every_active_outer_fold_precision_at_least_point_95": min(
            (value["precision"] for value in active), default=1.0) >= .95,
        "delete_rate_between_point_05_and_point_50": .05 <= metrics["delete_rate"] <= .50,
        "counterfact_validation_and_audit_passed": True,
        "zero_zsre_parameter_selection": True,
        "development_test_external_sealed": True,
    }
    result = {
        "protocol": PROTOCOL, "stage": "frozen_zsre_v7_transfer",
        "rows": len(transferred), "metrics": metrics,
        "arm": barrier.arm, "threshold": barrier.threshold,
        "gate": {"checks": checks, "offline_transfer_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "loading_info": model_meta["loading_info"],
        "hashes": {
            "plan": sha(PLAN), "independent_summary": sha(independent / "summary.json"),
            "independent_audit": sha(independent / "execution_audit.json"),
            "checkpoint": sha(checkpoint_path), "span_heads": sha(span_path),
            "v7_summary": sha(source / "result" / "summary.json"),
            "v7_audit": sha(source / "execution_audit.json"),
            "v7_oof": sha(source / "result" / "oof_predictions.jsonl"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
            "features": sha(features_path), "predictions": sha(predictions_path),
        },
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "counterfact_loaded": True, "zsre_calibration_loaded": True,
        "zsre_test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--counterfact-source", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--output", required=True)
    transfer_parser = sub.add_parser("transfer")
    transfer_parser.add_argument("--independent-root", required=True)
    transfer_parser.add_argument("--v7-root", required=True)
    transfer_parser.add_argument("--data-root", required=True)
    transfer_parser.add_argument("--model", required=True)
    transfer_parser.add_argument("--tokenizer", required=True)
    transfer_parser.add_argument("--output", required=True)
    for item in (train, transfer_parser):
        item.add_argument("--extract-batch-size", type=int, default=24)
        item.add_argument("--semantic-batch-size", type=int, default=64)
        item.add_argument("--token-batch-size", type=int, default=65536)
        item.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    result = train_independent(args) if args.command == "train" else transfer(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
