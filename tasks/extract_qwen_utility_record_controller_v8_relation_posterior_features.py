"""Extract V8 frozen-Qwen relation-posterior compatibility features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gc
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    load_jsonl,
    relation_text_record,
)


PROTOCOL = "qwen_utility_record_controller_v8_relation_posterior_features"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V8_RELATION_POSTERIOR_PLAN.md"
SEEDS = (26101651, 26101652, 26101653)
HEAD_EPOCHS = 10
HEAD_LEARNING_RATE = .01
HEAD_WEIGHT_DECAY = .05
POSTERIOR_FEATURE_NAMES = (
    "relation_posterior_dot_mean",
    "relation_posterior_dot_std",
    "relation_logit_cosine_mean",
    "relation_logit_cosine_std",
    "relation_top_class_agreement_fraction",
    "relation_incoming_confidence_mean",
    "relation_posterior_competitor_margin",
    "relation_posterior_rank_fraction",
)
FEATURE_NAMES = (*BASE_FEATURE_NAMES, *POSTERIOR_FEATURE_NAMES)

DOMAIN_SPECS = {
    "templama_train": {
        "episode_file": "train_episodes.jsonl",
        "data_audit": "data_audit.json",
        "feature_audit": "train_feature_audit.json",
        "training": True,
    },
    "lama_train": {
        "episode_file": "formal_train_episodes.jsonl",
        "data_audit": "formal_data_audit.json",
        "feature_audit": "feature_audit.json",
        "training": True,
    },
    "templama_dev": {
        "episode_file": "development_episodes.jsonl",
        "data_audit": "data_audit.json",
        "feature_audit": "development_feature_audit.json",
        "training": False,
    },
    "lama_dev": {
        "episode_file": "development_episodes.jsonl",
        "data_audit": "development_data_audit.json",
        "feature_audit": "development_feature_audit.json",
        "training": False,
    },
}


def passed_audit(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not value.get("passed"):
        raise ValueError(f"upstream audit did not pass: {path}")
    return value


def load_domains(args) -> tuple[dict, dict]:
    domains, audit_hashes = {}, {}
    for name, spec in DOMAIN_SPECS.items():
        data_root = Path(getattr(args, f"{name}_data_root"))
        feature_root = Path(getattr(args, f"{name}_feature_root"))
        data_audit = data_root / spec["data_audit"]
        feature_audit = feature_root / spec["feature_audit"]
        passed_audit(data_audit); passed_audit(feature_audit)
        episodes = load_jsonl(data_root / spec["episode_file"])
        features = []
        for shard in (0, 1):
            features.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
        episode_by_id = {row["episode_id"]: row for row in episodes}
        if len(episode_by_id) != len(episodes):
            raise ValueError(f"duplicate episode identity in {name}")
        if any(row["episode_id"] not in episode_by_id for row in features):
            raise ValueError(f"feature episode missing from {name}")
        domains[name] = {
            "episodes": episodes,
            "episode_by_id": episode_by_id,
            "features": features,
            "data_root": data_root,
            "feature_root": feature_root,
            "training": spec["training"],
        }
        audit_hashes[f"{name}_data"] = sha256(data_audit)
        audit_hashes[f"{name}_features"] = sha256(feature_audit)
    return domains, audit_hashes


def relation_mentions(domains: dict) -> dict:
    relations_by_text = defaultdict(set)
    required_texts = set()
    for domain in domains.values():
        for episode in domain["episodes"]:
            for event in episode["events"]:
                record = event["record"]
                text = relation_text_record(record)
                required_texts.add(text)
                if domain["training"]:
                    relations_by_text[text].add(record["audit"]["relation_id"])
    ambiguous = sorted(
        text for text, relations in relations_by_text.items()
        if len(relations) > 1
    )
    examples = sorted(
        (text, next(iter(relations)))
        for text, relations in relations_by_text.items()
        if len(relations) == 1
    )
    return {
        "required_texts": sorted(required_texts),
        "training_examples": examples,
        "ambiguous_training_texts": ambiguous,
    }


def extract_embeddings(model, tokenizer, texts: list[str], batch_size: int,
                       max_length: int) -> tuple[object, list[int]]:
    import torch
    import torch.nn.functional as F

    vectors, lengths = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=False, add_special_tokens=True,
        ).to("cuda")
        if int(encoded.input_ids.shape[1]) > max_length:
            raise ValueError("V8 relation text exceeds registered maximum")
        with torch.inference_mode():
            hidden = model.model(
                **encoded, use_cache=False
            ).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = F.normalize(
                (hidden * mask).sum(1) / mask.sum(1), dim=1
            )
        vectors.append(pooled.cpu().to(torch.float16))
        lengths.extend(int(value) for value in encoded.attention_mask.sum(1).cpu())
    return torch.cat(vectors), lengths


def prepare_relation_training(mentions: dict, text_to_index: dict,
                              vectors) -> dict:
    import torch

    examples = mentions["training_examples"]
    relations = sorted({relation for _, relation in examples})
    relation_to_index = {relation: index for index, relation in enumerate(relations)}
    x = torch.stack([
        vectors[text_to_index[text]].float() for text, _ in examples
    ])
    y = torch.tensor([
        relation_to_index[relation] for _, relation in examples
    ], dtype=torch.long)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    counts = torch.bincount(y, minlength=len(relations)).float().clamp_min(1)
    weights = (counts.mean() / counts).sqrt()
    weights /= weights.mean()
    return {
        "x": x, "y": y, "mean": mean, "std": std,
        "weights": weights, "relations": relations,
        "counts": {relation: int(counts[index])
                   for index, relation in enumerate(relations)},
    }


def fit_relation_heads(training: dict, device) -> list[dict]:
    import torch

    x = ((training["x"] - training["mean"]) / training["std"]).to(device)
    y = training["y"].to(device)
    weights = training["weights"].to(device)
    heads = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        head = torch.nn.Linear(x.shape[1], len(training["relations"])).to(device)
        torch.nn.init.normal_(head.weight, std=.001)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=HEAD_LEARNING_RATE,
            weight_decay=HEAD_WEIGHT_DECAY,
        )
        trace = []
        for epoch in range(1, HEAD_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            trace.append({"epoch": epoch, "loss": float(loss),
                          "gradient_norm": float(norm)})
        heads.append({
            "seed": seed,
            "state_dict": copy.deepcopy({
                key: value.detach().cpu() for key, value in head.state_dict().items()
            }),
            "trace": trace,
        })
    return heads


def predict_relation_logits(vectors, training: dict, heads: list[dict],
                            device, batch_size: int = 2048) -> list[object]:
    import torch

    normalized = (vectors.float() - training["mean"]) / training["std"]
    outputs = []
    for artifact in heads:
        head = torch.nn.Linear(normalized.shape[1], len(training["relations"]))
        head.load_state_dict(artifact["state_dict"])
        head.to(device).eval()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(normalized), batch_size):
                rows.append(head(normalized[start:start + batch_size].to(device)).cpu())
        outputs.append(torch.cat(rows))
    return outputs


def compatibility_features(candidate_logits: list, incoming_logits: list,
                           competitor_scores: list[float], slot: int) -> list[float]:
    import torch
    import torch.nn.functional as F

    dots, cosines, agreements, confidences = [], [], [], []
    for candidate, incoming in zip(candidate_logits, incoming_logits):
        candidate_probability = candidate.softmax(-1)
        incoming_probability = incoming.softmax(-1)
        dots.append(float(candidate_probability @ incoming_probability))
        cosines.append(float(F.cosine_similarity(
            candidate.unsqueeze(0), incoming.unsqueeze(0)
        )[0]))
        agreements.append(float(candidate.argmax() == incoming.argmax()))
        confidences.append(float(incoming_probability.max()))
    score = statistics.fmean(dots)
    alternatives = [
        value for index, value in enumerate(competitor_scores) if index != slot
    ]
    competitor = max(alternatives) if alternatives else 0.0
    ordered = sorted(
        range(len(competitor_scores)),
        key=lambda index: (-competitor_scores[index], index),
    )
    rank = ordered.index(slot) + 1
    values = [
        score,
        statistics.pstdev(dots),
        statistics.fmean(cosines),
        statistics.pstdev(cosines),
        statistics.fmean(agreements),
        statistics.fmean(confidences),
        score - competitor,
        rank / len(competitor_scores),
    ]
    if len(values) != len(POSTERIOR_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError("invalid V8 compatibility features")
    return values


def pair_records(domain: dict, row: dict) -> tuple[list[dict], dict]:
    episode = domain["episode_by_id"][row["episode_id"]]
    incoming = episode["events"][row["event_index"]]["record"]
    prior = {
        event["record"]["record_id"]: event["record"]
        for event in episode["events"][:row["event_index"]]
    }
    if set(row["candidate_record_ids"]) - set(prior):
        raise ValueError("V8 candidate missing from prior memory")
    return [prior[identity] for identity in row["candidate_record_ids"]], incoming


def augment_row(domain: dict, row: dict, text_to_index: dict,
                logits: list) -> dict:
    candidates, incoming = pair_records(domain, row)
    incoming_index = text_to_index[relation_text_record(incoming)]
    incoming_logits = [head[incoming_index] for head in logits]
    candidate_logits = [
        [head[text_to_index[relation_text_record(record)]] for head in logits]
        for record in candidates
    ]
    scores = [statistics.fmean(float(
        candidate.softmax(-1) @ incoming_value.softmax(-1)
    ) for candidate, incoming_value in zip(values, incoming_logits))
              for values in candidate_logits]
    features = [
        [*base, *compatibility_features(
            candidate_logits[slot], incoming_logits, scores, slot
        )]
        for slot, base in enumerate(row["features"])
    ]
    if len(features) != len(row["candidate_record_ids"]):
        raise ValueError("V8 feature/candidate count mismatch")
    return {**row, "feature_names": list(FEATURE_NAMES), "features": features,
            "protocol": PROTOCOL}


def add_arguments(parser) -> None:
    for name in DOMAIN_SPECS:
        parser.add_argument(f"--{name.replace('_', '-')}-data-root", required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-feature-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)


def extract(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    domains, audit_hashes = load_domains(args)
    mentions = relation_mentions(domains)
    output_root.mkdir(parents=True)
    torch.manual_seed(2610168)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    started = time.perf_counter()
    vectors, lengths = extract_embeddings(
        model, tokenizer, mentions["required_texts"],
        args.embedding_batch_size, args.max_text_length,
    )
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    del model
    gc.collect(); torch.cuda.empty_cache()
    text_to_index = {
        text: index for index, text in enumerate(mentions["required_texts"])
    }
    training = prepare_relation_training(mentions, text_to_index, vectors)
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    head_started = time.perf_counter()
    heads = fit_relation_heads(training, device)
    logits = predict_relation_logits(vectors, training, heads, device)
    head_seconds = time.perf_counter() - head_started

    embedding_path = output_root / "relation_embeddings.pt"
    head_path = output_root / "relation_heads.pt"
    torch.save({
        "protocol": PROTOCOL,
        "texts": mentions["required_texts"],
        "vectors": vectors,
        "token_lengths": lengths,
        "model_meta": model_meta,
        "upstream_audit_hashes": audit_hashes,
    }, embedding_path)
    torch.save({
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "relations": training["relations"],
        "relation_counts": training["counts"],
        "mean": training["mean"],
        "std": training["std"],
        "heads": heads,
        "training_examples": len(training["y"]),
        "ambiguous_training_texts": mentions["ambiguous_training_texts"],
        "embedding_sha256": sha256(embedding_path),
    }, head_path)

    counts = {}
    feature_hashes = {}
    for name, domain in domains.items():
        domain_root = output_root / name
        domain_root.mkdir()
        rows = [augment_row(domain, row, text_to_index, logits)
                for row in domain["features"]]
        identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"duplicate V8 feature identity in {name}")
        by_shard = {0: [], 1: []}
        original_shards = {}
        for shard in (0, 1):
            for row in load_jsonl(
                domain["feature_root"] / f"features_shard{shard}.jsonl"
            ):
                original_shards[(row["episode_id"], row["boundary_index"])] = shard
        for row in rows:
            by_shard[original_shards[(row["episode_id"], row["boundary_index"])]].append(row)
        for shard, values in by_shard.items():
            path = domain_root / f"features_shard{shard}.jsonl"
            with path.open("x") as sink:
                for row in values:
                    sink.write(json.dumps(row) + "\n")
            feature_hashes[f"{name}_shard{shard}"] = sha256(path)
        counts[name] = {
            "rows": len(rows),
            "candidates": sum(len(row["features"]) for row in rows),
            "shard_rows": {str(shard): len(values)
                           for shard, values in by_shard.items()},
        }
    checks = {
        "upstream_audits_passed": True,
        "registered_feature_dimension": len(FEATURE_NAMES) == 74,
        "three_relation_heads": len(heads) == 3,
        "finite_head_training": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for head in heads for epoch in head["trace"]
        ),
        "all_feature_rows_written": counts == {
            "templama_train": {"rows": 1536, "candidates": 4392,
                                "shard_rows": {"0": 768, "1": 768}},
            "lama_train": {"rows": 1536, "candidates": 4684,
                            "shard_rows": {"0": 768, "1": 768}},
            "templama_dev": {"rows": 768, "candidates": 2204,
                              "shard_rows": {"0": 384, "1": 384}},
            "lama_dev": {"rows": 192, "candidates": 588,
                          "shard_rows": {"0": 96, "1": 96}},
        },
        "heldout_sources_sealed": True,
    }
    summary = {
        "protocol": PROTOCOL,
        "plan": PLAN,
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "feature_names": list(FEATURE_NAMES),
        "relations": len(training["relations"]),
        "relation_counts": training["counts"],
        "relation_training_examples": len(training["y"]),
        "unique_relation_texts": len(mentions["required_texts"]),
        "ambiguous_training_texts": len(mentions["ambiguous_training_texts"]),
        "token_length_min": min(lengths),
        "token_length_max": max(lengths),
        "embedding_seconds": extraction_seconds,
        "relation_head_seconds": head_seconds,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "embedding_sha256": sha256(embedding_path),
        "relation_heads_sha256": sha256(head_path),
        "feature_sha256": feature_hashes,
        "upstream_audit_hashes": audit_hashes,
        "model_meta": model_meta,
        "templama_test_loaded": False,
        "lama_confirmation_loaded": False,
        "longmemeval_loaded": False,
        "locomo_loaded": False,
        "next_stage": "v8_relation_posterior_feature_audit",
        "terminal_status": "v8_relation_posterior_features_completed",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--max-text-length", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = extract(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
