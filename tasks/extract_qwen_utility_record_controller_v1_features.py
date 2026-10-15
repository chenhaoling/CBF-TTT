"""Extract frozen, inference-safe features and binary teachers for controller V1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from scripts.audit_qwen_utility_semantic_record_gate_v1_formal_labels import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
)
from tasks.evaluate_qwen_natural_value_bridge_v1 import candidate_scores, prepare
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    extract_embeddings,
    load_jsonl,
    relation_text_record,
)


PROTOCOL = "qwen_utility_record_controller_v1_features"
BASE_FEATURE_NAMES = (
    "exact_subject",
    "exact_value",
    "subject_value_conflict",
    "exact_statement",
    "relation_masked_cosine",
    "full_statement_cosine",
    "subject_masked_cosine",
    "value_masked_cosine",
    "relation_competitor_margin",
    "relation_rank_fraction",
    "pair_yes_no_margin",
    "age_fraction",
    "recency_rank_fraction",
    "prior_exact_statement_fraction",
    "prior_subject_fraction",
    "prior_subject_value_fraction",
    "memory_size_fraction",
    "prior_access_fraction",
    "selector_membership",
    "record_coefficient",
    "explicit_reset",
    "explicit_revoke_target",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(f"context_mean_{name}" for name in BASE_FEATURE_NAMES),
    *(f"context_max_{name}" for name in BASE_FEATURE_NAMES),
)
EXPECTED_LABELS = 3072


def full_text(record: dict) -> str:
    return "Represent this session fact: " + record["input"]["statement"]


def subject_masked_text(record: dict) -> str:
    text = record["input"]["statement"].replace(
        record["input"]["subject"], "[SUBJECT]", 1
    )
    return "Represent the fact with its subject hidden: " + text


def value_masked_text(record: dict) -> str:
    text = record["input"]["statement"].replace(
        record["input"]["value"], "[VALUE]", 1
    )
    return "Represent the fact with its value hidden: " + text


def pair_prompt(candidate: dict, incoming: dict) -> str:
    return (
        "Do these two session facts update the same property? "
        "Answer with only yes or no.\n"
        f"Fact A: {candidate['input']['statement']}\n"
        f"Fact B: {incoming['input']['statement']}"
    )


def endpoint_action_names(label: dict) -> list[str]:
    return [
        name for name, action in label["actions"].items()
        if action["spec"]["kind"] == "global_clear"
        or all(value in (0.0, 1.0) for value in action["spec"]["alphas"])
    ]


def action_key(label: dict, name: str):
    action = label["actions"][name]
    return (
        action["correct"],
        -action["total_answer_nll"],
        -action["mean_memory_mass"],
        sum(action["spec"]["alphas"]),
        name,
    )


def safe_binary_action(label: dict) -> str:
    baseline = label["actions"][label["retain_action"]]
    names = [
        name for name in endpoint_action_names(label)
        if label["actions"][name]["unrelated_correct"] >= baseline["unrelated_correct"]
    ]
    if not names:
        raise ValueError("no safe binary endpoint action")
    return max(names, key=lambda name: action_key(label, name))


def action_alphas(label: dict, name: str) -> list[float]:
    spec = label["actions"][name]["spec"]
    if spec["kind"] == "global_clear":
        return [0.0] * label["candidate_count"]
    return [float(value) for value in spec["alphas"]]


def single_delete_name(size: int, slot: int) -> str:
    values = [1] * size; values[slot] = 0
    return "grid_" + "_".join(map(str, values))


def teacher_targets(label: dict) -> dict:
    teacher = safe_binary_action(label)
    alphas = action_alphas(label, teacher)
    baseline = label["actions"][label["retain_action"]]
    eligible = [
        name for name in endpoint_action_names(label)
        if label["actions"][name]["unrelated_correct"] >= baseline["unrelated_correct"]
    ]
    risks, margins = [], []
    for slot, alpha in enumerate(alphas):
        single = label["actions"][single_delete_name(len(alphas), slot)]
        risks.append(int(single["unrelated_correct"] < baseline["unrelated_correct"]))
        alternatives = [
            name for name in eligible
            if action_alphas(label, name)[slot] != alpha
        ]
        if not alternatives:
            margins.append(0.0)
            continue
        other = max(alternatives, key=lambda name: action_key(label, name))
        teacher_row, other_row = label["actions"][teacher], label["actions"][other]
        correct_margin = teacher_row["correct"] - other_row["correct"]
        nll_margin = other_row["total_answer_nll"] - teacher_row["total_answer_nll"]
        margins.append(float(max(0, correct_margin) + min(10.0, max(0.0, nll_margin)) / 10.0))
    return {
        "safe_binary_action": teacher,
        "keep_targets": [int(value == 1.0) for value in alphas],
        "delete_targets": [int(value == 0.0) for value in alphas],
        "risk_targets": risks,
        "utility_margins": margins,
    }


def dot(left, right) -> float:
    return float((left * right).sum())


def required_embedding_texts(episodes: list[dict], labels: list[dict]) -> list[str]:
    episode_by_id = {row["episode_id"]: row for row in episodes}
    texts = set()
    for label in labels:
        episode = episode_by_id[label["episode_id"]]
        event_index = label["event_index"]
        incoming = episode["events"][event_index]["record"]
        state = {event["record"]["record_id"]: event["record"]
                 for event in episode["events"][:event_index]}
        records = [incoming, *(state[identity] for identity in label["candidate_record_ids"])]
        for record in records:
            texts.update((relation_text_record(record), full_text(record),
                          subject_masked_text(record), value_masked_text(record)))
    return sorted(texts)


def base_pair_margin(model, tokenizer, candidate: dict, incoming: dict) -> float:
    import torch

    prepared = prepare(model, tokenizer, pair_prompt(candidate, incoming), ["yes", "no"])
    length = max(len(ids) for ids in prepared["encoded"].values())
    residual = torch.zeros(
        (length, model.config.hidden_size), dtype=torch.float32,
        device=model.lm_head.weight.device,
    )
    scores = candidate_scores(model, prepared, residual, 0.0)
    return float(scores["no"] - scores["yes"])


def candidate_features(
    episode: dict,
    label: dict,
    embeddings: dict[str, object],
    pair_margins: dict[tuple[str, str], float],
) -> list[list[float]]:
    event_index = label["event_index"]
    boundary = episode["boundaries"][label["boundary_index"]]
    incoming = episode["events"][event_index]["record"]
    prior_events = episode["events"][:event_index]
    prior_by_id = {event["record"]["record_id"]: (index, event["record"])
                   for index, event in enumerate(prior_events)}
    candidates = [prior_by_id[identity] for identity in label["candidate_record_ids"]]
    relation_scores = [
        dot(embeddings[relation_text_record(record)],
            embeddings[relation_text_record(incoming)]) for _, record in candidates
    ]
    ranked = sorted(range(len(candidates)), key=lambda index: (-relation_scores[index], index))
    ranks = {index: rank + 1 for rank, index in enumerate(ranked)}
    memory_size = max(1, len(prior_events))
    rows = []
    for slot, (candidate_index, record) in enumerate(candidates):
        subject = normalize(record["input"]["subject"])
        value = normalize(record["input"]["value"])
        incoming_subject = normalize(incoming["input"]["subject"])
        incoming_value = normalize(incoming["input"]["value"])
        statement = normalize(record["input"]["statement"])
        incoming_statement = normalize(incoming["input"]["statement"])
        competitors = [score for index, score in enumerate(relation_scores) if index != slot]
        competitor = max(competitors) if competitors else 0.0
        exact_repetitions = sum(
            normalize(event["record"]["input"]["statement"]) == statement
            for event in prior_events
        )
        subject_repetitions = sum(
            normalize(event["record"]["input"]["subject"]) == subject
            for event in prior_events
        )
        subject_value_repetitions = sum(
            normalize(event["record"]["input"]["subject"]) == subject
            and normalize(event["record"]["input"]["value"]) == value
            for event in prior_events
        )
        prior_accesses = sum(
            any(normalize(query["input"]["subject"]) == subject
                for query in previous["queries"])
            for previous in episode["boundaries"]
            if previous["event_index"] < event_index
        )
        values = [
            float(subject == incoming_subject),
            float(value == incoming_value),
            float(subject == incoming_subject and value != incoming_value),
            float(statement == incoming_statement),
            relation_scores[slot],
            dot(embeddings[full_text(record)], embeddings[full_text(incoming)]),
            dot(embeddings[subject_masked_text(record)],
                embeddings[subject_masked_text(incoming)]),
            dot(embeddings[value_masked_text(record)],
                embeddings[value_masked_text(incoming)]),
            relation_scores[slot] - competitor,
            ranks[slot] / len(candidates),
            pair_margins[(record["record_id"], incoming["record_id"])],
            min(1.0, (event_index - candidate_index) / 63.0),
            (event_index - candidate_index) / memory_size,
            exact_repetitions / memory_size,
            subject_repetitions / memory_size,
            subject_value_repetitions / memory_size,
            min(1.0, memory_size / 64.0),
            min(1.0, prior_accesses / 3.0),
            1.0,
            1.0,
            float(boundary["reset"]),
            float(record["audit"]["semantic_key"] in boundary["revoke_keys"]),
        ]
        if len(values) != len(BASE_FEATURE_NAMES) or not all(math.isfinite(x) for x in values):
            raise ValueError("invalid controller feature row")
        rows.append(values)
    means = [statistics.fmean(row[index] for row in rows)
             for index in range(len(BASE_FEATURE_NAMES))]
    maxima = [max(row[index] for row in rows)
              for index in range(len(BASE_FEATURE_NAMES))]
    return [row + means + maxima for row in rows]


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, label_root = Path(args.data_root), Path(args.label_root)
    data_audit = json.loads((data_root / "formal_data_audit.json").read_text())
    label_audit = json.loads((label_root / "formal_labels_audit.json").read_text())
    if data_audit.get("protocol") != DATA_AUDIT_PROTOCOL or not data_audit.get("passed"):
        raise ValueError("formal data audit did not pass")
    if label_audit.get("protocol") != LABEL_AUDIT_PROTOCOL or not label_audit.get("passed"):
        raise ValueError("formal label audit did not pass")
    episodes = load_jsonl(data_root / "formal_train_episodes.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    label_path = label_root / f"labels_shard{args.shard}.jsonl"
    labels = load_jsonl(label_path)
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("controller feature shard output exists")

    torch.manual_seed(2610161)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = required_embedding_texts(list(episode_by_id.values()), labels)
    started = time.perf_counter()
    embeddings, lengths = extract_embeddings(
        model, tokenizer, texts, args.embedding_batch_size, args.max_text_length
    )
    torch.cuda.synchronize()
    embedding_seconds = time.perf_counter() - started

    rows = []; pair_seconds = 0.0
    with output.open("x") as sink, torch.inference_mode():
        for index, label in enumerate(labels):
            episode = episode_by_id[label["episode_id"]]
            incoming = episode["events"][label["event_index"]]["record"]
            prior = {event["record"]["record_id"]: event["record"]
                     for event in episode["events"][:label["event_index"]]}
            pair_values = {}
            pair_started = time.perf_counter()
            for identity in label["candidate_record_ids"]:
                record = prior[identity]
                pair_values[(identity, incoming["record_id"])] = base_pair_margin(
                    model, tokenizer, record, incoming
                )
            pair_seconds += time.perf_counter() - pair_started
            features = candidate_features(episode, label, embeddings, pair_values)
            targets = teacher_targets(label)
            row = {
                "protocol": PROTOCOL,
                "episode_id": label["episode_id"],
                "source_group": label["source_group"],
                "boundary_index": label["boundary_index"],
                "event_index": label["event_index"],
                "candidate_record_ids": label["candidate_record_ids"],
                "feature_names": list(FEATURE_NAMES),
                "features": features,
                "targets": targets,
                "audit": {
                    "target_relation": episode["audit"]["target_relation"],
                    "mechanism": episode["mechanism"],
                    "twin": episode["twin"],
                },
                "partition": "train",
                "shard": args.shard,
            }
            sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            if (index + 1) % 25 == 0:
                print(json.dumps({"shard": args.shard, "completed": index + 1,
                                  "total": len(labels)}), flush=True)
    torch.cuda.synchronize()
    resource = {
        "protocol": PROTOCOL,
        "partition": "train",
        "shard": args.shard,
        "num_shards": args.num_shards,
        "boundaries": len(rows),
        "record_examples": sum(len(row["features"]) for row in rows),
        "feature_dimension": len(FEATURE_NAMES),
        "embedding_texts": len(texts),
        "token_min": min(lengths),
        "token_max": max(lengths),
        "embedding_seconds": embedding_seconds,
        "pair_seconds": pair_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_root / "formal_data_audit.json"),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "label_audit_sha256": sha256(label_root / "formal_labels_audit.json"),
        "label_shard_sha256": sha256(label_path),
        "model": model_meta,
        "tokenizer": args.tokenizer,
        "development_loaded": False,
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }
    resource_output.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    return resource


def summarize(args) -> dict:
    root = Path(args.root); rows = []; resources = []
    for shard in range(args.num_shards):
        row_path = root / f"features_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(row_path)
        if resource["rows_sha256"] != sha256(row_path) or resource["boundaries"] != len(part):
            raise ValueError("controller feature shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    features = [value for row in rows for candidate in row["features"] for value in candidate]
    deletes = [value for row in rows for value in row["targets"]["delete_targets"]]
    risks = [value for row in rows for value in row["targets"]["risk_targets"]]
    checks = {
        "complete_unique_boundaries": len(rows) == EXPECTED_LABELS and len(identities) == len(rows),
        "fixed_feature_schema": all(tuple(row["feature_names"]) == FEATURE_NAMES
                                    and all(len(candidate) == len(FEATURE_NAMES)
                                            for candidate in row["features"])
                                    for row in rows),
        "finite_features": all(math.isfinite(value) for value in features),
        "binary_targets": set(deletes) <= {0, 1} and set(risks) <= {0, 1},
        "both_utility_classes": set(deletes) == {0, 1},
        "both_risk_classes": set(risks) == {0, 1},
        "locked_inputs": len({resource["data_manifest_sha256"] for resource in resources}) == 1
                         and len({resource["label_audit_sha256"] for resource in resources}) == 1,
        "heldouts_sealed": all(not resource["development_loaded"]
                               and not resource["confirmation_loaded"]
                               and not resource["external_benchmarks_loaded"]
                               for resource in resources),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "checks": checks,
        "passed": passed,
        "boundaries": len(rows),
        "record_examples": len(deletes),
        "feature_dimension": len(FEATURE_NAMES),
        "delete_fraction": statistics.fmean(deletes),
        "risk_fraction": statistics.fmean(risks),
        "safe_binary_actions": dict(Counter(
            row["targets"]["safe_binary_action"] for row in rows
        )),
        "resource": {
            "two_gpu_critical_seconds": max(row["total_seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
        },
        "gate": {"checks": checks, "passed": passed},
        "next_stage": "independent_feature_audit" if passed else "stop",
        "development_loaded": False,
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--label-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--embedding-batch-size", type=int, default=16)
    collect_parser.add_argument("--max-text-length", type=int, default=256)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps(collect(args), indent=2, sort_keys=True))
    else:
        result = summarize(args)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
