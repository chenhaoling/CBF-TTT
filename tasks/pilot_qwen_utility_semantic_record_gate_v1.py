"""Qwen counterfactual resource pilot for utility-supervised record forgetting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    candidate_scores,
    prepare,
    value_rows,
)


PROTOCOL = "qwen_utility_semantic_record_gate_v1_resource_pilot"
DATA_PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data"
DATA_AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data_audit"
ALPHAS = (0.0, 0.5, 1.0)
FORMAL_CHOICES = (128, 256, 512)
VALUE_SCALE = 8.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610167


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def relation_text_record(record: dict) -> str:
    text = record["input"]["statement"]
    text = text.replace(record["input"]["subject"], "[SUBJECT]", 1)
    text = text.replace(record["input"]["value"], "[VALUE]", 1)
    return "Represent the relation slot only: " + text


def relation_text_query(query: dict) -> str:
    text = query["input"]["query"]
    text = text.replace(query["input"]["subject"], "[SUBJECT]", 1)
    text = text.replace("____", "[VALUE]", 1)
    return "Represent the relation slot only: " + text


def dot(left, right) -> float:
    return float((left * right).sum())


def action_grid(size: int, alphas=ALPHAS) -> list[dict]:
    if not 0 <= size <= 3:
        raise ValueError("candidate slot count must be at most three")
    rows = [
        {"name": "grid_" + "_".join("05" if value == .5 else str(int(value))
                                      for value in values),
         "kind": "grid", "alphas": list(values)}
        for values in itertools.product(alphas, repeat=size)
    ]
    rows.append({"name": "global_clear", "kind": "global_clear", "alphas": []})
    return rows


def retain_action_name(size: int) -> str:
    return "grid_" + "_".join("1" for _ in range(size))


def state_before(episode: dict, event_index: int) -> list[dict]:
    return [
        {"record": event["record"], "strength": 1.0, "event_index": index}
        for index, event in enumerate(episode["events"][:event_index])
    ]


def candidate_indices(state: list[dict], incoming: dict, embeddings: dict[str, object]) -> tuple[int, ...]:
    if not state:
        return ()
    incoming_subject = normalize(incoming["input"]["subject"])
    incoming_vector = embeddings[relation_text_record(incoming)]
    scores = [dot(embeddings[relation_text_record(row["record"])], incoming_vector)
              for row in state]
    same_subject = [index for index, row in enumerate(state)
                    if normalize(row["record"]["input"]["subject"]) == incoming_subject]
    same_subject.sort(key=lambda index: (scores[index], index), reverse=True)
    selected = same_subject[:3]
    if len(selected) < 3:
        remaining = [index for index in range(len(state)) if index not in selected]
        remaining.sort(key=lambda index: (scores[index], index), reverse=True)
        selected.extend(remaining[:3 - len(selected)])
    return tuple(selected)


def apply_action(state: list[dict], slots: tuple[int, ...], action: dict, incoming: dict,
                 event_index: int) -> list[dict]:
    output = [{**row} for row in state]
    if action["kind"] == "global_clear":
        for row in output:
            row["strength"] = 0.0
    else:
        for slot, state_index in enumerate(slots):
            output[state_index]["strength"] *= float(action["alphas"][slot])
    output.append({"record": incoming, "strength": 1.0, "event_index": event_index})
    return output


def select_record(memory: list[dict], query: dict, embeddings: dict[str, object]):
    subject = normalize(query["input"]["subject"])
    eligible = [row for row in memory if row["strength"] > 1e-12
                and normalize(row["record"]["input"]["subject"]) == subject]
    if not eligible:
        return None
    query_vector = embeddings[relation_text_query(query)]
    return max(eligible, key=lambda row: (
        dot(embeddings[relation_text_record(row["record"])], query_vector),
        row["event_index"],
    ))


class OutcomeCache:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prepared = {}
        self.value_ids = {}
        self.results = {}
        self.hits = 0
        self.misses = 0

    def score(self, identity: str, query: dict, selected) -> dict:
        selected_value = "unknown" if selected is None else selected["record"]["input"]["value"]
        strength = 1.0 if selected is None else float(selected["strength"])
        key = (identity, selected_value, strength)
        if key in self.results:
            self.hits += 1
            return self.results[key]
        self.misses += 1
        candidates = query["candidates"]
        if identity not in self.prepared:
            self.prepared[identity] = prepare(
                self.model, self.tokenizer, query["input"]["query"], candidates
            )
        prepared = self.prepared[identity]
        value_key = (identity, selected_value)
        if value_key not in self.value_ids:
            prompt_ids, encoded = encode_candidates(
                self.tokenizer, prepared["prompt"], [selected_value]
            )
            if prompt_ids != prepared["prompt_ids"]:
                raise ValueError("selected value changed the stable prompt prefix")
            self.value_ids[value_key] = encoded[selected_value]
        residual = value_rows(self.model, self.value_ids[value_key])
        scores = candidate_scores(self.model, prepared, residual, VALUE_SCALE * strength)
        prediction = min(candidates, key=lambda value: (scores[value], value))
        result = {
            "prediction": prediction,
            "correct": int(prediction == query["answer"]),
            "answer_nll": scores[query["answer"]],
            "selected_value": selected_value,
            "selected_strength": strength,
            "selected_record_id": None if selected is None else selected["record"]["record_id"],
        }
        self.results[key] = result
        return result


def score_action(episode: dict, boundary_offset: int, slots: tuple[int, ...], action: dict,
                 embeddings: dict[str, object], cache: OutcomeCache) -> dict:
    boundary = episode["boundaries"][boundary_offset]
    event_index = boundary["event_index"]
    incoming = episode["events"][event_index]["record"]
    incoming_key = incoming["audit"]["semantic_key"]
    memory = apply_action(state_before(episode, event_index), slots, action, incoming, event_index)
    boundary_by_event = {row["event_index"]: row for row in episode["boundaries"]}
    outcomes, masses = [], []
    for index in range(event_index, len(episode["events"])):
        if index > event_index:
            memory.append({"record": episode["events"][index]["record"],
                           "strength": 1.0, "event_index": index})
        future = boundary_by_event.get(index)
        if future is None:
            continue
        masses.append(sum(row["strength"] for row in memory))
        for query_index, query in enumerate(future["queries"]):
            selected = select_record(memory, query, embeddings)
            identity = f"{episode['episode_id']}|{index}|{query_index}"
            result = cache.score(identity, query, selected)
            related = query["audit"]["semantic_key"] == incoming_key
            outcomes.append({**result, "related": related,
                             "query_role": query["role"], "event_index": index})
    unrelated = [row for row in outcomes if not row["related"]]
    return {
        "queries": len(outcomes),
        "correct": sum(row["correct"] for row in outcomes),
        "total_answer_nll": sum(row["answer_nll"] for row in outcomes),
        "unrelated_queries": len(unrelated),
        "unrelated_correct": sum(row["correct"] for row in unrelated),
        "mean_memory_mass": statistics.fmean(masses),
        "selected_null": sum(row["selected_record_id"] is None for row in outcomes),
    }


def choose_safe_oracle(outcomes: dict[str, dict], actions: list[dict], retain_name: str) -> str:
    baseline = outcomes[retain_name]
    eligible = [action for action in actions
                if outcomes[action["name"]]["unrelated_correct"] >= baseline["unrelated_correct"]]
    if not eligible:
        raise ValueError("safe barrier removed every action")
    return max(eligible, key=lambda action: (
        outcomes[action["name"]]["correct"],
        -outcomes[action["name"]]["total_answer_nll"],
        -outcomes[action["name"]]["mean_memory_mass"],
        sum(action["alphas"]),
        action["name"],
    ))["name"]


def build_label(episode: dict, boundary_offset: int, embeddings, cache, use_cuda: bool,
                alphas=ALPHAS) -> dict:
    import torch

    boundary = episode["boundaries"][boundary_offset]
    event_index = boundary["event_index"]
    incoming = episode["events"][event_index]["record"]
    state = state_before(episode, event_index)
    slots = candidate_indices(state, incoming, embeddings)
    actions = action_grid(len(slots), alphas=alphas)
    retain_name = retain_action_name(len(slots))
    if use_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); hits, misses = cache.hits, cache.misses
    outcomes = {
        action["name"]: score_action(
            episode, boundary_offset, slots, action, embeddings, cache
        ) for action in actions
    }
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    label = choose_safe_oracle(outcomes, actions, retain_name)
    ordered = sorted(actions, key=lambda action: (
        -outcomes[action["name"]]["correct"],
        outcomes[action["name"]]["total_answer_nll"],
        outcomes[action["name"]]["mean_memory_mass"],
        -sum(action["alphas"]), action["name"],
    ))
    runner_up = next((row["name"] for row in ordered if row["name"] != label), label)
    action_by_name = {row["name"]: row for row in actions}
    label_action = action_by_name[label]
    endpoint_alternatives = [row["name"] for row in actions
                             if row["kind"] == "global_clear"
                             or all(value in (0.0, 1.0) for value in row["alphas"])]
    best_endpoint = max(endpoint_alternatives, key=lambda name: (
        outcomes[name]["correct"], -outcomes[name]["total_answer_nll"],
        -outcomes[name]["mean_memory_mass"], name,
    ))
    return {
        "protocol": PROTOCOL,
        "episode_id": episode["episode_id"],
        "source_group": episode["source_group"],
        "twin": episode["twin"],
        "mechanism": episode["mechanism"],
        "length": episode["length"],
        "boundary_index": boundary_offset,
        "event_index": event_index,
        "active_historical_records": len(state),
        "candidate_record_ids": [state[index]["record"]["record_id"] for index in slots],
        "candidate_count": len(slots),
        "actions": {row["name"]: {"spec": row, **outcomes[row["name"]]} for row in actions},
        "retain_action": retain_name,
        "safe_oracle_action": label,
        "runner_up_action": runner_up,
        "best_endpoint_action": best_endpoint,
        "safe_oracle_contains_half": (
            label_action["kind"] == "grid" and any(value == .5 for value in label_action["alphas"])
        ),
        "safe_oracle_correct_gain": outcomes[label]["correct"] - outcomes[retain_name]["correct"],
        "safe_oracle_nll_gain": (
            outcomes[retain_name]["total_answer_nll"] - outcomes[label]["total_answer_nll"]
        ),
        "half_vs_endpoint_correct_gain": outcomes[label]["correct"] - outcomes[best_endpoint]["correct"],
        "half_vs_endpoint_nll_gain": (
            outcomes[best_endpoint]["total_answer_nll"] - outcomes[label]["total_answer_nll"]
        ),
        "runner_up_correct_margin": outcomes[label]["correct"] - outcomes[runner_up]["correct"],
        "runner_up_nll_margin": (
            outcomes[runner_up]["total_answer_nll"] - outcomes[label]["total_answer_nll"]
        ),
        "label_seconds": elapsed,
        "seconds_per_unique_action": elapsed / len(actions),
        "cache_hits": cache.hits - hits,
        "cache_misses": cache.misses - misses,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if use_cuda else 0.0,
    }


def unique_relation_texts(episodes: list[dict]) -> list[str]:
    values = set()
    for episode in episodes:
        values.update(relation_text_record(event["record"]) for event in episode["events"])
        for boundary in episode["boundaries"]:
            values.update(relation_text_query(query) for query in boundary["queries"])
    return sorted(values)


def extract_embeddings(model, tokenizer, texts: list[str], batch_size: int, max_length: int):
    import torch
    import torch.nn.functional as F

    output, lengths = {}, []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=False,
                            add_special_tokens=True).to("cuda")
        width = int(encoded.input_ids.shape[1])
        if width > max_length:
            raise ValueError(f"relation text exceeds {max_length}: {width}")
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False).last_hidden_state.float()
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1)
        for text, vector in zip(batch, pooled):
            output[text] = vector
        lengths.extend(encoded.attention_mask.sum(1).cpu().tolist())
    return output, [int(value) for value in lengths]


def collect(args) -> dict:
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root = Path(args.data_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    audit = json.loads((data_root / "data_audit.json").read_text())
    if manifest.get("protocol") != DATA_PROTOCOL:
        raise ValueError("unexpected utility pilot data protocol")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL or not audit.get("passed"):
        raise ValueError("utility pilot data audit did not pass")
    if audit.get("manifest_sha256") != sha256(data_root / "manifest.json"):
        raise ValueError("utility pilot manifest changed after audit")
    episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    selected = [row for index, row in enumerate(episodes)
                if index % args.num_shards == args.shard]
    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("utility pilot shard output exists")

    torch.manual_seed(261016)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = unique_relation_texts(selected)
    extraction_started = time.perf_counter()
    embeddings, token_lengths = extract_embeddings(
        model, tokenizer, texts, args.embedding_batch_size, args.max_relation_length
    )
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - extraction_started
    extraction_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_reserved = torch.cuda.max_memory_reserved() / 2**30

    cache = OutcomeCache(model, tokenizer)
    started = time.perf_counter(); rows = []
    with output.open("x") as sink, torch.inference_mode():
        for episode_index, episode in enumerate(selected):
            for boundary_offset in range(len(episode["boundaries"])):
                row = build_label(episode, boundary_offset, embeddings, cache, True)
                row["shard"] = args.shard
                sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            print(json.dumps({"shard": args.shard, "completed": episode_index + 1,
                              "total": len(selected)}), flush=True)
    label_seconds = time.perf_counter() - started
    label_allocated = max(row["peak_allocated_gib"] for row in rows)
    label_reserved = max(row["peak_reserved_gib"] for row in rows)
    resource = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "episodes": len(selected),
        "labels": len(rows),
        "relation_texts": len(texts),
        "relation_token_min": min(token_lengths),
        "relation_token_max": max(token_lengths),
        "embedding_seconds": extraction_seconds,
        "label_seconds": label_seconds,
        "total_seconds": extraction_seconds + label_seconds,
        "extraction_peak_allocated_gib": extraction_allocated,
        "extraction_peak_reserved_gib": extraction_reserved,
        "label_peak_allocated_gib": label_allocated,
        "label_peak_reserved_gib": label_reserved,
        "overall_peak_allocated_gib": max(extraction_allocated, label_allocated),
        "overall_peak_reserved_gib": max(extraction_reserved, label_reserved),
        "mean_label_seconds": statistics.fmean(row["label_seconds"] for row in rows),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "rows_sha256": sha256(output),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_root / "data_audit.json"),
        "model": model_meta,
        "tokenizer": args.tokenizer,
        "value_scale": VALUE_SCALE,
        "gpu": torch.cuda.get_device_name(),
        "development_loaded": False,
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }
    resource_output.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    return resource


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def bootstrap_group_gain(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    gains = {
        group: sum(row["safe_oracle_correct_gain"] for row in values)
        / sum(row["actions"][row["retain_action"]]["queries"] for row in values)
        for group, values in grouped.items()
    }
    groups = sorted(gains); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(gains[group] for group in sample))
    draws.sort()
    return [percentile(draws, .025), percentile(draws, .975)]


def unrelated_effects(rows: list[dict]) -> dict[str, int | bool]:
    deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"]
        for row in rows
    ]
    return {
        "added_errors": sum(max(0, -delta) for delta in deltas),
        "additional_correct": sum(max(0, delta) for delta in deltas),
        "net_correct_gain": sum(deltas),
        "rowwise_barrier_passed": all(delta >= 0 for delta in deltas),
    }


def summarize(args) -> dict:
    root = Path(args.root); rows, resources = [], []
    for shard in range(args.num_shards):
        path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) or resource["labels"] != len(part):
            raise ValueError("utility pilot shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    if len({(row["episode_id"], row["boundary_index"]) for row in rows}) != len(rows):
        raise ValueError("duplicate utility pilot labels")
    total_queries = sum(row["actions"][row["retain_action"]]["queries"] for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"] for row in rows)
    oracle_accuracy, retain_accuracy = oracle_correct / total_queries, retain_correct / total_queries
    per_mechanism = {}
    for mechanism in sorted({row["mechanism"] for row in rows}):
        part = [row for row in rows if row["mechanism"] == mechanism]
        queries = sum(row["actions"][row["retain_action"]]["queries"] for row in part)
        per_mechanism[mechanism] = {
            "labels": len(part),
            "safe_oracle_accuracy": sum(
                row["actions"][row["safe_oracle_action"]]["correct"] for row in part
            ) / queries,
            "retain_all_accuracy": sum(
                row["actions"][row["retain_action"]]["correct"] for row in part
            ) / queries,
            "optimal_actions": len({row["safe_oracle_action"] for row in part}),
        }
    half_rows = [row for row in rows if row["safe_oracle_contains_half"]]
    half_unique = [row for row in half_rows
                   if row["half_vs_endpoint_correct_gain"] > 0
                   or (row["half_vs_endpoint_correct_gain"] == 0
                       and row["half_vs_endpoint_nll_gain"] >= .01)]
    half_changes_prediction = any(row["half_vs_endpoint_correct_gain"] != 0 for row in half_rows)
    learned_action_space = (
        "ternary" if len(half_unique) / len(rows) >= .05 and half_changes_prediction
        else "binary_with_half_ablation"
    )
    critical_seconds = max(resource["total_seconds"] for resource in resources)
    bytes_per_group = sum((root / f"labels_shard{s}.jsonl").stat().st_size
                          for s in range(args.num_shards)) / 56
    projections = {
        str(choice): {
            "two_gpu_seconds": critical_seconds * choice / 56,
            "output_gib": bytes_per_group * choice / 2**30,
        } for choice in FORMAL_CHOICES
    }
    feasible = [choice for choice in FORMAL_CHOICES
                if projections[str(choice)]["two_gpu_seconds"] < 6 * 3600
                and projections[str(choice)]["output_gib"] < 50]
    selected_size = max(feasible) if feasible else None
    label_times = [row["label_seconds"] for row in rows]
    unrelated = unrelated_effects(rows)
    interval = bootstrap_group_gain(rows)
    checks = {
        "finite_complete_rows": len(rows) == 336 and all(
            math.isfinite(row["label_seconds"]) and row["label_seconds"] > 0 for row in rows
        ),
        "oracle_gain_at_least_point_05": oracle_accuracy - retain_accuracy >= .05,
        "oracle_gain_ci_lower_above_zero": interval[0] > 0,
        "zero_added_unrelated_errors": (
            unrelated["added_errors"] == 0 and unrelated["rowwise_barrier_passed"]
        ),
        "every_mechanism_has_two_actions": all(
            value["optimal_actions"] >= 2 for value in per_mechanism.values()
        ),
        "peak_reserved_below_24_gib": max(
            resource["overall_peak_reserved_gib"] for resource in resources
        ) < 24,
        "formal_size_available": selected_size is not None,
        "heldout_sources_sealed": all(
            not resource["development_loaded"] and not resource["confirmation_loaded"]
            and not resource["external_benchmarks_loaded"] for resource in resources
        ),
    }
    return {
        "protocol": PROTOCOL,
        "labels": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "source_groups": len({row["source_group"] for row in rows}),
        "future_queries": total_queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": unrelated["added_errors"],
        "additional_unrelated_correct": unrelated["additional_correct"],
        "net_unrelated_correct_gain": unrelated["net_correct_gain"],
        "unrelated_barrier_rowwise_passed": unrelated["rowwise_barrier_passed"],
        "safe_oracle_actions": dict(Counter(row["safe_oracle_action"] for row in rows)),
        "per_mechanism": per_mechanism,
        "half_strength": {
            "selected_labels": len(half_rows),
            "unique_margin_labels": len(half_unique),
            "unique_fraction": len(half_unique) / len(rows),
            "changes_prediction": half_changes_prediction,
            "primary_action_space": learned_action_space,
        },
        "label_time_seconds": {
            "mean": statistics.fmean(label_times), "median": statistics.median(label_times),
            "p90": percentile(label_times, .90), "p95": percentile(label_times, .95),
            "max": max(label_times),
        },
        "resource": {
            "two_gpu_critical_seconds": critical_seconds,
            "peak_allocated_gib": max(r["overall_peak_allocated_gib"] for r in resources),
            "peak_reserved_gib": max(r["overall_peak_reserved_gib"] for r in resources),
            "formal_projections": projections,
            "selected_formal_groups": selected_size,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
        "next_stage": "formal_label_construction" if all(checks.values()) else "stop_resource_pilot",
        "development_loaded": False,
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--embedding-batch-size", type=int, default=16)
    collect_parser.add_argument("--max-relation-length", type=int, default=256)
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
