"""Run the preregistered actual-Qwen causal-ledger resource/content pilot."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_causal_ledger_residual import (
    FAMILIES,
    PILOT_SEED,
    PROTOCOL as DATA_PROTOCOL,
    load_pilot,
    parse_visible_block,
)
from tasks.cbf_qwen_lowrank_bridge import unit_key, unit_value
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates, normalize_generation
from tasks.evaluate_v18_lowrank_fastmemory_bridge import (
    FastMemoryReadSession,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_dynamic_record_forgetting import (
    ACTIONS,
    candidate_indices,
    valid_actions,
)
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_causal_ledger_residual_pilot_v1"
SCALES = (4.0, 8.0, 16.0)
PRIMARY_SCALE = 8.0
FORMAL_CHOICES = (16, 32, 64)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610159
HEADER = "Use the fast-memory ledger and reply with only the requested code."


def assignment(parsed):
    return {
        "batch": parsed["record"],
        "facts": dict(parsed["facts"]),
        "old_refs": list(parsed["old_refs"]),
        "reset": bool(parsed["reset"]),
    }


def semantic_apply(state, event, block_index):
    output = copy.deepcopy(state)
    if event["reset"]:
        for row in output:
            row["weight"] = 0.0
    for reference in event["old_refs"]:
        for row in output:
            if row["batch"] == reference:
                row["weight"] = 0.0
    output.append({
        "batch": event["batch"],
        "facts": dict(event["facts"]),
        "weight": 1.0,
        "block_index": int(block_index),
    })
    return output


def apply_grid_action(state, event, block_index, slots, action):
    output = copy.deepcopy(state)
    clear, retention = action
    if clear:
        for row in output:
            row["weight"] = 0.0
    else:
        for slot, state_index in enumerate(slots):
            output[state_index]["weight"] *= float(retention[slot])
    output.append({
        "batch": event["batch"],
        "facts": dict(event["facts"]),
        "weight": 1.0,
        "block_index": int(block_index),
    })
    return output


def visible_facts(state):
    """Return newest active fact per key without discarding restorable records."""
    output = {}
    for row in state:
        if row["weight"] <= 1e-12:
            continue
        for key, value in row["facts"].items():
            output[key] = {
                "batch": row["batch"], "key": key, "value": value,
                "weight": float(row["weight"]), "block_index": row["block_index"],
            }
    return output


class KeyCache:
    seed = 0

    def __init__(self, keys):
        self.keys = keys

    def get(self, kind, text):
        if kind != "entity":
            raise ValueError("causal-ledger candidate selection requests entity keys only")
        return self.keys[text]


def build_factor_cache(model, tokenizer, inp, target):
    import torch

    keys, values = set(), {inp["null_value"]}
    for events in target["semantic_events"]:
        for event in events:
            if event["op"] == "assign":
                keys.update(event["facts"])
                values.update(event["facts"].values())
    for query in inp["queries"]:
        keys.add(query["key"]); values.update(query["candidates"])
    key_cache = {key: unit_key(model, tokenizer, key) for key in sorted(keys)}
    value_cache = {}
    token_ids = {}
    for value in sorted(values):
        ids = tokenizer.encode(value, add_special_tokens=False)
        if len(ids) != 5:
            raise ValueError("causal-ledger opaque value is not exactly five tokens")
        token_ids[value] = ids
        value_cache[value] = torch.stack([unit_value(model, token_id) for token_id in ids])
    return {
        "keys": key_cache,
        "values": value_cache,
        "token_ids": token_ids,
        "selector": KeyCache(key_cache),
    }


def render_query(text):
    return f"{HEADER}\n\nQuery: {text}\nAnswer:\n"


def prepare_query(model, tokenizer, query):
    import torch

    prompt = render_query(query["text"])
    prompt_ids, encoded = encode_candidates(tokenizer, prompt, query["candidates"])
    if any(len(ids) != 5 for ids in encoded.values()):
        raise ValueError("causal-ledger query candidate length changed")
    session = FastMemoryReadSession(model)
    prefix_hidden = session._forward(prompt_ids)
    hidden = {}
    for answer, ids in encoded.items():
        branch = session.clone()
        tail = branch._forward(ids[:-1])
        hidden[answer] = torch.cat((prefix_hidden[:, -1:], tail), dim=1)[0].float()
    return {
        "session": session,
        "prefix_hidden": prefix_hidden,
        "hidden": hidden,
        "encoded": encoded,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
    }


def score_residuals(model, prepared, candidates, residuals, chunk_size=18):
    """Return full-vocabulary five-token NLL for residual/action batches."""
    import torch
    import torch.nn.functional as F

    action_count = residuals.shape[0]
    scores = {
        str(scale): {candidate: [None] * action_count for candidate in candidates}
        for scale in SCALES
    }
    scale_tensor = torch.tensor(SCALES, dtype=torch.float32, device=residuals.device)
    for candidate in candidates:
        base = prepared["hidden"][candidate].to(residuals.device)
        labels = torch.tensor(
            prepared["encoded"][candidate], dtype=torch.long, device=residuals.device
        )
        for start in range(0, action_count, chunk_size):
            stop = min(action_count, start + chunk_size)
            block = residuals[start:stop]
            adjusted = base[None, None] + scale_tensor[None, :, None, None] * block[:, None]
            logits = model.lm_head(
                adjusted.reshape(-1, adjusted.shape[-1]).to(model.lm_head.weight.dtype)
            ).float()
            repeated = labels.repeat((stop - start) * len(SCALES))
            losses = F.cross_entropy(logits, repeated, reduction="none").reshape(
                stop - start, len(SCALES), 5
            ).mean(2)
            for scale_index, scale in enumerate(SCALES):
                values = losses[:, scale_index].detach().cpu().tolist()
                scores[str(scale)][candidate][start:stop] = values
            del adjusted, logits, losses
    output = []
    for index in range(action_count):
        by_scale = {}
        for scale in SCALES:
            table = {
                candidate: scores[str(scale)][candidate][index]
                for candidate in candidates
            }
            prediction = min(candidates, key=lambda value: (table[value], value))
            by_scale[str(scale)] = {"scores": table, "prediction": prediction}
        output.append(by_scale)
    return output


def generate_with_residual(model, tokenizer, prepared, residual, scale, max_new_tokens=13):
    import torch

    session = prepared["session"].clone()
    hidden = prepared["prefix_hidden"]
    generated = []
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
    stop_reason = "max_new_tokens"
    for index in range(max_new_tokens):
        value = residual[index] if index < residual.shape[0] else residual.new_zeros(
            residual.shape[1]
        )
        adjusted = hidden[:, -1].float() + float(scale) * value.unsqueeze(0)
        logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
        next_id = int(logits.argmax(-1).item()); generated.append(next_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if next_id in eos_ids:
            stop_reason = "eos"; break
        if "\n" in text:
            stop_reason = "newline"; break
        if index + 1 < max_new_tokens:
            hidden = session._forward([next_id])
    return {
        "ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "tokens": len(generated),
        "stop_reason": stop_reason,
        "max_new_tokens": max_new_tokens,
        "scale": float(scale),
    }


def weighted_residual(state, query_key, factors, null_code, visible_only=False):
    import torch

    rows = list(visible_facts(state).values()) if visible_only else [
        {"key": key, "value": value, "weight": float(row["weight"])}
        for row in state if row["weight"] > 1e-12
        for key, value in row["facts"].items()
    ]
    template = next(iter(factors["values"].values()))
    residual = torch.zeros_like(template)
    query = factors["keys"][query_key]
    exact_present = False
    for row in rows:
        similarity = float(torch.dot(query, factors["keys"][row["key"]]))
        residual += float(row["weight"]) * similarity * factors["values"][row["value"]]
        exact_present |= row["key"] == query_key
    if not exact_present:
        residual += factors["values"][null_code]
    return residual


def single_component_residual(query_key, value, factors):
    import torch
    similarity = float(torch.dot(factors["keys"][query_key], factors["keys"][query_key]))
    return similarity * factors["values"][value]


def randomize_residual(residual, identity):
    import torch

    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    signs = torch.randint(0, 2, residual.shape, generator=generator, dtype=torch.int8)
    signs = signs.to(residual.device, dtype=residual.dtype).mul_(2).sub_(1)
    return residual * signs


def lexicographic_label(outcomes, actions, scale=PRIMARY_SCALE):
    key = str(scale)
    best = None
    for index, action in enumerate(actions):
        row = outcomes[index][key]
        rank = (
            round(row["accuracy"] / 1e-12) * 1e-12,
            -round(row["nll"] / 1e-6) * 1e-6,
            -row["removed_mass"],
            -ACTIONS.index(action),
        )
        if best is None or rank > best[0]:
            best = (rank, index, action)
    return best[1], best[2]


def bootstrap_interval(group_rows, control, draws=BOOTSTRAP_DRAWS):
    groups = sorted(group_rows)
    rng = random.Random(BOOTSTRAP_SEED + {"wrong": 1, "zero": 2, "random": 3}[control])
    values = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(statistics.fmean(group_rows[group][control] for group in sample))
    values.sort()
    return [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    data_root, labels_path, content_path = (
        Path(args.data_root), Path(args.labels_output), Path(args.content_output)
    )
    if labels_path.exists() or content_path.exists():
        raise FileExistsError("causal-ledger pilot shard output exists")
    manifest, pairs = load_pilot(data_root)
    if manifest.get("seed") != PILOT_SEED or manifest.get("groups_per_family") != 8:
        raise ValueError("causal-ledger pilot dependency changed")
    selected = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard]
    torch.manual_seed(202610159); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    extraction_peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    started = time.perf_counter(); label_rows, content_rows = [], []
    with labels_path.open("x") as label_sink, content_path.open("x") as content_sink, torch.no_grad():
        for pair_index, (inp, target) in enumerate(selected):
            factors = build_factor_cache(model, tokenizer, inp, target)
            answers = {row["id"]: row for row in target["answers"]}
            queries_by_boundary = {}
            for query in inp["queries"]:
                queries_by_boundary.setdefault(query["after_block"], []).append(query)
            state = []
            for block in inp["blocks"]:
                event = assignment(parse_visible_block(block["text"], "pilot"))
                if block["index"] not in set(inp["decision_boundaries"]):
                    state = semantic_apply(state, event, block["index"])
                    continue
                slots = candidate_indices(state, event, factors["selector"])
                actions = valid_actions(event, slots)
                action_states = [
                    apply_grid_action(
                        state, event, block["index"], slots, action
                    ) for action in actions
                ]
                queries = queries_by_boundary[block["index"]]
                prepared = [prepare_query(model, tokenizer, query) for query in queries]
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                label_started = time.perf_counter(); query_scores = []
                for query, prep in zip(queries, prepared):
                    residuals = torch.stack([
                        weighted_residual(
                            candidate_state, query["key"], factors, inp["null_value"]
                        ) for candidate_state in action_states
                    ])
                    query_scores.append(score_residuals(
                        model, prep, query["candidates"], residuals, args.action_chunk
                    ))
                torch.cuda.synchronize(); label_seconds = time.perf_counter() - label_started
                peak_allocated = torch.cuda.max_memory_allocated() / 2**30
                peak_reserved = torch.cuda.max_memory_reserved() / 2**30
                outcomes = []
                for action_index, action in enumerate(actions):
                    by_scale = {}
                    for scale in SCALES:
                        key = str(scale); correct, nll = [], []
                        for query, scores in zip(queries, query_scores):
                            truth = answers[query["id"]]["answer"]
                            table = scores[action_index][key]
                            correct.append(int(table["prediction"] == truth))
                            nll.append(table["scores"][truth])
                        by_scale[key] = {
                            "accuracy": statistics.fmean(correct),
                            "nll": statistics.fmean(nll),
                            "removed_mass": sum(1.0 - value for value in action[1]),
                        }
                    outcomes.append(by_scale)
                label_index, label = lexicographic_label(outcomes, actions)
                row = {
                    "protocol": PROTOCOL,
                    "shard": args.shard,
                    "episode_id": inp["id"],
                    "group_id": inp["group_id"],
                    "family": target["family"],
                    "block_index": block["index"],
                    "slots": list(slots),
                    "valid_action_count": len(actions),
                    "actions": [[action[0], list(action[1])] for action in actions],
                    "outcomes": outcomes,
                    "label_index": label_index,
                    "label": [label[0], list(label[1])],
                    "label_seconds": label_seconds,
                    "seconds_per_action": label_seconds / len(actions),
                    "prompt_tokens_max": max(value["prompt_tokens"] for value in prepared),
                    "answer_tokens": 5,
                    "candidate_count": 6,
                    "active_records": sum(row["weight"] > 1e-12 for row in state),
                    "peak_allocated_gib": peak_allocated,
                    "peak_reserved_gib": peak_reserved,
                }
                label_sink.write(json.dumps(row) + "\n"); label_sink.flush(); label_rows.append(row)

                for query, prep in zip(queries, prepared):
                    truth = answers[query["id"]]
                    correct = single_component_residual(query["key"], truth["answer"], factors)
                    wrong_value = next(
                        value for value in query["candidates"]
                        if value not in {truth["answer"], inp["null_value"]}
                    )
                    wrong = single_component_residual(query["key"], wrong_value, factors)
                    zero = torch.zeros_like(correct)
                    randomized = randomize_residual(
                        correct, f"{inp['id']}|{query['id']}|random"
                    )
                    controls = torch.stack([correct, wrong, zero, randomized])
                    scored = score_residuals(
                        model, prep, query["candidates"], controls, args.action_chunk
                    )
                    generations = {}
                    for scale in SCALES:
                        generation = generate_with_residual(
                            model, tokenizer, prep, correct, scale
                        )
                        generation["normalized_exact"] = int(
                            normalize_generation(generation["text"])
                            == normalize_generation(truth["answer"])
                        )
                        generations[str(scale)] = generation
                    control_row = {
                        "protocol": PROTOCOL,
                        "shard": args.shard,
                        "episode_id": inp["id"],
                        "group_id": inp["group_id"],
                        "family": target["family"],
                        "block_index": block["index"],
                        "query_id": query["id"],
                        "role": query["role"],
                        "absent": truth["absent"],
                        "answer": truth["answer"],
                        "candidates": query["candidates"],
                        "controls": {
                            name: {
                                "by_scale": scored[index],
                                "answer_nll": {
                                    str(scale): scored[index][str(scale)]["scores"][truth["answer"]]
                                    for scale in SCALES
                                },
                            }
                            for index, name in enumerate(("correct", "wrong", "zero", "random"))
                        },
                        "correct_generation": generations,
                        "prompt_sha256": hashlib.sha256(prep["prompt"].encode()).hexdigest(),
                        "prompt_tokens": prep["prompt_tokens"],
                    }
                    content_sink.write(json.dumps(control_row) + "\n"); content_sink.flush()
                    content_rows.append(control_row)
                # Pilot labels never control the causal state followed by later blocks.
                state = semantic_apply(state, event, block["index"])
            print(json.dumps({
                "completed": pair_index + 1, "total": len(selected), "episode": inp["id"]
            }), flush=True)
    resource = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "pairs": len(selected),
        "labels": len(label_rows),
        "content_rows": len(content_rows),
        "seconds": time.perf_counter() - started,
        "labels_sha256": sha(labels_path),
        "content_sha256": sha(content_path),
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "extraction_peak_allocated_gib": extraction_peak_allocated,
        "extraction_peak_reserved_gib": extraction_peak_reserved,
        "label_peak_allocated_gib": max(row["peak_allocated_gib"] for row in label_rows),
        "label_peak_reserved_gib": max(row["peak_reserved_gib"] for row in label_rows),
        "overall_peak_allocated_gib": max(
            extraction_peak_allocated, max(row["peak_allocated_gib"] for row in label_rows)
        ),
        "overall_peak_reserved_gib": max(
            extraction_peak_reserved, max(row["peak_reserved_gib"] for row in label_rows)
        ),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "scales": list(SCALES),
        "primary_scale": PRIMARY_SCALE,
        "gpu": torch.cuda.get_device_name(),
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
    }
    write_json(args.resource_output, resource)
    return resource


def paired_interval(content_rows, control):
    grouped = {}
    for row in content_rows:
        key = str(PRIMARY_SCALE)
        gain = (
            row["controls"][control]["answer_nll"][key]
            - row["controls"]["correct"]["answer_nll"][key]
        )
        grouped.setdefault(row["group_id"], []).append(gain)
    group_rows = {
        group: {control: statistics.fmean(values)} for group, values in grouped.items()
    }
    mean = statistics.fmean(value[control] for value in group_rows.values())
    return {
        "mean": mean,
        "bootstrap_95ci": bootstrap_interval(group_rows, control),
    }


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    _, pairs = load_pilot(root / "data")
    labels, content, resources = [], [], []
    for shard in (0, 1):
        label_path = root / f"labels_shard{shard}.jsonl"
        content_path = root / f"content_shard{shard}.jsonl"
        resource_path = root / f"resource_shard{shard}.json"
        resource = json.loads(resource_path.read_text())
        if (
            resource.get("protocol") != PROTOCOL
            or resource.get("labels_sha256") != sha(label_path)
            or resource.get("content_sha256") != sha(content_path)
        ):
            raise ValueError("causal-ledger pilot shard hash mismatch")
        labels.extend(json.loads(line) for line in label_path.read_text().splitlines() if line)
        content.extend(json.loads(line) for line in content_path.read_text().splitlines() if line)
        resources.append(resource)
    if len(pairs) != 112 or len(labels) != 336 or len(content) != 1008:
        raise ValueError("causal-ledger pilot merged coverage mismatch")
    scale_table = {}
    for scale in SCALES:
        key = str(scale)
        scale_table[key] = {}
        for name in ("correct", "wrong", "zero", "random"):
            scale_table[key][name] = {
                "candidate_accuracy": statistics.fmean(
                    row["controls"][name]["by_scale"][key]["prediction"] == row["answer"]
                    for row in content
                ),
                "answer_nll": statistics.fmean(
                    row["controls"][name]["answer_nll"][key] for row in content
                ),
            }
        scale_table[key]["correct"]["greedy_exact"] = statistics.fmean(
            row["correct_generation"][key]["normalized_exact"] for row in content
        )
        scale_table[key]["correct"]["eos_rate"] = statistics.fmean(
            row["correct_generation"][key]["stop_reason"] == "eos" for row in content
        )
    primary = scale_table[str(PRIMARY_SCALE)]
    intervals = {name: paired_interval(content, name) for name in ("wrong", "zero", "random")}
    family_accuracy = {
        family: statistics.fmean(
            row["controls"]["correct"]["by_scale"][str(PRIMARY_SCALE)]["prediction"]
            == row["answer"] for row in content if row["family"] == family
        ) for family in FAMILIES
    }
    sensitivity = all(
        scale_table[str(scale)]["correct"]["candidate_accuracy"]
        >= primary["correct"]["candidate_accuracy"] - 0.05
        and scale_table[str(scale)]["correct"]["eos_rate"]
        <= primary["correct"]["eos_rate"] + 0.05
        for scale in (4.0, 16.0)
    )
    checks = {
        "correct_candidate_at_least_point_95": primary["correct"]["candidate_accuracy"] >= 0.95,
        "correct_greedy_at_least_point_85": primary["correct"]["greedy_exact"] >= 0.85,
        "wrong_candidate_at_most_point_10": primary["wrong"]["candidate_accuracy"] <= 0.10,
        "zero_candidate_at_most_point_30": primary["zero"]["candidate_accuracy"] <= 0.30,
        "all_nll_gain_ci_lowers_above_zero": all(
            value["bootstrap_95ci"][0] > 0 for value in intervals.values()
        ),
        "every_family_correct_at_least_point_90": min(family_accuracy.values()) >= 0.90,
        "scale_sensitivity_within_five_points": sensitivity,
    }
    wall = max(row["seconds"] for row in resources)
    allocated = max(row["overall_peak_allocated_gib"] for row in resources)
    reserved = max(row["overall_peak_reserved_gib"] for row in resources)
    estimates = {str(choice): wall * choice / 8.0 for choice in FORMAL_CHOICES}
    eligible = [
        choice for choice in FORMAL_CHOICES
        if all(checks.values()) and allocated < 24.0 and estimates[str(choice)] < 6 * 3600
    ]
    result = {
        "protocol": PROTOCOL,
        "content": scale_table,
        "nll_gain_intervals": intervals,
        "family_correct_accuracy": family_accuracy,
        "content_gate": {"checks": checks, "passed": all(checks.values())},
        "labels": {
            "rows": len(labels),
            "mean_seconds": statistics.fmean(row["label_seconds"] for row in labels),
            "p95_seconds": sorted(row["label_seconds"] for row in labels)[
                math.ceil(0.95 * len(labels)) - 1
            ],
            "mean_seconds_per_action": statistics.fmean(
                row["seconds_per_action"] for row in labels
            ),
            "valid_action_counts": dict(Counter(
                str(row["valid_action_count"]) for row in labels
            )),
            "label_counts": dict(Counter(str(row["label"]) for row in labels)),
        },
        "resources": {
            "two_gpu_critical_path_seconds": wall,
            "overall_peak_allocated_gib": allocated,
            "overall_peak_reserved_gib": reserved,
            "estimated_development_seconds": estimates,
            "selected_groups_per_family": max(eligible) if eligible else None,
            "resource_gate_passed": bool(eligible),
            "shards": resources,
        },
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
        "next_stage": (
            "independent_pilot_audit_before_development"
            if eligible else "stop_causal_ledger_residual_branch"
        ),
        "terminal_status": "completed_qwen_causal_ledger_residual_pilot",
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--action-chunk", type=int, default=18)
    collect_parser.add_argument("--labels-output", required=True)
    collect_parser.add_argument("--content-output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
