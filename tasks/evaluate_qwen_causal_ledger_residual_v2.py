"""Evaluate the fresh scale-32 causal-ledger content validation V2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_causal_ledger_residual import FAMILIES, parse_visible_block
from tasks.build_qwen_causal_ledger_residual_v2 import PHASE, RESOURCE_V2_SEED, load
from tasks.cbf_qwen_lowrank_bridge import unit_value
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    assignment,
    build_factor_cache,
    generate_with_residual,
    prepare_query,
    randomize_residual,
    semantic_apply,
    single_component_residual,
    visible_facts,
    weighted_residual,
)
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_causal_ledger_residual_v2_validation_v1"
SCALE = 32.0
CONTROLS = (
    "single_correct", "single_wrong", "zero", "random",
    "dense_visible", "addressable_top1", "exact_lookup",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610163


def ensure_value(model, tokenizer, factors, value):
    import torch

    if value in factors["values"]:
        return
    ids = tokenizer.encode(value, add_special_tokens=False)
    if len(ids) != 5:
        raise ValueError("V2 twin value is not five tokens")
    factors["token_ids"][value] = ids
    factors["values"][value] = torch.stack([unit_value(model, token_id) for token_id in ids])


def addressable_top1(state, query_key, factors, null_code):
    import torch

    visible = visible_facts(state)
    if query_key not in visible:
        return factors["values"][null_code]
    query = factors["keys"][query_key]
    best = max(
        visible.values(),
        key=lambda row: (float(torch.dot(query, factors["keys"][row["key"]])), row["block_index"], row["key"]),
    )
    similarity = float(torch.dot(query, factors["keys"][best["key"]]))
    return similarity * factors["values"][best["value"]]


def score_candidates(model, prepared, residual):
    import torch
    import torch.nn.functional as F

    scores = {}
    for candidate, hidden in prepared["hidden"].items():
        adjusted = hidden.to(residual.device) + SCALE * residual
        logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
        labels = torch.tensor(prepared["encoded"][candidate], dtype=torch.long, device=logits.device)
        scores[candidate] = float(F.cross_entropy(logits, labels, reduction="mean"))
    prediction = min(scores, key=lambda value: (scores[value], value))
    return {"scores": scores, "prediction": prediction}


def bootstrap_interval(group_values, control):
    groups = sorted(group_values)
    rng = random.Random(BOOTSTRAP_SEED + {"single_wrong": 1, "zero": 2, "random": 3}[control])
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(statistics.fmean(group_values[group] for group in sample))
    values.sort()
    return [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    data_root, output, resource_output = Path(args.data_root), Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("V2 validation shard output exists")
    manifest, pairs = load(data_root)
    selected = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard]
    twin_answers = {
        (inp["group_id"], target["twin"], row["id"]): row["answer"]
        for inp, target in pairs for row in target["answers"]
    }
    torch.manual_seed(RESOURCE_V2_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    started = time.perf_counter(); rows = []
    with output.open("x") as sink, torch.no_grad():
        for pair_index, (inp, target) in enumerate(selected):
            factors = build_factor_cache(model, tokenizer, inp, target)
            answers = {row["id"]: row for row in target["answers"]}
            queries = {}
            for query in inp["queries"]:
                queries.setdefault(query["after_block"], []).append(query)
            state = []
            for block in inp["blocks"]:
                event = assignment(parse_visible_block(block["text"], PHASE))
                state = semantic_apply(state, event, block["index"])
                if block["index"] not in queries:
                    continue
                for query in queries[block["index"]]:
                    answer = answers[query["id"]]
                    prepared = prepare_query(model, tokenizer, query)
                    correct = single_component_residual(query["key"], answer["answer"], factors)
                    wrong_value = next(
                        value for value in query["candidates"]
                        if value not in {answer["answer"], inp["null_value"]}
                    )
                    wrong = single_component_residual(query["key"], wrong_value, factors)
                    zero = torch.zeros_like(correct)
                    random_control = randomize_residual(
                        correct, f"v2|{inp['id']}|{query['id']}|random"
                    )
                    dense = weighted_residual(
                        state, query["key"], factors, inp["null_value"], visible_only=True
                    )
                    top1 = addressable_top1(state, query["key"], factors, inp["null_value"])
                    exact = factors["values"][answer["answer"]]
                    residuals = {
                        "single_correct": correct,
                        "single_wrong": wrong,
                        "zero": zero,
                        "random": random_control,
                        "dense_visible": dense,
                        "addressable_top1": top1,
                        "exact_lookup": exact,
                    }
                    controls = {}
                    for name, residual in residuals.items():
                        scored = score_candidates(model, prepared, residual)
                        generation = generate_with_residual(
                            model, tokenizer, prepared, residual, SCALE, max_new_tokens=5,
                            stop_on_eos_or_newline=False,
                        )
                        generation["truth_exact"] = int(
                            generation["ids"] == factors["token_ids"][answer["answer"]]
                        )
                        controls[name] = {
                            "candidate": scored,
                            "answer_nll": scored["scores"][answer["answer"]],
                            "generation": generation,
                        }
                    wrong_ids = factors["token_ids"][wrong_value]
                    controls["single_wrong"]["generation"]["injected_exact"] = int(
                        controls["single_wrong"]["generation"]["ids"] == wrong_ids
                    )
                    other_twin = twin_answers[(inp["group_id"], 1 - target["twin"], query["id"])]
                    twin = None
                    if other_twin != answer["answer"]:
                        ensure_value(model, tokenizer, factors, other_twin)
                        residual = single_component_residual(query["key"], other_twin, factors)
                        generation = generate_with_residual(
                            model, tokenizer, prepared, residual, SCALE, max_new_tokens=5,
                            stop_on_eos_or_newline=False,
                        )
                        twin = {
                            "answer": other_twin,
                            "truth_exact": int(generation["ids"] == factors["token_ids"][answer["answer"]]),
                            "twin_exact": int(generation["ids"] == factors["token_ids"][other_twin]),
                            "generation": generation,
                        }
                    row = {
                        "protocol": PROTOCOL,
                        "shard": args.shard,
                        "episode_id": inp["id"],
                        "group_id": inp["group_id"],
                        "twin": target["twin"],
                        "family": target["family"],
                        "block_index": block["index"],
                        "query_id": query["id"],
                        "role": query["role"],
                        "absent": answer["absent"],
                        "answer": answer["answer"],
                        "wrong_value": wrong_value,
                        "candidates": query["candidates"],
                        "controls": controls,
                        "twin_counterfactual": twin,
                        "twin_causally_equal": other_twin == answer["answer"],
                        "visible_facts": len(visible_facts(state)),
                        "prompt_sha256": hashlib.sha256(prepared["prompt"].encode()).hexdigest(),
                    }
                    sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            print(json.dumps({"completed": pair_index + 1, "total": len(selected), "episode": inp["id"]}), flush=True)
    resource = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "episodes": len(selected),
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(output),
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "scale": SCALE,
        "seed": manifest["seed"],
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
    }
    write_json(resource_output, resource)
    return resource


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = [], []
    for shard in (0, 1):
        path = root / f"v2_shard{shard}.jsonl"
        resource = json.loads((root / f"v2_resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in path.read_text().splitlines() if line]
        if resource["rows_sha256"] != sha(path) or resource["rows"] != len(part) or len(part) != 504:
            raise ValueError("V2 validation shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != 1008 or len({(row["episode_id"], row["query_id"]) for row in rows}) != 1008:
        raise ValueError("V2 validation coverage mismatch")
    metrics = {}
    for name in CONTROLS:
        metrics[name] = {
            "candidate_accuracy": statistics.fmean(
                row["controls"][name]["candidate"]["prediction"] == row["answer"] for row in rows
            ),
            "truth_exact": statistics.fmean(
                row["controls"][name]["generation"]["truth_exact"] for row in rows
            ),
            "answer_nll": statistics.fmean(row["controls"][name]["answer_nll"] for row in rows),
        }
    metrics["single_wrong"]["injected_exact"] = statistics.fmean(
        row["controls"]["single_wrong"]["generation"]["injected_exact"] for row in rows
    )
    primary = "addressable_top1"
    family = {
        name: statistics.fmean(
            row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["family"] == name
        ) for name in FAMILIES
    }
    role = {
        name: statistics.fmean(
            row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["role"] == name
        ) for name in ("current", "historical", "anchor")
    }
    absent = statistics.fmean(
        row["controls"][primary]["generation"]["truth_exact"] for row in rows if row["absent"]
    )
    oracle_agreement = statistics.fmean(
        row["controls"][primary]["candidate"]["prediction"]
        == row["controls"]["exact_lookup"]["candidate"]["prediction"]
        and row["controls"][primary]["generation"]["ids"]
        == row["controls"]["exact_lookup"]["generation"]["ids"]
        for row in rows
    )
    twin_rows = [row for row in rows if row["twin_counterfactual"] is not None]
    twin = {
        "rows": len(twin_rows),
        "groups": len({row["group_id"] for row in twin_rows}),
        "families": sorted({row["family"] for row in twin_rows}),
        "truth_exact": statistics.fmean(row["twin_counterfactual"]["truth_exact"] for row in twin_rows),
        "twin_exact": statistics.fmean(row["twin_counterfactual"]["twin_exact"] for row in twin_rows),
    }
    intervals = {}
    for control in ("single_wrong", "zero", "random"):
        grouped = {}
        for row in rows:
            grouped.setdefault(row["group_id"], []).append(
                row["controls"][control]["answer_nll"]
                - row["controls"]["single_correct"]["answer_nll"]
            )
        group_values = {group: statistics.fmean(values) for group, values in grouped.items()}
        intervals[control] = {
            "mean": statistics.fmean(group_values.values()),
            "bootstrap_95ci": bootstrap_interval(group_values, control),
        }
    checks = {
        "primary_candidate_at_least_point_95": metrics[primary]["candidate_accuracy"] >= 0.95,
        "primary_exact_at_least_point_85": metrics[primary]["truth_exact"] >= 0.85,
        "every_family_exact_at_least_point_80": min(family.values()) >= 0.80,
        "current_anchor_at_least_point_90": min(role["current"], role["anchor"]) >= 0.90,
        "historical_at_least_point_80": role["historical"] >= 0.80,
        "absent_at_least_point_85": absent >= 0.85,
        "exact_oracle_agreement": oracle_agreement == 1.0,
        "single_correct_candidate_exact_floor": metrics["single_correct"]["candidate_accuracy"] >= 0.95 and metrics["single_correct"]["truth_exact"] >= 0.90,
        "wrong_truth_ceiling_and_injected_floor": metrics["single_wrong"]["candidate_accuracy"] <= 0.10 and metrics["single_wrong"]["truth_exact"] <= 0.10 and metrics["single_wrong"]["injected_exact"] >= 0.85,
        "zero_random_candidate_at_most_point_30": max(metrics["zero"]["candidate_accuracy"], metrics["random"]["candidate_accuracy"]) <= 0.30,
        "twin_truth_ceiling_and_twin_floor": twin["rows"] > 0 and twin["truth_exact"] <= 0.10 and twin["twin_exact"] >= 0.85,
        "all_nll_gain_ci_lowers_above_zero": all(value["bootstrap_95ci"][0] > 0 for value in intervals.values()),
    }
    result = {
        "protocol": PROTOCOL,
        "scale": SCALE,
        "rows": len(rows),
        "metrics": metrics,
        "primary_family_exact": family,
        "primary_role_exact": role,
        "primary_absent_exact": absent,
        "oracle_agreement": oracle_agreement,
        "twin_counterfactual": twin,
        "nll_gain_intervals": intervals,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
        "next_stage": "independent_v2_audit_before_development" if all(checks.values()) else "stop_v2_before_development",
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
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
