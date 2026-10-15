"""Evaluate frozen lifecycle policies on causal-ledger development V2."""

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

from tasks.build_qwen_causal_ledger_development_v2 import load, PHASE
from tasks.build_qwen_causal_ledger_residual import FAMILIES, parse_visible_block
from tasks.evaluate_qwen_causal_ledger_residual_v2 import (
    addressable_top1,
    score_candidates,
    SCALE,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    assignment,
    build_factor_cache,
    generate_with_residual,
    prepare_query,
    semantic_apply,
    visible_facts,
    weighted_residual,
)
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_causal_ledger_development_v2_v1"
POLICIES = (
    "no_memory", "retain_latest", "global_boundary_clear", "last1", "last2",
    "dense_full", "full_ledger", "semantic_oracle",
)
PRIMARY = "full_ledger"
ORACLE = "semantic_oracle"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610160


def append_record(state, event, block_index):
    output = copy.deepcopy(state)
    output.append({
        "batch": event["batch"], "facts": dict(event["facts"]),
        "weight": 1.0, "block_index": int(block_index),
    })
    return output


def lookup_value(state, query_key, null_code):
    row = visible_facts(state).get(query_key)
    return null_code if row is None else row["value"]


def bootstrap_comparison(primary_groups, comparator_groups):
    groups = sorted(primary_groups)
    rng = random.Random(BOOTSTRAP_SEED)
    exact, nll = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        exact.append(statistics.fmean(
            primary_groups[group]["exact"] - comparator_groups[group]["exact"] for group in sample
        ))
        nll.append(statistics.fmean(
            comparator_groups[group]["nll"] - primary_groups[group]["nll"] for group in sample
        ))
    exact.sort(); nll.sort()
    bounds = lambda values: [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]
    return {
        "exact_gain": statistics.fmean(
            primary_groups[group]["exact"] - comparator_groups[group]["exact"] for group in groups
        ),
        "exact_bootstrap_95ci": bounds(exact),
        "nll_gain": statistics.fmean(
            comparator_groups[group]["nll"] - primary_groups[group]["nll"] for group in groups
        ),
        "nll_bootstrap_95ci": bounds(nll),
    }


def collect(
    args,
    scale=SCALE,
    protocol=PROTOCOL,
    loader=load,
    phase=PHASE,
    confirmation_generated=False,
    ood_generated=False,
):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    data_root, output, resource_output = Path(args.data_root), Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("development V2 shard output exists")
    manifest, pairs = loader(data_root)
    selected = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard]
    torch.manual_seed(202610160)
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
            states = {name: [] for name in ("full_ledger", "retain_latest", "global_boundary_clear", "last1", "last2")}
            boundaries = set(inp["decision_boundaries"])
            for block in inp["blocks"]:
                event = assignment(parse_visible_block(block["text"], phase))
                states["full_ledger"] = semantic_apply(states["full_ledger"], event, block["index"])
                states["retain_latest"] = append_record(states["retain_latest"], event, block["index"])
                global_state = [] if block["index"] in boundaries else states["global_boundary_clear"]
                states["global_boundary_clear"] = append_record(global_state, event, block["index"])
                states["last1"] = append_record(states["last1"], event, block["index"])[-1:]
                states["last2"] = append_record(states["last2"], event, block["index"])[-2:]
                if block["index"] not in queries:
                    continue
                for query in queries[block["index"]]:
                    answer = answers[query["id"]]
                    prepared = prepare_query(model, tokenizer, query)
                    returned = {
                        "no_memory": inp["null_value"],
                        "retain_latest": lookup_value(states["retain_latest"], query["key"], inp["null_value"]),
                        "global_boundary_clear": lookup_value(states["global_boundary_clear"], query["key"], inp["null_value"]),
                        "last1": lookup_value(states["last1"], query["key"], inp["null_value"]),
                        "last2": lookup_value(states["last2"], query["key"], inp["null_value"]),
                        "full_ledger": lookup_value(states["full_ledger"], query["key"], inp["null_value"]),
                        "semantic_oracle": answer["answer"],
                    }
                    dense = weighted_residual(
                        states["full_ledger"], query["key"], factors, inp["null_value"], visible_only=True
                    )
                    cached = {}
                    policy_rows = {}
                    for policy in POLICIES:
                        cache_key = "__dense__" if policy == "dense_full" else returned[policy]
                        if cache_key not in cached:
                            residual = dense if policy == "dense_full" else factors["values"][returned[policy]]
                            scored = score_candidates(model, prepared, residual, scale=scale)
                            generation = generate_with_residual(
                                model, tokenizer, prepared, residual, scale, max_new_tokens=5,
                                stop_on_eos_or_newline=False,
                            )
                            cached[cache_key] = {
                                "candidate": scored,
                                "generation": generation,
                            }
                        value = cached[cache_key]
                        policy_rows[policy] = {
                            "returned_value": None if policy == "dense_full" else returned[policy],
                            "candidate_prediction": value["candidate"]["prediction"],
                            "candidate_correct": int(value["candidate"]["prediction"] == answer["answer"]),
                            "nll": value["candidate"]["scores"][answer["answer"]],
                            "generated_ids": value["generation"]["ids"],
                            "generated_text": value["generation"]["text"],
                            "exact": int(value["generation"]["ids"] == factors["token_ids"][answer["answer"]]),
                        }
                    row = {
                        "protocol": protocol,
                        "shard": args.shard,
                        "episode_id": inp["id"],
                        "group_id": inp["group_id"],
                        "family": target["family"],
                        "length": target["length"],
                        "block_index": block["index"],
                        "query_id": query["id"],
                        "role": query["role"],
                        "absent": answer["absent"],
                        "answer": answer["answer"],
                        "candidates": query["candidates"],
                        "visible_facts": len(visible_facts(states["full_ledger"])),
                        "prompt_sha256": hashlib.sha256(prepared["prompt"].encode()).hexdigest(),
                        "policies": policy_rows,
                    }
                    sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            print(json.dumps({"completed": pair_index + 1, "total": len(selected), "episode": inp["id"]}), flush=True)
    resource = {
        "protocol": protocol,
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
        "scale": float(scale),
        "seed": manifest["seed"],
        "confirmation_generated": bool(confirmation_generated),
        "ood_generated": bool(ood_generated),
    }
    write_json(resource_output, resource)
    return resource


def summarize(
    args,
    scale=SCALE,
    protocol=PROTOCOL,
    confirmation_generated=False,
    ood_generated=False,
    passing_next_stage="independent_development_audit_before_confirmation",
    failure_next_stage="stop_before_confirmation",
):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = [], []
    for shard in (0, 1):
        path = root / f"development_shard{shard}.jsonl"
        resource = json.loads((root / f"development_resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in path.read_text().splitlines() if line]
        if resource["rows_sha256"] != sha(path) or resource["rows"] != len(part) or len(part) != 4032:
            raise ValueError("development V2 shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != 8064 or len({(row["episode_id"], row["query_id"]) for row in rows}) != 8064:
        raise ValueError("development V2 coverage mismatch")
    metrics = {}
    group_metrics = {}
    for policy in POLICIES:
        metrics[policy] = {
            "candidate_accuracy": statistics.fmean(row["policies"][policy]["candidate_correct"] for row in rows),
            "exact": statistics.fmean(row["policies"][policy]["exact"] for row in rows),
            "nll": statistics.fmean(row["policies"][policy]["nll"] for row in rows),
            "roles": {role: statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["role"] == role) for role in ("current", "historical", "anchor")},
            "families": {family: statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["family"] == family) for family in FAMILIES},
            "lengths": {str(length): statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["length"] == length) for length in (8, 12, 16)},
            "absent": statistics.fmean(row["policies"][policy]["exact"] for row in rows if row["absent"]),
        }
        group_metrics[policy] = {}
        for group in sorted({row["group_id"] for row in rows}):
            part = [row for row in rows if row["group_id"] == group]
            group_metrics[policy][group] = {
                "exact": statistics.fmean(row["policies"][policy]["exact"] for row in part),
                "nll": statistics.fmean(row["policies"][policy]["nll"] for row in part),
            }
    candidates = [name for name in POLICIES if name not in {PRIMARY, ORACLE}]
    strongest = max(candidates, key=lambda name: metrics[name]["exact"])
    comparison = bootstrap_comparison(group_metrics[PRIMARY], group_metrics[strongest])
    oracle_agreement = statistics.fmean(
        row["policies"][PRIMARY]["candidate_prediction"] == row["policies"][ORACLE]["candidate_prediction"]
        and row["policies"][PRIMARY]["generated_ids"] == row["policies"][ORACLE]["generated_ids"]
        for row in rows
    )
    family_gains = {family: metrics[PRIMARY]["families"][family] - metrics[strongest]["families"][family] for family in FAMILIES}
    length_gains = {length: metrics[PRIMARY]["lengths"][length] - metrics[strongest]["lengths"][length] for length in metrics[PRIMARY]["lengths"]}
    primary = metrics[PRIMARY]
    checks = {
        "primary_candidate_at_least_point_95": primary["candidate_accuracy"] >= 0.95,
        "primary_exact_at_least_point_85": primary["exact"] >= 0.85,
        "current_anchor_at_least_point_95": min(primary["roles"]["current"], primary["roles"]["anchor"]) >= 0.95,
        "historical_at_least_point_85": primary["roles"]["historical"] >= 0.85,
        "absent_at_least_point_90": primary["absent"] >= 0.90,
        "every_family_at_least_point_85": min(primary["families"].values()) >= 0.85,
        "gain_at_least_point_05": comparison["exact_gain"] >= 0.05,
        "gain_ci_lower_above_point_02": comparison["exact_bootstrap_95ci"][0] > 0.02,
        "nll_gain_ci_lower_above_zero": comparison["nll_bootstrap_95ci"][0] > 0,
        "no_negative_family_or_length_gain": min(list(family_gains.values()) + list(length_gains.values())) >= 0,
        "exact_oracle_agreement": oracle_agreement == 1.0,
    }
    result = {
        "protocol": protocol,
        "scale": float(scale),
        "rows": len(rows),
        "metrics": metrics,
        "strongest_non_oracle_comparator": strongest,
        "comparison": comparison,
        "family_gains": family_gains,
        "length_gains": length_gains,
        "oracle_agreement": oracle_agreement,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "confirmation_generated": bool(confirmation_generated),
        "ood_generated": bool(ood_generated),
        "next_stage": passing_next_stage if all(checks.values()) else failure_next_stage,
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
