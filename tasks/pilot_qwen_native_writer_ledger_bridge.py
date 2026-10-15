"""Test whether the trained native TTT writer binds one repeated ledger fact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_causal_ledger_residual import (
    FAMILIES,
    RESOURCE_V2_SEED,
    generate_group,
)
from tasks.evaluate_qwen_causal_ledger_residual_v2 import score_candidates
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    build_factor_cache,
    generate_with_residual,
    prepare_query,
    render_query,
)
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_native_writer_ledger_bridge_v1"
SCALE = 48.0
CHUNK_SIZE = 4096
GROUPS_PER_FAMILY = 2
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610164
ARMS = ("zero", "correct_writer", "wrong_writer", "external_residual")


def select_samples():
    samples = []
    for family in FAMILIES:
        for group_index in range(GROUPS_PER_FAMILY):
            inp, target = generate_group(
                RESOURCE_V2_SEED, "resource_v2", family, group_index
            )[0]
            answers = {row["id"]: row for row in target["answers"]}
            query = min(
                (
                    row for row in inp["queries"]
                    if row["role"] == "current" and not answers[row["id"]]["absent"]
                ),
                key=lambda row: (row["after_block"], row["id"]),
            )
            answer = answers[query["id"]]["answer"]
            wrong = next(
                value for value in query["candidates"]
                if value not in {answer, inp["null_value"]}
            )
            samples.append({
                "group_id": inp["group_id"],
                "family": family,
                "group_index": group_index,
                "inp": inp,
                "target": target,
                "query": query,
                "answer": answer,
                "wrong": wrong,
            })
    if len(samples) != len(FAMILIES) * GROUPS_PER_FAMILY:
        raise ValueError("native writer pilot sample count changed")
    return samples


def mapping_text(key, value):
    return (
        "Memorize this session fact exactly. "
        f"For index {key}, the active symbol is {value}. "
        "Use this mapping when the index is queried.\n"
    )


def fill_chunk(tokenizer, key, value):
    unit = tokenizer.encode(mapping_text(key, value), add_special_tokens=False)
    if not unit or len(unit) >= CHUNK_SIZE:
        raise ValueError("invalid native writer mapping unit")
    return (unit * math.ceil(CHUNK_SIZE / len(unit)))[:CHUNK_SIZE]


def memory_stats(model, session):
    import torch

    relative = {}
    for layer in model.config.ttt_layers:
        memory = session.cache.cbf_memory[layer]
        base = model.model.layers[layer].mlp.down_proj.weight
        relative[str(layer)] = float(
            torch.linalg.vector_norm(memory.float())
            / torch.linalg.vector_norm(base.float()).clamp_min(1e-12)
        )
    return {
        "layers": len(relative),
        "relative_norm": relative,
        "mean_relative_norm": statistics.fmean(relative.values()),
        "max_relative_norm": max(relative.values()),
    }


def writer_session(model, ids):
    import torch
    from cbf_ttt.runtime import CBFSession

    session = CBFSession(model, update_rule="write")
    started = time.perf_counter()
    with torch.inference_mode():
        coefficient = session.step(ids, policy="baseline")
    if coefficient != 1.0:
        raise RuntimeError("baseline native writer did not fully admit the delta")
    return session, time.perf_counter() - started


def score_session(model, tokenizer, session, query, answer_ids):
    query_ids = tokenizer.encode(render_query(query["text"]), add_special_tokens=False)
    read = session.clone_memory_only()
    started = time.perf_counter()
    scores = {
        candidate: read.score_answer(
            query_ids, tokenizer.encode(candidate, add_special_tokens=False)
        )
        for candidate in query["candidates"]
    }
    prediction = min(scores, key=lambda value: (scores[value], value))
    generated = read.clone_memory_only().generate(query_ids, 5)
    return {
        "scores": scores,
        "prediction": prediction,
        "generated_ids": generated,
        "exact": int(generated == answer_ids),
        "read_seconds": time.perf_counter() - started,
    }


def bootstrap_interval(rows, control):
    rng = random.Random(BOOTSTRAP_SEED + {"wrong_writer": 1, "zero": 2}[control])
    gains = [
        row["arms"][control]["scores"][row["answer"]]
        - row["arms"]["correct_writer"]["scores"][row["answer"]]
        for row in rows
    ]
    values = []
    for _ in range(BOOTSTRAP_DRAWS):
        values.append(statistics.fmean(gains[rng.randrange(len(gains))] for _ in gains))
    values.sort()
    return [
        values[max(0, math.ceil(0.025 * len(values)) - 1)],
        values[min(len(values) - 1, math.ceil(0.975 * len(values)) - 1)],
    ]


def aggregate(rows):
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {
            "candidate_accuracy": statistics.fmean(
                row["arms"][arm]["prediction"] == row["answer"] for row in rows
            ),
            "exact": statistics.fmean(row["arms"][arm]["exact"] for row in rows),
            "answer_nll": statistics.fmean(
                row["arms"][arm]["scores"][row["answer"]] for row in rows
            ),
        }
    metrics["wrong_writer"]["injected_wrong_accuracy"] = statistics.fmean(
        row["arms"]["wrong_writer"]["prediction"] == row["wrong"] for row in rows
    )
    intervals = {
        control: bootstrap_interval(rows, control)
        for control in ("wrong_writer", "zero")
    }
    checks = {
        "correct_candidate_at_least_point_80": metrics["correct_writer"]["candidate_accuracy"] >= .80,
        "correct_exact_at_least_point_60": metrics["correct_writer"]["exact"] >= .60,
        "wrong_truth_at_most_point_20": metrics["wrong_writer"]["candidate_accuracy"] <= .20,
        "wrong_injected_at_least_point_80": metrics["wrong_writer"]["injected_wrong_accuracy"] >= .80,
        "zero_truth_at_most_point_30": metrics["zero"]["candidate_accuracy"] <= .30,
        "correct_nll_lower_bounds_positive": min(value[0] for value in intervals.values()) > 0,
        "external_candidate_at_least_point_95": metrics["external_residual"]["candidate_accuracy"] >= .95,
        "external_exact_at_least_point_85": metrics["external_residual"]["exact"] >= .85,
    }
    return {
        "metrics": metrics,
        "correct_nll_gain_bootstrap_95ci": intervals,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.runtime import CBFSession
    from tasks.cbf_checkpoint_native import load_model

    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("native writer bridge shard exists")
    samples = [
        row for index, row in enumerate(select_samples())
        if index % args.num_shards == args.shard
    ]
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    versions = [parameter._version for parameter in model.parameters()]
    started = time.perf_counter(); rows = []
    with output.open("x") as sink:
        for index, sample in enumerate(samples):
            inp, target, query = sample["inp"], sample["target"], sample["query"]
            factors = build_factor_cache(model, tokenizer, inp, target)
            answer_ids = factors["token_ids"][sample["answer"]]
            correct_ids = fill_chunk(tokenizer, query["key"], sample["answer"])
            wrong_ids = fill_chunk(tokenizer, query["key"], sample["wrong"])
            correct_session, correct_write_seconds = writer_session(model, correct_ids)
            wrong_session, wrong_write_seconds = writer_session(model, wrong_ids)
            zero_session = CBFSession(model, update_rule="write")
            arms = {
                "correct_writer": score_session(
                    model, tokenizer, correct_session, query, answer_ids
                ),
                "wrong_writer": score_session(
                    model, tokenizer, wrong_session, query, answer_ids
                ),
                "zero": score_session(model, tokenizer, zero_session, query, answer_ids),
            }
            prepared = prepare_query(model, tokenizer, query)
            residual = factors["values"][sample["answer"]]
            external_score = score_candidates(model, prepared, residual, scale=SCALE)
            external_generation = generate_with_residual(
                model, tokenizer, prepared, residual, SCALE,
                max_new_tokens=5, stop_on_eos_or_newline=False,
            )
            arms["external_residual"] = {
                "scores": external_score["scores"],
                "prediction": external_score["prediction"],
                "generated_ids": external_generation["ids"],
                "exact": int(external_generation["ids"] == answer_ids),
                "read_seconds": None,
            }
            row = {
                "protocol": PROTOCOL,
                "shard": args.shard,
                "group_id": sample["group_id"],
                "family": sample["family"],
                "group_index": sample["group_index"],
                "query_id": query["id"],
                "query_key": query["key"],
                "answer": sample["answer"],
                "wrong": sample["wrong"],
                "candidates": query["candidates"],
                "answer_ids": answer_ids,
                "correct_chunk_sha256": hashlib.sha256(bytes().join(
                    int(value).to_bytes(4, "little") for value in correct_ids
                )).hexdigest(),
                "wrong_chunk_sha256": hashlib.sha256(bytes().join(
                    int(value).to_bytes(4, "little") for value in wrong_ids
                )).hexdigest(),
                "chunk_tokens": CHUNK_SIZE,
                "write_seconds": {
                    "correct": correct_write_seconds,
                    "wrong": wrong_write_seconds,
                },
                "memory": {
                    "correct": memory_stats(model, correct_session),
                    "wrong": memory_stats(model, wrong_session),
                },
                "arms": arms,
            }
            sink.write(json.dumps(row) + "\n"); sink.flush(); rows.append(row)
            print(json.dumps({"completed": index + 1, "total": len(samples)}), flush=True)
            del correct_session, wrong_session, zero_session
    if versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("native writer pilot modified model parameters")
    resource = {
        "protocol": PROTOCOL,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(output),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "source_files_sha256": model_meta["source_files_sha256"],
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "chunk_size": model.config.ttt_chunk,
        "ttt_layers": list(model.config.ttt_layers),
        "ttt_lr": model.config.ttt_lr,
        "confirmation_scored": False,
        "natural_benchmark_scored": False,
    }
    write_json(resource_output, resource)
    return resource


def summarize(args):
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = [], []
    for shard in (0, 1):
        path = root / f"bridge_shard{shard}.jsonl"
        resource = json.loads((root / f"bridge_resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in path.read_text().splitlines() if line]
        if resource["rows_sha256"] != sha(path) or resource["rows"] != len(part) or len(part) != 7:
            raise ValueError("native writer bridge shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != 14 or len({row["group_id"] for row in rows}) != 14:
        raise ValueError("native writer bridge coverage mismatch")
    result = {
        "protocol": PROTOCOL,
        "rows": len(rows),
        **aggregate(rows),
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "confirmation_scored": False,
        "natural_benchmark_scored": False,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("model", "tokenizer", "output", "resource-output"):
        collect_parser.add_argument("--" + name, required=True)
    collect_parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
