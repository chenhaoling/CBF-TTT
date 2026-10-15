"""Score variable-length natural values through frozen-Qwen LM-head residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_value_bridge_v1 import EXPECTED_COUNTS, load
from tasks.cbf_qwen_lowrank_bridge import unit_value
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates, normalize_generation
from tasks.evaluate_v18_lowrank_fastmemory_bridge import FastMemoryReadSession
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_value_bridge_v1"
SCALES = (4.0, 8.0, 16.0, 32.0, 48.0)
ARMS = ("correct", "wrong", "zero", "random")
QUERY_TYPES = ("exact", "rephrase")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610164
HEADER = "Use the selected session memory and answer with only the requested value."


def render_query(text):
    return f"{HEADER}\n\nQuery: {text}\nAnswer: "


def randomize_rows(value, identity):
    import torch

    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    signs = torch.randint(0, 2, value.shape, generator=generator, dtype=torch.int8)
    signs = signs.to(value.device, dtype=value.dtype).mul_(2).sub_(1)
    output = value * signs
    ratios = (output.norm(dim=1) / value.norm(dim=1).clamp_min(1e-12)).tolist()
    return output, [float(item) for item in ratios]


def prepare(model, tokenizer, query, candidates):
    import torch

    prompt = render_query(query)
    prompt_ids, encoded = encode_candidates(tokenizer, prompt, candidates)
    if any(not ids for ids in encoded.values()):
        raise ValueError("empty natural-value answer tokenization")
    session = FastMemoryReadSession(model)
    prefix_hidden = session._forward(prompt_ids)
    cache_length = session.cache.get_seq_length()
    hidden = {}
    for answer, ids in encoded.items():
        if len(ids) == 1:
            hidden[answer] = prefix_hidden[:, -1:].float()
        else:
            branch = session.clone(); tail = branch._forward(ids[:-1])
            hidden[answer] = torch.cat((prefix_hidden[:, -1:], tail), dim=1).float()
    if session.cache.get_seq_length() != cache_length:
        raise RuntimeError("candidate branch mutated natural-value prompt cache")
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "encoded": encoded,
        "session": session,
        "prefix_hidden": prefix_hidden,
        "hidden": hidden,
    }


def value_rows(model, token_ids):
    import torch
    return torch.stack([unit_value(model, token_id) for token_id in token_ids])


def pad_rows(value, length):
    output = value.new_zeros((length, value.shape[1]))
    output[:min(length, value.shape[0])] = value[:length]
    return output


def candidate_scores(model, prepared, residual, scale):
    import torch
    import torch.nn.functional as F

    scores = {}
    for answer, ids in prepared["encoded"].items():
        hidden = prepared["hidden"][answer][0]
        adjusted = hidden + float(scale) * pad_rows(residual, len(ids))
        logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
        labels = torch.tensor(ids, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        scores[answer] = float(loss)
    return scores


def generate_fixed(model, tokenizer, prepared, residual, scale, length):
    session = prepared["session"].clone()
    hidden = prepared["prefix_hidden"][:, -1:]
    generated = []
    for index in range(length):
        row = residual[index] if index < residual.shape[0] else residual.new_zeros(residual.shape[1])
        adjusted = hidden[:, -1].float() + float(scale) * row.unsqueeze(0)
        logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
        token = int(logits.argmax(-1).item()); generated.append(token)
        if index + 1 < length:
            hidden = session._forward([token])
    return {
        "ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "tokens": len(generated),
        "conditioned_length": int(length),
    }


def score_row(model, tokenizer, episode, query_type, scales):
    import torch

    query = episode["queries"][query_type]
    prepared = prepare(model, tokenizer, query, episode["candidates"])
    correct_ids = prepared["encoded"][episode["correct"]]
    wrong_ids = prepared["encoded"][episode["wrong"]]
    correct = value_rows(model, correct_ids)
    wrong = value_rows(model, wrong_ids)
    randomized, norm_ratios = randomize_rows(
        correct, f"{episode['id']}|{query_type}|random"
    )
    residuals = {
        "correct": correct,
        "wrong": wrong,
        "zero": correct.new_zeros(correct.shape),
        "random": randomized,
    }
    started = time.perf_counter(); by_scale = {}
    for scale in scales:
        arms = {}
        for arm in ARMS:
            scores = candidate_scores(model, prepared, residuals[arm], scale)
            prediction = min(episode["candidates"], key=lambda value: (scores[value], value))
            arms[arm] = {
                "scores": scores,
                "prediction": prediction,
                "candidate_correct": int(prediction == episode["correct"]),
                "injected_wrong": int(prediction == episode["wrong"]),
                "answer_nll": scores[episode["correct"]],
            }
        generation = generate_fixed(
            model, tokenizer, prepared, correct, scale, len(correct_ids)
        )
        generation.update({
            "token_exact": int(generation["ids"] == correct_ids),
            "normalized_exact": int(
                normalize_generation(generation["text"])
                == normalize_generation(episode["correct"])
            ),
        })
        arms["correct"]["generation"] = generation
        by_scale[str(float(scale))] = {"arms": arms}
    torch.cuda.synchronize()
    return {
        "protocol": PROTOCOL,
        "episode_id": episode["id"],
        "split": episode["split"],
        "subject_sha256": hashlib.sha256(episode["subject"].encode()).hexdigest(),
        "source_row_sha256": episode["source_row_sha256"],
        "query_type": query_type,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prepared["prompt"].encode()).hexdigest(),
        "correct": episode["correct"],
        "wrong": episode["wrong"],
        "candidates": episode["candidates"],
        "candidate_token_ids": prepared["encoded"],
        "prompt_tokens": len(prepared["prompt_ids"]),
        "correct_tokens": len(correct_ids),
        "wrong_tokens": len(wrong_ids),
        "random_row_norm_ratios": norm_ratios,
        "scales": by_scale,
        "seconds": time.perf_counter() - started,
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    output, resource_output = Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("natural-value shard output exists")
    manifest, episodes = load(args.data_root, args.stage)
    selected = [
        row for index, row in enumerate(episodes)
        if index % args.num_shards == args.shard
    ]
    if args.stage == "calibration":
        scales = SCALES
    else:
        selection = json.loads(Path(args.selection).read_text())
        if not selection["gate"]["passed"] or selection["selected_scale"] not in SCALES:
            raise ValueError("development requires passing frozen calibration")
        scales = (float(selection["selected_scale"]),)
    torch.manual_seed(202610164)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    started = time.perf_counter(); rows = []
    with output.open("x") as sink, torch.no_grad():
        for index, episode in enumerate(selected):
            for query_type in QUERY_TYPES:
                row = score_row(model, tokenizer, episode, query_type, scales)
                row["shard"] = args.shard
                sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush()
                rows.append(row)
            print(json.dumps({
                "stage": args.stage, "shard": args.shard,
                "completed": index + 1, "total": len(selected),
            }), flush=True)
    resource = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "episodes": len(selected),
        "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "mean_row_seconds": statistics.fmean(row["seconds"] for row in rows),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(output),
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "scales": list(scales),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "calibration_scored": True,
        "development_scored": args.stage == "dev",
        "test_loaded": False,
        "test_scored": False,
        "external_benchmarks_loaded": False,
    }
    write_json(resource_output, resource)
    return resource


def _interval(values, seed):
    groups = sorted(values); rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    draws.sort()
    return [
        draws[max(0, math.ceil(.025 * len(draws)) - 1)],
        draws[min(len(draws) - 1, math.ceil(.975 * len(draws)) - 1)],
    ]


def aggregate(rows, scale):
    key = str(float(scale)); metrics = {}
    for arm in ARMS:
        metrics[arm] = {
            "candidate_accuracy": statistics.fmean(
                row["scales"][key]["arms"][arm]["candidate_correct"] for row in rows
            ),
            "answer_nll": statistics.fmean(
                row["scales"][key]["arms"][arm]["answer_nll"] for row in rows
            ),
            "by_query_type": {
                query_type: statistics.fmean(
                    row["scales"][key]["arms"][arm]["candidate_correct"] for row in rows
                    if row["query_type"] == query_type
                ) for query_type in QUERY_TYPES
            },
        }
    metrics["correct"]["generation_token_exact"] = statistics.fmean(
        row["scales"][key]["arms"]["correct"]["generation"]["token_exact"]
        for row in rows
    )
    metrics["correct"]["generation_by_query_type"] = {
        query_type: statistics.fmean(
            row["scales"][key]["arms"]["correct"]["generation"]["token_exact"]
            for row in rows if row["query_type"] == query_type
        ) for query_type in QUERY_TYPES
    }
    metrics["wrong"]["injected_wrong_accuracy"] = statistics.fmean(
        row["scales"][key]["arms"]["wrong"]["injected_wrong"] for row in rows
    )
    comparisons = {}
    groups = sorted({row["episode_id"] for row in rows})
    for offset, control in enumerate(("wrong", "zero", "random")):
        values = {}
        for group in groups:
            part = [row for row in rows if row["episode_id"] == group]
            values[group] = statistics.fmean(
                row["scales"][key]["arms"][control]["answer_nll"]
                - row["scales"][key]["arms"]["correct"]["answer_nll"]
                for row in part
            )
        comparisons[control] = {
            "nll_gain": statistics.fmean(values.values()),
            "bootstrap_95ci": _interval(values, BOOTSTRAP_SEED + offset),
        }
    strongest = max(("wrong", "zero", "random"), key=lambda arm: metrics[arm]["candidate_accuracy"])
    query_gains = {
        query_type: metrics["correct"]["by_query_type"][query_type]
        - metrics[strongest]["by_query_type"][query_type]
        for query_type in QUERY_TYPES
    }
    checks = {
        "correct_candidate_at_least_point_90": metrics["correct"]["candidate_accuracy"] >= .90,
        "correct_each_query_candidate_at_least_point_85": min(
            metrics["correct"]["by_query_type"].values()
        ) >= .85,
        "correct_generation_at_least_point_70": metrics["correct"]["generation_token_exact"] >= .70,
        "correct_each_query_generation_at_least_point_65": min(
            metrics["correct"]["generation_by_query_type"].values()
        ) >= .65,
        "wrong_truth_at_most_point_20": metrics["wrong"]["candidate_accuracy"] <= .20,
        "wrong_injected_at_least_point_70": metrics["wrong"]["injected_wrong_accuracy"] >= .70,
        "zero_truth_at_most_point_35": metrics["zero"]["candidate_accuracy"] <= .35,
        "all_nll_gain_ci_lower_above_zero": min(
            value["bootstrap_95ci"][0] for value in comparisons.values()
        ) > 0,
    }
    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "strongest_control": strongest,
        "query_type_candidate_gains": query_gains,
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def _read_stage(root, stage):
    root = Path(root); rows, resources = [], []
    expected_episodes = EXPECTED_COUNTS[stage]
    for shard in (0, 1):
        row_path = root / f"{stage}_shard{shard}.jsonl"
        resource_path = root / f"{stage}_resource_shard{shard}.json"
        part = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        resource = json.loads(resource_path.read_text())
        if (
            resource["rows_sha256"] != sha(row_path)
            or resource["episodes"] != expected_episodes // 2
            or resource["rows"] != len(part)
            or len(part) != expected_episodes
        ):
            raise ValueError("natural-value stage shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != 2 * expected_episodes or len({
        (row["episode_id"], row["query_type"]) for row in rows
    }) != 2 * expected_episodes:
        raise ValueError("natural-value stage coverage mismatch")
    return rows, resources


def summarize_calibration(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = _read_stage(args.root, "calibration")
    tables = {str(scale): aggregate(rows, scale) for scale in SCALES}
    passing = [scale for scale in SCALES if tables[str(scale)]["gate"]["passed"]]
    selected = min(passing) if passing else None
    result = {
        "protocol": PROTOCOL,
        "stage": "calibration",
        "tables": tables,
        "selected_scale": selected,
        "selection_rule": "smallest scale passing every frozen content gate",
        "gate": {
            "passed": selected is not None,
            "checks": {} if selected is None else tables[str(selected)]["gate"]["checks"],
        },
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "development_scored": False,
        "test_loaded": False,
        "test_scored": False,
    }
    write_json(output, result); return result


def summarize_development(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    calibration = json.loads(Path(args.calibration).read_text())
    if not calibration["gate"]["passed"]:
        raise ValueError("natural-value calibration did not release development")
    scale = float(calibration["selected_scale"])
    rows, resources = _read_stage(args.root, "dev")
    result = aggregate(rows, scale)
    nonnegative = min(result["query_type_candidate_gains"].values()) >= 0
    result["gate"]["checks"]["no_negative_query_type_candidate_gain"] = nonnegative
    result["gate"]["passed"] = all(result["gate"]["checks"].values())
    result.update({
        "protocol": PROTOCOL,
        "stage": "development",
        "selected_scale": scale,
        "rows": len(rows),
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "development_scored": True,
        "test_loaded": False,
        "test_scored": False,
        "next_stage": (
            "natural_answer_causal_ledger_sequences"
            if result["gate"]["passed"] else "stop_natural_value_writer"
        ),
    })
    write_json(output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--stage", choices=("calibration", "dev"), required=True)
    collect_parser.add_argument("--selection")
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--resource-output", required=True)
    calibration_parser = sub.add_parser("summarize-calibration")
    calibration_parser.add_argument("--root", required=True)
    calibration_parser.add_argument("--output", required=True)
    development_parser = sub.add_parser("summarize-development")
    development_parser.add_argument("--root", required=True)
    development_parser.add_argument("--calibration", required=True)
    development_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args)
    elif args.command == "summarize-calibration":
        result = summarize_calibration(args)
    else:
        result = summarize_development(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
