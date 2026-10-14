"""Measure contextual full-vocabulary generation across fixed residual scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.build_qwen_causal_ledger_residual import load_pilot
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import (
    build_factor_cache,
    generate_with_residual,
    prepare_query,
    PROTOCOL as SOURCE_PROTOCOL,
    single_component_residual,
)
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_causal_ledger_context_scale_diagnostic_v1"
SCALES = (16.0, 20.0, 24.0, 32.0, 48.0, 64.0)


def normalize(value):
    return " ".join(value.strip().casefold().split())


def select_scale(table):
    eligible = [
        scale for scale in SCALES
        if table[str(scale)]["free_exact"] >= 0.85
        and table[str(scale)]["teacher_sequence_exact"] >= 0.95
    ]
    return min(eligible) if eligible else None


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root, output, resource_output = Path(args.root), Path(args.output), Path(args.resource_output)
    if output.exists() or resource_output.exists():
        raise FileExistsError("context-scale shard output exists")
    manifest, pairs = load_pilot(root / "data")
    selected = [pair for index, pair in enumerate(pairs) if index % args.num_shards == args.shard]
    torch.manual_seed(202610159)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    started = time.perf_counter(); rows = []
    with output.open("x") as sink, torch.no_grad():
        for pair_index, (inp, target) in enumerate(selected):
            factors = build_factor_cache(model, tokenizer, inp, target)
            answers = {row["id"]: row for row in target["answers"]}
            for query in inp["queries"]:
                answer = answers[query["id"]]
                prepared = prepare_query(model, tokenizer, query)
                residual = single_component_residual(query["key"], answer["answer"], factors)
                target_ids = prepared["encoded"][answer["answer"]]
                if len(target_ids) != 5:
                    raise ValueError("context-scale answer is not five tokens")
                by_scale = {}
                for scale in SCALES:
                    base = prepared["hidden"][answer["answer"]].to(residual.device)
                    adjusted = base + scale * residual
                    logits = model.lm_head(adjusted.to(model.lm_head.weight.dtype)).float()
                    labels = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
                    prediction = logits.argmax(-1)
                    teacher_correct = prediction.eq(labels)
                    nll = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                    generation = generate_with_residual(
                        model, tokenizer, prepared, residual, scale, max_new_tokens=5,
                        stop_on_eos_or_newline=False,
                    )
                    generated_ids = generation["ids"]
                    generated_text = generation["text"]
                    free_correct = [
                        int(observed == expected)
                        for observed, expected in zip(generated_ids, target_ids)
                    ]
                    by_scale[str(scale)] = {
                        "teacher_ids": prediction.detach().cpu().tolist(),
                        "teacher_position_correct": teacher_correct.int().cpu().tolist(),
                        "teacher_sequence_exact": int(bool(teacher_correct.all())),
                        "answer_nll": float(nll.mean()),
                        "generated_ids": generated_ids,
                        "generated_text": generated_text,
                        "free_position_correct": free_correct,
                        "free_token_accuracy": statistics.fmean(free_correct),
                        "free_exact": int(generated_ids == target_ids),
                        "free_text_exact": int(normalize(generated_text) == normalize(answer["answer"])),
                    }
                row = {
                    "protocol": PROTOCOL,
                    "source_protocol": SOURCE_PROTOCOL,
                    "shard": args.shard,
                    "episode_id": inp["id"],
                    "group_id": inp["group_id"],
                    "family": target["family"],
                    "query_id": query["id"],
                    "role": query["role"],
                    "answer": answer["answer"],
                    "target_ids": target_ids,
                    "prompt_sha256": __import__("hashlib").sha256(prepared["prompt"].encode()).hexdigest(),
                    "residual_norms": residual.float().norm(dim=1).cpu().tolist(),
                    "hidden_norms": prepared["hidden"][answer["answer"]].float().norm(dim=1).cpu().tolist(),
                    "by_scale": by_scale,
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
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(),
        "scales": list(SCALES),
        "pilot_seed": manifest["seed"],
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
        row_path = root / f"scale_shard{shard}.jsonl"
        resource = json.loads((root / f"scale_resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        if (
            resource["protocol"] != PROTOCOL
            or resource["rows_sha256"] != sha(row_path)
            or resource["rows"] != len(part)
            or len(part) != 504
        ):
            raise ValueError("context-scale shard mismatch")
        rows.extend(part); resources.append(resource)
    if len(rows) != 1008 or len({(row["episode_id"], row["query_id"]) for row in rows}) != 1008:
        raise ValueError("context-scale merged coverage mismatch")
    table = {}
    for scale in SCALES:
        key = str(scale)
        table[key] = {
            "free_exact": statistics.fmean(row["by_scale"][key]["free_exact"] for row in rows),
            "free_text_exact": statistics.fmean(row["by_scale"][key]["free_text_exact"] for row in rows),
            "free_token_accuracy": statistics.fmean(row["by_scale"][key]["free_token_accuracy"] for row in rows),
            "free_position_accuracy": [
                statistics.fmean(row["by_scale"][key]["free_position_correct"][index] for row in rows)
                for index in range(5)
            ],
            "teacher_sequence_exact": statistics.fmean(
                row["by_scale"][key]["teacher_sequence_exact"] for row in rows
            ),
            "teacher_position_accuracy": [
                statistics.fmean(row["by_scale"][key]["teacher_position_correct"][index] for row in rows)
                for index in range(5)
            ],
            "answer_nll": statistics.fmean(row["by_scale"][key]["answer_nll"] for row in rows),
            "family_free_exact": {
                family: statistics.fmean(
                    row["by_scale"][key]["free_exact"] for row in rows if row["family"] == family
                ) for family in sorted({row["family"] for row in rows})
            },
        }
    selected = select_scale(table)
    result = {
        "protocol": PROTOCOL,
        "rows": len(rows),
        "scales": list(SCALES),
        "table": table,
        "selection_rule": "smallest scale with free exact >=0.85 and teacher sequence exact >=0.95",
        "selected_scale": selected,
        "passed": selected is not None,
        "resources": {
            "two_gpu_critical_path_seconds": max(row["seconds"] for row in resources),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
            "shards": resources,
        },
        "development_generated": False,
        "confirmation_generated": False,
        "ood_generated": False,
        "next_stage": "versioned_content_revalidation" if selected is not None else "stop_scalar_scale_repair",
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--root", required=True)
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

