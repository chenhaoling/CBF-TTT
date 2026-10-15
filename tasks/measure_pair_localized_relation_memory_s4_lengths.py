"""Measure S4 token lengths before any Qwen window forward pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.extract_pair_localized_relation_memory_s4_embeddings import DATASETS, load_rows
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_pair_localized_relation_memory_s4_token_lengths_v1"


def quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def measure(args) -> dict:
    from transformers import AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    summaries, maximum = {}, 0
    for dataset in DATASETS:
        rows, meta = load_rows(args.data_root, dataset)
        lengths = []
        for start in range(0, len(rows), args.batch_size):
            encoded = tokenizer(
                [row["text"] for row in rows[start:start + args.batch_size]],
                add_special_tokens=True, truncation=False, padding=False,
            )["input_ids"]
            lengths.extend(map(len, encoded))
        maximum = max(maximum, max(lengths))
        summaries[dataset] = {
            "rows": len(lengths), "min": min(lengths), "median": quantile(lengths, .5),
            "p90": quantile(lengths, .9), "p95": quantile(lengths, .95),
            "p99": quantile(lengths, .99), "max": max(lengths), "data": meta,
        }
    result = {
        "protocol": PROTOCOL,
        "datasets": summaries,
        "global_max": maximum,
        "recommended_max_length": 1 << (maximum - 1).bit_length(),
        "tokenizer": args.tokenizer,
        "tokenizer_config_sha256": sha(Path(args.tokenizer) / "tokenizer_config.json"),
        "model_forward_passes": 0,
        "dialogre_dev_loaded": False,
        "dialogre_test_loaded": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
