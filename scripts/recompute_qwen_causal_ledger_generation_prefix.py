"""Recompute fixed-length generation EM from saved causal-ledger token traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_causal_ledger_residual import PROTOCOL, SCALES
from tasks.train_cbf_event_writer import write_json


def normalize(value):
    return " ".join(value.strip().casefold().split())


def run(args):
    from transformers import AutoTokenizer

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    paths = [root / f"content_shard{shard}.jsonl" for shard in (0, 1)]
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text().splitlines()
        if line
    ]
    if len(rows) != 1008 or any(row["protocol"] != PROTOCOL for row in rows):
        raise ValueError("unexpected causal-ledger content coverage")
    by_scale = {}
    for scale in SCALES:
        key = str(scale); token_exact, text_exact, examples = [], [], []
        for row in rows:
            target_ids = tokenizer.encode(row["answer"], add_special_tokens=False)
            if len(target_ids) != 5:
                raise ValueError("causal-ledger target is not five tokens")
            generated_ids = row["correct_generation"][key]["ids"][:5]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            token_match = generated_ids == target_ids
            text_match = normalize(generated_text) == normalize(row["answer"])
            token_exact.append(token_match); text_exact.append(text_match)
            if not token_match and len(examples) < 8:
                examples.append({
                    "episode_id": row["episode_id"],
                    "query_id": row["query_id"],
                    "target": row["answer"],
                    "generated": generated_text,
                    "target_ids": target_ids,
                    "generated_ids": generated_ids,
                })
        by_scale[key] = {
            "five_token_exact": statistics.fmean(token_exact),
            "five_text_exact": statistics.fmean(text_exact),
            "errors": len(rows) - sum(token_exact),
            "examples": examples,
        }
    result = {
        "protocol": "qwen_causal_ledger_generation_prefix_diagnostic_v1",
        "source_protocol": PROTOCOL,
        "posthoc_diagnostic": True,
        "changes_registered_gate": False,
        "rows": len(rows),
        "answer_tokens": 5,
        "content_sha256": {path.name: sha(path) for path in paths},
        "tokenizer": args.tokenizer,
        "by_scale": by_scale,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
