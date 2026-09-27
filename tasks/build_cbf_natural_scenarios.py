"""Build grouped joint-gate pilots from unseen continuations of pretraining records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


REGIMES = ("both_relevant", "old_only", "new_only", "neither_relevant")


def load_records(path: Path, tokenizer, count: int, minimum_tokens: int) -> tuple[list[list[int]], str]:
    records = []
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            content = row.get("content_split")
            if not isinstance(content, str) or not content.strip():
                continue
            ids = tokenizer.encode(content, add_special_tokens=False)
            if len(ids) < minimum_tokens:
                continue
            records.append(ids)
            digest.update(line.encode("utf-8"))
            if len(records) == count:
                break
    if len(records) != count:
        raise ValueError(f"needed {count} long records, found {len(records)}")
    return records, digest.hexdigest()


def build_scenarios(records: list[list[int]], split_counts: dict[str, int],
                    context_tokens: int, query_tokens: int, answer_tokens: int,
                    seed: int) -> tuple[list[dict], dict]:
    groups = sum(split_counts.values())
    if groups < 3 or any(split_counts.get(name, 0) < 1 for name in ("train", "dev", "test")):
        raise ValueError("positive train/dev/test group counts are required")
    if len(records) != 4 * groups:
        raise ValueError("four distinct records are required for every source group")
    if min(context_tokens, query_tokens, answer_tokens) < 1:
        raise ValueError("context, query, and answer token counts must be positive")
    if any(len(record) < context_tokens + query_tokens + answer_tokens for record in records):
        raise ValueError("a natural record is shorter than the held-out split")
    indices = list(range(groups))
    random.Random(seed).shuffle(indices)
    split_for_group = {}
    cursor = 0
    for split in ("train", "dev", "test"):
        for index in indices[cursor:cursor + split_counts[split]]:
            split_for_group[index] = split
        cursor += split_counts[split]

    def heldout(record: list[int], kind: str) -> dict:
        begin = context_tokens
        return {"kind": kind,
                "query_ids": record[begin:begin + query_tokens],
                "answer_ids": record[begin + query_tokens:begin + query_tokens + answer_tokens]}

    scenarios = []
    for index in range(groups):
        old, new, distractor, neutral = records[4 * index:4 * index + 4]
        group_id = f"natural-pack-{index:05d}"
        for variant, regime in enumerate(REGIMES):
            candidate = new if regime in ("both_relevant", "new_only") else distractor
            if regime == "both_relevant":
                queries = [heldout(old, "old_continuation"), heldout(new, "new_continuation")]
            elif regime == "old_only":
                queries = [heldout(old, "old_continuation")]
            elif regime == "new_only":
                queries = [heldout(new, "new_continuation")]
            else:
                queries = [heldout(neutral, "neutral_continuation")]
            context = old[:context_tokens] + candidate[:context_tokens]
            if any(query["answer_ids"] == context[start:start + answer_tokens]
                   for query in queries for start in range(len(context) - answer_tokens + 1)):
                raise ValueError(f"held-out target appears in context for {group_id}-{regime}")
            scenarios.append({
                "id": f"{group_id}-v{variant}", "group_id": group_id,
                "split": split_for_group[index], "regime": regime,
                "objective": "joint_natural_v1", "candidate_boundary": 2,
                "context_ids": context,
                "futures": [{"gap_chunks": 0, "continuation_ids": [], "queries": queries}],
            })
    metadata = {
        "protocol": "joint_natural_v1", "seed": seed,
        "context_tokens": context_tokens, "query_tokens": query_tokens,
        "answer_tokens": answer_tokens, "record_count": len(records),
        "source_groups": groups, "scenarios": len(scenarios),
        "split_group_counts": split_counts,
        "split_groups": {split: sorted(row["group_id"] for row in scenarios
                                        if row["split"] == split and row["regime"] == REGIMES[0])
                         for split in ("train", "dev", "test")},
    }
    return scenarios, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-groups", type=int, default=8)
    parser.add_argument("--dev-groups", type=int, default=2)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--answer-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=118)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    split_counts = {"train": args.train_groups, "dev": args.dev_groups, "test": args.test_groups}
    records, source_hash = load_records(args.data, tokenizer, 4 * sum(split_counts.values()),
                                        args.context_tokens + args.query_tokens + args.answer_tokens)
    scenarios, metadata = build_scenarios(records, split_counts, args.context_tokens,
                                          args.query_tokens, args.answer_tokens, args.seed)
    metadata.update({"source_sha256": source_hash, "source_path": str(args.data),
                     "tokenizer": args.tokenizer})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            sink.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    Path(str(args.output) + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "scenarios": len(scenarios),
                      "groups": sum(split_counts.values())}))


if __name__ == "__main__":
    main()
