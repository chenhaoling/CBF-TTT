"""Build four-regime title-recall scenarios with KV-intact and memory-only reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.download_postcutoff_arxiv import normalized_title


REGIMES = ("both_relevant", "old_only", "new_only", "neither_relevant")


def build_scenarios(documents: list[dict], source_meta: list[dict], tokenizer,
                    split_counts: dict[str, int], context_tokens: int = 4096,
                    seed: int = 120) -> tuple[list[dict], dict]:
    groups = sum(split_counts.values())
    if groups < 3 or any(split_counts.get(split, 0) < 1 for split in ("train", "dev", "test")):
        raise ValueError("positive train/dev/test group counts are required")
    if context_tokens < 1 or len(documents) != 4 * groups or len(source_meta) != len(documents):
        raise ValueError("four matching, long documents are required per source group")
    ids = [item["source_id"] for item in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("source documents must have distinct IDs")
    records = []
    for document, meta in zip(documents, source_meta):
        if document["source_id"] != meta["source_id"] or not meta.get("title"):
            raise ValueError("document/metadata source ID or title mismatch")
        tokens = tokenizer.encode(document["content_split"], add_special_tokens=False)
        if len(tokens) < context_tokens:
            raise ValueError(f"short paper: {document['source_id']}")
        prefix = tokens[:context_tokens]
        if normalized_title(meta["title"]) not in normalized_title(tokenizer.decode(prefix)):
            raise ValueError(f"title absent from visible prefix: {document['source_id']}")
        records.append((prefix, meta["title"]))
    if len(set(title for _, title in records)) != len(records):
        raise ValueError("duplicate titles would make recall ambiguous")

    indices = list(range(groups))
    random.Random(seed).shuffle(indices)
    split_for_group = {}
    cursor = 0
    for split in ("train", "dev", "test"):
        for index in indices[cursor:cursor + split_counts[split]]:
            split_for_group[index] = split
        cursor += split_counts[split]

    def query(title: str, ordinal: str, kind: str) -> dict:
        prompt = f"Question: What is the exact title of the {ordinal} paper in this reading session?\nAnswer:"
        return {"kind": kind,
                "query_ids": tokenizer.encode(prompt, add_special_tokens=False),
                "answer_ids": tokenizer.encode(" " + title, add_special_tokens=False)}

    scenarios = []
    for index in range(groups):
        old, new, distractor, neutral = records[4 * index:4 * index + 4]
        group_id = f"title-recall-{index:05d}"
        for variant, regime in enumerate(REGIMES):
            candidate = new if regime in ("both_relevant", "new_only") else distractor
            if regime == "both_relevant":
                queries = [query(old[1], "first", "old_title"), query(new[1], "second", "new_title")]
            elif regime == "old_only":
                queries = [query(old[1], "first", "old_title")]
            elif regime == "new_only":
                queries = [query(new[1], "second", "new_title")]
            else:
                queries = [query(neutral[1], "third", "unseen_title")]
            context = old[0] + candidate[0]
            # Every action sees the same question/answer in both read conditions.
            futures = [{"gap_chunks": 0, "reset_kv": reset,
                        "continuation_ids": [], "queries": queries}
                       for reset in (False, True)]
            scenarios.append({"id": f"{group_id}-v{variant}", "group_id": group_id,
                              "split": split_for_group[index], "regime": regime,
                              "objective": "joint_title_recall_v1", "candidate_boundary": 2,
                              "context_ids": context, "futures": futures})
    return scenarios, {
        "protocol": "joint_title_recall_v1", "seed": seed,
        "context_tokens_per_paper": context_tokens, "record_count": len(records),
        "source_groups": groups, "scenarios": len(scenarios),
        "split_group_counts": split_counts,
        "split_groups": {split: sorted(row["group_id"] for row in scenarios
                                if row["split"] == split and row["regime"] == REGIMES[0])
                         for split in ("train", "dev", "test")},
        "read_conditions": ["kv_intact", "memory_only"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-groups", type=int, default=8)
    parser.add_argument("--dev-groups", type=int, default=2)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=120)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    documents = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads(Path(str(args.data) + ".meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    scenarios, manifest = build_scenarios(
        documents, meta["accepted"], tokenizer,
        {"train": args.train_groups, "dev": args.dev_groups, "test": args.test_groups},
        args.context_tokens, args.seed,
    )
    manifest.update({"source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                     "source_path": str(args.data), "tokenizer": args.tokenizer})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            sink.write(json.dumps(scenario, ensure_ascii=False) + "\n")
    Path(str(args.output) + ".meta.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "scenarios": len(scenarios), "groups": manifest["source_groups"]}))


if __name__ == "__main__":
    main()
