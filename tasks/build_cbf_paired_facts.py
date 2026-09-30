"""Build balanced, source-disjoint twin contexts differing in one session fact."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


COLORS = ("red", "blue", "green", "yellow", "black", "white", "orange", "purple")
PROTOCOL = "paired_fact_writer_v1"


def build(documents, tokenizer, counts, chunk_size=4096, seed=132):
    if set(counts) != {"train", "dev", "test"} or any(n < 8 or n % 8 for n in counts.values()):
        raise ValueError("each split needs a positive multiple of eight source papers")
    if len(documents) != sum(counts.values()) or len({r["source_id"] for r in documents}) != len(documents):
        raise ValueError("source count mismatch or duplicate source")
    options = [tokenizer.encode(" " + color, add_special_tokens=False) for color in COLORS]
    if any(len(ids) != 1 for ids in options) or len({ids[0] for ids in options}) != 8:
        raise ValueError("colors must have distinct single-token answers")
    suffixes = [tokenizer.encode("\n\nSession record: The access color is " + color + ".\n",
                                add_special_tokens=False) for color in COLORS]
    if len({len(ids) for ids in suffixes}) != 1:
        raise ValueError("fact suffix token lengths must match")
    query = tokenizer.encode("Question: What is the access color recorded in this session?\nAnswer:",
                             add_special_tokens=False)
    ordered = list(documents)
    random.Random(seed).shuffle(ordered)
    result, cursor = [], 0
    for split, count in counts.items():
        for i, doc in enumerate(ordered[cursor:cursor + count]):
            prefix = tokenizer.encode(doc["content_split"], add_special_tokens=False)[:chunk_size-len(suffixes[0])]
            if len(prefix) + len(suffixes[0]) != chunk_size:
                raise ValueError("source too short for full chunk")
            labels = (i % 8, (i + 4) % 8)
            contexts = [prefix + suffixes[label] for label in labels]
            if sum(a != b for a, b in zip(*contexts)) != 1:
                raise ValueError("twins must differ at exactly one token")
            for variant, label in enumerate(labels):
                source = doc["source_id"]
                result.append({"protocol": PROTOCOL, "id": source + ("a", "b")[variant],
                               "group_id": source, "twin_id": source + ("b", "a")[variant],
                               "split": split, "context_ids": contexts[variant], "query_ids": query,
                               "answer_ids": options[label], "answer_index": label,
                               "choice_ids": [ids[0] for ids in options]})
        cursor += count
    validate(result)
    return result


def validate(rows):
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or {r["split"] for r in rows} != {"train", "dev", "test"}:
        raise ValueError("duplicate rows or incomplete splits")
    for row in rows:
        twin = indexed.get(row["twin_id"])
        if row["protocol"] != PROTOCOL or not row["id"].replace(".", "").isalnum():
            raise ValueError("invalid episode protocol or ID")
        if not twin or twin["twin_id"] != row["id"] or row["group_id"] != twin["group_id"] or row["split"] != twin["split"]:
            raise ValueError("invalid or cross-split twins")
        if row["answer_index"] == twin["answer_index"] or row["query_ids"] != twin["query_ids"]:
            raise ValueError("invalid twin fact/query")
        if row["choice_ids"] != twin["choice_ids"] or len(set(row["choice_ids"])) != 8:
            raise ValueError("invalid answer choices")
        if row["answer_ids"] != [row["choice_ids"][row["answer_index"]]]:
            raise ValueError("answer index mismatch")
        if len(row["context_ids"]) != len(twin["context_ids"]) or sum(a != b for a, b in zip(row["context_ids"], twin["context_ids"])) != 1:
            raise ValueError("twins must differ at exactly one context token")
    for split in ("train", "dev", "test"):
        counts = Counter(r["answer_index"] for r in rows if r["split"] == split)
        if len(counts) != 8 or len(set(counts.values())) != 1:
            raise ValueError("labels must be balanced in each split")
    if len({tuple(r["query_ids"]) for r in rows}) != 1:
        raise ValueError("query must be constant across episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seed", type=int, default=132)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError("refusing to overwrite episodes")
    documents = [json.loads(line) for line in args.data.read_text().splitlines()]
    rows = build(documents, AutoTokenizer.from_pretrained(args.tokenizer),
                 {"train": 32, "dev": 8, "test": 8}, seed=args.seed)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"protocol": PROTOCOL, "seed": args.seed, "colors": COLORS,
                "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "episodes_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "split_source_ids": {split: sorted({r["group_id"] for r in rows if r["split"] == split})
                                     for split in ("train", "dev", "test")},
                "label_counts": {split: dict(Counter(r["answer_index"] for r in rows if r["split"] == split))
                                 for split in ("train", "dev", "test")}}
    Path(str(args.output) + ".meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "groups": len(documents)}))


if __name__ == "__main__":
    main()
