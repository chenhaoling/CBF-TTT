"""Make disjoint single-paper writer train/dev/test episodes from dated documents."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.download_postcutoff_arxiv import normalized_title


def build(documents, metadata, tokenizer, counts, chunk_size=4096, seed=122):
    if set(counts) != {"train", "dev", "test"} or min(counts.values()) < 1:
        raise ValueError("positive train/dev/test counts required")
    if len(documents) != sum(counts.values()) or len(metadata) != len(documents):
        raise ValueError("document/metadata counts do not match budget")
    if len({row["source_id"] for row in documents}) != len(documents):
        raise ValueError("duplicate paper IDs")
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    splits, cursor = {}, 0
    for split, count in counts.items():
        splits.update({index: split for index in indices[cursor:cursor + count]})
        cursor += count
    prompt = "Question: What is the exact title of the paper in this reading session?\nAnswer:"
    query = tokenizer.encode(prompt, add_special_tokens=False)
    rows = []
    for index, (document, meta) in enumerate(zip(documents, metadata)):
        if document["source_id"] != meta["source_id"] or not meta.get("title"):
            raise ValueError("document and metadata identity mismatch")
        prefix = tokenizer.encode(document["content_split"], add_special_tokens=False)[:chunk_size]
        if len(prefix) != chunk_size or normalized_title(meta["title"]) not in normalized_title(tokenizer.decode(prefix)):
            raise ValueError("title is absent from a complete input chunk")
        rows.append({"protocol": "task_writer_v1", "id": document["source_id"],
                     "group_id": document["source_id"], "split": splits[index],
                     "context_ids": prefix, "query_ids": query,
                     "answer_ids": tokenizer.encode(" " + meta["title"], add_special_tokens=False)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--dev", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=122)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    documents = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    metadata = json.loads(Path(str(args.data) + ".meta.json").read_text())
    counts = {"train": args.train, "dev": args.dev, "test": args.test}
    rows = build(documents, metadata["accepted"], AutoTokenizer.from_pretrained(args.tokenizer),
                 counts, args.chunk_size, args.seed)
    if args.output.exists():
        raise ValueError("refusing to overwrite episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"protocol": "task_writer_v1", "seed": args.seed, "counts": counts,
                "chunk_size": args.chunk_size, "tokenizer": args.tokenizer,
                "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "episodes_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "split_source_ids": {split: [row["id"] for row in rows if row["split"] == split]
                                     for split in counts}}
    Path(str(args.output) + ".meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "counts": counts}))


if __name__ == "__main__":
    main()
