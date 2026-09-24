"""Shorten an existing balanced plaintext pilot without redownloading sources."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.build_fineweb_pretrain_pilot import pack_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("--input and --output must differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    counts = Counter()
    lengths = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row["content_split"]
            source_name = row["source"]
            packed = next(pack_documents([{"text": text}], tokenizer, args.seq_len,
                                         args.chunk_size, 1), None)
            if packed is None:
                raise ValueError(f"source row from {source_name} is shorter than {args.seq_len}")
            lengths.append(packed.pop("token_count_with_eos"))
            packed["source"] = source_name
            counts[source_name] += 1
            sink.write(json.dumps(packed, ensure_ascii=False) + "\n")
    if len(counts) != 2 or len(set(counts.values())) != 1:
        raise ValueError(f"expected equal record counts for two sources, got {dict(counts)}")
    print(json.dumps({"output": str(output), "source_records": dict(counts),
                      "min_tokens": min(lengths), "max_tokens": max(lengths)}))


if __name__ == "__main__":
    main()
