"""Stream a small FineWeb-Edu sample into long plaintext records for VeOmni.

The full FineWeb-Edu corpus is not downloaded. Each emitted JSONL record is
validated after decoding because VeOmni tokenizes `content_split` again.
"""

import argparse
import json
from pathlib import Path


DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"


def pack_documents(rows, tokenizer, seq_len: int, chunk_size: int, max_records: int):
    """Yield full two-chunk plaintext records from streamed source documents."""
    if seq_len <= chunk_size or chunk_size < 2 or max_records < 1:
        raise ValueError("seq_len must exceed chunk_size, and counts must be positive")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer needs an EOS token")

    target = seq_len - 1  # VeOmni appends one EOS token to each plaintext row.
    buffer = []
    emitted = 0
    for row in rows:
        source_text = row.get("text")
        if not isinstance(source_text, str) or not source_text.strip():
            continue
        buffer.extend(tokenizer.encode(source_text, add_special_tokens=False))
        buffer.append(tokenizer.eos_token_id)
        while len(buffer) >= target and emitted < max_records:
            count = target
            while True:
                text = tokenizer.decode(buffer[:count], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                encoded_len = len(tokenizer.encode(text, add_special_tokens=False))
                if encoded_len <= target:
                    break
                count -= 1
                if count <= chunk_size:
                    raise ValueError("tokenizer round trip exceeds sequence limit")
            if encoded_len <= chunk_size:
                raise ValueError("packed row does not reach a second TTT chunk")
            yield {"content_split": text, "token_count_with_eos": encoded_len + 1}
            emitted += 1
            del buffer[:count]
        if emitted >= max_records:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="Local Qwen3-4B tokenizer directory")
    parser.add_argument("--output", required=True, help="VeOmni plaintext JSONL output")
    parser.add_argument("--max-records", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-documents", type=int, default=10000)
    args = parser.parse_args()
    if args.max_documents < 1:
        parser.error("--max-documents must be positive")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    source = load_dataset(DATASET, name=CONFIG, split="train", streaming=True,
                          revision=args.revision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with output.open("w", encoding="utf-8") as sink:
        for record in pack_documents(source.take(args.max_documents), tokenizer,
                                     args.seq_len, args.chunk_size, args.max_records):
            lengths.append(record.pop("token_count_with_eos"))
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(lengths) != args.max_records:
        raise RuntimeError(f"only produced {len(lengths)} of {args.max_records} records; "
                           "increase --max-documents")
    metadata = {
        "source": DATASET, "config": CONFIG, "revision": args.revision,
        "license": "odc-by", "tokenizer": args.tokenizer,
        "records": len(lengths), "seq_len": args.seq_len,
        "chunk_size": args.chunk_size, "min_token_count_with_eos": min(lengths),
        "max_token_count_with_eos": max(lengths),
        "total_token_count_with_eos": sum(lengths),
        "note": "Downloaded through Hugging Face streaming; documents packed with EOS boundaries",
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
