"""Retokenize a small LongCrawl64 parquet shard for Qwen3 VeOmni training.

The publisher's Hugging Face parquet has a text column. Token-ID-only copies
use GPT-2 TikToken; those IDs must never be passed to Qwen3 directly.
"""

import argparse
import json
from pathlib import Path

from scripts.build_fineweb_pretrain_pilot import pack_documents


def iter_decoded_documents(parquet_paths, max_documents):
    import pyarrow.parquet as pq

    seen = 0
    tiktoken_encoder = None
    for parquet_path in parquet_paths:
        source = pq.ParquetFile(parquet_path)
        column = next((key for key in ("text", "tokens", "input_ids")
                       if key in source.schema.names), None)
        if column is None:
            raise ValueError(f"LongCrawl64 parquet lacks text/tokens/input_ids: {source.schema.names}")
        if column != "text" and tiktoken_encoder is None:
            import tiktoken
            tiktoken_encoder = tiktoken.get_encoding("gpt2")
        for batch in source.iter_batches(batch_size=1, columns=[column]):
            for row in batch.to_pylist():
                if column == "text":
                    text = row[column]
                    if not isinstance(text, str) or not text.strip():
                        continue
                else:
                    ids = row[column]
                    if not isinstance(ids, list) or not ids:
                        continue
                    if any(not isinstance(token, int) or token < 0 or
                           token >= tiktoken_encoder.n_vocab for token in ids):
                        raise ValueError("LongCrawl64 contains a token outside GPT-2 TikToken vocabulary")
                    text = tiktoken_encoder.decode(ids)
                yield {"text": text}
                seen += 1
                if seen >= max_documents:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more local LongCrawl64 parquet shards, in deterministic order")
    parser.add_argument("--tokenizer", required=True, help="Local Qwen3-4B tokenizer directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--max-documents", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.max_documents < 1:
        parser.error("--max-documents must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rows = iter_decoded_documents(args.input, args.max_documents)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lengths = []
    with output.open("w", encoding="utf-8") as sink:
        for record in pack_documents(rows, tokenizer, args.seq_len,
                                     args.chunk_size, args.max_records):
            lengths.append(record.pop("token_count_with_eos"))
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(lengths) != args.max_records:
        raise RuntimeError(f"only produced {len(lengths)} of {args.max_records} rows; "
                           "increase --max-documents")
    metadata = {
        "source": "manifestai/longcrawl64", "parquet_shards": args.input,
        "source_format": "publisher_parquet_text_or_gpt2_ids", "target_tokenizer": args.tokenizer,
        "records": len(lengths), "seq_len": args.seq_len,
        "chunk_size": args.chunk_size, "min_token_count_with_eos": min(lengths),
        "max_token_count_with_eos": max(lengths),
        "total_token_count_with_eos": sum(lengths),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}))


if __name__ == "__main__":
    main()
