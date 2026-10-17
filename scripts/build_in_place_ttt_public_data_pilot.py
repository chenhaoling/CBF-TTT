"""Build the frozen, auditable 32K public-data pilot for In-Place TTT.

This entry point is opt-in and publishes one source per output directory.  The
training JSONL remains compatible with VeOmni's plaintext transform, while the
separate provenance JSONL never stores source text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL = "in_place_ttt_public_data_pilot_v1"
FINEWEB_REPOSITORY = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_PUBLISHER_FILE = "sample/10BT/000_00000.parquet"
LONGCRAWL_REPOSITORY = "manifestai/longcrawl64"
LONGCRAWL_REVISION = "8e7af027525b3597852746ee11e4baa0e98492d7"
DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "IN_PLACE_TTT_BASELINE_PUBLIC_DATA_PILOT_ADDENDUM.md"
TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def token_bytes(token_ids: list[int]) -> bytes:
    if any(not isinstance(token_id, int) or token_id < 0 or token_id > 0xFFFFFFFF
           for token_id in token_ids):
        raise ValueError("token IDs must be uint32 values")
    return struct.pack(f"<{len(token_ids)}I", *token_ids)


def token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(token_bytes(token_ids)).hexdigest()


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return value / divisor


def tokenizer_asset_hashes(tokenizer_path: Path) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_ASSETS:
        path = tokenizer_path / name
        if path.is_file():
            assets[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    for required in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if required not in assets:
            raise FileNotFoundError(f"required tokenizer asset is missing: {tokenizer_path / required}")
    return assets


def contains_forbidden_provenance_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"}
                   or contains_forbidden_provenance_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_provenance_key(item) for item in value)
    return False


def validate_output_paths(output_root: Path) -> Path:
    staging = output_root.with_name(output_root.name + ".partial")
    for path in (output_root, staging):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {path}")
    if not output_root.parent.exists():
        output_root.parent.mkdir(parents=True)
    staging.mkdir()
    return staging


def fineweb_uri() -> str:
    return (f"hf://datasets/{FINEWEB_REPOSITORY}@{FINEWEB_REVISION}/"
            f"{FINEWEB_PUBLISHER_FILE}")


def fineweb_file_metadata() -> dict[str, Any]:
    from huggingface_hub import HfFileSystem

    info = HfFileSystem().info(fineweb_uri())
    lfs = info.get("lfs") or {}
    if not isinstance(info.get("size"), int) or info["size"] <= 0:
        raise ValueError("FineWeb publisher file has no valid byte size")
    if not isinstance(lfs.get("sha256"), str) or len(lfs["sha256"]) != 64:
        raise ValueError("FineWeb publisher file has no LFS SHA256")
    if lfs.get("size") != info["size"]:
        raise ValueError("FineWeb publisher file size and LFS size disagree")
    return {
        "repository_path": FINEWEB_PUBLISHER_FILE,
        "publisher_file": Path(FINEWEB_PUBLISHER_FILE).name,
        "size_bytes": info["size"],
        "lfs_sha256": lfs["sha256"],
        "blob_id": info.get("blob_id"),
    }


def iter_fineweb_documents(max_documents: int) -> Iterator[dict[str, Any]]:
    """Range-read pinned publisher parquet without Datasets fragment threads."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    accepted = 0
    parquet_row_index = 0
    with filesystem.open(fineweb_uri(), "rb") as source:
        parquet = pq.ParquetFile(source)
        if "text" not in parquet.schema.names:
            raise ValueError("FineWeb publisher parquet lacks a text column")
        columns = ["text"] + (["id"] if "id" in parquet.schema.names else [])
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                text = row.get("text")
                locator: dict[str, Any] = {
                    "publisher_file": FINEWEB_PUBLISHER_FILE,
                    "parquet_row_index": parquet_row_index,
                }
                parquet_row_index += 1
                source_identity = row.get("id")
                if isinstance(source_identity, str) and source_identity:
                    locator["source_identity_sha256"] = hashlib.sha256(
                        source_identity.encode("utf-8")
                    ).hexdigest()
                if not isinstance(text, str) or not text.strip():
                    continue
                yield {"text": text, "locator": locator}
                accepted += 1
                if accepted >= max_documents:
                    return


def _decode_gpt2_ids(ids: Any, encoder: Any) -> str | None:
    if not isinstance(ids, list) or not ids:
        return None
    if any(not isinstance(token_id, int) or token_id < 0 or token_id >= encoder.n_vocab
           for token_id in ids):
        raise ValueError("LongCrawl64 contains a token outside GPT-2 TikToken vocabulary")
    return encoder.decode(ids)


def iter_longcrawl_documents(paths: list[Path], max_documents: int) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    accepted = 0
    gpt2_encoder = None
    for file_index, path in enumerate(paths):
        parquet = pq.ParquetFile(path)
        column = next((name for name in ("text", "tokens", "input_ids")
                       if name in parquet.schema.names), None)
        if column is None:
            raise ValueError(f"LongCrawl64 parquet lacks text/tokens/input_ids: {parquet.schema.names}")
        if column != "text" and gpt2_encoder is None:
            import tiktoken
            gpt2_encoder = tiktoken.get_encoding("gpt2")
        absolute_row_index = 0
        for batch in parquet.iter_batches(batch_size=16, columns=[column]):
            for row in batch.to_pylist():
                if column == "text":
                    text = row[column]
                else:
                    text = _decode_gpt2_ids(row[column], gpt2_encoder)
                locator = {
                    "file_index": file_index,
                    "publisher_file": path.name,
                    "parquet_row_index": absolute_row_index,
                }
                absolute_row_index += 1
                if not isinstance(text, str) or not text.strip():
                    continue
                yield {"text": text, "locator": locator}
                accepted += 1
                if accepted >= max_documents:
                    return


def _document_entry(document_index: int, locator: dict[str, Any], text: str,
                    content_token_count: int) -> dict[str, Any]:
    return {
        "document_index": document_index,
        "locator": locator,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_token_count": content_token_count,
        "token_count_with_boundary": content_token_count + 1,
    }


def _consume_segments(segments: deque[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    remaining = count
    while remaining:
        segment = segments[0]
        available = segment["token_count_with_boundary"] - segment["cursor"]
        take = min(remaining, available)
        entry = {key: value for key, value in segment.items() if key != "cursor"}
        entry["token_start"] = segment["cursor"]
        entry["token_end"] = segment["cursor"] + take
        spans.append(entry)
        segment["cursor"] += take
        remaining -= take
        if segment["cursor"] == segment["token_count_with_boundary"]:
            segments.popleft()
    return spans


def iter_packed_records(documents: Iterable[dict[str, Any]], tokenizer: Any,
                        seq_len: int, max_records: int) -> Iterator[tuple[dict[str, str], dict[str, Any]]]:
    """Yield VeOmni rows and text-free provenance from an ordered document stream."""
    if seq_len < 2 or max_records < 1:
        raise ValueError("seq_len must be at least 2 and max_records must be positive")
    if tokenizer.eos_token_id is None or not isinstance(tokenizer.eos_token, str):
        raise ValueError("tokenizer must define eos_token_id and eos_token")
    if tokenizer.encode(tokenizer.eos_token, add_special_tokens=False) != [tokenizer.eos_token_id]:
        raise ValueError("the tokenizer EOS string must encode to exactly eos_token_id")

    target = seq_len - 1
    pending: list[int] = []
    segments: deque[dict[str, Any]] = deque()
    emitted = 0
    document_index = 0
    for document in documents:
        text = document.get("text")
        locator = document.get("locator")
        if not isinstance(text, str) or not text.strip() or not isinstance(locator, dict):
            raise ValueError("every source document must contain nonempty text and a locator dict")
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        if not content_ids:
            raise ValueError("a nonempty source document encoded to zero tokens")
        entry = _document_entry(document_index, locator, text, len(content_ids))
        pending.extend(content_ids)
        pending.append(tokenizer.eos_token_id)
        segments.append({**entry, "cursor": 0})
        document_index += 1

        while len(pending) >= target and emitted < max_records:
            raw_count = target
            while raw_count > 0:
                packed_text = tokenizer.decode(
                    pending[:raw_count],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                packed_ids = tokenizer.encode(packed_text, add_special_tokens=False)
                if len(packed_ids) <= target:
                    break
                raw_count -= 1
            if raw_count == 0:
                raise ValueError("no stable tokenizer round trip exists for the packed record")
            synthetic_eos = target - len(packed_ids)
            if synthetic_eos:
                packed_text += tokenizer.eos_token * synthetic_eos
                packed_ids = tokenizer.encode(packed_text, add_special_tokens=False)
            if len(packed_ids) != target:
                raise ValueError("EOS padding did not produce the exact content token count")
            training_ids = packed_ids + [tokenizer.eos_token_id]
            spans = _consume_segments(segments, raw_count)
            del pending[:raw_count]
            provenance = {
                "record_index": emitted,
                "content_token_count": target,
                "training_token_count": seq_len,
                "raw_source_tokens_consumed": raw_count,
                "synthetic_eos_tokens": synthetic_eos,
                "training_token_sha256": token_ids_sha256(training_ids),
                "spans": spans,
            }
            if contains_forbidden_provenance_key(provenance):
                raise AssertionError("provenance contains a forbidden plaintext key")
            yield {"content_split": packed_text}, provenance
            emitted += 1
        if emitted >= max_records:
            return
    raise RuntimeError(f"source ended after producing {emitted} of {max_records} records")


def _source_manifest(args: argparse.Namespace) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    if args.source == "fineweb":
        return iter_fineweb_documents(args.max_documents), {
            "name": "fineweb",
            "repository": FINEWEB_REPOSITORY,
            "config": FINEWEB_CONFIG,
            "revision": FINEWEB_REVISION,
            "split": "train",
            "access": "pinned_huggingface_parquet_range_read",
            "input_files": [fineweb_file_metadata()],
        }
    paths = [Path(path).expanduser().resolve() for path in args.longcrawl_input]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all --longcrawl-input paths must be existing files")
    files = [{
        "path": str(path),
        "publisher_file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in paths]
    return iter_longcrawl_documents(paths, args.max_documents), {
        "name": "longcrawl",
        "repository": LONGCRAWL_REPOSITORY,
        "revision": LONGCRAWL_REVISION,
        "split": "train",
        "input_files": files,
    }


def _document_stream_entry(span: dict[str, Any]) -> dict[str, Any]:
    return {key: span[key] for key in (
        "document_index", "locator", "document_sha256",
        "content_token_count", "token_count_with_boundary",
    )}


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(args.output_root).expanduser().resolve()
    staging = validate_output_paths(output_root)
    try:
        plan_path = Path(args.plan).expanduser().resolve()
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        tokenizer_assets = tokenizer_asset_hashes(tokenizer_path)
        documents, source = _source_manifest(args)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            fix_mistral_regex=False,
            local_files_only=True,
        )

        data_path = staging / "data.jsonl"
        provenance_path = staging / "provenance.jsonl"
        ordered_hasher = hashlib.sha256()
        document_hasher = hashlib.sha256()
        seen_documents: set[int] = set()
        records = 0
        synthetic_eos_tokens = 0
        raw_source_tokens = 0
        try:
            with data_path.open("x", encoding="utf-8") as data_sink, \
                    provenance_path.open("x", encoding="utf-8") as provenance_sink:
                for row, provenance in iter_packed_records(
                        documents, tokenizer, args.seq_len, args.max_records):
                    data_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    provenance_sink.write(json.dumps(provenance, ensure_ascii=False, sort_keys=True) + "\n")
                    training_ids = tokenizer.encode(row["content_split"], add_special_tokens=False)
                    training_ids.append(tokenizer.eos_token_id)
                    ordered_hasher.update(token_bytes(training_ids))
                    synthetic_eos_tokens += provenance["synthetic_eos_tokens"]
                    raw_source_tokens += provenance["raw_source_tokens_consumed"]
                    for span in provenance["spans"]:
                        index = span["document_index"]
                        if index not in seen_documents:
                            if index != len(seen_documents):
                                raise AssertionError("document order is not contiguous")
                            seen_documents.add(index)
                            document_hasher.update(
                                (canonical_json(_document_stream_entry(span)) + "\n").encode("utf-8")
                            )
                    records += 1
        finally:
            close = getattr(documents, "close", None)
            if callable(close):
                close()

        if records != args.max_records:
            raise AssertionError(f"built {records} records, expected {args.max_records}")
        manifest = {
            "protocol": PROTOCOL,
            "source": source,
            "packing": {
                "records": records,
                "seq_len": args.seq_len,
                "content_tokens_per_record": args.seq_len - 1,
                "total_training_tokens": records * args.seq_len,
                "documents_read": len(seen_documents),
                "raw_source_tokens_consumed": raw_source_tokens,
                "synthetic_eos_tokens": synthetic_eos_tokens,
                "terminal_eos_tokens": records,
            },
            "bindings": {
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "builder": {"path": str(Path(__file__).resolve()),
                            "sha256": sha256_file(Path(__file__).resolve())},
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_assets": tokenizer_assets,
            },
            "artifacts": {
                "data.jsonl": {"sha256": sha256_file(data_path), "size_bytes": data_path.stat().st_size},
                "provenance.jsonl": {
                    "sha256": sha256_file(provenance_path),
                    "size_bytes": provenance_path.stat().st_size,
                },
                "ordered_training_tokens_sha256": ordered_hasher.hexdigest(),
                "document_stream_sha256": document_hasher.hexdigest(),
            },
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "peak_host_rss_mib": peak_rss_mib(),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        if manifest["resources"]["wall_seconds"] <= 0 or manifest["resources"]["peak_host_rss_mib"] <= 0:
            raise AssertionError("invalid resource measurement")
        os.replace(staging, output_root)
        print(json.dumps({"output_root": str(output_root), **manifest["packing"],
                          **manifest["resources"]}, ensure_ascii=False, sort_keys=True))
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("fineweb", "longcrawl"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--longcrawl-input", nargs="*", default=[])
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--max-documents", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.seq_len < 2 or args.max_documents < 1:
        parser.error("record, sequence, and document limits must be positive")
    if args.source == "longcrawl" and not args.longcrawl_input:
        parser.error("--longcrawl-input is required for --source longcrawl")
    if args.source == "fineweb" and args.longcrawl_input:
        parser.error("--longcrawl-input is only valid for --source longcrawl")
    return args


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
