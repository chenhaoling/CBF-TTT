"""Build the continuous two-stage Level-B public corpus or its frozen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL = "in_place_ttt_public_corpus_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md"
PROFILES = {
    "pilot": (
        {"name": "stage1_32k", "seq_len": 32768, "records_per_source": 64},
        {"name": "stage2_128k", "seq_len": 131072, "records_per_source": 8},
    ),
    # Exact corpus consumed by the registered 2xRTX5090 100M-tier configs:
    # 27 global batches at 32K followed by 5 global batches at 128K.
    "preliminary_100m": (
        {"name": "stage1_32k", "seq_len": 32768, "records_per_source": 864},
        {"name": "stage2_128k", "seq_len": 131072, "records_per_source": 160},
    ),
    "full": (
        {"name": "stage1_32k", "seq_len": 32768, "records_per_source": 305184},
        {"name": "stage2_128k", "seq_len": 131072, "records_per_source": 57216},
    ),
}
TRAINING_PROFILES = frozenset({"preliminary_100m", "full"})
SOURCES = ("fineweb", "longcrawl")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "config.json")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def token_bytes(ids: list[int]) -> bytes:
    if any(not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF for value in ids):
        raise ValueError("token IDs must be uint32 values")
    return struct.pack(f"<{len(ids)}I", *ids)


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(token_bytes(ids)).hexdigest()


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return value / divisor


def tokenizer_assets(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ASSETS:
        asset = path / name
        if asset.is_file():
            result[name] = {"sha256": sha256_file(asset), "size_bytes": asset.stat().st_size}
    for required in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if required not in result:
            raise FileNotFoundError(path / required)
    return result


def forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden_key(item) for item in value)
    return False


def _lfs_value(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def validate_remote_file(filesystem: Any, repository: str, revision: str,
                         record: dict[str, Any]) -> str:
    uri = f"hf://datasets/{repository}@{revision}/{record['repository_path']}"
    info = filesystem.info(uri)
    lfs = info.get("lfs")
    observed = {
        "size_bytes": info.get("size"),
        "lfs_sha256": _lfs_value(lfs, "sha256"),
        "blob_id": info.get("blob_id"),
    }
    expected = {key: record[key] for key in observed}
    if observed != expected or _lfs_value(lfs, "size") != info.get("size"):
        raise ValueError(f"publisher metadata drift: {record['repository_path']}")
    return uri


def iter_source_documents(source_name: str, source: dict[str, Any],
                          filesystem: Any) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    gpt2_encoder = None
    for file_index, file_record in enumerate(source["files"]):
        uri = validate_remote_file(filesystem, source["repository"], source["revision"], file_record)
        with filesystem.open(uri, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            column = next((name for name in ("text", "tokens", "input_ids")
                           if name in parquet.schema.names), None)
            if column is None:
                raise ValueError(f"publisher parquet has no supported content column: {uri}")
            if column != "text" and gpt2_encoder is None:
                import tiktoken
                gpt2_encoder = tiktoken.get_encoding("gpt2")
            columns = [column]
            if "id" in parquet.schema.names and column == "text":
                columns.append("id")
            row_index = 0
            for batch in parquet.iter_batches(batch_size=16, columns=columns):
                for row in batch.to_pylist():
                    if column == "text":
                        text = row[column]
                    else:
                        ids = row[column]
                        if (not isinstance(ids, list) or not ids
                                or any(not isinstance(value, int) or value < 0
                                       or value >= gpt2_encoder.n_vocab for value in ids)):
                            raise ValueError("invalid GPT-2 token row in LongCrawl")
                        text = gpt2_encoder.decode(ids)
                    locator = {
                        "file_index": file_index,
                        "repository_path": file_record["repository_path"],
                        "parquet_row_index": row_index,
                    }
                    row_index += 1
                    identity = row.get("id")
                    if isinstance(identity, str) and identity:
                        locator["source_identity_sha256"] = hashlib.sha256(
                            identity.encode("utf-8")
                        ).hexdigest()
                    if not isinstance(text, str) or not text.strip():
                        continue
                    yield {"source": source_name, "text": text, "locator": locator}


def document_entry(source: str, index: int, locator: dict[str, Any], text: str,
                   content_tokens: int) -> dict[str, Any]:
    return {
        "source": source,
        "document_index": index,
        "locator": locator,
        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content_token_count": content_tokens,
        "token_count_with_boundary": content_tokens + 1,
    }


def public_document_entry(segment: dict[str, Any]) -> dict[str, Any]:
    return {key: segment[key] for key in (
        "source", "document_index", "locator", "document_sha256",
        "content_token_count", "token_count_with_boundary",
    )}


class PersistentSourcePacker:
    def __init__(self, source: str, documents: Iterable[dict[str, Any]], tokenizer: Any):
        self.source = source
        self.documents = iter(documents)
        self.tokenizer = tokenizer
        self.pending: list[int] = []
        self.segments: deque[dict[str, Any]] = deque()
        self.documents_loaded = 0
        self.document_hasher = hashlib.sha256()

    def _load_document(self) -> None:
        try:
            row = next(self.documents)
        except StopIteration as error:
            raise RuntimeError(f"{self.source} source exhausted") from error
        text, locator = row.get("text"), row.get("locator")
        if row.get("source") != self.source or not isinstance(text, str) \
                or not text.strip() or not isinstance(locator, dict):
            raise ValueError(f"malformed {self.source} document")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"{self.source} document encoded to zero tokens")
        entry = document_entry(self.source, self.documents_loaded, locator, text, len(ids))
        self.document_hasher.update((canonical(entry) + "\n").encode("utf-8"))
        self.pending.extend(ids)
        self.pending.append(self.tokenizer.eos_token_id)
        self.segments.append({**entry, "cursor": 0})
        self.documents_loaded += 1

    def _consume(self, count: int) -> list[dict[str, Any]]:
        spans = []
        remaining = count
        while remaining:
            segment = self.segments[0]
            available = segment["token_count_with_boundary"] - segment["cursor"]
            take = min(remaining, available)
            span = public_document_entry(segment)
            span["token_start"] = segment["cursor"]
            span["token_end"] = segment["cursor"] + take
            spans.append(span)
            segment["cursor"] += take
            remaining -= take
            if segment["cursor"] == segment["token_count_with_boundary"]:
                self.segments.popleft()
        return spans

    def next_record(self, seq_len: int) -> tuple[str, list[int], dict[str, Any]]:
        target = seq_len - 1
        while len(self.pending) < target:
            self._load_document()
        raw_count = target
        while raw_count > 0:
            text = self.tokenizer.decode(
                self.pending[:raw_count], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= target:
                break
            raw_count -= 1
        if raw_count == 0:
            raise ValueError("no stable tokenizer round trip")
        synthetic_eos = target - len(ids)
        if synthetic_eos:
            text += self.tokenizer.eos_token * synthetic_eos
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != target:
            raise ValueError("EOS padding failed to reach exact content length")
        training_ids = ids + [self.tokenizer.eos_token_id]
        spans = self._consume(raw_count)
        del self.pending[:raw_count]
        provenance = {
            "source": self.source,
            "content_token_count": target,
            "training_token_count": seq_len,
            "raw_source_tokens_consumed": raw_count,
            "synthetic_eos_tokens": synthetic_eos,
            "training_token_sha256": token_hash(training_ids),
            "spans": spans,
        }
        if forbidden_key(provenance):
            raise AssertionError("plaintext key entered provenance")
        return text, training_ids, provenance

    def snapshot(self) -> dict[str, Any]:
        segments = []
        for segment in self.segments:
            value = public_document_entry(segment)
            value["cursor"] = segment["cursor"]
            segments.append(value)
        return {
            "source": self.source,
            "documents_loaded": self.documents_loaded,
            "pending_token_count": len(self.pending),
            "pending_tokens_sha256": token_hash(self.pending),
            "pending_segments": segments,
            "document_stream_sha256": self.document_hasher.copy().hexdigest(),
        }

    def close(self) -> None:
        close = getattr(self.documents, "close", None)
        if callable(close):
            close()


def prepare_root(output_root: Path) -> Path:
    staging = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or staging.exists():
        raise FileExistsError(f"refusing to overwrite final or partial corpus: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    return staging


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = Path(args.output_root).expanduser().resolve()
    staging = prepare_root(output_root)
    packers: dict[str, PersistentSourcePacker] = {}
    try:
        inventory_path = Path(args.inventory).expanduser().resolve()
        inventory_audit_path = Path(args.inventory_audit).expanduser().resolve()
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
        plan_path = Path(args.plan).expanduser().resolve()
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory_audit = json.loads(inventory_audit_path.read_text(encoding="utf-8"))
        if (not inventory_audit.get("passed")
                or inventory_audit.get("bindings", {}).get("inventory_sha256") != sha256_file(inventory_path)):
            raise ValueError("inventory independent audit is absent, failed, or bound to another inventory")

        from huggingface_hub import HfFileSystem
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, fix_mistral_regex=False, local_files_only=True,
        )
        if tokenizer.encode(tokenizer.eos_token, add_special_tokens=False) != [tokenizer.eos_token_id]:
            raise ValueError("tokenizer EOS is not a stable single token")
        filesystem = HfFileSystem()
        for source in SOURCES:
            documents = iter_source_documents(source, inventory["sources"][source], filesystem)
            packers[source] = PersistentSourcePacker(source, documents, tokenizer)

        combined_hasher = hashlib.sha256()
        stages = []
        total_records = total_tokens = total_synthetic = 0
        for stage in PROFILES[args.profile]:
            stage_started = time.perf_counter()
            name, seq_len, count = stage["name"], stage["seq_len"], stage["records_per_source"]
            data_path = staging / f"{name}.jsonl"
            provenance_path = staging / f"{name}.provenance.jsonl"
            stage_hasher = hashlib.sha256()
            combined_hasher.update((name + "\n").encode("utf-8"))
            source_seconds = {source: 0.0 for source in SOURCES}
            source_synthetic = {source: 0 for source in SOURCES}
            before = {source: packers[source].snapshot() for source in SOURCES}
            records = 0
            with data_path.open("x", encoding="utf-8") as data_sink, \
                    provenance_path.open("x", encoding="utf-8") as provenance_sink:
                for source_record_index in range(count):
                    for source in SOURCES:
                        call_started = time.perf_counter()
                        text, training_ids, provenance = packers[source].next_record(seq_len)
                        source_seconds[source] += time.perf_counter() - call_started
                        combined_record_index = records
                        provenance.update({
                            "stage": name,
                            "stage_record_index": combined_record_index,
                            "source_record_index": source_record_index,
                        })
                        data_sink.write(json.dumps({"content_split": text}, ensure_ascii=False) + "\n")
                        provenance_sink.write(json.dumps(provenance, ensure_ascii=False,
                                                         sort_keys=True) + "\n")
                        payload = token_bytes(training_ids)
                        stage_hasher.update(payload)
                        combined_hasher.update(payload)
                        source_synthetic[source] += provenance["synthetic_eos_tokens"]
                        records += 1
            after = {source: packers[source].snapshot() for source in SOURCES}
            stage_tokens = records * seq_len
            stage_record = {
                **stage,
                "combined_records": records,
                "combined_training_tokens": stage_tokens,
                "global_batch_size": 64 if args.profile in TRAINING_PROFILES else None,
                "training_steps": records // 64 if args.profile in TRAINING_PROFILES else None,
                "source_records": {source: count for source in SOURCES},
                "source_training_tokens": {source: count * seq_len for source in SOURCES},
                "synthetic_eos_tokens": source_synthetic,
                "source_state_before": before,
                "source_state_after": after,
                "ordered_training_tokens_sha256": stage_hasher.hexdigest(),
                "artifacts": {
                    data_path.name: {"sha256": sha256_file(data_path),
                                     "size_bytes": data_path.stat().st_size},
                    provenance_path.name: {"sha256": sha256_file(provenance_path),
                                           "size_bytes": provenance_path.stat().st_size},
                },
                "resources": {
                    "wall_seconds": time.perf_counter() - stage_started,
                    "source_pack_seconds": source_seconds,
                    "source_tokens_per_second": {
                        source: count * seq_len / source_seconds[source] for source in SOURCES
                    },
                    "peak_host_rss_mib": peak_rss_mib(),
                },
            }
            if (records != 2 * count
                    or (args.profile in TRAINING_PROFILES and records % 64 != 0)):
                raise AssertionError("stage record count does not match the frozen profile")
            if any(not math.isfinite(value) or value <= 0
                   for value in stage_record["resources"]["source_tokens_per_second"].values()):
                raise AssertionError("invalid source throughput")
            stages.append(stage_record)
            total_records += records
            total_tokens += stage_tokens
            total_synthetic += sum(source_synthetic.values())

        manifest = {
            "protocol": PROTOCOL,
            "profile": args.profile,
            "stages": stages,
            "totals": {
                "records": total_records,
                "training_tokens": total_tokens,
                "synthetic_eos_tokens": total_synthetic,
                "combined_curriculum_sha256": combined_hasher.hexdigest(),
            },
            "bindings": {
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "builder": {"path": str(Path(__file__).resolve()),
                            "sha256": sha256_file(Path(__file__).resolve())},
                "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
                "inventory_audit": {"path": str(inventory_audit_path),
                                    "sha256": sha256_file(inventory_audit_path)},
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_assets": tokenizer_assets(tokenizer_path),
            },
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "peak_host_rss_mib": peak_rss_mib(),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                             sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output_root)
        print(json.dumps({"output_root": str(output_root), "profile": args.profile,
                          **manifest["totals"], **manifest["resources"]}, sort_keys=True))
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for packer in packers.values():
            packer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    return parser.parse_args(argv)


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
