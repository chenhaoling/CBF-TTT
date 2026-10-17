"""Independently audit an In-Place TTT public-data pilot directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from itertools import zip_longest
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_data_pilot_v1"
FINEWEB = {
    "repository": "HuggingFaceFW/fineweb-edu",
    "config": "sample-10BT",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
}
LONGCRAWL = {
    "repository": "manifestai/longcrawl64",
    "revision": "8e7af027525b3597852746ee11e4baa0e98492d7",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPOSITORY_ROOT / "IN_PLACE_TTT_BASELINE_PUBLIC_DATA_PILOT_ADDENDUM.md"
DEFAULT_BUILDER = REPOSITORY_ROOT / "scripts/build_in_place_ttt_public_data_pilot.py"
TOKENIZER_ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "config.json",
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


def token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(token_bytes(token_ids)).hexdigest()


def forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden_key(item) for item in value)
    return False


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def _document_entry(span: dict[str, Any]) -> dict[str, Any]:
    return {key: span[key] for key in (
        "document_index", "locator", "document_sha256",
        "content_token_count", "token_count_with_boundary",
    )}


def _audit_source(source: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    name = source.get("name")
    if name == "fineweb":
        passed = (
            source.get("repository") == FINEWEB["repository"]
            and source.get("config") == FINEWEB["config"]
            and source.get("revision") == FINEWEB["revision"]
            and source.get("split") == "train"
            and source.get("input_files") == []
        )
        return passed, {"name": name, "revision": source.get("revision"),
                        "input_files_rehashed": 0}
    if name != "longcrawl":
        return False, {"name": name, "error": "unknown source"}
    files = source.get("input_files")
    if not isinstance(files, list) or not files:
        return False, {"name": name, "error": "missing input files"}
    observed = []
    passed = (source.get("repository") == LONGCRAWL["repository"]
              and source.get("revision") == LONGCRAWL["revision"]
              and source.get("split") == "train")
    for record in files:
        path = Path(record.get("path", ""))
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = sha256_file(path) if exists else None
        file_passed = (exists and size == record.get("size_bytes")
                       and digest == record.get("sha256")
                       and path.name == record.get("publisher_file"))
        passed = passed and file_passed
        observed.append({"path": str(path), "exists": exists, "size_bytes": size,
                         "sha256": digest, "passed": file_passed})
    return passed, {"name": name, "revision": source.get("revision"),
                    "input_files_rehashed": len(files), "files": observed}


def audit(root: Path, tokenizer_path: Path, plan_path: Path,
          builder_path: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    data_path = root / "data.jsonl"
    provenance_path = root / "provenance.jsonl"
    for required in (manifest_path, data_path, provenance_path, tokenizer_path,
                     plan_path, builder_path):
        if not required.exists():
            raise FileNotFoundError(required)
    if not tokenizer_path.is_dir():
        raise NotADirectoryError(tokenizer_path)
    manifest = read_json(manifest_path)
    packing = manifest.get("packing", {})
    artifacts = manifest.get("artifacts", {})
    bindings = manifest.get("bindings", {})
    resources = manifest.get("resources", {})

    checks: dict[str, bool] = {}
    checks["protocol"] = manifest.get("protocol") == PROTOCOL
    checks["no_partial_paths"] = not any(root.parent.glob(root.name + ".partial*"))
    checks["plan_binding"] = (
        bindings.get("plan", {}).get("sha256") == sha256_file(plan_path)
    )
    checks["builder_binding"] = (
        bindings.get("builder", {}).get("sha256") == sha256_file(builder_path)
    )
    source_passed, source_observed = _audit_source(manifest.get("source", {}))
    checks["source_binding"] = source_passed

    expected_assets = bindings.get("tokenizer_assets", {})
    observed_assets: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_ASSETS:
        path = tokenizer_path / name
        if path.is_file():
            observed_assets[name] = {"sha256": sha256_file(path),
                                     "size_bytes": path.stat().st_size}
    checks["tokenizer_assets"] = (
        all(name in observed_assets for name in ("tokenizer.json", "tokenizer_config.json", "config.json"))
        and observed_assets == expected_assets
    )
    checks["artifact_hashes"] = (
        artifacts.get("data.jsonl", {}).get("sha256") == sha256_file(data_path)
        and artifacts.get("data.jsonl", {}).get("size_bytes") == data_path.stat().st_size
        and artifacts.get("provenance.jsonl", {}).get("sha256") == sha256_file(provenance_path)
        and artifacts.get("provenance.jsonl", {}).get("size_bytes") == provenance_path.stat().st_size
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        fix_mistral_regex=False,
        local_files_only=True,
    )
    eos_id = tokenizer.eos_token_id
    expected_records = packing.get("records")
    seq_len = packing.get("seq_len")
    ordered_hasher = hashlib.sha256()
    document_hasher = hashlib.sha256()
    document_entries: dict[int, dict[str, Any]] = {}
    document_cursors: dict[int, int] = {}
    record_count = 0
    total_tokens = 0
    total_synthetic_eos = 0
    total_raw_tokens = 0
    rows_valid = True
    provenance_has_no_plaintext = True
    previous_document_index = -1
    for record_index, pair in enumerate(zip_longest(
            read_jsonl(data_path), read_jsonl(provenance_path))):
        row, provenance = pair
        if row is None or provenance is None:
            rows_valid = False
            break
        provenance_has_no_plaintext &= not forbidden_key(provenance)
        if set(row) != {"content_split"} or not isinstance(row["content_split"], str):
            rows_valid = False
            continue
        content_ids = tokenizer.encode(row["content_split"], add_special_tokens=False)
        training_ids = content_ids + [eos_id]
        row_hash = token_sha256(training_ids)
        ordered_hasher.update(token_bytes(training_ids))
        row_spans = provenance.get("spans")
        row_raw = 0
        if not isinstance(row_spans, list) or not row_spans:
            rows_valid = False
            row_spans = []
        for span in row_spans:
            try:
                document_index = span["document_index"]
                start, end = span["token_start"], span["token_end"]
                total = span["token_count_with_boundary"]
                entry = _document_entry(span)
            except (KeyError, TypeError):
                rows_valid = False
                continue
            if (not isinstance(document_index, int) or document_index < 0
                    or not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end <= total
                    or span.get("content_token_count") + 1 != total
                    or document_index < previous_document_index):
                rows_valid = False
                continue
            previous_document_index = document_index
            if document_index not in document_entries:
                if document_index != len(document_entries) or start != 0:
                    rows_valid = False
                document_entries[document_index] = entry
                document_cursors[document_index] = 0
                document_hasher.update((canonical_json(entry) + "\n").encode("utf-8"))
            if document_entries[document_index] != entry \
                    or start != document_cursors[document_index]:
                rows_valid = False
            document_cursors[document_index] = end
            row_raw += end - start
        synthetic = provenance.get("synthetic_eos_tokens")
        if not isinstance(synthetic, int) or synthetic < 0:
            rows_valid = False
            synthetic = 0
        rows_valid &= (
            provenance.get("record_index") == record_index
            and isinstance(seq_len, int)
            and len(content_ids) == seq_len - 1
            and len(training_ids) == seq_len
            and provenance.get("content_token_count") == seq_len - 1
            and provenance.get("training_token_count") == seq_len
            and provenance.get("raw_source_tokens_consumed") == row_raw
            and provenance.get("training_token_sha256") == row_hash
        )
        record_count += 1
        total_tokens += len(training_ids)
        total_synthetic_eos += synthetic
        total_raw_tokens += row_raw

    checks["records_and_token_hashes"] = bool(
        rows_valid
        and isinstance(expected_records, int)
        and record_count == expected_records
        and total_tokens == packing.get("total_training_tokens")
        and total_tokens == expected_records * seq_len
        and ordered_hasher.hexdigest() == artifacts.get("ordered_training_tokens_sha256")
    )
    checks["provenance_has_no_plaintext"] = provenance_has_no_plaintext
    checks["document_stream"] = (
        len(document_entries) == packing.get("documents_read")
        and document_hasher.hexdigest() == artifacts.get("document_stream_sha256")
        and total_raw_tokens == packing.get("raw_source_tokens_consumed")
    )
    checks["eos_accounting"] = (
        total_synthetic_eos == packing.get("synthetic_eos_tokens")
        and packing.get("terminal_eos_tokens") == record_count
    )
    checks["resources"] = (
        isinstance(resources.get("wall_seconds"), (int, float))
        and math.isfinite(resources["wall_seconds"]) and resources["wall_seconds"] > 0
        and isinstance(resources.get("peak_host_rss_mib"), (int, float))
        and math.isfinite(resources["peak_host_rss_mib"]) and resources["peak_host_rss_mib"] > 0
    )
    return {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "records": record_count,
            "training_tokens": total_tokens,
            "documents": len(document_entries),
            "synthetic_eos_tokens": total_synthetic_eos,
            "raw_source_tokens_consumed": total_raw_tokens,
            "ordered_training_tokens_sha256": ordered_hasher.hexdigest(),
            "document_stream_sha256": document_hasher.hexdigest(),
            "source": source_observed,
            "tokenizer_assets": observed_assets,
        },
        "bindings": {
            "manifest_sha256": sha256_file(manifest_path),
            "plan_sha256": sha256_file(plan_path),
            "builder_sha256": sha256_file(builder_path),
            "auditor_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--builder", default=str(DEFAULT_BUILDER))
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "independent_audit.json"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {output}")
    result = audit(root, Path(args.tokenizer).expanduser().resolve(),
                   Path(args.plan).expanduser().resolve(),
                   Path(args.builder).expanduser().resolve())
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "checks": result["checks"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
