#!/usr/bin/env python3
"""Read-only smoke audit of a growing Base-tokenized Stage-1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import struct
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_partial_corpus_smoke_v1"
STAGE = "stage1_32k"
SEQ_LEN = 32768
SOURCES = ("fineweb", "longcrawl")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "config.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def token_sha(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or forbidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden(item) for item in value)
    return False


def first_complete_lines(path: Path, count: int) -> list[bytes]:
    rows = []
    with path.open("rb") as handle:
        for _ in range(count):
            line = handle.readline()
            if not line.endswith(b"\n"):
                raise ValueError(f"{path.name} has fewer than {count} complete lines")
            rows.append(line)
    return rows


def last_complete_lines(path: Path, count: int) -> list[bytes]:
    with path.open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        position = end
        blocks: list[bytes] = []
        newlines = 0
        while position and newlines <= count:
            take = min(1 << 20, position)
            position -= take
            handle.seek(position)
            block = handle.read(take)
            blocks.append(block)
            newlines += block.count(b"\n")
        payload = b"".join(reversed(blocks))
    lines = payload.splitlines(keepends=True)
    complete = [line for line in lines if line.endswith(b"\n")]
    if len(complete) < count:
        raise ValueError(f"{path.name} has fewer than {count} complete lines")
    return complete[-count:]


def provenance_errors(provenance: Any, record_index: int) -> list[str]:
    errors: list[str] = []
    source = SOURCES[record_index % 2]
    spans = provenance.get("spans") if isinstance(provenance, dict) else None
    raw_count = 0
    if not isinstance(spans, list) or not spans:
        errors.append("spans are absent")
        spans = []
    for span in spans:
        if not isinstance(span, dict):
            errors.append("span is not an object")
            continue
        start, end = span.get("token_start"), span.get("token_end")
        total = span.get("token_count_with_boundary")
        if (span.get("source") != source or not isinstance(start, int)
                or not isinstance(end, int) or not isinstance(total, int)
                or start < 0 or end <= start or end > total):
            errors.append("span boundary or source is invalid")
        else:
            raw_count += end - start
    expected = {
        "source": source,
        "stage": STAGE,
        "stage_record_index": record_index,
        "source_record_index": record_index // 2,
        "content_token_count": SEQ_LEN - 1,
        "training_token_count": SEQ_LEN,
    }
    for key, value in expected.items():
        if not isinstance(provenance, dict) or provenance.get(key) != value:
            errors.append(f"{key} differs")
    if isinstance(provenance, dict):
        if provenance.get("raw_source_tokens_consumed") != raw_count:
            errors.append("raw source span accounting differs")
        synthetic = provenance.get("synthetic_eos_tokens")
        if not isinstance(synthetic, int) or synthetic < 0:
            errors.append("synthetic EOS count is invalid")
        digest = provenance.get("training_token_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("training-token digest is invalid")
        if forbidden(provenance):
            errors.append("plaintext key entered provenance")
    return errors


def record_errors(row: Any, provenance: Any, record_index: int,
                  tokenizer: Any) -> tuple[list[str], dict[str, Any]]:
    errors = provenance_errors(provenance, record_index)
    if not isinstance(row, dict) or set(row) != {"content_split"} \
            or not isinstance(row.get("content_split"), str):
        return errors + ["data row schema is invalid"], {}
    ids = tokenizer.encode(row["content_split"], add_special_tokens=False)
    training_ids = ids + [tokenizer.eos_token_id]
    observed = token_sha(training_ids)
    if len(ids) != SEQ_LEN - 1:
        errors.append("retokenized content length differs")
    if provenance.get("training_token_sha256") != observed:
        errors.append("training-token digest differs")
    return errors, {
        "stage_record_index": record_index,
        "source": SOURCES[record_index % 2],
        "data_line_sha256": hashlib.sha256(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        ).hexdigest(),
        "provenance_sha256": hashlib.sha256(
            (json.dumps(provenance, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest(),
        "training_token_sha256": observed,
        "content_tokens": len(ids),
    }


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if platform.node() != args.expected_hostname:
        raise RuntimeError("partial-corpus smoke is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("partial-corpus smoke must hide all GPUs")
    final_root = args.output_root.resolve()
    partial_root = final_root.with_name(final_root.name + ".partial")
    if final_root.exists() or not partial_root.is_dir():
        raise RuntimeError("expected a growing partial root and absent final root")
    data_path = partial_root / f"{STAGE}.jsonl"
    provenance_path = partial_root / f"{STAGE}.provenance.jsonl"
    for path in (data_path, provenance_path, args.plan, args.builder):
        if not path.is_file():
            raise FileNotFoundError(path)
    process_cmdline = Path(f"/proc/{args.builder_pid}/cmdline")
    if not process_cmdline.is_file():
        raise RuntimeError("registered corpus builder PID is not live")
    command = process_cmdline.read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="strict"
    )
    if str(args.builder.resolve()) not in command or str(final_root) not in command:
        raise RuntimeError("live PID does not bind the registered builder and output root")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.resolve(), use_fast=True, fix_mistral_regex=False,
        local_files_only=True,
    )
    if tokenizer.eos_token_id != 151643:
        raise ValueError("tokenizer is not the registered Qwen3-4B-Base tokenizer")

    first_data = first_complete_lines(data_path, args.sample_pairs * 2)
    first_provenance = first_complete_lines(provenance_path, args.sample_pairs * 2)
    errors: list[str] = []
    samples = []
    for index, (data_line, provenance_line) in enumerate(
            zip(first_data, first_provenance, strict=True)):
        row = json.loads(data_line)
        provenance = json.loads(provenance_line)
        row_errors, summary = record_errors(row, provenance, index, tokenizer)
        errors.extend(f"record {index}: {error}" for error in row_errors)
        samples.append(summary)

    recent_lines = last_complete_lines(provenance_path, args.recent_provenance)
    recent = [json.loads(line) for line in recent_lines]
    recent_indices = [row.get("stage_record_index") for row in recent]
    if (not all(isinstance(index, int) for index in recent_indices)
            or recent_indices != list(range(recent_indices[0], recent_indices[0]
                                            + len(recent_indices)))):
        errors.append("recent provenance indices are not consecutive")
    else:
        for provenance, index in zip(recent, recent_indices, strict=True):
            errors.extend(f"recent record {index}: {error}"
                          for error in provenance_errors(provenance, index))

    assets = {}
    for name in ASSETS:
        path = args.tokenizer.resolve() / name
        if path.is_file():
            assets[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    result = {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "checks": {
            "live_registered_builder": True,
            "partial_atomicity": True,
            "first_records": not any(error.startswith("record ") for error in errors),
            "recent_provenance": not any(error.startswith("recent ") for error in errors)
                and "recent provenance indices are not consecutive" not in errors,
            "no_plaintext_artifact": True,
        },
        "sampled_first_records": samples,
        "recent_provenance": {
            "records": len(recent_indices),
            "first_stage_record_index": recent_indices[0],
            "last_stage_record_index": recent_indices[-1],
            "ordered_provenance_sha256": hashlib.sha256(
                b"".join(recent_lines)
            ).hexdigest(),
        },
        "observed_files": {
            data_path.name: {"size_bytes": data_path.stat().st_size},
            provenance_path.name: {"size_bytes": provenance_path.stat().st_size},
        },
        "bindings": {
            "host": platform.node(),
            "builder_pid": args.builder_pid,
            "builder_cmdline_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "builder_sha256": sha256(args.builder.resolve()),
            "plan_sha256": sha256(args.plan.resolve()),
            "auditor_sha256": sha256(Path(__file__).resolve()),
            "tokenizer_path": str(args.tokenizer.resolve()),
            "tokenizer_assets": assets,
        },
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "peak_host_rss_mib": peak_rss_mib(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "claim_boundary": (
            "read-only partial smoke only; complete independent corpus audit remains required"
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--builder-pid", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-pairs", type=int, default=8)
    parser.add_argument("--recent-provenance", type=int, default=32)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_pairs < 1 or args.recent_provenance < 2:
        raise ValueError("sample sizes are too small")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": len(result["errors"]),
                      "last_stage_record_index":
                      result["recent_provenance"]["last_stage_record_index"]},
                     sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
