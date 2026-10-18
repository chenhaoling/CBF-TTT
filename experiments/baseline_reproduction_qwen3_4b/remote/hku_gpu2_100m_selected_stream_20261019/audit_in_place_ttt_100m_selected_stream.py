from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


PROTOCOL = "in_place_ttt_100m_selected_stream_audit_v1"
SEED = 42
BUFFER_SIZE = 10_000
WORLD_SIZE = 2
PER_RANK_BATCH = 32
PREFIX_RECORDS = 15_000
STAGES = {
    "stage1_32k": {"seq_len": 32_768, "steps": 27},
    "stage2_128k": {"seq_len": 131_072, "steps": 5},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def make_lightweight_prefix(source: Path, target: Path, stage: str, seq_len: int) -> dict:
    digest = hashlib.sha256()
    rows = 0
    with source.open(encoding="utf-8") as reader, target.open("x", encoding="utf-8") as writer:
        for line in itertools.islice(reader, PREFIX_RECORDS):
            row = json.loads(line)
            if (
                row.get("stage") != stage
                or row.get("stage_record_index") != rows
                or row.get("source") not in {"fineweb", "longcrawl"}
                or row.get("training_token_count") != seq_len
                or not isinstance(row.get("training_token_sha256"), str)
                or len(row["training_token_sha256"]) != 64
            ):
                raise ValueError(f"invalid provenance prefix at {stage} row {rows}")
            compact = {
                "stage_record_index": row["stage_record_index"],
                "source": row["source"],
                "training_token_sha256": row["training_token_sha256"],
            }
            payload = json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n"
            writer.write(payload)
            digest.update(payload.encode())
            rows += 1
    if rows != PREFIX_RECORDS:
        raise ValueError(f"{stage} has only {rows} complete prefix records")
    return {"records": rows, "compact_sha256": digest.hexdigest()}


def rank_schedule(path: Path, rank: int, count: int) -> list[dict]:
    dataset = load_dataset("json", data_files=[str(path)], split="train", streaming=True)
    dataset = dataset.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=WORLD_SIZE)
    return list(itertools.islice(dataset, count))


def audit_stage(path: Path, stage: str, seq_len: int, steps: int) -> dict:
    count = steps * PER_RANK_BATCH
    first = [rank_schedule(path, rank, count) for rank in range(WORLD_SIZE)]
    second = [rank_schedule(path, rank, count) for rank in range(WORLD_SIZE)]
    indices = [[row["stage_record_index"] for row in rows] for rows in first]
    token_hashes = [[row["training_token_sha256"] for row in rows] for rows in first]
    sources = [[row["source"] for row in rows] for rows in first]
    flat_indices = [value for rank_values in indices for value in rank_values]
    flat_hashes = [value for rank_values in token_hashes for value in rank_values]
    flat_sources = [value for rank_values in sources for value in rank_values]
    source_counts = Counter(flat_sources)
    total = len(flat_sources)
    expected = steps * PER_RANK_BATCH * WORLD_SIZE
    checks = {
        "repeat_schedule_exact": first == second,
        "rank_lengths_exact": all(len(rows) == count for rows in first),
        "global_records_exact": total == expected,
        "stage_indices_unique": len(set(flat_indices)) == expected,
        "training_hashes_unique": len(set(flat_hashes)) == expected,
        "rank_indices_disjoint": set(indices[0]).isdisjoint(indices[1]),
        "rank_hashes_disjoint": set(token_hashes[0]).isdisjoint(token_hashes[1]),
        "sources_valid": set(flat_sources) <= {"fineweb", "longcrawl"},
        "prefix_sufficient": PREFIX_RECORDS >= BUFFER_SIZE + expected,
    }
    return {
        "stage": stage,
        "sequence_length": seq_len,
        "steps": steps,
        "records_per_rank": count,
        "selected_records": total,
        "selected_training_tokens": total * seq_len,
        "source_records": dict(sorted(source_counts.items())),
        "source_training_tokens": {
            source: records * seq_len for source, records in sorted(source_counts.items())
        },
        "absolute_source_record_difference": abs(
            source_counts["fineweb"] - source_counts["longcrawl"]
        ),
        "absolute_source_fraction_difference": abs(
            source_counts["fineweb"] - source_counts["longcrawl"]
        ) / total,
        "rank_source_records": [dict(sorted(Counter(values).items())) for values in sources],
        "ordered_stage_indices_sha256": canonical_hash(indices),
        "ordered_training_hashes_sha256": canonical_hash(token_hashes),
        "ordered_sources_sha256": canonical_hash(sources),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-provenance", type=Path, required=True)
    parser.add_argument("--stage2-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        raise FileExistsError(args.output)

    provenance = {
        "stage1_32k": args.stage1_provenance.resolve(),
        "stage2_128k": args.stage2_provenance.resolve(),
    }
    with tempfile.TemporaryDirectory(prefix="in_place_ttt_selected_stream_") as temp:
        root = Path(temp)
        prefixes = {}
        results = {}
        for stage, spec in STAGES.items():
            compact = root / f"{stage}.jsonl"
            prefixes[stage] = make_lightweight_prefix(
                provenance[stage], compact, stage, spec["seq_len"]
            )
            results[stage] = audit_stage(
                compact, stage, spec["seq_len"], spec["steps"]
            )

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": SEED,
        "shuffle_buffer_size": BUFFER_SIZE,
        "world_size": WORLD_SIZE,
        "per_rank_batch": PER_RANK_BATCH,
        "prefix_records": PREFIX_RECORDS,
        "provenance_paths": {stage: str(path) for stage, path in provenance.items()},
        "prefixes": prefixes,
        "stages": results,
        "script_sha256": sha256_file(Path(__file__)),
    }
    payload["total_selected_records"] = sum(
        row["selected_records"] for row in results.values()
    )
    payload["total_selected_training_tokens"] = sum(
        row["selected_training_tokens"] for row in results.values()
    )
    payload["passed"] = all(row["passed"] for row in results.values())
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, args.output)


if __name__ == "__main__":
    main()
