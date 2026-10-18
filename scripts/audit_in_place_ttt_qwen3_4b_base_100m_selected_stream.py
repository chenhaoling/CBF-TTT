#!/usr/bin/env python3
"""Independently audit the corrected Qwen3-4B-Base selected stream."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import platform
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_selected_stream_audit_v1"
REPLAY_PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_selected_stream_replay_v1"
STAGES = {"stage1_32k": (32_768, 27), "stage2_128k": (131_072, 5)}
SEED, BUFFER, WORLD, BATCH, WORKERS, PREFETCH, PREFIX = 42, 10_000, 2, 32, 2, 2, 15_000
FORBIDDEN = {"content", "content_split", "text", "input_ids", "token_ids"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def keep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def has_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN or has_forbidden_key(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(has_forbidden_key(child) for child in value)
    return False


def build_prefix(source: Path, target: Path, stage: str, seq_len: int) -> str:
    digest = hashlib.sha256()
    seen = 0
    with source.open(encoding="utf-8") as reader, target.open("x", encoding="utf-8") as writer:
        for expected, line in enumerate(itertools.islice(reader, PREFIX)):
            row = json.loads(line)
            if (row.get("stage") != stage or row.get("stage_record_index") != expected
                    or row.get("training_token_count") != seq_len
                    or row.get("source") not in {"fineweb", "longcrawl"}
                    or len(row.get("training_token_sha256", "")) != 64):
                raise ValueError(f"invalid provenance row {stage}:{expected}")
            compact = {name: row[name] for name in
                       ("stage_record_index", "source", "training_token_sha256")}
            encoded = json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n"
            writer.write(encoded)
            digest.update(encoded.encode())
            seen += 1
    if seen != PREFIX:
        raise ValueError(f"short provenance prefix for {stage}")
    return digest.hexdigest()


def schedule(path: Path, rank: int, steps: int) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from torchdata.stateful_dataloader import StatefulDataLoader

    stream = load_dataset("json", data_files=[str(path)], split="train", streaming=True)
    stream = stream.shuffle(seed=SEED, buffer_size=BUFFER)
    stream = split_dataset_by_node(stream, rank=rank, world_size=WORLD)
    if stream.n_shards != 1:
        raise ValueError(f"expected one JSON shard, observed {stream.n_shards}")
    loader = StatefulDataLoader(
        stream, batch_size=BATCH, num_workers=WORKERS, drop_last=True,
        pin_memory=False, prefetch_factor=PREFETCH, collate_fn=keep_rows,
    )
    iterator = iter(loader)
    rows: list[dict[str, Any]] = []
    try:
        for _ in range(steps):
            rows += next(iterator)
    finally:
        del iterator, loader, stream
        gc.collect()
    return rows


def audit(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    replay_path = args.replay.resolve()
    try:
        replay = json.loads(replay_path.read_text())
    except Exception as error:
        replay = {}
        errors.append(f"cannot read replay: {error}")

    checks["host_and_gpu_policy"] = (
        platform.node() == args.expected_hostname
        and os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}
    )
    checks["replay_protocol_and_pass"] = (
        replay.get("protocol") == REPLAY_PROTOCOL and replay.get("passed") is True
    )
    checks["frozen_loader_contract"] = all([
        replay.get("seed") == SEED, replay.get("shuffle_buffer_size") == BUFFER,
        replay.get("world_size") == WORLD, replay.get("per_rank_batch") == BATCH,
        replay.get("num_workers") == WORKERS, replay.get("prefetch_factor") == PREFETCH,
        replay.get("dataset_n_shards") == 1, replay.get("effective_num_workers") == 1,
        replay.get("drop_last") is True, replay.get("prefix_records") == PREFIX,
    ])
    checks["artifact_has_no_plaintext_or_token_arrays"] = not has_forbidden_key(replay)

    bindings = replay.get("bindings", {})
    expected_files = {
        "corpus_manifest_sha256": args.corpus_root / "manifest.json",
        "corpus_audit_sha256": args.corpus_audit,
        "model_identity_sha256": args.model_identity,
        "plan_sha256": args.plan,
        "replay_script_sha256": args.replay_script,
    }
    checks["primary_bindings_current"] = all(
        Path(path).is_file() and bindings.get(name) == sha256(Path(path))
        for name, path in expected_files.items()
    )
    config_paths = {
        "stage1_no_ttt": args.stage1_no_ttt_config,
        "stage1_ttt": args.stage1_ttt_config,
        "stage2_no_ttt": args.stage2_no_ttt_config,
        "stage2_ttt": args.stage2_ttt_config,
    }
    checks["config_bindings_current"] = all(
        bindings.get("config_sha256", {}).get(name) == sha256(path.resolve())
        for name, path in config_paths.items()
    )
    provenance_paths = {
        stage: args.corpus_root.resolve() / f"{stage}.provenance.jsonl" for stage in STAGES
    }
    checks["provenance_bindings_current"] = all(
        bindings.get("provenance_sha256", {}).get(stage) == sha256(path)
        for stage, path in provenance_paths.items()
    )

    independent: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="selected_stream_independent_") as temporary:
            root = Path(temporary)
            for stage, (seq_len, steps) in STAGES.items():
                compact = root / f"{stage}.jsonl"
                prefix_hash = build_prefix(provenance_paths[stage], compact, stage, seq_len)
                ranks = [schedule(compact, rank, steps) for rank in range(WORLD)]
                indices = [[row["stage_record_index"] for row in rows] for rows in ranks]
                token_hashes = [[row["training_token_sha256"] for row in rows] for rows in ranks]
                sources = [[row["source"] for row in rows] for rows in ranks]
                flat_indices = list(itertools.chain.from_iterable(indices))
                flat_hashes = list(itertools.chain.from_iterable(token_hashes))
                flat_sources = list(itertools.chain.from_iterable(sources))
                expected = steps * BATCH * WORLD
                independent[stage] = {
                    "compact_prefix_sha256": prefix_hash,
                    "ordered_stage_indices_sha256": canonical(indices),
                    "ordered_training_hashes_sha256": canonical(token_hashes),
                    "ordered_sources_sha256": canonical(sources),
                    "selected_records": len(flat_indices),
                    "selected_training_tokens": len(flat_indices) * seq_len,
                    "source_records": dict(sorted(Counter(flat_sources).items())),
                    "rank_lengths_exact": all(len(rows) == steps * BATCH for rows in ranks),
                    "indices_unique": len(set(flat_indices)) == expected,
                    "hashes_unique": len(set(flat_hashes)) == expected,
                    "ranks_disjoint": set(indices[0]).isdisjoint(indices[1]),
                }
    except Exception as error:
        errors.append(f"independent replay failed: {error}")

    for stage, result in independent.items():
        saved = replay.get("stages", {}).get(stage, {})
        prefix = replay.get("prefixes", {}).get(stage, {})
        checks[f"{stage}_prefix_matches"] = (
            result["compact_prefix_sha256"] == prefix.get("compact_sha256")
        )
        checks[f"{stage}_schedule_matches"] = all([
            result["ordered_stage_indices_sha256"] == saved.get("ordered_stage_indices_sha256"),
            result["ordered_training_hashes_sha256"] == saved.get("ordered_training_hashes_sha256"),
            result["ordered_sources_sha256"] == saved.get("ordered_sources_sha256"),
            result["selected_records"] == saved.get("selected_records"),
            result["selected_training_tokens"] == saved.get("selected_training_tokens"),
            result["source_records"] == saved.get("source_records"),
            result["rank_lengths_exact"], result["indices_unique"],
            result["hashes_unique"], result["ranks_disjoint"],
        ])
    checks["all_stages_replayed"] = set(independent) == set(STAGES)

    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "passed": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "independent_stages": independent,
        "bindings": {
            "replay_sha256": sha256(replay_path) if replay_path.is_file() else None,
            "auditor_sha256": sha256(Path(__file__)),
            "corpus_manifest_sha256": sha256(args.corpus_root / "manifest.json"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-script", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage1-no-ttt-config", type=Path, required=True)
    parser.add_argument("--stage1-ttt-config", type=Path, required=True)
    parser.add_argument("--stage2-no-ttt-config", type=Path, required=True)
    parser.add_argument("--stage2-ttt-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    result = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
