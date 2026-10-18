#!/usr/bin/env python3
"""Replay the corrected 100M native iterable stream without corpus plaintext."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_selected_stream_replay_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
SEED = 42
BUFFER_SIZE = 10_000
WORLD_SIZE = 2
PER_RANK_BATCH = 32
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
PREFIX_RECORDS = 15_000
STAGES = {
    "stage1_32k": {"seq_len": 32_768, "steps": 27, "config_stage": "stage1"},
    "stage2_128k": {"seq_len": 131_072, "steps": 5, "config_stage": "stage2"},
}
FORBIDDEN_OUTPUT_KEYS = {"content", "content_split", "text", "input_ids", "token_ids"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def identity_collate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep compact rows intact; a top-level function is worker-picklable."""
    return rows


def compact_prefix(source: Path, target: Path, stage: str, seq_len: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    with source.open(encoding="utf-8") as reader, target.open("x", encoding="utf-8") as writer:
        for line in itertools.islice(reader, PREFIX_RECORDS):
            row = json.loads(line)
            if (row.get("stage") != stage or row.get("stage_record_index") != rows
                    or row.get("source") not in {"fineweb", "longcrawl"}
                    or row.get("training_token_count") != seq_len
                    or not isinstance(row.get("training_token_sha256"), str)
                    or len(row["training_token_sha256"]) != 64):
                raise ValueError(f"invalid {stage} provenance row {rows}")
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
        raise ValueError(f"{stage} contains only {rows} complete provenance rows")
    return {"records": rows, "compact_sha256": digest.hexdigest()}


def replay_rank(path: Path, rank: int, steps: int) -> list[dict[str, Any]]:
    # Imports remain inside the execution path so repository unit tests do not
    # require the frozen remote training environment.
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from torchdata.stateful_dataloader import StatefulDataLoader

    dataset = load_dataset("json", data_files=[str(path)], split="train", streaming=True)
    dataset = dataset.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=WORLD_SIZE)
    loader = StatefulDataLoader(
        dataset,
        batch_size=PER_RANK_BATCH,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=False,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=identity_collate,
    )
    iterator = iter(loader)
    selected: list[dict[str, Any]] = []
    try:
        for _ in range(steps):
            selected.extend(next(iterator))
    finally:
        del iterator, loader, dataset
        gc.collect()
    return selected


def summarize_schedule(path: Path, stage: str, spec: dict[str, Any]) -> dict[str, Any]:
    first = [replay_rank(path, rank, spec["steps"]) for rank in range(WORLD_SIZE)]
    second = [replay_rank(path, rank, spec["steps"]) for rank in range(WORLD_SIZE)]
    indices = [[int(row["stage_record_index"]) for row in rows] for rows in first]
    hashes = [[str(row["training_token_sha256"]) for row in rows] for rows in first]
    sources = [[str(row["source"]) for row in rows] for rows in first]
    second_indices = [[int(row["stage_record_index"]) for row in rows] for rows in second]
    second_hashes = [[str(row["training_token_sha256"]) for row in rows] for rows in second]
    second_sources = [[str(row["source"]) for row in rows] for rows in second]
    expected_per_rank = spec["steps"] * PER_RANK_BATCH
    expected_global = expected_per_rank * WORLD_SIZE
    flat_indices = list(itertools.chain.from_iterable(indices))
    flat_hashes = list(itertools.chain.from_iterable(hashes))
    flat_sources = list(itertools.chain.from_iterable(sources))
    counts = Counter(flat_sources)
    checks = {
        "repeat_schedule_exact": (
            indices == second_indices and hashes == second_hashes and sources == second_sources
        ),
        "rank_lengths_exact": all(len(rows) == expected_per_rank for rows in first),
        "global_records_exact": len(flat_indices) == expected_global,
        "stage_indices_unique": len(set(flat_indices)) == expected_global,
        "training_hashes_unique": len(set(flat_hashes)) == expected_global,
        "rank_indices_disjoint": set(indices[0]).isdisjoint(indices[1]),
        "rank_hashes_disjoint": set(hashes[0]).isdisjoint(hashes[1]),
        "sources_valid": set(flat_sources) <= {"fineweb", "longcrawl"},
        "prefix_sufficient": PREFIX_RECORDS >= BUFFER_SIZE + expected_global,
    }
    return {
        "stage": stage,
        "sequence_length": spec["seq_len"],
        "steps": spec["steps"],
        "records_per_rank": expected_per_rank,
        "selected_records": len(flat_indices),
        "selected_training_tokens": len(flat_indices) * spec["seq_len"],
        "source_records": dict(sorted(counts.items())),
        "source_training_tokens": {
            name: count * spec["seq_len"] for name, count in sorted(counts.items())
        },
        "rank_source_records": [dict(sorted(Counter(values).items())) for values in sources],
        "ordered_stage_indices_sha256": canonical_hash(indices),
        "ordered_training_hashes_sha256": canonical_hash(hashes),
        "ordered_sources_sha256": canonical_hash(sources),
        "checks": checks,
        "passed": all(checks.values()),
    }


def runtime_bindings() -> tuple[dict[str, str], dict[str, str]]:
    modules = {
        "veomni_dataset": "veomni.data.dataset",
        "veomni_data_loader": "veomni.data.data_loader",
        "datasets_iterable": "datasets.iterable_dataset",
        "datasets_distributed": "datasets.distributed",
        "torchdata_stateful_loader": "torchdata.stateful_dataloader.stateful_dataloader",
    }
    hashes: dict[str, str] = {}
    for name, module_name in modules.items():
        module = importlib.import_module(module_name)
        source = Path(module.__file__).resolve()
        hashes[name] = sha256_file(source)
    versions = {name: importlib.metadata.version(name) for name in ("datasets", "torchdata", "veomni")}
    versions["python"] = platform.python_version()
    return hashes, versions


def load_config_contract(paths: dict[str, tuple[Path, Path]]) -> tuple[dict[str, Any], dict[str, str]]:
    import yaml

    contracts: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    expected = {
        "stage1": {"max_seq_len": 32_768, "train_size": 56_623_104, "max_steps": 27},
        "stage2": {"max_seq_len": 131_072, "train_size": 41_943_040, "max_steps": 5},
    }
    for stage, arms in paths.items():
        loaded = [yaml.safe_load(path.read_text()) for path in arms]
        for arm, path in zip(("no_ttt", "ttt"), arms):
            hashes[f"{stage}_{arm}"] = sha256_file(path)
        data_keys = ("train_size", "dataloader_type", "datasets_type", "data_type",
                     "max_seq_len", "text_keys", "drop_last", "num_workers",
                     "prefetch_factor", "pin_memory")
        train_keys = ("seed", "global_batch_size", "micro_batch_size", "max_steps")
        contracts_for_arms = [
            {"data": {key: cfg["data"].get(key) for key in data_keys},
             "train": {key: cfg["train"].get(key) for key in train_keys}}
            for cfg in loaded
        ]
        if contracts_for_arms[0] != contracts_for_arms[1]:
            raise ValueError(f"{stage} no-TTT and TTT stream contracts differ")
        contract = contracts_for_arms[0]
        frozen = expected[stage]
        if (contract["data"]["train_size"] != frozen["train_size"]
                or contract["data"]["max_seq_len"] != frozen["max_seq_len"]
                or contract["train"]["max_steps"] != frozen["max_steps"]
                or contract["data"]["dataloader_type"] != "native"
                or contract["data"]["datasets_type"] != "iterable"
                or contract["data"]["drop_last"] is not True
                or contract["data"]["num_workers"] != NUM_WORKERS
                or contract["data"]["prefetch_factor"] != PREFETCH_FACTOR
                or contract["train"]["seed"] != SEED
                or contract["train"]["global_batch_size"] != 64):
            raise ValueError(f"{stage} config differs from frozen selected-stream contract")
        contracts[stage] = contract
    return contracts, hashes


def contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_OUTPUT_KEYS or contains_forbidden_key(child)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_key(child) for child in value)
    return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    if platform.node() != args.expected_hostname:
        raise RuntimeError("selected-stream replay is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("selected-stream replay must hide all GPUs")
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    corpus_root = args.corpus_root.resolve()
    manifest_path = corpus_root / "manifest.json"
    audit_path = args.corpus_audit.resolve()
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    identity = json.loads(args.model_identity.read_text())
    if (manifest.get("profile") != "full"
            or manifest.get("totals", {}).get("records") != 724_800
            or manifest.get("totals", {}).get("training_tokens") != 34_999_369_728
            or audit.get("passed") is not True
            or audit.get("bindings", {}).get("manifest_sha256") != sha256_file(manifest_path)
            or identity.get("repository") != "Qwen/Qwen3-4B-Base"
            or identity.get("revision") != MODEL_REVISION
            or identity.get("passed") is not True):
        raise ValueError("corrected corpus or Base model identity did not pass preflight")

    config_paths = {
        "stage1": (args.stage1_no_ttt_config.resolve(), args.stage1_ttt_config.resolve()),
        "stage2": (args.stage2_no_ttt_config.resolve(), args.stage2_ttt_config.resolve()),
    }
    contracts, config_hashes = load_config_contract(config_paths)
    runtime_hashes, versions = runtime_bindings()
    provenance_paths = {
        stage: corpus_root / f"{stage}.provenance.jsonl" for stage in STAGES
    }
    with tempfile.TemporaryDirectory(prefix="qwen3_base_selected_stream_") as temporary:
        temp = Path(temporary)
        prefixes: dict[str, Any] = {}
        stages: dict[str, Any] = {}
        for stage, spec in STAGES.items():
            compact = temp / f"{stage}.jsonl"
            prefixes[stage] = compact_prefix(
                provenance_paths[stage], compact, stage, spec["seq_len"]
            )
            stages[stage] = summarize_schedule(compact, stage, spec)

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "shuffle_buffer_size": BUFFER_SIZE,
        "world_size": WORLD_SIZE,
        "per_rank_batch": PER_RANK_BATCH,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "drop_last": True,
        "pin_memory_in_order_audit": False,
        "prefix_records": PREFIX_RECORDS,
        "config_contracts": contracts,
        "prefixes": prefixes,
        "stages": stages,
        "versions": versions,
        "bindings": {
            "replay_script_sha256": sha256_file(Path(__file__)),
            "plan_sha256": sha256_file(args.plan.resolve()),
            "corpus_manifest_sha256": sha256_file(manifest_path),
            "corpus_audit_sha256": sha256_file(audit_path),
            "model_identity_sha256": sha256_file(args.model_identity.resolve()),
            "provenance_sha256": {
                stage: sha256_file(path) for stage, path in provenance_paths.items()
            },
            "config_sha256": config_hashes,
            "runtime_source_sha256": runtime_hashes,
        },
    }
    payload["total_selected_records"] = sum(row["selected_records"] for row in stages.values())
    payload["total_selected_training_tokens"] = sum(
        row["selected_training_tokens"] for row in stages.values()
    )
    payload["passed"] = all(row["passed"] for row in stages.values())
    if contains_forbidden_key(payload):
        raise AssertionError("plaintext or token-array key entered selected-stream artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output.resolve()),
                      "passed": result["passed"],
                      "total_selected_training_tokens":
                      result["total_selected_training_tokens"]}, sort_keys=True))


if __name__ == "__main__":
    main()
