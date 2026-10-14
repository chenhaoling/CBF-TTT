"""Merge and verify two-GPU dynamic-record resource-pilot shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_dynamic_record_forgetting import PROTOCOL, summarize
from tasks.train_cbf_event_writer import write_json


def run(args):
    root = Path(args.root)
    shards = [root / name for name in args.shards.split(",") if name]
    if len(shards) != 2:
        raise ValueError("registered resource pilot requires exactly two GPU shards")
    rows, manifests, summaries, seeds = [], [], [], set()
    data_hash = None
    for shard in shards:
        manifest = json.loads((shard / "manifest.json").read_text())
        summary = json.loads((shard / "summary.json").read_text())
        if manifest.get("protocol") != PROTOCOL or summary.get("protocol") != PROTOCOL:
            raise ValueError("unexpected resource-pilot shard protocol")
        if data_hash is None:
            data_hash = manifest["data_manifest_sha256"]
        if manifest["data_manifest_sha256"] != data_hash:
            raise ValueError("resource shards used different data")
        for seed in manifest["projection_seeds"]:
            if seed in seeds:
                raise ValueError("projection seed appears in multiple shards")
            seeds.add(seed)
            name = f"labels_seed{seed}.jsonl"
            path = shard / name
            if sha(path) != manifest["raw_rows_sha256"].get(name):
                raise ValueError("resource label hash mismatch")
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        manifests.append(manifest)
        summaries.append(summary)
    if seeds != {2301, 2302, 2303}:
        raise ValueError("resource pilot must cover all three projection seeds")
    wall_seconds = max(summary["wall_seconds"] for summary in summaries)
    merged = summarize(
        rows,
        wall_seconds,
        max(summary["extraction_peak_allocated_gib"] for summary in summaries),
        max(summary["extraction_peak_reserved_gib"] for summary in summaries),
    )
    merged["two_gpu_shards"] = [str(path) for path in shards]
    merged["shard_wall_seconds"] = [summary["wall_seconds"] for summary in summaries]
    merged["all_label_rows_hashed"] = True
    write_json(root / "summary.json", merged)
    write_json(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "data_manifest_sha256": data_hash,
            "projection_seeds": sorted(seeds),
            "shards": [
                {
                    "path": path.name,
                    "manifest_sha256": sha(path / "manifest.json"),
                    "summary_sha256": sha(path / "summary.json"),
                }
                for path in shards
            ],
            "all_label_rows_hashed": True,
        },
    )
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--shards", default="shard_gpu0,shard_gpu1")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
