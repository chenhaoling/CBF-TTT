"""Create the frozen publisher-file inventory for the Level-B public corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_corpus_inventory_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md"
TOKENIZER_ASSETS = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "config.json",
)
SOURCES = {
    "fineweb": {
        "repository": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "glob": "sample/100BT/*.parquet",
        "ordering": "lexicographic_repository_path",
    },
    "longcrawl": {
        "repository": "manifestai/longcrawl64",
        "revision": "8e7af027525b3597852746ee11e4baa0e98492d7",
        "glob": "train/*.parquet",
        "ordering": "numeric_shard_index_0_through_255",
    },
}
LONGCRAWL_PATTERN = re.compile(r"^train/(\d+)-of-256\.parquet$")
EXPECTED_COUNTS = {"fineweb": 140, "longcrawl": 256}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return value / divisor


def tokenizer_assets(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in TOKENIZER_ASSETS:
        asset = path / name
        if asset.is_file():
            result[name] = {"sha256": sha256_file(asset), "size_bytes": asset.stat().st_size}
    for required in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if required not in result:
            raise FileNotFoundError(path / required)
    return result


def _lfs_value(lfs: Any, name: str) -> Any:
    return lfs.get(name) if isinstance(lfs, dict) else getattr(lfs, name, None)


def normalize_file(repository: str, revision: str, path: str,
                   info: dict[str, Any]) -> dict[str, Any]:
    prefix = f"datasets/{repository}@{revision}/"
    repository_path = path[len(prefix):] if path.startswith(prefix) else path
    lfs = info.get("lfs")
    size = info.get("size")
    lfs_size = _lfs_value(lfs, "size")
    lfs_sha = _lfs_value(lfs, "sha256")
    if (not repository_path.endswith(".parquet") or not isinstance(size, int) or size <= 0
            or lfs_size != size or not isinstance(lfs_sha, str) or len(lfs_sha) != 64
            or not isinstance(info.get("blob_id"), str) or not info["blob_id"]):
        raise ValueError(f"invalid publisher metadata for {path}")
    return {
        "repository_path": repository_path,
        "size_bytes": size,
        "lfs_sha256": lfs_sha,
        "blob_id": info["blob_id"],
    }


def order_files(source: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = [record["repository_path"] for record in files]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{source} inventory contains duplicate paths")
    if source == "fineweb":
        ordered = sorted(files, key=lambda record: record["repository_path"])
        if not ordered or any(not record["repository_path"].startswith("sample/100BT/")
                              for record in ordered):
            raise ValueError("FineWeb inventory does not match sample/100BT")
        return ordered
    indexed = []
    for record in files:
        match = LONGCRAWL_PATTERN.fullmatch(record["repository_path"])
        if match is None:
            raise ValueError(f"invalid LongCrawl publisher path: {record['repository_path']}")
        indexed.append((int(match.group(1)), record))
    indices = [index for index, _ in indexed]
    if len(indices) != 256 or set(indices) != set(range(256)):
        raise ValueError("LongCrawl inventory must contain each shard index 0 through 255")
    return [record for _, record in sorted(indexed)]


def collect_source(filesystem: Any, name: str, spec: dict[str, str]) -> dict[str, Any]:
    pattern = (f"hf://datasets/{spec['repository']}@{spec['revision']}/"
               f"{spec['glob']}")
    details = filesystem.glob(pattern, detail=True)
    if not isinstance(details, dict):
        details = {path: filesystem.info(path) for path in details}
    files = [normalize_file(spec["repository"], spec["revision"], path, info)
             for path, info in details.items()]
    files = order_files(name, files)
    if len(files) != EXPECTED_COUNTS[name]:
        raise ValueError(f"{name} inventory has {len(files)} files; expected {EXPECTED_COUNTS[name]}")
    files_hash = hashlib.sha256(
        ("\n".join(canonical_json(record) for record in files) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        **spec,
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(record["size_bytes"] for record in files),
        "ordered_files_sha256": files_hash,
    }


def build_inventory(tokenizer_path: Path, plan_path: Path) -> dict[str, Any]:
    from huggingface_hub import HfFileSystem

    started = time.perf_counter()
    filesystem = HfFileSystem()
    sources = {name: collect_source(filesystem, name, spec)
               for name, spec in SOURCES.items()}
    core = {
        "protocol": PROTOCOL,
        "sources": sources,
        "bindings": {
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "inventory_code": {"path": str(Path(__file__).resolve()),
                               "sha256": sha256_file(Path(__file__).resolve())},
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_assets": tokenizer_assets(tokenizer_path),
        },
    }
    core["inventory_core_sha256"] = hashlib.sha256(
        canonical_json(core).encode("utf-8")
    ).hexdigest()
    core["resources"] = {
        "wall_seconds": time.perf_counter() - started,
        "peak_host_rss_mib": peak_rss_mib(),
    }
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite inventory output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    if not tokenizer_path.is_dir() or not plan_path.is_file():
        raise FileNotFoundError("tokenizer directory or plan is missing")
    inventory = build_inventory(tokenizer_path, plan_path)
    partial.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({
        "output": str(output),
        "fineweb_files": inventory["sources"]["fineweb"]["file_count"],
        "longcrawl_files": inventory["sources"]["longcrawl"]["file_count"],
        **inventory["resources"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
