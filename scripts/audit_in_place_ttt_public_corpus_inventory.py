"""Independently re-resolve and audit a Level-B public-corpus inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_corpus_inventory_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md"
DEFAULT_INVENTORY_CODE = ROOT / "scripts/inventory_in_place_ttt_public_corpus.py"
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "config.json")
SOURCE_SPECS = {
    "fineweb": {
        "repository": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "glob": "sample/100BT/*.parquet",
        "ordering": "lexicographic_repository_path",
        "count": 140,
    },
    "longcrawl": {
        "repository": "manifestai/longcrawl64",
        "revision": "8e7af027525b3597852746ee11e4baa0e98492d7",
        "glob": "train/*.parquet",
        "ordering": "numeric_shard_index_0_through_255",
        "count": 256,
    },
}
LONGCRAWL_PATTERN = re.compile(r"^train/(\d+)-of-256\.parquet$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def lfs_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def normalize(repository: str, revision: str, path: str,
              info: dict[str, Any]) -> dict[str, Any]:
    prefix = f"datasets/{repository}@{revision}/"
    relative = path[len(prefix):] if path.startswith(prefix) else path
    lfs = info.get("lfs")
    return {
        "repository_path": relative,
        "size_bytes": info.get("size"),
        "lfs_sha256": lfs_value(lfs, "sha256"),
        "blob_id": info.get("blob_id"),
    }


def order(name: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if name == "fineweb":
        return sorted(files, key=lambda record: record["repository_path"])
    indexed = []
    for record in files:
        match = LONGCRAWL_PATTERN.fullmatch(record["repository_path"])
        if match is None:
            return []
        indexed.append((int(match.group(1)), record))
    if {index for index, _ in indexed} != set(range(256)) or len(indexed) != 256:
        return []
    return [record for _, record in sorted(indexed)]


def resolve(filesystem: Any, name: str) -> dict[str, Any]:
    spec = SOURCE_SPECS[name]
    repository, revision, glob = spec["repository"], spec["revision"], spec["glob"]
    pattern = f"hf://datasets/{repository}@{revision}/{glob}"
    details = filesystem.glob(pattern, detail=True)
    if not isinstance(details, dict):
        details = {path: filesystem.info(path) for path in details}
    files = order(name, [normalize(repository, revision, path, info)
                         for path, info in details.items()])
    digest = hashlib.sha256(
        ("\n".join(canonical(record) for record in files) + "\n").encode("utf-8")
    ).hexdigest()
    return {"files": files, "file_count": len(files),
            "total_size_bytes": sum(record.get("size_bytes") or 0 for record in files),
            "ordered_files_sha256": digest}


def audit(inventory_path: Path, tokenizer: Path, plan: Path,
          inventory_code: Path) -> dict[str, Any]:
    from huggingface_hub import HfFileSystem

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {"protocol": inventory.get("protocol") == PROTOCOL}
    bindings = inventory.get("bindings", {})
    checks["plan_binding"] = bindings.get("plan", {}).get("sha256") == sha(plan)
    checks["inventory_code_binding"] = (
        bindings.get("inventory_code", {}).get("sha256") == sha(inventory_code)
    )
    observed_assets = {}
    for name in ASSETS:
        path = tokenizer / name
        if path.is_file():
            observed_assets[name] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    checks["tokenizer_assets"] = (
        all(name in observed_assets for name in ("tokenizer.json", "tokenizer_config.json", "config.json"))
        and observed_assets == bindings.get("tokenizer_assets")
    )
    filesystem = HfFileSystem()
    resolved = {name: resolve(filesystem, name) for name in SOURCE_SPECS}
    source_checks = {}
    for name, spec in SOURCE_SPECS.items():
        repository, revision, glob = spec["repository"], spec["revision"], spec["glob"]
        saved = inventory.get("sources", {}).get(name, {})
        source_checks[name] = (
            saved.get("repository") == repository
            and saved.get("revision") == revision
            and saved.get("glob") == glob
            and saved.get("ordering") == spec["ordering"]
            and saved.get("files") == resolved[name]["files"]
            and saved.get("file_count") == resolved[name]["file_count"] == spec["count"]
            and saved.get("total_size_bytes") == resolved[name]["total_size_bytes"]
            and saved.get("ordered_files_sha256") == resolved[name]["ordered_files_sha256"]
        )
    checks["source_files"] = all(source_checks.values())
    core = {key: inventory[key] for key in ("protocol", "sources", "bindings")}
    checks["inventory_core_hash"] = inventory.get("inventory_core_sha256") == hashlib.sha256(
        canonical(core).encode("utf-8")
    ).hexdigest()
    resources = inventory.get("resources", {})
    checks["resources"] = all(
        isinstance(resources.get(key), (int, float))
        and math.isfinite(resources[key]) and resources[key] > 0
        for key in ("wall_seconds", "peak_host_rss_mib")
    )
    return {
        "protocol": PROTOCOL + "_independent_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "source_checks": source_checks,
        "observed": {name: {key: value for key, value in result.items() if key != "files"}
                     for name, result in resolved.items()},
        "bindings": {
            "inventory_sha256": sha(inventory_path),
            "plan_sha256": sha(plan),
            "inventory_code_sha256": sha(inventory_code),
            "auditor_sha256": sha(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--inventory-code", default=str(DEFAULT_INVENTORY_CODE))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    result = audit(
        Path(args.inventory).expanduser().resolve(),
        Path(args.tokenizer).expanduser().resolve(),
        Path(args.plan).expanduser().resolve(),
        Path(args.inventory_code).expanduser().resolve(),
    )
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "checks": result["checks"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
