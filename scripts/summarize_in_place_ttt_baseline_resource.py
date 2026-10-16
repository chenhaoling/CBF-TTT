"""Summarize one clean-upstream baseline resource probe without reading efficacy data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
PLAN = "IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_matches(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def summarize(root: Path, output: Path) -> dict:
    metadata = json.loads((root / "run_metadata.json").read_text())
    log_path, monitor_path = root / "train.log", root / "gpu.csv"
    exit_code = int((root / "exit_code.txt").read_text().strip())
    log = log_path.read_text(errors="replace")
    with monitor_path.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("empty GPU monitor")
    peaks = {}
    for row in rows:
        index = row["index"]
        peaks[index] = max(peaks.get(index, 0.0), float(row["memory_used_mib"]))
    losses = numeric_matches(r"['\"]?loss['\"]?\s*[:=]\s*([0-9.eE+-]+)", log)
    grad_norms = numeric_matches(r"['\"]?grad(?:ient)?_norm['\"]?\s*[:=]\s*([0-9.eE+-]+)", log)
    # GNU time emits h:mm:ss or m:ss. Parse directly when present.
    wall_seconds = None
    match = re.search(
        r"Elapsed \(wall clock\) time.*\):\s*((?:[0-9]+:){1,2}[0-9.]+)", log
    )
    if match:
        parts = [float(value) for value in match.group(1).split(":")]
        wall_seconds = sum(value * (60 ** index) for index, value in enumerate(reversed(parts)))
    result = {
        "protocol": "in_place_ttt_baseline_resource_v1",
        "upstream_commit": metadata["upstream_commit"],
        "arm": metadata["arm"],
        "sequence_length": metadata["sequence_length"],
        "global_batch_size": metadata["global_batch_size"],
        "nominal_tokens": metadata["sequence_length"] * metadata["global_batch_size"],
        "exit_code": exit_code,
        "completed": exit_code == 0 and bool(losses),
        "last_loss": losses[-1] if losses else None,
        "last_gradient_norm": grad_norms[-1] if grad_norms else None,
        "wall_seconds": wall_seconds,
        "tokens_per_second": (
            metadata["sequence_length"] * metadata["global_batch_size"] / wall_seconds
            if exit_code == 0 and losses and wall_seconds else None
        ),
        "peak_gpu_memory_mib": peaks,
        "dependencies": {
            "plan_sha256": metadata["plan_sha256"],
            "config_sha256": metadata["config_sha256"],
            "data_sha256": metadata["data_sha256"],
            "model_config_sha256": metadata["model_config_sha256"],
            "train_log_sha256": sha256(log_path),
            "gpu_monitor_sha256": sha256(monitor_path),
        },
    }
    if result["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("unexpected upstream commit")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()

