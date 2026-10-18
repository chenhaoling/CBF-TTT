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
    host_path = root / "host.csv"
    exit_code = int((root / "exit_code.txt").read_text().strip())
    log = log_path.read_text(errors="replace")
    with monitor_path.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("empty GPU monitor")
    peaks = {}
    minimum_gpu_free_fraction = {}
    for row in rows:
        index = row["index"]
        used, total = float(row["memory_used_mib"]), float(row["memory_total_mib"])
        peaks[index] = max(peaks.get(index, 0.0), used)
        free_fraction = (total - used) / total
        minimum_gpu_free_fraction[index] = min(
            minimum_gpu_free_fraction.get(index, 1.0), free_fraction
        )
    host_resources = None
    if host_path.exists():
        with host_path.open() as source:
            host_rows = list(csv.DictReader(source))
        if not host_rows:
            raise ValueError("empty host monitor")
        host_resources = {
            "peak_memory_used_mib": max(float(row["memory_used_mib"]) for row in host_rows),
            "minimum_memory_available_mib": min(
                float(row["memory_available_mib"]) for row in host_rows
            ),
            "peak_swap_used_mib": max(float(row["swap_used_mib"]) for row in host_rows),
        }
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
    optimizer_step_matches = numeric_matches(
        r"\[[0-9:]+<[^\]]*,\s*([0-9.]+)s/it,\s*loss:", log
    )
    optimizer_step_seconds = min(optimizer_step_matches) if optimizer_step_matches else None
    activation_host_gate = (
        host_resources is not None
        and host_resources["minimum_memory_available_mib"] >= 8192
        and host_resources["peak_swap_used_mib"] <= 3584
    )
    result = {
        "protocol": metadata.get("protocol", "in_place_ttt_baseline_resource_v1"),
        "upstream_commit": metadata["upstream_commit"],
        "arm": metadata["arm"],
        "resource_profile": metadata.get("resource_profile", "native"),
        "pytorch_cuda_alloc_conf": metadata.get("pytorch_cuda_alloc_conf"),
        "sequence_length": metadata["sequence_length"],
        "global_batch_size": metadata["global_batch_size"],
        "nominal_tokens": metadata["sequence_length"] * metadata["global_batch_size"],
        "exit_code": exit_code,
        "completed": exit_code == 0 and bool(losses) and bool(grad_norms),
        "last_loss": losses[-1] if losses else None,
        "last_gradient_norm": grad_norms[-1] if exit_code == 0 and losses and grad_norms else None,
        "wall_seconds": wall_seconds,
        "optimizer_step_seconds": optimizer_step_seconds,
        "steady_tokens_per_second": (
            metadata["sequence_length"] * metadata["global_batch_size"]
            / optimizer_step_seconds if optimizer_step_seconds else None
        ),
        "tokens_per_second": (
            metadata["sequence_length"] * metadata["global_batch_size"] / wall_seconds
            if exit_code == 0 and losses and wall_seconds else None
        ),
        "peak_gpu_memory_mib": peaks,
        "minimum_gpu_free_fraction": minimum_gpu_free_fraction,
        "checkpoint_bytes": sum(
            path.stat().st_size for path in (root / "checkpoint").rglob("*") if path.is_file()
        ),
        "host_resources": host_resources,
        "cpu_offload_host_gate_passed": (
            activation_host_gate
            if metadata.get("resource_profile") in {"cpu_offload", "fsdp1_cpu_offload"}
            else None
        ),
        "activation_offload_gate_passed": (
            exit_code == 0 and bool(losses) and bool(grad_norms)
            and activation_host_gate
            and bool(minimum_gpu_free_fraction)
            and all(value >= 0.05 for value in minimum_gpu_free_fraction.values())
            if metadata.get("resource_profile") == "activation_offload"
            else None
        ),
        "dependencies": {
            "plan_sha256": metadata["plan_sha256"],
            "config_sha256": metadata["config_sha256"],
            "data_sha256": metadata["data_sha256"],
            "model_config_sha256": metadata["model_config_sha256"],
            "train_log_sha256": sha256(log_path),
            "gpu_monitor_sha256": sha256(monitor_path),
            "host_monitor_sha256": sha256(host_path) if host_path.exists() else None,
            "addendum_sha256": metadata.get("addendum_sha256"),
            "runner_sha256": metadata.get("runner_sha256"),
            "summarizer_sha256": metadata.get("summarizer_sha256"),
            "activation_offloading_sha256": metadata.get("activation_offloading_sha256"),
            "preflight_gpu_sha256": metadata.get("preflight_gpu_sha256"),
            "preflight_processes_sha256": metadata.get("preflight_processes_sha256"),
            "preflight_host_sha256": metadata.get("preflight_host_sha256"),
            "preflight_conflicts_sha256": metadata.get("preflight_conflicts_sha256"),
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

