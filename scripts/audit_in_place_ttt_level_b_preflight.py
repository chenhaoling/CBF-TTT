"""Independently audit a matched Level-B 32K or 128K one-step preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_level_b_preflight_audit_v1"
ARMS = ("no_ttt", "ttt")
STAGE_SPECS = {
    "stage1": {"seq_len": 32768, "one_step_tokens": 2_097_152},
    "stage2": {"seq_len": 131072, "one_step_tokens": 8_388_608},
}
METRIC_PATTERN = re.compile(
    r"loss:\s*([-+0-9.eE]+),\s*grad_norm:\s*([-+0-9.eE]+),\s*lr:\s*([-+0-9.eE]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def parse_metrics(log: str) -> dict[str, float] | None:
    matches = METRIC_PATTERN.findall(log.replace("\r", "\n"))
    if not matches:
        return None
    loss, grad_norm, lr = (float(value) for value in matches[-1])
    return {"loss": loss, "grad_norm": grad_norm, "lr": lr}


def parse_gpu_trace(path: Path, selected: list[int]) -> dict[str, Any]:
    peaks: dict[int, float] = {}
    totals: dict[int, float] = {}
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            if index not in selected:
                continue
            used, total = float(row["memory_used_mib"]), float(row["memory_total_mib"])
            if index in totals and totals[index] != total:
                raise ValueError(f"GPU {index} total memory changed")
            totals[index] = total
            peaks[index] = max(peaks.get(index, 0.0), used)
            rows += 1
    complete = set(peaks) == set(selected) and rows >= len(selected)
    fractions = {
        str(index): (totals[index] - peaks[index]) / totals[index]
        for index in peaks
    }
    return {
        "complete": complete,
        "rows": rows,
        "peak_memory_mib": {str(key): value for key, value in sorted(peaks.items())},
        "memory_total_mib": {str(key): value for key, value in sorted(totals.items())},
        "free_fraction_at_peak": fractions,
        "minimum_free_fraction": min(fractions.values()) if fractions else None,
    }


def audit(root: Path, stage: str) -> dict[str, Any]:
    errors: list[str] = []
    spec = STAGE_SPECS[stage]
    metadata_path = root / "run_metadata.json"
    config_audit_path = root / "config_audit.json"
    if not metadata_path.is_file() or not config_audit_path.is_file():
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["run metadata or config audit is missing"]}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_audit = json.loads(config_audit_path.read_text(encoding="utf-8"))
    selected = metadata.get("selected_gpus")
    if (metadata.get("protocol") != "in_place_ttt_level_b_preflight_v1"
            or metadata.get("stage") != stage
            or not isinstance(selected, list) or len(selected) != 8
            or len(set(selected)) != 8 or not all(isinstance(item, int) for item in selected)):
        errors.append("run metadata protocol/stage/GPU selection is invalid")
        selected = selected if isinstance(selected, list) else []
    gpu_inventory = metadata.get("gpu_inventory", [])
    if (len(gpu_inventory) != 8
            or any(item.get("index") not in selected
                   or item.get("memory_total_mib", 0) < 80_000
                   or item.get("memory_used_mib", math.inf) > 512
                   or item.get("compute_process_count") != 0
                   for item in gpu_inventory)):
        errors.append("preflight did not bind eight idle >=80GB GPUs")
    if metadata.get("disk_free_bytes", 0) < 400 * 1024**3:
        errors.append("preflight disk free space was below 400 GiB")
    if metadata.get("upstream_commit") != "be2324829b0e91c8fd10a74d4b43714fde6676e1" \
            or metadata.get("upstream_clean") is not True:
        errors.append("official checkout binding is invalid")
    if not config_audit.get("passed") or config_audit.get("corpus") is None:
        errors.append("formal config/corpus audit did not pass")
    if metadata.get("config_audit_sha256") != sha256(config_audit_path):
        errors.append("config audit hash binding is wrong")
    if metadata.get("seq_len") != spec["seq_len"] \
            or metadata.get("one_step_tokens") != spec["one_step_tokens"]:
        errors.append("stage length or one-step token budget is wrong")

    arm_results: dict[str, Any] = {}
    for arm in ARMS:
        arm_root = root / arm
        log_path, exit_path = arm_root / "train.log", arm_root / "exit_code.txt"
        trace_path, arm_metadata_path = arm_root / "gpu.csv", arm_root / "metadata.json"
        if not all(path.is_file() for path in (log_path, exit_path, trace_path, arm_metadata_path)):
            errors.append(f"{arm}: required artifacts are missing")
            continue
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        log = log_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_metrics(log)
        trace = parse_gpu_trace(trace_path, selected)
        arm_metadata = json.loads(arm_metadata_path.read_text(encoding="utf-8"))
        finite = metrics is not None and all(math.isfinite(value) for value in metrics.values())
        arm_ok = (
            exit_code == 0 and finite and trace["complete"]
            and trace["minimum_free_fraction"] is not None
            and trace["minimum_free_fraction"] >= 0.05
            and "out of memory" not in log.lower()
            and arm_metadata.get("arm") == arm
            and arm_metadata.get("stage") == stage
            and arm_metadata.get("data_sha256") == metadata.get("data_sha256")
            and arm_metadata.get("config_sha256")
            == metadata.get("config_sha256", {}).get(arm)
        )
        if not arm_ok:
            errors.append(f"{arm}: exit, finite metric, input binding, or 5% headroom gate failed")
        arm_results[arm] = {
            "passed": arm_ok,
            "exit_code": exit_code,
            "metrics": metrics,
            "gpu": trace,
            "log_sha256": sha256(log_path),
            "metadata_sha256": sha256(arm_metadata_path),
        }
    return {
        "protocol": PROTOCOL,
        "stage": stage,
        "passed": not errors and set(arm_results) == set(ARMS),
        "errors": errors,
        "arms": arm_results,
        "bindings": {
            "run_metadata_sha256": sha256(metadata_path),
            "config_audit_sha256": sha256(config_audit_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_SPECS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.stage)
    output = args.output.resolve() if args.output else args.root.resolve() / "independent_audit.json"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
