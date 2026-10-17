"""Independently audit one formal Level-B training stage and arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.audit_in_place_ttt_level_b_preflight import parse_gpu_trace


PROTOCOL = "in_place_ttt_level_b_training_audit_v1"
SPECS = {
    "stage1": {"steps": 9537, "seq_len": 32768, "train_size": 20_000_538_624},
    "stage2": {"steps": 1788, "seq_len": 131072, "train_size": 14_998_831_104},
}
ARMS = ("no_ttt", "ttt")
METRIC_PATTERN = re.compile(
    r"loss:\s*([-+0-9.eE]+),\s*grad_norm:\s*([-+0-9.eE]+),\s*lr:\s*([-+0-9.eE]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def expected_checkpoint_steps(final_step: int) -> list[int]:
    return list(range(500, final_step + 1, 500)) + (
        [] if final_step % 500 == 0 else [final_step]
    )


def all_metrics(log: str) -> list[dict[str, float]]:
    return [
        {"loss": float(loss), "grad_norm": float(grad), "lr": float(lr)}
        for loss, grad, lr in METRIC_PATTERN.findall(log.replace("\r", "\n"))
    ]


def host_resources(path: Path) -> dict[str, Any]:
    rows = 0
    minimum_available = math.inf
    maximum_used = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            minimum_available = min(minimum_available, int(row["memory_available_bytes"]))
            maximum_used = max(maximum_used, int(row["memory_used_bytes"]))
    return {
        "rows": rows,
        "minimum_available_bytes": None if rows == 0 else minimum_available,
        "maximum_used_bytes": None if rows == 0 else maximum_used,
    }


def audit(root: Path, stage: str, arm: str, addendum: Path) -> dict[str, Any]:
    errors: list[str] = []
    spec = SPECS[stage]
    required = {
        name: root / name for name in (
            "run_metadata.json", "config_audit.json", "train.log", "exit_code.txt",
            "gpu.csv", "host.csv",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required artifacts: {missing}"]}
    metadata = json.loads(required["run_metadata.json"].read_text(encoding="utf-8"))
    config_audit = json.loads(required["config_audit.json"].read_text(encoding="utf-8"))
    exit_code = int(required["exit_code.txt"].read_text(encoding="utf-8").strip())
    log = required["train.log"].read_text(encoding="utf-8", errors="replace")
    metrics = all_metrics(log)
    selected = metadata.get("selected_gpus", [])

    if (metadata.get("protocol") != "in_place_ttt_level_b_training_v1"
            or metadata.get("stage") != stage or metadata.get("arm") != arm):
        errors.append("run metadata protocol/stage/arm mismatch")
    if (metadata.get("expected_steps") != spec["steps"]
            or metadata.get("seq_len") != spec["seq_len"]
            or metadata.get("train_size") != spec["train_size"]):
        errors.append("registered stage counts differ")
    if (not isinstance(selected, list) or len(selected) != 8
            or len(set(selected)) != 8 or not all(isinstance(value, int) for value in selected)):
        errors.append("selected GPU binding is invalid")
        selected = []
    if metadata.get("upstream_commit") != "be2324829b0e91c8fd10a74d4b43714fde6676e1" \
            or metadata.get("upstream_clean") is not True:
        errors.append("official checkout binding is invalid")
    if not config_audit.get("passed") or config_audit.get("corpus") is None \
            or metadata.get("config_audit_sha256") != sha256(required["config_audit.json"]):
        errors.append("config/corpus audit binding is invalid")
    if metadata.get("training_addendum_sha256") != sha256(addendum):
        errors.append("training addendum binding is invalid")
    preflight_path = Path(metadata.get("preflight_audit_path", ""))
    if not preflight_path.is_file() \
            or metadata.get("preflight_audit_sha256") != sha256(preflight_path):
        errors.append("preflight audit binding is invalid")
    else:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (not preflight.get("passed") or preflight.get("stage") != stage
                or not preflight.get("arms", {}).get(arm, {}).get("passed")):
            errors.append("matched preflight did not release this stage/arm")
    preflight_metadata_path = Path(metadata.get("preflight_run_metadata_path", ""))
    if not preflight_metadata_path.is_file() \
            or metadata.get("preflight_run_metadata_sha256") != sha256(preflight_metadata_path):
        errors.append("preflight run-metadata binding is invalid")
    else:
        preflight_metadata = json.loads(preflight_metadata_path.read_text(encoding="utf-8"))
        if (preflight_metadata.get("hostname") != metadata.get("hostname")
                or preflight_metadata.get("selected_gpus") != selected
                or preflight_metadata.get("stage") != stage):
            errors.append("training host/GPU selection differs from preflight")
    if exit_code != 0:
        errors.append(f"training exit code is {exit_code}")
    if not metrics or any(not all(math.isfinite(value) for value in row.values()) for row in metrics):
        errors.append("training has no finite parsed metrics")
    if "out of memory" in log.lower():
        errors.append("training log contains out-of-memory")

    trace = parse_gpu_trace(required["gpu.csv"], selected)
    if (not trace["complete"] or trace["minimum_free_fraction"] is None
            or trace["minimum_free_fraction"] < 0.05):
        errors.append("GPU trace is incomplete or violates 5% headroom")
    host = host_resources(required["host.csv"])
    if host["rows"] == 0:
        errors.append("host resource trace is empty")

    training_output = root / "training_output"
    observed_dirs: dict[int, Path] = {}
    for path in training_output.rglob("global_step_*") if training_output.is_dir() else []:
        if path.is_dir() and re.fullmatch(r"global_step_(\d+)", path.name):
            observed_dirs[int(path.name.rsplit("_", 1)[1])] = path
    expected_steps = expected_checkpoint_steps(spec["steps"])
    if sorted(observed_dirs) != expected_steps:
        errors.append(f"checkpoint steps differ: {sorted(observed_dirs)}")
    checkpoint_records = {}
    for step, path in sorted(observed_dirs.items()):
        metadata_file = path / ".metadata"
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        valid = metadata_file.is_file() and size >= 40_000_000_000
        if not valid:
            errors.append(f"checkpoint {step} is missing metadata or unexpectedly small")
        checkpoint_records[str(step)] = {
            "path": str(path), "size_bytes": size,
            "metadata_sha256": sha256(metadata_file) if metadata_file.is_file() else None,
        }
    final_path = observed_dirs.get(spec["steps"])
    hf_path = final_path / "hf_ckpt" if final_path else None
    hf_valid = bool(
        hf_path and (hf_path / "config.json").is_file()
        and ((hf_path / "model.safetensors.index.json").is_file()
             or (hf_path / "model.safetensors").is_file())
    )
    if not hf_valid:
        errors.append("final HF checkpoint is missing")
    else:
        config = json.loads((hf_path / "config.json").read_text(encoding="utf-8"))
        if bool(config.get("ttt_mode", False)) != (arm == "ttt"):
            errors.append("final HF ttt_mode differs from arm")
        if config.get("max_position_embeddings") != spec["seq_len"]:
            errors.append("final HF max_position_embeddings differs from stage")

    return {
        "protocol": PROTOCOL,
        "stage": stage,
        "arm": arm,
        "passed": not errors,
        "errors": errors,
        "metrics": {
            "parsed_count": len(metrics),
            "first": metrics[0] if metrics else None,
            "last": metrics[-1] if metrics else None,
        },
        "gpu": trace,
        "host": host,
        "checkpoints": checkpoint_records,
        "final_hf_path": str(hf_path) if hf_path else None,
        "bindings": {name + "_sha256": sha256(path) for name, path in required.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(SPECS), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--training-addendum", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root, args.stage, args.arm, args.training_addendum.resolve())
    output = args.output.resolve() if args.output else root / "independent_audit.json"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
