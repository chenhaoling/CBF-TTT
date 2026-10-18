"""Independently audit one hku-gpu2 activation-offload resource boundary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_level_b_configs import read_config, sha256
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_level_b_configs import read_config, sha256


PROTOCOL = "in_place_ttt_2x5090_resource_audit_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
SEQUENCE_LENGTHS = {8192, 16384, 32768, 65536, 131072}
EXPECTED_HOSTNAME = "user-Super-Server"


def _preflight_gpus(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) != 2:
        return records, [f"preflight observed {len(rows)} GPUs instead of two"]
    for expected_index, row in enumerate(rows):
        if len(row) != 4:
            errors.append(f"malformed preflight GPU row: {row}")
            continue
        index, name, total, used = [value.strip() for value in row]
        record = {
            "index": int(index), "name": name,
            "memory_total_mib": float(total), "memory_used_mib": float(used),
        }
        records.append(record)
        if (record["index"] != expected_index or "RTX 5090" not in name
                or record["memory_total_mib"] < 32_000
                or record["memory_used_mib"] > 128):
            errors.append(f"preflight GPU {expected_index} is not the registered idle 5090")
    return records, errors


def _mem_available_mib(path: Path) -> float | None:
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", path.read_text(), re.MULTILINE)
    return int(match.group(1)) / 1024 if match else None


def _file_binding(
    metadata: dict[str, Any], key: str, path: Path, errors: list[str]
) -> str:
    digest = sha256(path)
    if metadata.get(key) != digest:
        errors.append(f"{key} binding mismatch")
    return digest


def audit(
    root: Path,
    *,
    plan: Path,
    addendum: Path,
    config: Path,
    data: Path,
    model_config: Path,
    runner: Path,
    summarizer: Path,
    offloading: Path,
    data_manifest: Path | None = None,
    data_audit: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    names = (
        "run_metadata.json", "summary.json", "train.log", "exit_code.txt",
        "gpu.csv", "host.csv", "preflight_gpu.csv", "preflight_processes.csv",
        "preflight_host.txt", "preflight_conflicts.txt",
    )
    required = {name: root / name for name in names}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required artifacts: {missing}"]}

    metadata = json.loads(required["run_metadata.json"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))
    log = required["train.log"].read_text(encoding="utf-8", errors="replace")
    try:
        exit_code = int(required["exit_code.txt"].read_text().strip())
    except ValueError:
        exit_code = -1
        errors.append("exit code is malformed")

    arm = metadata.get("arm")
    if (metadata.get("protocol") != "in_place_ttt_baseline_resource_v2"
            or metadata.get("resource_profile") != "activation_offload"
            or metadata.get("activation_offload") is not True):
        errors.append("resource metadata protocol/profile mismatch")
    if metadata.get("hostname") != EXPECTED_HOSTNAME:
        errors.append("run was not executed on hku-gpu2")
    if (metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True):
        errors.append("official upstream checkout binding is invalid")
    if arm not in {"no_ttt", "ttt"}:
        errors.append("arm is invalid")
    if metadata.get("sequence_length") not in SEQUENCE_LENGTHS:
        errors.append("sequence length is outside the frozen sparse ladder")
    if metadata.get("global_batch_size") != 2:
        errors.append("resource boundary global batch must equal two")
    if metadata.get("pytorch_cuda_alloc_conf") != "expandable_segments:True":
        errors.append("expandable-segments allocator is not bound")

    expected_bindings = {
        "plan_sha256": plan,
        "addendum_sha256": addendum,
        "config_sha256": config,
        "data_sha256": data,
        "model_config_sha256": model_config,
        "runner_sha256": runner,
        "summarizer_sha256": summarizer,
        "activation_offloading_sha256": offloading,
        "preflight_gpu_sha256": required["preflight_gpu.csv"],
        "preflight_processes_sha256": required["preflight_processes.csv"],
        "preflight_host_sha256": required["preflight_host.txt"],
        "preflight_conflicts_sha256": required["preflight_conflicts.txt"],
    }
    observed_bindings = {
        key: _file_binding(metadata, key, path, errors)
        for key, path in expected_bindings.items()
    }
    seq_len = metadata.get("sequence_length")
    if (data_manifest is None) != (data_audit is None):
        errors.append("resource input manifest/audit must be supplied together")
    if seq_len in SEQUENCE_LENGTHS and seq_len > 8192 \
            and (data_manifest is None or data_audit is None):
        errors.append("audited exact-length resource input is required above 8K")
    if data_manifest is not None and data_audit is not None:
        manifest, input_audit = (
            json.loads(data_manifest.read_text()), json.loads(data_audit.read_text())
        )
        manifest_sha, audit_sha = sha256(data_manifest), sha256(data_audit)
        saved = manifest.get("outputs", {}).get(str(seq_len), {})
        artifact = saved.get("artifact", {})
        audited = input_audit.get("outputs", {}).get(str(seq_len), {})
        if (manifest.get("protocol") != "in_place_ttt_2x5090_resource_inputs_v1"
                or input_audit.get("passed") is not True
                or input_audit.get("bindings", {}).get("manifest_sha256") != manifest_sha
                or metadata.get("data_manifest_sha256") != manifest_sha
                or metadata.get("data_audit_sha256") != audit_sha
                or metadata.get("data_manifest_path") != str(data_manifest.resolve())
                or metadata.get("data_audit_path") != str(data_audit.resolve())
                or data.resolve() != (data_manifest.parent / artifact.get("path", "")).resolve()
                or artifact.get("sha256") != sha256(data)
                or saved.get("records") != 2
                or saved.get("training_tokens") != 2 * seq_len
                or audited.get("artifact_sha256") != artifact.get("sha256")
                or audited.get("records") != 2
                or audited.get("training_tokens") != 2 * seq_len):
            errors.append("resource input manifest/audit binding is invalid")
        observed_bindings["data_manifest_sha256"] = manifest_sha
        observed_bindings["data_audit_sha256"] = audit_sha
    dependencies = summary.get("dependencies", {})
    for key, digest in observed_bindings.items():
        if dependencies.get(key) != digest:
            errors.append(f"summary {key} binding mismatch")
    current_outputs = {
        "train_log_sha256": required["train.log"],
        "gpu_monitor_sha256": required["gpu.csv"],
        "host_monitor_sha256": required["host.csv"],
    }
    for key, path in current_outputs.items():
        if dependencies.get(key) != sha256(path):
            errors.append(f"summary {key} is stale")

    config_data = read_config(config)
    train = config_data.get("train", {})
    foundation = config_data.get("model", {}).get("foundation", {})
    if (train.get("data_parallel_mode") != "fsdp2"
            or train.get("enable_gradient_checkpointing") is not True
            or train.get("enable_activation_offload") is not True
            or train.get("activation_gpu_limit") != 0.0
            or train.get("enable_fsdp_offload") is not False):
        errors.append("config activation-offload/FSDP2 settings differ")
    if isinstance(arm, str) and foundation.get("ttt_mode") is not (arm == "ttt"):
        errors.append("config ttt_mode differs from arm")

    gpus, gpu_errors = _preflight_gpus(required["preflight_gpu.csv"])
    errors.extend(gpu_errors)
    if required["preflight_processes.csv"].read_text().strip():
        errors.append("preflight CUDA process list is not empty")
    if required["preflight_conflicts.txt"].read_text().strip():
        errors.append("preflight corpus-builder/auditor conflict list is not empty")
    preflight_available = _mem_available_mib(required["preflight_host.txt"])
    if preflight_available is None or preflight_available < 48 * 1024:
        errors.append("preflight host available memory is below 48 GiB")

    if exit_code != 0 or summary.get("exit_code") != 0:
        errors.append(f"training exit code is {exit_code}")
    if summary.get("completed") is not True:
        errors.append("resource boundary did not complete")
    if summary.get("activation_offload_gate_passed") is not True:
        errors.append("activation-offload resource gate did not pass")
    for key in ("last_loss", "last_gradient_norm", "optimizer_step_seconds"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{key} is absent or nonfinite")
    free = summary.get("minimum_gpu_free_fraction", {})
    if set(free) != {"0", "1"} or any(float(value) < 0.05 for value in free.values()):
        errors.append("GPU trace does not retain 5% headroom on both registered devices")
    host = summary.get("host_resources") or {}
    if (host.get("minimum_memory_available_mib", 0) < 8192
            or host.get("peak_swap_used_mib", math.inf) > 3584):
        errors.append("host trace violates memory/swap gate")
    lowered = log.lower()
    required_log_settings = (
        r"data_parallel_mode['\"]?\s*[:=]\s*['\"]?fsdp2",
        r"enable_activation_offload['\"]?\s*[:=]\s*true",
        r"activation_gpu_limit['\"]?\s*[:=]\s*0(?:\.0)?",
        r"enable_fsdp_offload['\"]?\s*[:=]\s*false",
    )
    if any(re.search(pattern, lowered) is None for pattern in required_log_settings):
        errors.append("effective training log lacks frozen activation-offload settings")
    if "out of memory" in lowered or re.search(r"\b(?:nan|inf)\b", lowered):
        errors.append("training log contains OOM or nonfinite markers")

    checkpoint = root / "checkpoint"
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    ) if checkpoint.is_dir() else 0
    metadata_files = list(checkpoint.rglob(".metadata")) if checkpoint.is_dir() else []
    if checkpoint_bytes < 40_000_000_000 or not metadata_files:
        errors.append("one-step DCP checkpoint is missing or unexpectedly small")
    if summary.get("checkpoint_bytes") != checkpoint_bytes:
        errors.append("summary checkpoint byte count is stale")

    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "arm": arm,
        "sequence_length": metadata.get("sequence_length"),
        "preflight_gpus": gpus,
        "preflight_host_available_mib": preflight_available,
        "checkpoint_bytes": checkpoint_bytes,
        "bindings": observed_bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument("--offloading", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--data-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {key: value.expanduser().resolve() for key, value in vars(args).items()
             if isinstance(value, Path)}
    root = paths.pop("root")
    output = paths.pop("output", None) or root / "independent_audit.json"
    result = audit(root, **paths)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
