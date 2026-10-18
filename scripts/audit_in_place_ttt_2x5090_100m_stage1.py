"""Independently audit paired formal 2x5090 100M-tier Stage-1 training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_2x5090_100m_stage1_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_100m_stage1_v1"
RELEASE_PROTOCOL = "in_place_ttt_2x5090_100m_stage1_release_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
HOSTNAME = "user-Super-Server"
ARMS = ("no_ttt", "ttt")
STEPS = 27
SEQ_LEN = 32768
GLOBAL_BATCH = 64
TRAIN_SIZE = STEPS * SEQ_LEN * GLOBAL_BATCH
MIN_DCP_BYTES = 40_000_000_000
MIN_HF_BYTES = 7_000_000_000
PROGRESS_PATTERN = re.compile(
    r"Epoch 1/1:[^\r\n]*\|\s*(\d+)/27\s*"
    r"\[([0-9:]+)<[^\]]*?,\s*([0-9.]+)s/it,\s*"
    r"loss:\s*([-+0-9.eE]+),\s*grad_norm:\s*([-+0-9.eE]+),\s*"
    r"lr:\s*([-+0-9.eE]+)\]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def elapsed_seconds(value: str) -> float:
    fields = [float(item) for item in value.split(":")]
    if len(fields) not in {2, 3}:
        raise ValueError(f"invalid elapsed value: {value}")
    return sum(item * 60 ** index for index, item in enumerate(reversed(fields)))


def parse_progress(log: str) -> list[dict[str, float | int]]:
    records: dict[int, dict[str, float | int]] = {}
    for match in PROGRESS_PATTERN.finditer(log.replace("\r", "\n")):
        step = int(match.group(1))
        if step not in records:
            records[step] = {
                "step": step,
                "cumulative_seconds": elapsed_seconds(match.group(2)),
                "displayed_seconds_per_iteration": float(match.group(3)),
                "loss": float(match.group(4)),
                "gradient_norm": float(match.group(5)),
                "learning_rate": float(match.group(6)),
            }
    result = [records[step] for step in sorted(records)]
    previous = 0.0
    for row in result:
        cumulative = float(row["cumulative_seconds"])
        row["individual_seconds"] = cumulative - previous
        previous = cumulative
    return result


def gpu_resources(path: Path) -> dict[str, Any]:
    rows = 0
    peak: dict[str, float] = {}
    minimum_free: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            index = row["index"]
            used = float(row["memory_used_mib"])
            total = float(row["memory_total_mib"])
            peak[index] = max(peak.get(index, 0.0), used)
            minimum_free[index] = min(
                minimum_free.get(index, 1.0), (total - used) / total
            )
    return {"rows": rows, "peak_memory_mib": peak,
            "minimum_free_fraction": minimum_free}


def host_resources(path: Path) -> dict[str, Any]:
    rows = 0
    minimum_available = math.inf
    peak_swap = 0.0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            minimum_available = min(
                minimum_available, float(row["memory_available_mib"])
            )
            peak_swap = max(peak_swap, float(row["swap_used_mib"]))
    return {
        "rows": rows,
        "minimum_memory_available_mib": None if rows == 0 else minimum_available,
        "peak_swap_used_mib": None if rows == 0 else peak_swap,
    }


def validate_preflight(root: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    with (root / "preflight_gpu.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    valid_gpus = len(rows) == 2
    if valid_gpus:
        for expected, row in enumerate(rows):
            try:
                index, name, total, used = [item.strip() for item in row]
                valid_gpus &= (
                    int(index) == expected and "RTX 5090" in name
                    and float(total) >= 32000 and float(used) <= 128
                )
            except (ValueError, IndexError):
                valid_gpus = False
    match = re.search(
        r"^MemAvailable:\s+(\d+)\s+kB$",
        (root / "preflight_host.txt").read_text(), re.MULTILINE,
    )
    available_mib = int(match.group(1)) / 1024 if match else None
    processes_empty = not (root / "preflight_processes.csv").read_text().strip()
    conflicts_empty = not (root / "preflight_conflicts.txt").read_text().strip()
    if not valid_gpus:
        errors.append("preflight GPU binding/idle gate failed")
    if available_mib is None or available_mib < 48 * 1024:
        errors.append("preflight host memory gate failed")
    if not processes_empty or not conflicts_empty:
        errors.append("preflight process/conflict list is nonempty")
    return errors, {
        "gpu_valid": valid_gpus,
        "host_available_mib": available_mib,
        "processes_empty": processes_empty,
        "conflicts_empty": conflicts_empty,
    }


def audit_arm(root: Path, arm: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    names = (
        "train.log", "exit_code.txt", "gpu.csv", "host.csv",
        "preflight_gpu.csv", "preflight_processes.csv",
        "preflight_host.txt", "preflight_conflicts.txt", "disk_before.txt",
    )
    required = {name: root / name for name in names}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [f"{arm}: missing {missing}"], {}
    log = required["train.log"].read_text(encoding="utf-8", errors="replace")
    try:
        exit_code = int(required["exit_code.txt"].read_text().strip())
        disk_free = int(required["disk_before.txt"].read_text().strip())
    except ValueError:
        exit_code, disk_free = -1, -1
    progress = parse_progress(log)
    progress_valid = (
        [row["step"] for row in progress] == list(range(1, STEPS + 1))
        and all(
            all(math.isfinite(float(row[key])) for key in (
                "cumulative_seconds", "individual_seconds", "loss",
                "gradient_norm", "learning_rate",
            )) and float(row["individual_seconds"]) > 0
            for row in progress
        )
    )
    wall_match = re.search(
        r"Elapsed \(wall clock\) time.*\):\s*((?:[0-9]+:){1,2}[0-9.]+)", log
    )
    command_wall = elapsed_seconds(wall_match.group(1)) if wall_match else None
    gpu, host = gpu_resources(required["gpu.csv"]), host_resources(required["host.csv"])
    preflight_errors, preflight = validate_preflight(root)
    errors.extend(f"{arm}: {item}" for item in preflight_errors)
    if exit_code != 0:
        errors.append(f"{arm}: exit code is {exit_code}")
    if disk_free < 80_000_000_000:
        errors.append(f"{arm}: pre-run free disk is below 80 GB")
    if not progress_valid:
        errors.append(f"{arm}: 27-step progress trace is invalid")
    if command_wall is None or not math.isfinite(command_wall) or command_wall <= 0:
        errors.append(f"{arm}: command wall time is invalid")
    if (gpu["rows"] == 0 or set(gpu["minimum_free_fraction"]) != {"0", "1"}
            or any(value < 0.05 for value in gpu["minimum_free_fraction"].values())):
        errors.append(f"{arm}: GPU resource gate failed")
    if (host["rows"] == 0 or host["minimum_memory_available_mib"] is None
            or host["minimum_memory_available_mib"] < 8192
            or host["peak_swap_used_mib"] is None
            or host["peak_swap_used_mib"] > 3584):
        errors.append(f"{arm}: host resource gate failed")
    lowered = log.lower()
    settings = (
        r"data_parallel_mode['\"]?\s*[:=]\s*['\"]?fsdp2",
        r"enable_activation_offload['\"]?\s*[:=]\s*true",
        r"activation_gpu_limit['\"]?\s*[:=]\s*0(?:\.0)?",
        r"enable_fsdp_offload['\"]?\s*[:=]\s*false",
    )
    if any(re.search(pattern, lowered) is None for pattern in settings):
        errors.append(f"{arm}: effective log lacks frozen offload settings")
    if "out of memory" in lowered or re.search(r"\b(?:nan|inf)\b", lowered):
        errors.append(f"{arm}: log contains OOM/nonfinite marker")

    step_dirs = [
        path for path in root.rglob("global_step_*")
        if path.is_dir() and re.fullmatch(r"global_step_\d+", path.name)
    ]
    if [path.name for path in step_dirs] != ["global_step_27"]:
        errors.append(f"{arm}: final checkpoint set is not exactly global_step_27")
    final = step_dirs[0] if len(step_dirs) == 1 else None
    dcp_files = [
        path for path in final.rglob("*") if path.is_file()
        and "hf_ckpt" not in path.parts
    ] if final else []
    dcp_bytes = sum(path.stat().st_size for path in dcp_files)
    if final is None or not (final / ".metadata").is_file() \
            or dcp_bytes < MIN_DCP_BYTES:
        errors.append(f"{arm}: final DCP is missing or unexpectedly small")
    hf = final / "hf_ckpt" if final else None
    hf_files = [path for path in hf.rglob("*") if path.is_file()] \
        if hf and hf.is_dir() else []
    hf_bytes = sum(path.stat().st_size for path in hf_files)
    hf_config_path = hf / "config.json" if hf else None
    if not hf_config_path or not hf_config_path.is_file() or hf_bytes < MIN_HF_BYTES:
        errors.append(f"{arm}: converted HF checkpoint is missing or unexpectedly small")
        hf_config = {}
    else:
        hf_config = json.loads(hf_config_path.read_text(encoding="utf-8"))
        if (bool(hf_config.get("ttt_mode", False)) != (arm == "ttt")
                or hf_config.get("max_position_embeddings") != SEQ_LEN):
            errors.append(f"{arm}: converted HF config differs from arm/Stage 1")
    return errors, {
        "arm": arm,
        "exit_code": exit_code,
        "disk_free_bytes_before": disk_free,
        "progress": progress,
        "command_wall_seconds": command_wall,
        "gpu": gpu,
        "host": host,
        "preflight": preflight,
        "final_checkpoint_relative_path": (
            str(final.relative_to(root)) if final else None
        ),
        "final_hf_relative_path": (
            str(hf.relative_to(root)) if hf and hf.is_dir() else None
        ),
        "dcp_bytes": dcp_bytes,
        "hf_bytes": hf_bytes,
        "dcp_manifest": {
            str(path.relative_to(final)): {"sha256": sha256(path),
                                           "size_bytes": path.stat().st_size}
            for path in dcp_files
        } if final else {},
        "hf_manifest": {
            str(path.relative_to(hf)): {"sha256": sha256(path),
                                        "size_bytes": path.stat().st_size}
            for path in hf_files
        } if hf else {},
        "bindings": {name + "_sha256": sha256(path)
                     for name, path in required.items()},
    }


def audit(
    root: Path,
    *,
    formal_plan: Path,
    execution_addendum: Path,
    timing_audit: Path,
    release: Path,
    corpus_manifest: Path,
    corpus_audit: Path,
    data: Path,
    model_config: Path,
    configs: dict[str, Path],
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = root / "run_metadata.json"
    config_audit_path = root / "config_audit.json"
    if not metadata_path.is_file() or not config_audit_path.is_file():
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["run metadata or config audit is missing"]}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_audit = json.loads(config_audit_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_audit.read_text(encoding="utf-8"))
    release_data = json.loads(release.read_text(encoding="utf-8"))
    manifest = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    corpus_check = json.loads(corpus_audit.read_text(encoding="utf-8"))
    if (metadata.get("protocol") != RUN_PROTOCOL
            or metadata.get("hostname") != HOSTNAME
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata["tool_commit"]) != 40
            or metadata.get("steps_per_arm") != STEPS
            or metadata.get("sequence_length") != SEQ_LEN
            or metadata.get("global_batch_size") != GLOBAL_BATCH
            or metadata.get("tokens_per_arm") != TRAIN_SIZE):
        errors.append("run metadata protocol/host/count/revision binding is invalid")
    expected = {
        "formal_plan_sha256": sha256(formal_plan),
        "execution_addendum_sha256": sha256(execution_addendum),
        "timing_audit_sha256": sha256(timing_audit),
        "release_sha256": sha256(release),
        "corpus_manifest_sha256": sha256(corpus_manifest),
        "corpus_audit_sha256": sha256(corpus_audit),
        "data_sha256": sha256(data),
        "model_config_sha256": sha256(model_config),
        "no_ttt_config_sha256": sha256(configs["no_ttt"]),
        "ttt_config_sha256": sha256(configs["ttt"]),
        "config_audit_sha256": sha256(config_audit_path),
        "runner_sha256": sha256(runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        errors.append("run dependency hash mismatch")
    if config_audit.get("passed") is not True:
        errors.append("registered 2x5090 config audit did not pass")
    if (timing.get("passed") is not True
            or timing.get("protocol")
            != "in_place_ttt_2x5090_stage1_timing_audit_v1"
            or timing.get("projected_sequential_stage1_seconds") is None):
        errors.append("paired Stage-1 timing audit did not pass")
    if (release_data.get("protocol") != RELEASE_PROTOCOL
            or release_data.get("accepted") is not True
            or release_data.get("timing_audit_sha256") != sha256(timing_audit)):
        errors.append("formal Stage-1 release record is invalid")
    stage = next((item for item in manifest.get("stages", [])
                  if item.get("name") == "stage1_32k"), {})
    artifact = stage.get("artifacts", {}).get(data.name, {})
    if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
            or manifest.get("profile") != "full"
            or corpus_check.get("passed") is not True
            or corpus_check.get("profile") != "full"
            or corpus_check.get("bindings", {}).get("manifest_sha256")
            != expected["corpus_manifest_sha256"]
            or artifact.get("sha256") != expected["data_sha256"]
            or artifact.get("size_bytes") != data.stat().st_size):
        errors.append("full Stage-1 corpus binding is invalid")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_errors, record = audit_arm(root / arm, arm)
        errors.extend(arm_errors)
        arms[arm] = record
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "arms": arms,
        "bindings": expected | {"run_metadata_sha256": sha256(metadata_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--no-ttt-config", type=Path, required=True)
    parser.add_argument("--ttt-config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.expanduser().resolve() for key, value in vars(args).items()
             if isinstance(value, Path)}
    output = paths.pop("output")
    configs = {"no_ttt": paths.pop("no_ttt_config"),
               "ttt": paths.pop("ttt_config")}
    root = paths.pop("root")
    result = audit(root, configs=configs, **paths)
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
