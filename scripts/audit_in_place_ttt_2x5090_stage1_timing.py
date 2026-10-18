"""Independently audit the hku-gpu2 three-step Stage-1 timing pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any


PROTOCOL = "in_place_ttt_2x5090_stage1_timing_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_stage1_timing_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
HOSTNAME = "user-Super-Server"
ARMS = ("no_ttt", "ttt")
STEPS = 3
SEQ_LEN = 32768
GLOBAL_BATCH = 64
TOKENS_PER_STEP = SEQ_LEN * GLOBAL_BATCH
FORMAL_STAGE1_STEPS = 27
PROGRESS_PATTERN = re.compile(
    r"Epoch 1/1:[^\r\n]*\|\s*(\d+)/3\s*"
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
        raise ValueError(f"invalid tqdm elapsed value: {value}")
    return sum(item * 60 ** index for index, item in enumerate(reversed(fields)))


def parse_progress(log: str) -> list[dict[str, float | int]]:
    """Return the first completed trace for each optimizer step."""
    observed: dict[int, dict[str, float | int]] = {}
    for match in PROGRESS_PATTERN.finditer(log.replace("\r", "\n")):
        step = int(match.group(1))
        if step not in observed:
            observed[step] = {
                "step": step,
                "cumulative_seconds": elapsed_seconds(match.group(2)),
                "displayed_seconds_per_iteration": float(match.group(3)),
                "loss": float(match.group(4)),
                "gradient_norm": float(match.group(5)),
                "learning_rate": float(match.group(6)),
            }
    records = [observed[step] for step in sorted(observed)]
    previous = 0.0
    for record in records:
        cumulative = float(record["cumulative_seconds"])
        record["individual_seconds"] = cumulative - previous
        previous = cumulative
    return records


def gpu_resources(path: Path) -> dict[str, Any]:
    peaks: dict[str, float] = {}
    free: dict[str, float] = {}
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            index = row["index"]
            used = float(row["memory_used_mib"])
            total = float(row["memory_total_mib"])
            peaks[index] = max(peaks.get(index, 0.0), used)
            free[index] = min(free.get(index, 1.0), (total - used) / total)
    return {"rows": rows, "peak_memory_mib": peaks,
            "minimum_free_fraction": free}


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
        "minimum_memory_available_mib": (
            None if rows == 0 else minimum_available
        ),
        "peak_swap_used_mib": None if rows == 0 else peak_swap,
    }


def audit_arm(root: Path, arm: str) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    names = (
        "train.log", "exit_code.txt", "gpu.csv", "host.csv",
        "preflight_gpu.csv", "preflight_processes.csv",
        "preflight_host.txt", "preflight_conflicts.txt",
    )
    required = {name: root / name for name in names}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [f"{arm}: missing {missing}"], {}
    log = required["train.log"].read_text(encoding="utf-8", errors="replace")
    try:
        exit_code = int(required["exit_code.txt"].read_text().strip())
    except ValueError:
        exit_code = -1
    progress = parse_progress(log)
    finite = all(
        all(math.isfinite(float(row[key])) for key in (
            "cumulative_seconds", "individual_seconds", "loss",
            "gradient_norm", "learning_rate",
        )) for row in progress
    )
    monotonic = (
        [row["step"] for row in progress] == [1, 2, 3]
        and all(float(row["individual_seconds"]) > 0 for row in progress)
    )
    wall_match = re.search(
        r"Elapsed \(wall clock\) time.*\):\s*((?:[0-9]+:){1,2}[0-9.]+)", log
    )
    command_wall_seconds = (
        elapsed_seconds(wall_match.group(1)) if wall_match else None
    )
    steady_seconds = (
        median(float(row["individual_seconds"]) for row in progress[1:])
        if len(progress) == STEPS and finite and monotonic else None
    )
    gpu = gpu_resources(required["gpu.csv"])
    host = host_resources(required["host.csv"])
    with required["preflight_gpu.csv"].open(newline="", encoding="utf-8") as handle:
        preflight_rows = [row for row in csv.reader(handle) if row]
    preflight_gpu_valid = len(preflight_rows) == 2
    if preflight_gpu_valid:
        for expected_index, row in enumerate(preflight_rows):
            if len(row) != 4:
                preflight_gpu_valid = False
                break
            try:
                index, name, total, used = [item.strip() for item in row]
                preflight_gpu_valid &= (
                    int(index) == expected_index and "RTX 5090" in name
                    and float(total) >= 32000 and float(used) <= 128
                )
            except ValueError:
                preflight_gpu_valid = False
                break
    host_match = re.search(
        r"^MemAvailable:\s+(\d+)\s+kB$",
        required["preflight_host.txt"].read_text(), re.MULTILINE,
    )
    preflight_available_mib = (
        int(host_match.group(1)) / 1024 if host_match else None
    )
    checkpoint_files = [
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and (
            path.name == ".metadata" or "global_step_" in str(path)
            or path.suffix == ".distcp"
        )
    ]
    if exit_code != 0:
        errors.append(f"{arm}: exit code is {exit_code}")
    if len(progress) != STEPS or not finite or not monotonic:
        errors.append(f"{arm}: progress trace is incomplete, nonfinite or nonmonotonic")
    if steady_seconds is None or not math.isfinite(steady_seconds) \
            or steady_seconds <= 0:
        errors.append(f"{arm}: warm-excluded timing is invalid")
    if command_wall_seconds is None or not math.isfinite(command_wall_seconds) \
            or command_wall_seconds <= 0:
        errors.append(f"{arm}: command wall time is absent or invalid")
    if set(gpu["minimum_free_fraction"]) != {"0", "1"} \
            or gpu["rows"] == 0 \
            or any(value < 0.05 for value in gpu["minimum_free_fraction"].values()):
        errors.append(f"{arm}: GPU trace violates the registered headroom gate")
    if (host["rows"] == 0
            or host["minimum_memory_available_mib"] is None
            or host["minimum_memory_available_mib"] < 8192
            or host["peak_swap_used_mib"] is None
            or host["peak_swap_used_mib"] > 3584):
        errors.append(f"{arm}: host trace violates the memory/swap gate")
    if required["preflight_processes.csv"].read_text().strip() \
            or required["preflight_conflicts.txt"].read_text().strip():
        errors.append(f"{arm}: preflight process/conflict list is nonempty")
    if not preflight_gpu_valid or preflight_available_mib is None \
            or preflight_available_mib < 48 * 1024:
        errors.append(f"{arm}: preflight GPU/host gate is invalid")
    if checkpoint_files:
        errors.append(f"{arm}: timing pilot wrote checkpoint files")
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
        errors.append(f"{arm}: log contains OOM or nonfinite marker")
    return errors, {
        "arm": arm,
        "progress": progress,
        "command_wall_seconds": command_wall_seconds,
        "warm_excluded_median_seconds": steady_seconds,
        "steady_tokens_per_second": (
            TOKENS_PER_STEP / steady_seconds if steady_seconds else None
        ),
        "projected_formal_stage1_seconds": (
            FORMAL_STAGE1_STEPS * steady_seconds if steady_seconds else None
        ),
        "gpu": gpu,
        "host": host,
        "preflight_gpu_valid": preflight_gpu_valid,
        "preflight_host_available_mib": preflight_available_mib,
        "checkpoint_files": checkpoint_files,
        "bindings": {name + "_sha256": sha256(path)
                     for name, path in required.items()},
    }


def audit(
    root: Path,
    *,
    timing_plan: Path,
    execution_addendum: Path,
    ladder_audit: Path,
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
    ladder = json.loads(ladder_audit.read_text(encoding="utf-8"))
    manifest = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    corpus_check = json.loads(corpus_audit.read_text(encoding="utf-8"))
    if (metadata.get("protocol") != RUN_PROTOCOL
            or metadata.get("hostname") != HOSTNAME
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata["tool_commit"]) != 40
            or metadata.get("steps") != STEPS
            or metadata.get("sequence_length") != SEQ_LEN
            or metadata.get("global_batch_size") != GLOBAL_BATCH
            or metadata.get("tokens_per_arm") != STEPS * TOKENS_PER_STEP):
        errors.append("run metadata protocol/host/count/revision binding is invalid")
    expected = {
        "timing_plan_sha256": sha256(timing_plan),
        "execution_addendum_sha256": sha256(execution_addendum),
        "ladder_audit_sha256": sha256(ladder_audit),
        "corpus_manifest_sha256": sha256(corpus_manifest),
        "corpus_audit_sha256": sha256(corpus_audit),
        "data_sha256": sha256(data),
        "model_config_sha256": sha256(model_config),
        "runner_sha256": sha256(runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "no_ttt_config_sha256": sha256(configs["no_ttt"]),
        "ttt_config_sha256": sha256(configs["ttt"]),
        "config_audit_sha256": sha256(config_audit_path),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        errors.append("run dependency hash mismatch")
    if config_audit.get("passed") is not True:
        errors.append("registered 2x5090 configuration audit did not pass")
    stage = next((item for item in manifest.get("stages", [])
                  if item.get("name") == "stage1_32k"), {})
    artifact = stage.get("artifacts", {}).get(data.name, {})
    if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
            or manifest.get("profile") != "full"
            or corpus_check.get("passed") is not True
            or corpus_check.get("profile") != "full"
            or corpus_check.get("bindings", {}).get("manifest_sha256")
            != sha256(corpus_manifest)
            or artifact.get("sha256") != expected["data_sha256"]
            or artifact.get("size_bytes") != data.stat().st_size):
        errors.append("full public-corpus manifest/audit/artifact is invalid")
    if (ladder.get("passed") is not True
            or set(ladder.get("boundaries", {}))
            != {"16384_no_ttt", "16384_ttt", "32768_no_ttt", "32768_ttt"}):
        errors.append("Stage-1 resource ladder did not release timing")

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_errors, record = audit_arm(root / arm, arm)
        errors.extend(arm_errors)
        arms[arm] = record
    projections = [arms[arm].get("projected_formal_stage1_seconds") for arm in ARMS]
    projected_total = (
        sum(float(value) for value in projections)
        if all(isinstance(value, (int, float)) and math.isfinite(value)
               and value > 0 for value in projections) else None
    )
    if projected_total is None:
        errors.append("paired Stage-1 wall-time projection is invalid")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "arms": arms,
        "projected_sequential_stage1_seconds": projected_total,
        "bindings": expected | {"run_metadata_sha256": sha256(metadata_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--ladder-audit", type=Path, required=True)
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
    configs = {
        "no_ttt": paths.pop("no_ttt_config"),
        "ttt": paths.pop("ttt_config"),
    }
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
