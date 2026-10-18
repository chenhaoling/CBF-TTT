"""Independently audit paired three-step 128K Stage-2 timing."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_stage1_timing import (
        elapsed_seconds, gpu_resources, host_resources, parse_progress, sha256,
    )
    from scripts.audit_in_place_ttt_2x5090_100m_stage1 import validate_preflight
    from scripts.audit_in_place_ttt_2x5090_stage2_ladder import validate_start_model
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_2x5090_stage1_timing import (
        elapsed_seconds, gpu_resources, host_resources, parse_progress, sha256,
    )
    from audit_in_place_ttt_2x5090_100m_stage1 import validate_preflight
    from audit_in_place_ttt_2x5090_stage2_ladder import validate_start_model


PROTOCOL = "in_place_ttt_2x5090_stage2_timing_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_stage2_timing_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
HOSTNAME = "user-Super-Server"
ARMS = ("no_ttt", "ttt")
STEPS = 3
SEQ_LEN = 131072
GLOBAL_BATCH = 64
TOKENS_PER_STEP = SEQ_LEN * GLOBAL_BATCH
TRAIN_SIZE = STEPS * TOKENS_PER_STEP
FORMAL_STAGE2_STEPS = 5


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
    steady = (
        median(float(row["individual_seconds"]) for row in progress[1:])
        if len(progress) == STEPS and finite and monotonic else None
    )
    wall_match = re.search(
        r"Elapsed \(wall clock\) time.*\):\s*((?:[0-9]+:){1,2}[0-9.]+)", log
    )
    command_wall = elapsed_seconds(wall_match.group(1)) if wall_match else None
    gpu, host = gpu_resources(required["gpu.csv"]), host_resources(required["host.csv"])
    preflight_errors, preflight = validate_preflight(root)
    errors.extend(f"{arm}: {item}" for item in preflight_errors)
    checkpoint_files = [
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and (
            path.name == ".metadata" or path.suffix == ".distcp"
            or "global_step_" in str(path)
        )
    ]
    if exit_code != 0:
        errors.append(f"{arm}: exit code is {exit_code}")
    if len(progress) != STEPS or not finite or not monotonic:
        errors.append(f"{arm}: three-step progress trace is invalid")
    if steady is None or not math.isfinite(steady) or steady <= 0:
        errors.append(f"{arm}: warm-excluded timing is invalid")
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
        errors.append(f"{arm}: log contains OOM/nonfinite marker")
    return errors, {
        "arm": arm,
        "progress": progress,
        "command_wall_seconds": command_wall,
        "warm_excluded_median_seconds": steady,
        "steady_tokens_per_second": TOKENS_PER_STEP / steady if steady else None,
        "projected_formal_stage2_seconds": (
            FORMAL_STAGE2_STEPS * steady if steady else None
        ),
        "gpu": gpu, "host": host, "preflight": preflight,
        "checkpoint_files": checkpoint_files,
        "bindings": {name + "_sha256": sha256(path)
                     for name, path in required.items()},
    }


def audit(
    root: Path,
    *,
    stage1_root: Path,
    stage2_ladder_root: Path,
    timing_plan: Path,
    execution_addendum: Path,
    corpus_manifest: Path,
    corpus_audit: Path,
    data: Path,
    configs: dict[str, Path],
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = root / "run_metadata.json"
    config_audit_path = root / "config_audit.json"
    start_models_path = root / "start_models.json"
    stage1_audit_path = stage1_root / "independent_audit.json"
    stage1_state_path = stage1_root / "state.txt"
    ladder_audit_path = stage2_ladder_root / "independent_ladder_audit.json"
    ladder_state_path = stage2_ladder_root / "state.txt"
    required_top = (
        metadata_path, config_audit_path, start_models_path,
        stage1_audit_path, stage1_state_path, ladder_audit_path, ladder_state_path,
    )
    if any(not path.is_file() for path in required_top):
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["timing/Stage-1/Stage-2 ladder prerequisite is missing"]}
    metadata = json.loads(metadata_path.read_text())
    config_check = json.loads(config_audit_path.read_text())
    saved_models = json.loads(start_models_path.read_text())
    stage1_audit = json.loads(stage1_audit_path.read_text())
    ladder_audit = json.loads(ladder_audit_path.read_text())
    manifest = json.loads(corpus_manifest.read_text())
    corpus_check = json.loads(corpus_audit.read_text())
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
            or metadata.get("tokens_per_arm") != TRAIN_SIZE
            or metadata.get("stage1_root") != str(stage1_root.resolve())
            or metadata.get("stage2_ladder_root") != str(stage2_ladder_root.resolve())):
        errors.append("Stage-2 timing metadata host/count/revision binding is invalid")
    expected = {
        "timing_plan_sha256": sha256(timing_plan),
        "execution_addendum_sha256": sha256(execution_addendum),
        "stage1_audit_sha256": sha256(stage1_audit_path),
        "stage2_ladder_audit_sha256": sha256(ladder_audit_path),
        "corpus_manifest_sha256": sha256(corpus_manifest),
        "corpus_audit_sha256": sha256(corpus_audit),
        "data_sha256": sha256(data),
        "no_ttt_config_sha256": sha256(configs["no_ttt"]),
        "ttt_config_sha256": sha256(configs["ttt"]),
        "config_audit_sha256": sha256(config_audit_path),
        "start_models_sha256": sha256(start_models_path),
        "runner_sha256": sha256(runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        errors.append("Stage-2 timing dependency hash mismatch")
    if config_check.get("passed") is not True:
        errors.append("registered 2x5090 config audit did not pass")
    if (stage1_audit.get("passed") is not True
            or stage1_state_path.read_text().strip() != "completed_100m_stage1"):
        errors.append("paired formal Stage-1 did not pass")
    if (ladder_audit.get("protocol")
            != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
            or ladder_audit.get("passed") is not True
            or ladder_state_path.read_text().strip() != "completed_stage2_ladder"):
        errors.append("arm-specific Stage-2 resource ladder did not pass")
    current_models: dict[str, Any] = {}
    for arm in ARMS:
        model_errors, record = validate_start_model(stage1_root, stage1_audit, arm)
        errors.extend(model_errors); current_models[arm] = record
    if saved_models != current_models \
            or ladder_audit.get("start_models") != current_models:
        errors.append("Stage-1 start-model binding is stale or differs from ladder")
    stage = next((item for item in manifest.get("stages", [])
                  if item.get("name") == "stage2_128k"), {})
    artifact = stage.get("artifacts", {}).get(data.name, {})
    if (manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
            or manifest.get("profile") != "full"
            or corpus_check.get("passed") is not True
            or corpus_check.get("profile") != "full"
            or corpus_check.get("bindings", {}).get("manifest_sha256")
            != expected["corpus_manifest_sha256"]
            or artifact.get("sha256") != expected["data_sha256"]
            or artifact.get("size_bytes") != data.stat().st_size):
        errors.append("full Stage-2 corpus binding is invalid")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_errors, record = audit_arm(root / arm, arm)
        errors.extend(arm_errors); arms[arm] = record
    projections = [arms[arm].get("projected_formal_stage2_seconds") for arm in ARMS]
    paired = (
        sum(float(value) for value in projections)
        if all(isinstance(value, (int, float)) and math.isfinite(value)
               and value > 0 for value in projections) else None
    )
    if paired is None:
        errors.append("paired Stage-2 wall-time projection is invalid")
    return {
        "protocol": PROTOCOL, "passed": not errors, "errors": errors,
        "arms": arms, "start_models": current_models,
        "projected_sequential_stage2_seconds": paired,
        "bindings": expected | {"run_metadata_sha256": sha256(metadata_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-ladder-root", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
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
