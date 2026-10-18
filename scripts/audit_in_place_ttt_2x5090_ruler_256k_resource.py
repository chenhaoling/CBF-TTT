#!/usr/bin/env python3
"""Audit the paired one-sample RULER-256K extrapolation resource gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_ruler import (
        evaluation_input_sha256, sha256,
    )
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_2x5090_ruler import evaluation_input_sha256, sha256


PROTOCOL = "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_ruler_256k_resource_run_v1"
ARMS = ("no_ttt", "ttt")
TASK_COUNT = 13


def _prediction_dir(root: Path) -> Path | None:
    found = sorted(
        path for path in (root / "opencompass").glob("*/predictions/*")
        if path.is_dir()
    )
    return found[0] if len(found) == 1 else None


def audit(
    root: Path,
    *,
    training_root: Path,
    matrix_root: Path,
    public_corpus_plan: Path,
    evaluation_plan: Path,
    extrapolation_plan: Path,
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = root / "resource_metadata.json"
    state_path = root / "state.txt"
    training_path = training_root / "independent_audit.json"
    matrix_path = matrix_root / "independent_matrix_audit.json"
    required = (metadata_path, state_path, training_path, matrix_path)
    if any(not path.is_file() for path in required):
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["resource metadata/state or dependency audit is missing"]}
    metadata = json.loads(metadata_path.read_text())
    training = json.loads(training_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    tool_root = Path(__file__).resolve().parents[1]
    expected = {
        "training_audit_sha256": sha256(training_path),
        "matrix_audit_sha256": sha256(matrix_path),
        "public_corpus_plan_sha256": sha256(public_corpus_plan),
        "evaluation_plan_sha256": sha256(evaluation_plan),
        "extrapolation_plan_sha256": sha256(extrapolation_plan),
        "runner_sha256": sha256(runner),
        "single_runner_sha256": sha256(
            tool_root / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
        ),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    if (metadata.get("protocol") != RUN_PROTOCOL
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata.get("tool_commit", "")) != 40
            or metadata.get("training_root") != str(training_root.resolve())
            or metadata.get("matrix_root") != str(matrix_root.resolve())
            or metadata.get("context_length") != 262144
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("256K resource metadata/dependency binding is invalid")
    if (training.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
            or training.get("passed") is not True
            or training.get("errors") != []
            or set(training.get("arms", {})) != set(ARMS)):
        errors.append("paired formal Stage-2 audit did not pass cleanly")
    if (matrix.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
            or matrix.get("passed") is not True or matrix.get("errors") != []
            or set(matrix.get("recomputed_context_scores", {}))
            != {"4k", "8k", "16k", "32k", "64k", "128k"}):
        errors.append("registered 4K-128K matrix audit did not pass cleanly")
    if state_path.read_text().strip() not in {
            "auditing_256k_resource", "completed_ruler_256k_resource"}:
        errors.append("256K resource state is incomplete")

    boundaries: dict[str, Any] = {}
    paired_inputs: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        child = root / arm
        child_audit_path = child / "independent_resource_audit.json"
        child_metadata_path = child / "run_metadata.json"
        if not child_audit_path.is_file() or not child_metadata_path.is_file():
            errors.append(f"{arm}: child resource audit/metadata is missing")
            continue
        checked = json.loads(child_audit_path.read_text())
        child_metadata = json.loads(child_metadata_path.read_text())
        if (checked.get("protocol")
                != "in_place_ttt_2x5090_ruler_resource_audit_v1"
                or checked.get("passed") is not True
                or checked.get("errors") != [] or checked.get("arm") != arm
                or checked.get("context_length") != 262144
                or checked.get("extrapolation") is not True
                or child_metadata.get("protocol")
                != "in_place_ttt_2x5090_ruler_run_v1"
                or child_metadata.get("run_kind") != "resource"
                or child_metadata.get("num_samples_per_task") != 1
                or child_metadata.get("context_length") != 262144
                or child_metadata.get("extrapolation") is not True
                or child_metadata.get("arm") != arm):
            errors.append(f"{arm}: 256K child identity/completion differs")
        gpu = checked.get("gpu", {})
        try:
            free = 1 - float(gpu["peak_memory_used_mib"]) / float(
                gpu["memory_total_mib"]
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            free = -1
        if free < 0.05:
            errors.append(f"{arm}: 256K GPU headroom is below 5%")
        prediction_dir = _prediction_dir(child)
        input_hashes: dict[str, str] = {}
        if prediction_dir is None:
            errors.append(f"{arm}: 256K predictions are missing")
        else:
            files = sorted(prediction_dir.glob("*.json"))
            if len(files) != TASK_COUNT:
                errors.append(f"{arm}: 256K prediction task count differs")
            for path in files:
                try:
                    input_hashes[path.stem.removesuffix("_256k")] = (
                        evaluation_input_sha256(json.loads(path.read_text()))
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{arm}: {exc}")
        paired_inputs[arm] = input_hashes
        boundaries[arm] = checked
    if set(paired_inputs) == set(ARMS) \
            and paired_inputs["no_ttt"] != paired_inputs["ttt"]:
        errors.append("paired 256K resource inputs differ")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "context_length": 262144,
        "extrapolation": True,
        "scores_discarded": True,
        "formal_256k_released": not errors,
        "boundaries": boundaries,
        "bindings": {"resource_metadata_sha256": sha256(metadata_path), **expected},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--extrapolation-plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {key: value.expanduser().resolve() for key, value in vars(args).items()}
    output = paths.pop("output")
    root = paths.pop("root")
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
