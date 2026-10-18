#!/usr/bin/env python3
"""Independently audit paired formal public-data RULER-256K scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_ruler import (
        TASK_PREFIXES, _one_dir, evaluation_input_sha256, sha256,
    )
    from scripts.summarize_in_place_ttt_2x5090_ruler_256k import PAPER
except ModuleNotFoundError:  # Direct `python scripts/<name>.py` execution.
    from audit_in_place_ttt_2x5090_ruler import (
        TASK_PREFIXES, _one_dir, evaluation_input_sha256, sha256,
    )
    from summarize_in_place_ttt_2x5090_ruler_256k import PAPER


PROTOCOL = "in_place_ttt_2x5090_ruler_256k_audit_v1"
RUN_PROTOCOL = "in_place_ttt_2x5090_ruler_256k_formal_run_v1"
RELEASE_PROTOCOL = "in_place_ttt_2x5090_ruler_256k_formal_release_v1"
ARMS = ("no_ttt", "ttt")


def audit(
    root: Path,
    *,
    training_root: Path,
    matrix_root: Path,
    resource_root: Path,
    release: Path,
    public_corpus_plan: Path,
    evaluation_plan: Path,
    extrapolation_plan: Path,
    formal_plan: Path,
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = root / "formal_metadata.json"
    summary_path = root / "summary.json"
    state_path = root / "state.txt"
    training_path = training_root / "independent_audit.json"
    matrix_path = matrix_root / "independent_matrix_audit.json"
    resource_path = resource_root / "independent_resource_audit.json"
    required = (
        metadata_path, summary_path, state_path, training_path, matrix_path,
        resource_path, release,
    )
    if any(not path.is_file() for path in required):
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["formal 256K artifact or dependency is missing"]}
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    training = json.loads(training_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    resource = json.loads(resource_path.read_text())
    release_data = json.loads(release.read_text())
    tool = Path(__file__).resolve().parents[1]
    expected_bindings = {
        "training_audit_sha256": sha256(training_path),
        "matrix_audit_sha256": sha256(matrix_path),
        "resource_audit_sha256": sha256(resource_path),
        "release_sha256": sha256(release),
        "public_corpus_plan_sha256": sha256(public_corpus_plan),
        "evaluation_plan_sha256": sha256(evaluation_plan),
        "extrapolation_plan_sha256": sha256(extrapolation_plan),
        "formal_plan_sha256": sha256(formal_plan),
        "runner_sha256": sha256(runner),
        "single_runner_sha256": sha256(
            tool / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
        ),
        "summarizer_sha256": sha256(
            tool / "scripts/summarize_in_place_ttt_2x5090_ruler_256k.py"
        ),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    if (metadata.get("protocol") != RUN_PROTOCOL
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata.get("tool_commit", "")) != 40
            or metadata.get("context_length") != 262144
            or metadata.get("training_root") != str(training_root.resolve())
            or metadata.get("matrix_root") != str(matrix_root.resolve())
            or metadata.get("resource_root") != str(resource_root.resolve())
            or any(metadata.get(key) != value
                   for key, value in expected_bindings.items())):
        errors.append("formal 256K metadata/dependency binding is invalid")
    if (training.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
            or training.get("passed") is not True or training.get("errors") != []
            or set(training.get("arms", {})) != set(ARMS)):
        errors.append("paired formal Stage-2 audit did not pass cleanly")
    if (matrix.get("protocol") != "in_place_ttt_2x5090_ruler_matrix_audit_v1"
            or matrix.get("passed") is not True or matrix.get("errors") != []):
        errors.append("primary matrix audit did not pass cleanly")
    if (resource.get("protocol")
            != "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
            or resource.get("passed") is not True or resource.get("errors") != []
            or resource.get("formal_256k_released") is not True):
        errors.append("256K resource audit did not release formal scoring")
    if (release_data.get("protocol") != RELEASE_PROTOCOL
            or release_data.get("accepted") is not True
            or release_data.get("resource_audit_sha256") != sha256(resource_path)
            or not str(release_data.get("reviewed_by", "")).strip()
            or not str(release_data.get("decision_note", "")).strip()):
        errors.append("explicit 256K release is invalid")
    if state_path.read_text().strip() not in {
            "auditing_ruler_256k", "completed_ruler_256k_formal"}:
        errors.append("formal 256K state is incomplete")

    expected_tasks = {f"{prefix}_256k" for prefix in TASK_PREFIXES}
    arm_scores: dict[str, float] = {}
    arm_inputs: dict[str, dict[str, str]] = {}
    children: dict[str, Any] = {}
    for arm in ARMS:
        child = root / arm
        child_audit_path = child / "independent_audit.json"
        child_metadata_path = child / "run_metadata.json"
        child_summary_path = child / "summary.json"
        if any(not path.is_file() for path in (
                child_audit_path, child_metadata_path, child_summary_path)):
            errors.append(f"{arm}: formal child audit/metadata/summary is missing")
            continue
        checked = json.loads(child_audit_path.read_text())
        child_metadata = json.loads(child_metadata_path.read_text())
        child_summary = json.loads(child_summary_path.read_text())
        if (checked.get("protocol") != "in_place_ttt_2x5090_ruler_audit_v1"
                or checked.get("passed") is not True or checked.get("errors") != []
                or checked.get("arm") != arm
                or checked.get("context_length") != 262144
                or checked.get("extrapolation") is not True
                or child_metadata.get("run_kind") != "formal"
                or child_metadata.get("num_samples_per_task") != 100
                or child_metadata.get("arm") != arm
                or child_metadata.get("context_length") != 262144
                or child_metadata.get("extrapolation") is not True
                or child_summary.get("completed") is not True):
            errors.append(f"{arm}: formal child identity/completion differs")
        result_dir = _one_dir(child, "results")
        prediction_dir = _one_dir(child, "predictions")
        scores: list[float] = []
        inputs: dict[str, str] = {}
        if result_dir is None or prediction_dir is None:
            errors.append(f"{arm}: raw result/prediction directory is missing")
        else:
            result_files = sorted(result_dir.glob("*.json"))
            prediction_files = sorted(prediction_dir.glob("*.json"))
            if {path.stem for path in result_files} != expected_tasks:
                errors.append(f"{arm}: raw result task set differs")
            if {path.stem for path in prediction_files} != expected_tasks:
                errors.append(f"{arm}: raw prediction task set differs")
            for path in result_files:
                score = float(json.loads(path.read_text()).get("score", float("nan")))
                if not math.isfinite(score):
                    errors.append(f"{arm}: nonfinite score in {path.name}")
                scores.append(score)
            for path in prediction_files:
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict) \
                        or list(payload) != [str(index) for index in range(100)]:
                    errors.append(f"{arm}: sample set differs in {path.name}")
                    continue
                try:
                    inputs[path.stem.removesuffix("_256k")] = (
                        evaluation_input_sha256(payload)
                    )
                except ValueError as exc:
                    errors.append(f"{arm}: {exc}")
        if len(scores) == len(expected_tasks):
            arm_scores[arm] = sum(scores) / len(scores)
        arm_inputs[arm] = inputs
        children[arm] = {
            "independent_audit_sha256": sha256(child_audit_path),
            "run_metadata_sha256": sha256(child_metadata_path),
            "summary_sha256": sha256(child_summary_path),
        }
    if set(arm_inputs) == set(ARMS) and arm_inputs["no_ttt"] != arm_inputs["ttt"]:
        errors.append("paired formal 256K inputs differ")
    scores = None
    paper_comparison = None
    if set(arm_scores) == set(ARMS):
        scores = {
            "no_ttt": arm_scores["no_ttt"],
            "ttt": arm_scores["ttt"],
            "ttt_minus_no_ttt": arm_scores["ttt"] - arm_scores["no_ttt"],
        }
        paper_comparison = {
            "public_no_ttt_minus_paper_matched": scores["no_ttt"] - PAPER["no_ttt"],
            "public_ttt_minus_paper_ttt": scores["ttt"] - PAPER["ttt"],
            "public_gain_minus_paper_gain": (
                scores["ttt_minus_no_ttt"] - (PAPER["ttt"] - PAPER["no_ttt"])
            ),
        }
    if (summary.get("protocol") != "in_place_ttt_2x5090_ruler_256k_summary_v1"
            or summary.get("completed") is not True
            or summary.get("errors") != []
            or summary.get("scores") != scores
            or summary.get("paper_table1_reference") != PAPER
            or summary.get("public_replication_minus_paper") != paper_comparison
            or summary.get("claim_boundary")
            != "public_data_yarn_extrapolation_not_paper_exact"):
        errors.append("formal 256K summary does not reproduce raw artifacts")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "context_length": 262144,
        "extrapolation": True,
        "scores": scores,
        "paper_table1_reference": PAPER,
        "public_replication_minus_paper": paper_comparison,
        "claim_boundary": "public_data_yarn_extrapolation_not_paper_exact",
        "child_bindings": children,
        "dependency_bindings": expected_bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root", "training_root", "matrix_root", "resource_root", "release",
        "public_corpus_plan", "evaluation_plan", "extrapolation_plan",
        "formal_plan", "runner", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
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
