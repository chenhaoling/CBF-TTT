#!/usr/bin/env python3
"""Audit the paired 4K/32K/64K/128K 2x5090 RULER resource ladder."""

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


PROTOCOL = "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1"
BOUNDARIES = ((4096, "4k"), (32768, "32k"), (65536, "64k"),
              (131072, "128k"))
ARMS = ("no_ttt", "ttt")


def _one_prediction_dir(root: Path) -> Path | None:
    found = sorted(
        path for path in (root / "opencompass").glob("*/predictions/*")
        if path.is_dir()
    )
    return found[0] if len(found) == 1 else None


def audit(
    root: Path,
    *,
    public_corpus_plan: Path,
    evaluation_plan: Path,
    training_root: Path,
    runner: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    metadata_path = root / "ladder_metadata.json"
    training_audit_path = training_root / "independent_audit.json"
    if not metadata_path.is_file() or not training_audit_path.is_file():
        return {"protocol": PROTOCOL, "passed": False,
                "errors": ["resource ladder metadata/training audit is missing"]}
    metadata = json.loads(metadata_path.read_text())
    training = json.loads(training_audit_path.read_text())
    tool_root = Path(__file__).resolve().parents[1]
    expected = {
        "training_audit_sha256": sha256(training_audit_path),
        "public_corpus_plan_sha256": sha256(public_corpus_plan),
        "evaluation_plan_sha256": sha256(evaluation_plan),
        "runner_sha256": sha256(runner),
        "single_runner_sha256": sha256(
            tool_root / "scripts/run_in_place_ttt_2x5090_ruler_single.sh"),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    if (metadata.get("protocol")
            != "in_place_ttt_2x5090_ruler_resource_ladder_run_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("tool_clean") is not True
            or not isinstance(metadata.get("tool_commit"), str)
            or len(metadata.get("tool_commit", "")) != 40
            or metadata.get("training_root") != str(training_root.resolve())
            or metadata.get("boundaries") != [4096, 32768, 65536, 131072]
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("resource ladder metadata/dependency binding is invalid")
    if (training.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
            or training.get("passed") is not True):
        errors.append("paired formal Stage-2 audit did not pass")
    runs: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for context, suffix in BOUNDARIES:
        paired_inputs: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            key = f"{suffix}_{arm}"
            run = root / arm / suffix
            audit_path = run / "independent_resource_audit.json"
            metadata_path = run / "run_metadata.json"
            if not audit_path.is_file() or not metadata_path.is_file():
                errors.append(f"missing resource audit/metadata: {key}")
                continue
            checked = json.loads(audit_path.read_text())
            metadata = json.loads(metadata_path.read_text())
            if (checked.get("protocol")
                    != "in_place_ttt_2x5090_ruler_resource_audit_v1"
                    or checked.get("passed") is not True
                    or checked.get("arm") != arm
                    or checked.get("context_length") != context
                    or metadata.get("protocol")
                    != "in_place_ttt_2x5090_ruler_run_v1"
                    or metadata.get("run_kind") != "resource"
                    or metadata.get("num_samples_per_task") != 1
                    or metadata.get("arm") != arm
                    or metadata.get("context_length") != context):
                errors.append(f"resource identity/completion differs: {key}")
            gpu = checked.get("gpu", {})
            try:
                free_fraction = 1 - float(gpu["peak_memory_used_mib"]) / float(
                    gpu["memory_total_mib"]
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                free_fraction = -1
            if free_fraction < 0.05:
                errors.append(f"resource GPU headroom differs: {key}")
            prediction_dir = _one_prediction_dir(run)
            input_hashes = {}
            if prediction_dir is None:
                errors.append(f"resource predictions missing: {key}")
            else:
                for path in sorted(prediction_dir.glob("*.json")):
                    try:
                        input_hashes[path.stem] = evaluation_input_sha256(
                            json.loads(path.read_text())
                        )
                    except (ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"{key}: {exc}")
            paired_inputs[arm] = input_hashes
            runs[key] = checked
            bindings[key] = {
                "independent_resource_audit_sha256": sha256(audit_path),
                "run_metadata_sha256": sha256(metadata_path),
            }
        if set(paired_inputs) == set(ARMS):
            normalized = {}
            for arm, hashes in paired_inputs.items():
                token = "_" + suffix
                normalized[arm] = {
                    name[:-len(token)]: digest for name, digest in hashes.items()
                    if name.endswith(token)
                }
            if normalized["no_ttt"] != normalized["ttt"]:
                errors.append(f"paired resource inputs differ: {suffix}")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "boundaries": runs,
        "scores_discarded": True,
        "bindings": bindings | {
            "ladder_metadata_sha256": sha256(metadata_path),
            **expected,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-corpus-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = audit(
        args.root.expanduser().resolve(),
        public_corpus_plan=args.public_corpus_plan.expanduser().resolve(),
        evaluation_plan=args.evaluation_plan.expanduser().resolve(),
        training_root=args.training_root.expanduser().resolve(),
        runner=args.runner.expanduser().resolve(),
    )
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
