#!/usr/bin/env python3
"""Audit the corrected Base Level-A prerequisite and 16K/32K ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_in_place_ttt_2x5090_stage1_ladder import validate_child
except ModuleNotFoundError:  # Direct script execution.
    from audit_in_place_ttt_2x5090_stage1_ladder import validate_child


PREREQUISITE_PROTOCOL = "in_place_ttt_qwen3_4b_base_level_a_prerequisite_v1"
PROTOCOL = "in_place_ttt_qwen3_4b_base_stage1_ladder_audit_v1"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
ARMS = ("no_ttt", "ttt")
LENGTHS = (16_384, 32_768)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_inputs(manifest_path: Path, audit_path: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    if not manifest_path.is_file() or not audit_path.is_file():
        return ["resource input manifest/audit is missing"], {}
    manifest, audit = load(manifest_path), load(audit_path)
    manifest_sha = sha256(manifest_path)
    if (manifest.get("protocol") != "in_place_ttt_2x5090_resource_inputs_v1"
            or manifest.get("bindings", {}).get("required_source_profile") != "full"
            or audit.get("passed") is not True
            or not all(audit.get("checks", {}).values())
            or audit.get("bindings", {}).get("manifest_sha256") != manifest_sha):
        errors.append("corrected full-corpus resource input audit is invalid")
    return errors, manifest


def audit_prerequisite(
    level_a_root: Path,
    level_a_audit_path: Path,
    manifest_path: Path,
    input_audit_path: Path,
) -> dict[str, Any]:
    errors, manifest = validate_inputs(manifest_path, input_audit_path)
    state, status = level_a_root / "state.txt", level_a_root / "status.txt"
    if not state.is_file() or state.read_text().strip() != "completed_both_arms":
        errors.append("corrected Level-A resource arms are incomplete")
    if not status.is_file() or status.read_text().strip() != "COMPLETE":
        errors.append("corrected Level-A wrapper is not complete")
    if not level_a_audit_path.is_file():
        errors.append("corrected Level-A terminal audit is missing")
        level_a_audit = {}
    else:
        level_a_audit = load(level_a_audit_path)
        if (level_a_audit.get("passed") is not True
                or not all(level_a_audit.get("checks", {}).values())
                or level_a_audit.get("resource_manifest_sha256")
                != sha256(manifest_path)):
            errors.append("corrected Level-A terminal audit is invalid")
    saved = manifest.get("outputs", {}).get("8192", {})
    artifact = saved.get("artifact", {})
    data_path = manifest_path.parent / artifact.get("path", "missing")
    manifest_sha = sha256(manifest_path) if manifest_path.is_file() else ""
    input_audit_sha = sha256(input_audit_path) if input_audit_path.is_file() else ""
    if (not data_path.is_file() or saved.get("records") != 2
            or saved.get("training_tokens") != 16_384
            or artifact.get("sha256") != sha256(data_path)):
        errors.append("corrected 8K resource input is invalid")
    records = {}
    for arm in ARMS:
        child_errors, record = validate_child(
            level_a_root / f"resource_8k_{arm}", 8192, arm,
            artifact.get("sha256", ""), manifest_sha, input_audit_sha,
        )
        errors.extend(child_errors)
        records[arm] = record
    return {
        "protocol": PREREQUISITE_PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "level_a_root": str(level_a_root.resolve()),
        "level_a_audit_sha256": (
            sha256(level_a_audit_path) if level_a_audit_path.is_file() else None
        ),
        "resource_manifest_sha256": (
            sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "resource_audit_sha256": (
            sha256(input_audit_path) if input_audit_path.is_file() else None
        ),
        "arms": records,
    }


def audit_final(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    errors: list[str] = []
    required = {
        name: root / name for name in (
            "ladder_metadata.json", "level_a_prerequisite_audit.json",
            "config_audit.json", "state.txt",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing ladder artifacts: {missing}"]}
    metadata = load(required["ladder_metadata.json"])
    saved_prerequisite = load(required["level_a_prerequisite_audit.json"])
    current_prerequisite = audit_prerequisite(
        args.level_a_root.resolve(), args.level_a_audit.resolve(),
        args.data_manifest.resolve(), args.data_audit.resolve(),
    )
    config_audit = load(required["config_audit.json"])
    manifest = load(args.data_manifest)
    manifest_sha, input_audit_sha = sha256(args.data_manifest), sha256(args.data_audit)
    expected = {
        "parent_plan_sha256": sha256(args.parent_plan),
        "stage1_plan_sha256": sha256(args.stage1_plan),
        "input_addendum_sha256": sha256(args.input_addendum),
        "model_identity_sha256": sha256(args.model_identity),
        "data_manifest_sha256": manifest_sha,
        "data_audit_sha256": input_audit_sha,
        "runner_sha256": sha256(args.runner),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "config_audit_sha256": sha256(required["config_audit.json"]),
        "level_a_prerequisite_audit_sha256":
            sha256(required["level_a_prerequisite_audit.json"]),
    }
    if (metadata.get("protocol") != "in_place_ttt_qwen3_4b_base_stage1_ladder_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("upstream_commit") != UPSTREAM_COMMIT
            or metadata.get("upstream_clean") is not True
            or metadata.get("tool_clean") is not True
            or metadata.get("level_a_root") != str(args.level_a_root.resolve())
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected Stage-1 ladder metadata binding is invalid")
    if (saved_prerequisite != current_prerequisite
            or current_prerequisite.get("passed") is not True):
        errors.append("corrected Level-A prerequisite is failed or stale")
    if config_audit.get("passed") is not True:
        errors.append("2x5090 configuration audit failed")
    if required["state.txt"].read_text().strip() not in {
            "all_children_completed", "completed_stage1_ladder"}:
        errors.append("Stage-1 ladder children are incomplete")
    records = {}
    starts: list[str] = []
    for length in LENGTHS:
        saved = manifest.get("outputs", {}).get(str(length), {})
        artifact = saved.get("artifact", {})
        data_path = args.data_manifest.parent / artifact.get("path", "missing")
        if (not data_path.is_file() or saved.get("records") != 2
                or saved.get("training_tokens") != 2 * length
                or artifact.get("sha256") != sha256(data_path)):
            errors.append(f"{length}: corrected resource input is invalid")
        for arm in ARMS:
            child_errors, record = validate_child(
                root / f"resource_{length}_{arm}", length, arm,
                artifact.get("sha256", ""), manifest_sha, input_audit_sha,
            )
            errors.extend(child_errors)
            records[f"{length}_{arm}"] = record
            start = root / f"{length}_{arm}_started_at.txt"
            if not start.is_file():
                errors.append(f"{length}_{arm}: start timestamp is missing")
            else:
                starts.append(start.read_text().strip())
    if len(starts) != 4 or starts != sorted(starts):
        errors.append("Stage-1 ladder execution order differs from registration")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "level_a_prerequisite": current_prerequisite,
        "boundaries": records,
        "bindings": expected,
        "claim_boundary": (
            "Corrected Base 16K/32K resource evidence only; no formal training, "
            "RULER matrix, or update-admission release."
        ),
    }


def write(result: dict[str, Any], output: Path) -> None:
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial = output.with_name(output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prereq = sub.add_parser("prerequisite")
    prereq.add_argument("--level-a-root", type=Path, required=True)
    prereq.add_argument("--level-a-audit", type=Path, required=True)
    prereq.add_argument("--data-manifest", type=Path, required=True)
    prereq.add_argument("--data-audit", type=Path, required=True)
    prereq.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--level-a-root", type=Path, required=True)
    final.add_argument("--level-a-audit", type=Path, required=True)
    final.add_argument("--data-manifest", type=Path, required=True)
    final.add_argument("--data-audit", type=Path, required=True)
    final.add_argument("--stage1-plan", type=Path, required=True)
    final.add_argument("--parent-plan", type=Path, required=True)
    final.add_argument("--input-addendum", type=Path, required=True)
    final.add_argument("--model-identity", type=Path, required=True)
    final.add_argument("--runner", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prerequisite":
        result = audit_prerequisite(
            args.level_a_root.resolve(), args.level_a_audit.resolve(),
            args.data_manifest.resolve(), args.data_audit.resolve(),
        )
    else:
        result = audit_final(args)
    write(result, args.output.resolve())


if __name__ == "__main__":
    main()
