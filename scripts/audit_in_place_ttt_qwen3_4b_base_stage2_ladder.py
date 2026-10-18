#!/usr/bin/env python3
"""Bind the existing Stage-2 ladder audit to the corrected Base chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_base_stage2_ladder_audit_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(args: argparse.Namespace) -> dict[str, Any]:
    ladder_audit_path = args.root / "independent_ladder_audit.json"
    metadata_path = args.root / "ladder_metadata.json"
    stage1_audit_path = args.stage1_root / "independent_audit.json"
    manifest_path = args.resource_input_root / "manifest.json"
    input_audit_path = args.resource_input_root / "independent_audit.json"
    required = [
        ladder_audit_path, metadata_path, stage1_audit_path,
        args.corrected_stage1_audit, args.model_identity, manifest_path,
        input_audit_path, args.resource_plan, args.execution_addendum,
        args.input_addendum, args.runner, args.underlying_auditor,
        args.no_ttt_config, args.ttt_config,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}
    errors: list[str] = []
    ladder, metadata = load(ladder_audit_path), load(metadata_path)
    stage1, corrected_stage1 = load(stage1_audit_path), load(
        args.corrected_stage1_audit
    )
    identity, manifest = load(args.model_identity), load(manifest_path)
    input_audit = load(input_audit_path)
    if (identity.get("passed") is not True
            or not identity.get("checks")
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        errors.append("Base identity audit is invalid")
    if (stage1.get("protocol")
            != "in_place_ttt_2x5090_100m_stage1_audit_v1"
            or stage1.get("passed") is not True or stage1.get("errors") != []
            or set(stage1.get("arms", {})) != {"no_ttt", "ttt"}):
        errors.append("underlying formal Stage-1 audit is invalid")
    if (corrected_stage1.get("protocol")
            != "in_place_ttt_qwen3_4b_base_100m_stage1_audit_v1"
            or corrected_stage1.get("passed") is not True
            or corrected_stage1.get("errors") != []
            or corrected_stage1.get("model_revision") != MODEL_REVISION
            or corrected_stage1.get("underlying_formal_audit_sha256")
            != sha256(stage1_audit_path)):
        errors.append("corrected formal Stage-1 audit is invalid or stale")
    manifest_sha = sha256(manifest_path)
    if (manifest.get("protocol") != "in_place_ttt_2x5090_resource_inputs_v1"
            or manifest.get("bindings", {}).get("required_source_profile")
            != "full"
            or input_audit.get("passed") is not True
            or not input_audit.get("checks")
            or not all(input_audit.get("checks", {}).values())
            or input_audit.get("bindings", {}).get("manifest_sha256")
            != manifest_sha):
        errors.append("corrected full-corpus resource inputs are invalid")
    if (ladder.get("protocol")
            != "in_place_ttt_2x5090_stage2_ladder_audit_v1"
            or ladder.get("passed") is not True or ladder.get("errors") != []
            or set(ladder.get("boundaries", {})) != {
                "65536_no_ttt", "65536_ttt",
                "131072_no_ttt", "131072_ttt",
            }):
        errors.append("underlying Stage-2 ladder audit failed")
    expected = {
        "stage1_audit_sha256": sha256(stage1_audit_path),
        "resource_plan_sha256": sha256(args.resource_plan),
        "execution_addendum_sha256": sha256(args.execution_addendum),
        "input_addendum_sha256": sha256(args.input_addendum),
        "data_manifest_sha256": manifest_sha,
        "data_audit_sha256": sha256(input_audit_path),
        "no_ttt_config_sha256": sha256(args.no_ttt_config),
        "ttt_config_sha256": sha256(args.ttt_config),
        "runner_sha256": sha256(args.runner),
        "auditor_sha256": sha256(args.underlying_auditor),
    }
    if (metadata.get("protocol") != "in_place_ttt_2x5090_stage2_ladder_v1"
            or metadata.get("hostname") != "user-Super-Server"
            or metadata.get("stage1_root") != str(args.stage1_root.resolve())
            or any(metadata.get(key) != value for key, value in expected.items())):
        errors.append("corrected Stage-2 ladder metadata binding mismatch")
    if ladder.get("bindings", {}).get("stage1_audit_sha256") \
            != sha256(stage1_audit_path):
        errors.append("ladder audit does not bind the current Stage-1 audit")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": MODEL_REVISION,
        "corrected_stage1_audit_sha256": sha256(args.corrected_stage1_audit),
        "underlying_ladder_audit_sha256": sha256(ladder_audit_path),
        "start_models": ladder.get("start_models", {}),
        "boundaries": ladder.get("boundaries", {}),
        "bindings": expected,
        "claim_boundary": (
            "Corrected Base Stage-2 64K/128K resource evidence only; timing, "
            "formal Stage 2, RULER and update-admission remain unreleased."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--corrected-stage1-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--resource-input-root", type=Path, required=True)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--input-addendum", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--underlying-auditor", type=Path, required=True)
    parser.add_argument("--no-ttt-config", type=Path, required=True)
    parser.add_argument("--ttt-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    partial = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(args.output)
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
