#!/usr/bin/env python3
"""Create a non-authorizing corrected-Base RULER-256K review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts import create_in_place_ttt_2x5090_ruler_256k_release as generic
except ModuleNotFoundError:  # Direct script execution.
    import create_in_place_ttt_2x5090_ruler_256k_release as generic


PROTOCOL = "in_place_ttt_qwen3_4b_base_ruler_256k_release_review_v1"
CORRECTED_AUDIT_PROTOCOL = (
    "in_place_ttt_qwen3_4b_base_ruler_256k_resource_audit_v1"
)
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validated_review(
    *, resource_audit: Path, corrected_resource_audit: Path,
    model_identity: Path, release_tool: Path,
) -> dict[str, Any]:
    if Path(generic.__file__).resolve() != release_tool.resolve():
        raise ValueError("release-tool path differs from imported validator")
    generic_review = generic.review(resource_audit)
    corrected = json.loads(corrected_resource_audit.read_text(encoding="utf-8"))
    identity = json.loads(model_identity.read_text(encoding="utf-8"))
    if (corrected.get("protocol") != CORRECTED_AUDIT_PROTOCOL
            or corrected.get("passed") is not True
            or corrected.get("errors") != []
            or corrected.get("model_revision") != MODEL_REVISION
            or corrected.get("context_length") != 262144
            or corrected.get("extrapolation") is not True
            or corrected.get("scores_discarded") is not True
            or corrected.get("formal_256k_released") is not True
            or corrected.get("underlying_resource_audit_sha256")
            != sha256(resource_audit)):
        raise ValueError("corrected Base 256K resource audit is invalid")
    if (identity.get("passed") is not True
            or not identity.get("checks")
            or not all(identity.get("checks", {}).values())
            or identity.get("revision") != MODEL_REVISION):
        raise ValueError("Qwen3-4B-Base identity is invalid")
    return generic_review | {
        "corrected_resource_audit_path": str(corrected_resource_audit.resolve()),
        "corrected_resource_audit_sha256": sha256(corrected_resource_audit),
        "model_identity_path": str(model_identity.resolve()),
        "model_identity_sha256": sha256(model_identity),
        "model_revision": MODEL_REVISION,
        "claim_boundary": "public_data_yarn_extrapolation_review_only",
    }


def build_review(
    *, resource_audit: Path, corrected_resource_audit: Path,
    model_identity: Path, release_tool: Path, review_builder: Path,
    transition_tool: Path, plan: Path, target_release: Path, output: Path,
    expected_hostname: str,
) -> dict[str, Any]:
    if platform.node() != expected_hostname:
        raise RuntimeError("corrected 256K release review is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("corrected 256K release review must hide all GPUs")
    output, target_release = output.resolve(), target_release.resolve()
    partial = output.with_name(output.name + ".partial")
    release_partial = target_release.with_name(target_release.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    if target_release.exists() or release_partial.exists():
        raise FileExistsError("target formal release exists before review")
    required = (resource_audit, corrected_resource_audit, model_identity,
                release_tool, review_builder, transition_tool, plan)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"missing review inputs: {missing}")
    if review_builder.resolve() != Path(__file__).resolve():
        raise ValueError("review-builder path differs from running source")

    review = validated_review(
        resource_audit=resource_audit,
        corrected_resource_audit=corrected_resource_audit,
        model_identity=model_identity,
        release_tool=release_tool,
    )
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ready_for_human_review": True,
        "formal_release_created": False,
        "target_release_path": str(target_release),
        "review": review,
        "bindings": {
            "resource_audit_sha256": sha256(resource_audit),
            "corrected_resource_audit_sha256": sha256(corrected_resource_audit),
            "model_identity_sha256": sha256(model_identity),
            "release_tool_sha256": sha256(release_tool),
            "review_builder_sha256": sha256(review_builder),
            "transition_tool_sha256": sha256(transition_tool),
            "plan_sha256": sha256(plan),
        },
        "required_human_fields": ["accept", "reviewed_by", "decision_note"],
        "next_action": (
            "inspect the paired resource evidence, then separately invoke the "
            "review-bound transition if and only if accepted"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-audit", type=Path, required=True)
    parser.add_argument("--corrected-resource-audit", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--release-tool", type=Path, required=True)
    parser.add_argument("--review-builder", type=Path, required=True)
    parser.add_argument("--transition-tool", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_review(
        resource_audit=args.resource_audit.resolve(),
        corrected_resource_audit=args.corrected_resource_audit.resolve(),
        model_identity=args.model_identity.resolve(),
        release_tool=args.release_tool.resolve(),
        review_builder=args.review_builder.resolve(),
        transition_tool=args.transition_tool.resolve(),
        plan=args.plan.resolve(), target_release=args.target_release.resolve(),
        output=args.output.resolve(), expected_hostname=args.expected_hostname,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "ready_for_human_review": result["ready_for_human_review"],
        "formal_release_created": result["formal_release_created"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
