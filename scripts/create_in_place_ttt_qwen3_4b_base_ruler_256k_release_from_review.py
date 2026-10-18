#!/usr/bin/env python3
"""Atomically accept a corrected-Base RULER-256K review packet."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts import create_in_place_ttt_2x5090_ruler_256k_release as generic
    from scripts import create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review as review_module
except ModuleNotFoundError:  # Direct script execution.
    import create_in_place_ttt_2x5090_ruler_256k_release as generic
    import create_in_place_ttt_qwen3_4b_base_ruler_256k_release_review as review_module


CORRECTED_RELEASE_PROTOCOL = (
    "in_place_ttt_qwen3_4b_base_ruler_256k_formal_release_v1"
)


def create_from_review(
    *, review_packet: Path, transition_plan: Path, output: Path, accept: bool,
    reviewed_by: str, decision_note: str, expected_hostname: str,
) -> dict[str, Any]:
    if platform.node() != expected_hostname:
        raise RuntimeError("corrected 256K release is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("corrected 256K release transition must hide all GPUs")
    output, review_packet = output.resolve(), review_packet.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    packet = json.loads(review_packet.read_text(encoding="utf-8"))
    if (packet.get("protocol") != review_module.PROTOCOL
            or packet.get("ready_for_human_review") is not True
            or packet.get("formal_release_created") is not False
            or packet.get("target_release_path") != str(output)
            or set(packet.get("required_human_fields", []))
            != {"accept", "reviewed_by", "decision_note"}):
        raise ValueError("review packet is invalid or targets another release")
    embedded = packet.get("review", {})
    resource = Path(embedded.get("resource_audit_path", "")).resolve()
    corrected = Path(embedded.get("corrected_resource_audit_path", "")).resolve()
    identity = Path(embedded.get("model_identity_path", "")).resolve()
    release_tool = Path(generic.__file__).resolve()
    review_builder = Path(review_module.__file__).resolve()
    transition_tool = Path(__file__).resolve()
    required = (resource, corrected, identity, release_tool, review_builder,
                transition_tool, transition_plan.resolve())
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"missing review-bound inputs: {missing}")
    bindings = packet.get("bindings", {})
    current = {
        "resource_audit_sha256": review_module.sha256(resource),
        "corrected_resource_audit_sha256": review_module.sha256(corrected),
        "model_identity_sha256": review_module.sha256(identity),
        "release_tool_sha256": review_module.sha256(release_tool),
        "review_builder_sha256": review_module.sha256(review_builder),
        "transition_tool_sha256": review_module.sha256(transition_tool),
        "plan_sha256": review_module.sha256(transition_plan.resolve()),
    }
    if bindings != current:
        raise ValueError("review packet inputs or tools changed")
    recomputed = review_module.validated_review(
        resource_audit=resource, corrected_resource_audit=corrected,
        model_identity=identity, release_tool=release_tool,
    )
    if recomputed != embedded:
        raise ValueError("review content differs from independent recomputation")
    if not accept:
        raise PermissionError("--accept is required; no release was written")
    if not reviewed_by.strip() or not decision_note.strip():
        raise ValueError("--reviewed-by and --decision-note must be nonempty")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as temporary:
        staged = Path(temporary) / "generic_release.json"
        release = generic.create_release(
            resource_audit=resource, output=staged, accept=True,
            reviewed_by=reviewed_by, decision_note=decision_note,
        )
        release.update({
            "corrected_base_release_protocol": CORRECTED_RELEASE_PROTOCOL,
            "model_revision": review_module.MODEL_REVISION,
            "corrected_resource_audit_path": str(corrected),
            "corrected_resource_audit_sha256": review_module.sha256(corrected),
            "review_packet_path": str(review_packet),
            "review_packet_sha256": review_module.sha256(review_packet),
            "release_transition_plan_path": str(transition_plan.resolve()),
            "release_transition_plan_sha256": review_module.sha256(
                transition_plan.resolve()
            ),
        })
        partial.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output)
    return release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--transition-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--reviewed-by", default="")
    parser.add_argument("--decision-note", default="")
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_from_review(
        review_packet=args.review_packet.resolve(),
        transition_plan=args.transition_plan.resolve(),
        output=args.output.resolve(), accept=args.accept,
        reviewed_by=args.reviewed_by, decision_note=args.decision_note,
        expected_hostname=args.expected_hostname,
    )
    print(json.dumps({
        "accepted": result["accepted"],
        "output": str(args.output.resolve()),
        "release_sha256": review_module.sha256(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
