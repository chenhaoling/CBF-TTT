#!/usr/bin/env python3
"""Create a non-authorizing review packet for a corrected formal release."""

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
    from scripts import create_in_place_ttt_qwen3_4b_base_stage1_release as stage1_release
    from scripts import create_in_place_ttt_qwen3_4b_base_stage2_release as stage2_release
except ModuleNotFoundError:  # Direct script execution.
    import create_in_place_ttt_qwen3_4b_base_stage1_release as stage1_release
    import create_in_place_ttt_qwen3_4b_base_stage2_release as stage2_release


PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_release_review_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def build_review(
    *,
    stage: str,
    timing_audit: Path,
    corrected_timing_audit: Path,
    selected_stream_audit: Path,
    release_tool: Path,
    plan: Path,
    target_release: Path,
    output: Path,
    expected_hostname: str,
) -> dict[str, Any]:
    if platform.node() != expected_hostname:
        raise RuntimeError("formal-release review is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("formal-release review must hide all GPUs")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    if target_release.exists() or target_release.with_name(target_release.name + ".partial").exists():
        raise FileExistsError("target formal release exists before review packet")

    module = {"stage1": stage1_release, "stage2": stage2_release}.get(stage)
    if module is None:
        raise ValueError("stage must be stage1 or stage2")
    if Path(module.__file__).resolve() != release_tool.resolve():
        raise ValueError("release-tool path differs from the imported stage validator")
    timing = module.corrected_review(timing_audit, corrected_timing_audit)
    selected = module.selected_stream_review(selected_stream_audit)
    review = timing | selected
    if review.get("model_revision") != MODEL_REVISION:
        raise ValueError("review resolved another model revision")

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stage": stage,
        "ready_for_human_review": True,
        "formal_release_created": False,
        "target_release_path": str(target_release.resolve()),
        "review": review,
        "bindings": {
            "review_builder_sha256": sha256(Path(__file__).resolve()),
            "release_tool_sha256": sha256(release_tool.resolve()),
            "plan_sha256": sha256(plan.resolve()),
            "timing_audit_sha256": sha256(timing_audit.resolve()),
            "corrected_timing_audit_sha256": sha256(corrected_timing_audit.resolve()),
            "selected_stream_audit_sha256": sha256(selected_stream_audit.resolve()),
        },
        "required_human_fields": ["accept", "reviewed_by", "decision_note"],
        "next_action": (
            "inspect the measured projection, then separately invoke the "
            "stage-specific release creator if and only if accepted"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--corrected-timing-audit", type=Path, required=True)
    parser.add_argument("--selected-stream-audit", type=Path, required=True)
    parser.add_argument("--release-tool", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--target-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_review(
        stage=args.stage,
        timing_audit=args.timing_audit.resolve(),
        corrected_timing_audit=args.corrected_timing_audit.resolve(),
        selected_stream_audit=args.selected_stream_audit.resolve(),
        release_tool=args.release_tool.resolve(),
        plan=args.plan.resolve(),
        target_release=args.target_release.resolve(),
        output=args.output.resolve(),
        expected_hostname=args.expected_hostname,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "stage": result["stage"],
        "ready_for_human_review": result["ready_for_human_review"],
        "formal_release_created": result["formal_release_created"],
        "projected_sequential_hours": result["review"]["projected_sequential_hours"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
