#!/usr/bin/env python3
"""Create a formal release only from its exact audited review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts import create_in_place_ttt_qwen3_4b_base_stage1_release as stage1_release
    from scripts import create_in_place_ttt_qwen3_4b_base_stage2_release as stage2_release
except ModuleNotFoundError:  # Direct script execution.
    import create_in_place_ttt_qwen3_4b_base_stage1_release as stage1_release
    import create_in_place_ttt_qwen3_4b_base_stage2_release as stage2_release


REVIEW_PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_release_review_v1"
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def create_from_review(
    *,
    review_packet: Path,
    transition_plan: Path,
    output: Path,
    accept: bool,
    reviewed_by: str,
    decision_note: str,
    expected_hostname: str,
) -> dict[str, Any]:
    if platform.node() != expected_hostname:
        raise RuntimeError("reviewed release transition is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("reviewed release transition must hide all GPUs")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    packet = json.loads(review_packet.read_text(encoding="utf-8"))
    if (packet.get("protocol") != REVIEW_PROTOCOL
            or packet.get("ready_for_human_review") is not True
            or packet.get("formal_release_created") is not False
            or packet.get("target_release_path") != str(output)
            or set(packet.get("required_human_fields", []))
            != {"accept", "reviewed_by", "decision_note"}):
        raise ValueError("review packet is invalid or targets another release")
    stage = packet.get("stage")
    module = {"stage1": stage1_release, "stage2": stage2_release}.get(stage)
    if module is None:
        raise ValueError("review packet stage is invalid")
    embedded = packet.get("review", {})
    timing = Path(embedded.get("timing_audit_path", "")).resolve()
    corrected = Path(embedded.get("corrected_timing_audit_path", "")).resolve()
    selected = Path(embedded.get("selected_stream_audit_path", "")).resolve()
    for path in (timing, corrected, selected, transition_plan.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    bindings = packet.get("bindings", {})
    if (bindings.get("timing_audit_sha256") != sha256(timing)
            or bindings.get("corrected_timing_audit_sha256") != sha256(corrected)
            or bindings.get("selected_stream_audit_sha256") != sha256(selected)
            or bindings.get("release_tool_sha256")
            != sha256(Path(module.__file__).resolve())):
        raise ValueError("review packet inputs or release validator changed")
    recomputed = (module.corrected_review(timing, corrected)
                  | module.selected_stream_review(selected))
    if recomputed != embedded or recomputed.get("model_revision") != MODEL_REVISION:
        raise ValueError("review packet content differs from independent recomputation")
    if not accept:
        raise PermissionError("--accept is required; no formal release was written")
    if not reviewed_by.strip() or not decision_note.strip():
        raise ValueError("--reviewed-by and --decision-note must be nonempty")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging."
    ) as temporary:
        staged = Path(temporary) / "release.json"
        release = module.create_release(
            timing_audit=timing,
            corrected_timing_audit=corrected,
            selected_stream_audit=selected,
            output=staged,
            accept=True,
            reviewed_by=reviewed_by,
            decision_note=decision_note,
        )
        release.update({
            "review_packet_path": str(review_packet.resolve()),
            "review_packet_sha256": sha256(review_packet.resolve()),
            "review_packet_protocol": REVIEW_PROTOCOL,
            "release_transition_plan_path": str(transition_plan.resolve()),
            "release_transition_plan_sha256": sha256(transition_plan.resolve()),
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
        output=args.output.resolve(),
        accept=args.accept,
        reviewed_by=args.reviewed_by,
        decision_note=args.decision_note,
        expected_hostname=args.expected_hostname,
    )
    print(json.dumps({
        "accepted": result["accepted"],
        "stage": result["stage"],
        "output": str(args.output.resolve()),
        "release_sha256": sha256(args.output.resolve()),
        "review_packet_sha256": result["review_packet_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
