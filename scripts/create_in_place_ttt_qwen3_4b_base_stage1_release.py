#!/usr/bin/env python3
"""Create an explicit release bound to both corrected Base timing audits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.create_in_place_ttt_2x5090_timing_release import review, sha256
except ModuleNotFoundError:  # Direct script execution.
    from create_in_place_ttt_2x5090_timing_release import review, sha256


MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CORRECTED_PROTOCOL = "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1"
RELEASE_PROTOCOL = "in_place_ttt_2x5090_100m_stage1_release_v1"
SELECTED_STREAM_PROTOCOL = "in_place_ttt_qwen3_4b_base_100m_selected_stream_audit_v1"


def selected_stream_review(path: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    stage = audit.get("independent_stages", {}).get("stage1_32k", {})
    if (audit.get("protocol") != SELECTED_STREAM_PROTOCOL
            or audit.get("passed") is not True
            or audit.get("errors") != []
            or not audit.get("checks")
            or not all(audit["checks"].values())
            or stage.get("selected_records") != 1728
            or stage.get("selected_training_tokens") != 56_623_104):
        raise ValueError("corrected selected-stream audit is invalid or stale")
    return {
        "selected_stream_audit_path": str(path.resolve()),
        "selected_stream_audit_sha256": sha256(path),
        "selected_stream_audit_protocol": SELECTED_STREAM_PROTOCOL,
        "selected_stream_corpus_manifest_sha256":
            audit.get("bindings", {}).get("corpus_manifest_sha256"),
        "selected_stream_stage": "stage1_32k",
        "selected_stream_ordered_stage_indices_sha256":
            stage.get("ordered_stage_indices_sha256"),
        "selected_stream_ordered_training_hashes_sha256":
            stage.get("ordered_training_hashes_sha256"),
    }


def corrected_review(
    timing_audit: Path, corrected_timing_audit: Path
) -> dict[str, Any]:
    summary = review("stage1", timing_audit)
    corrected = json.loads(corrected_timing_audit.read_text(encoding="utf-8"))
    errors: list[str] = []
    if (corrected.get("protocol") != CORRECTED_PROTOCOL
            or corrected.get("passed") is not True
            or corrected.get("errors") != []
            or corrected.get("model_revision") != MODEL_REVISION
            or corrected.get("formal_release_created") is not False
            or corrected.get("underlying_timing_audit_sha256")
            != sha256(timing_audit)):
        errors.append("corrected Base timing audit is invalid or stale")
    if corrected.get("projected_sequential_stage1_seconds") \
            != summary["projected_sequential_seconds"]:
        errors.append("corrected and underlying timing projections differ")
    if errors:
        raise ValueError("; ".join(errors))
    return summary | {
        "corrected_timing_audit_path": str(corrected_timing_audit.resolve()),
        "corrected_timing_audit_sha256": sha256(corrected_timing_audit),
        "corrected_timing_audit_protocol": CORRECTED_PROTOCOL,
        "model_revision": MODEL_REVISION,
    }


def create_release(
    *,
    timing_audit: Path,
    corrected_timing_audit: Path,
    selected_stream_audit: Path,
    output: Path,
    accept: bool,
    reviewed_by: str,
    decision_note: str,
) -> dict[str, Any]:
    summary = (corrected_review(timing_audit, corrected_timing_audit)
               | selected_stream_review(selected_stream_audit))
    if not accept:
        raise PermissionError("--accept is required; no release was written")
    if not reviewed_by.strip() or not decision_note.strip():
        raise ValueError("--reviewed-by and --decision-note must be nonempty")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    release = {
        "protocol": RELEASE_PROTOCOL,
        "accepted": True,
        "stage": "stage1",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_by": reviewed_by.strip(),
        "decision_note": decision_note.strip(),
        **summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--corrected-timing-audit", type=Path, required=True)
    parser.add_argument("--selected-stream-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--reviewed-by", default="")
    parser.add_argument("--decision-note", default="")
    args = parser.parse_args()
    timing = args.timing_audit.expanduser().resolve()
    corrected = args.corrected_timing_audit.expanduser().resolve()
    selected_stream = args.selected_stream_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = (corrected_review(timing, corrected)
               | selected_stream_review(selected_stream))
    if not args.accept:
        print(json.dumps({
            "accepted": False,
            "release_written": False,
            "review": summary,
            "next_action": "review the measured projection and rerun with --accept",
        }, indent=2, sort_keys=True))
        raise SystemExit(2)
    result = create_release(
        timing_audit=timing, corrected_timing_audit=corrected,
        selected_stream_audit=selected_stream,
        output=output, accept=True, reviewed_by=args.reviewed_by,
        decision_note=args.decision_note,
    )
    print(json.dumps({
        "accepted": True,
        "release_written": True,
        "output": str(output),
        "release_sha256": sha256(output),
        "projected_sequential_hours": result["projected_sequential_hours"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
