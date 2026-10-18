"""Review an audited timing pilot and explicitly release formal 2x5090 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = {
    "stage1": {
        "audit_protocol": "in_place_ttt_2x5090_stage1_timing_audit_v1",
        "release_protocol": "in_place_ttt_2x5090_100m_stage1_release_v1",
        "paired_key": "projected_sequential_stage1_seconds",
        "arm_key": "projected_formal_stage1_seconds",
    },
    "stage2": {
        "audit_protocol": "in_place_ttt_2x5090_stage2_timing_audit_v1",
        "release_protocol": "in_place_ttt_2x5090_100m_stage2_release_v1",
        "paired_key": "projected_sequential_stage2_seconds",
        "arm_key": "projected_formal_stage2_seconds",
    },
}
ARMS = ("no_ttt", "ttt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def review(stage: str, timing_audit: Path) -> dict[str, Any]:
    spec = STAGES[stage]
    audit = json.loads(timing_audit.read_text(encoding="utf-8"))
    errors: list[str] = []
    if audit.get("protocol") != spec["audit_protocol"]:
        errors.append("timing audit protocol mismatch")
    if audit.get("passed") is not True or audit.get("errors") != []:
        errors.append("timing audit did not pass cleanly")
    arms = audit.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        errors.append("timing audit does not contain exactly both registered arms")
        arms = {}
    arm_projections: dict[str, float] = {}
    for arm in ARMS:
        record = arms.get(arm, {})
        projection = record.get(spec["arm_key"])
        steady = record.get("warm_excluded_median_seconds")
        if not positive_finite(projection) or not positive_finite(steady):
            errors.append(f"{arm} timing projection or steady timing is invalid")
            continue
        arm_projections[arm] = float(projection)
    paired = audit.get(spec["paired_key"])
    if not positive_finite(paired):
        errors.append("paired sequential projection is invalid")
    elif len(arm_projections) == len(ARMS) and not math.isclose(
        float(paired), sum(arm_projections.values()), rel_tol=1e-9, abs_tol=1e-6
    ):
        errors.append("paired projection does not equal the two arm projections")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "stage": stage,
        "timing_audit_path": str(timing_audit.resolve()),
        "timing_audit_sha256": sha256(timing_audit),
        "timing_audit_protocol": spec["audit_protocol"],
        "arm_projected_seconds": arm_projections,
        "projected_sequential_seconds": float(paired),
        "projected_sequential_hours": float(paired) / 3600.0,
    }


def create_release(
    *,
    stage: str,
    timing_audit: Path,
    output: Path,
    accept: bool,
    reviewed_by: str,
    decision_note: str,
) -> dict[str, Any]:
    summary = review(stage, timing_audit)
    if not accept:
        raise PermissionError(
            "review passed but --accept is required; no release was written"
        )
    if not reviewed_by.strip() or not decision_note.strip():
        raise ValueError("--reviewed-by and --decision-note must be nonempty")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    release = {
        "protocol": STAGES[stage]["release_protocol"],
        "accepted": True,
        "stage": stage,
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
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--timing-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--reviewed-by", default="")
    parser.add_argument("--decision-note", default="")
    args = parser.parse_args()
    timing_audit = args.timing_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = review(args.stage, timing_audit)
    if not args.accept:
        print(json.dumps({
            "accepted": False,
            "release_written": False,
            "review": summary,
            "next_action": "review the projection and rerun with --accept",
        }, indent=2, sort_keys=True))
        raise SystemExit(2)
    result = create_release(
        stage=args.stage,
        timing_audit=timing_audit,
        output=output,
        accept=True,
        reviewed_by=args.reviewed_by,
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
