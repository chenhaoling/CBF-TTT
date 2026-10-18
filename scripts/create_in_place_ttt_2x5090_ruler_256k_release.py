#!/usr/bin/env python3
"""Review the 256K resource audit and explicitly release formal scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_PROTOCOL = "in_place_ttt_2x5090_ruler_256k_resource_audit_v1"
RELEASE_PROTOCOL = "in_place_ttt_2x5090_ruler_256k_formal_release_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def review(resource_audit: Path) -> dict[str, Any]:
    audit = json.loads(resource_audit.read_text())
    if (audit.get("protocol") != AUDIT_PROTOCOL
            or audit.get("passed") is not True or audit.get("errors") != []
            or audit.get("context_length") != 262144
            or audit.get("extrapolation") is not True
            or audit.get("scores_discarded") is not True
            or audit.get("formal_256k_released") is not True
            or set(audit.get("boundaries", {})) != {"no_ttt", "ttt"}):
        raise ValueError("256K resource audit did not release formal scoring")
    return {
        "resource_audit_path": str(resource_audit.resolve()),
        "resource_audit_sha256": sha256(resource_audit),
        "context_length": 262144,
        "arms": ["no_ttt", "ttt"],
    }


def create_release(
    *, resource_audit: Path, output: Path, accept: bool,
    reviewed_by: str, decision_note: str,
) -> dict[str, Any]:
    summary = review(resource_audit)
    if not accept:
        raise PermissionError("review passed but --accept is required")
    if not reviewed_by.strip() or not decision_note.strip():
        raise ValueError("reviewer and decision note must be nonempty")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    release = {
        "protocol": RELEASE_PROTOCOL,
        "accepted": True,
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
    parser.add_argument("--resource-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--reviewed-by", default="")
    parser.add_argument("--decision-note", default="")
    args = parser.parse_args()
    resource_audit = args.resource_audit.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = review(resource_audit)
    if not args.accept:
        print(json.dumps({"accepted": False, "release_written": False,
                          "review": summary}, indent=2, sort_keys=True))
        raise SystemExit(2)
    create_release(
        resource_audit=resource_audit, output=output, accept=True,
        reviewed_by=args.reviewed_by, decision_note=args.decision_note,
    )
    print(json.dumps({"accepted": True, "release_written": True,
                      "output": str(output), "release_sha256": sha256(output)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
