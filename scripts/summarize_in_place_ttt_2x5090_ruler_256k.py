#!/usr/bin/env python3
"""Summarize the paired formal public-data RULER-256K extrapolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("no_ttt", "ttt")
PAPER = {"no_ttt": 41.7, "ttt": 43.9}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    records: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for arm in ARMS:
        summary_path = root / arm / "summary.json"
        audit_path = root / arm / "independent_audit.json"
        if not summary_path.is_file() or not audit_path.is_file():
            errors.append(f"{arm}: formal summary/audit is missing")
            continue
        summary = json.loads(summary_path.read_text())
        audit = json.loads(audit_path.read_text())
        if (summary.get("protocol") != "in_place_ttt_2x5090_ruler_summary_v1"
                or summary.get("completed") is not True
                or summary.get("arm") != arm
                or summary.get("context_length") != 262144
                or audit.get("protocol") != "in_place_ttt_2x5090_ruler_audit_v1"
                or audit.get("passed") is not True or audit.get("errors") != []
                or audit.get("arm") != arm
                or audit.get("context_length") != 262144
                or audit.get("extrapolation") is not True):
            errors.append(f"{arm}: formal 256K identity/completion differs")
        if len(summary.get("task_scores", {})) != 13 \
                or len(summary.get("evaluation_input_sha256", {})) != 13:
            errors.append(f"{arm}: formal 256K task/input count differs")
        records[arm] = summary
        bindings[arm] = {
            "summary_sha256": sha256(summary_path),
            "independent_audit_sha256": sha256(audit_path),
        }
    scores = None
    paper_comparison = None
    if set(records) == set(ARMS):
        no_ttt, ttt = records["no_ttt"], records["ttt"]
        if no_ttt.get("evaluation_input_sha256") != ttt.get("evaluation_input_sha256"):
            errors.append("paired formal 256K inputs differ")
        no_score = float(no_ttt["macro_accuracy"])
        ttt_score = float(ttt["macro_accuracy"])
        scores = {
            "no_ttt": no_score,
            "ttt": ttt_score,
            "ttt_minus_no_ttt": ttt_score - no_score,
        }
        paper_comparison = {
            "public_no_ttt_minus_paper_matched": no_score - PAPER["no_ttt"],
            "public_ttt_minus_paper_ttt": ttt_score - PAPER["ttt"],
            "public_gain_minus_paper_gain": (
                ttt_score - no_score - (PAPER["ttt"] - PAPER["no_ttt"])
            ),
        }
    return {
        "protocol": "in_place_ttt_2x5090_ruler_256k_summary_v1",
        "completed": not errors and scores is not None,
        "errors": errors,
        "context_length": 262144,
        "extrapolation": True,
        "scores": scores,
        "paper_table1_reference": PAPER,
        "public_replication_minus_paper": paper_comparison,
        "claim_boundary": "public_data_yarn_extrapolation_not_paper_exact",
        "bindings": bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root.resolve())
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "completed": result["completed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["completed"] else 1)


if __name__ == "__main__":
    main()
