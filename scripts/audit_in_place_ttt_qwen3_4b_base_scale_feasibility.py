#!/usr/bin/env python3
"""Recompute the corrected Base 20B+15B scale-feasibility conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts import summarize_in_place_ttt_qwen3_4b_base_scale_feasibility as reporter
except ModuleNotFoundError:  # Direct script execution.
    import summarize_in_place_ttt_qwen3_4b_base_scale_feasibility as reporter


PROTOCOL = "in_place_ttt_qwen3_4b_base_scale_feasibility_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    required = [args.stage1_audit, args.stage2_audit, args.plan, args.reporter,
                args.summary_json, args.report_md]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}
    errors: list[str] = []
    if args.reporter.resolve() != Path(reporter.__file__).resolve():
        errors.append("reporter path differs from imported implementation")
    try:
        recomputed = reporter.summarize(
            args.stage1_audit, args.stage2_audit, args.plan
        )
    except Exception as exc:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"independent recomputation failed: {exc}"]}
    observed = json.loads(args.summary_json.read_text(encoding="utf-8"))
    if observed != recomputed:
        errors.append("saved feasibility JSON differs from recomputation")
    if args.report_md.read_text(encoding="utf-8") != reporter.render_markdown(
            recomputed):
        errors.append("saved feasibility Markdown differs from recomputation")
    if (observed.get("completed") is not True
            or observed.get("matched_pair_processed_tokens") != 70_000_000_000
            or observed.get("paper_scale_executed") is not False
            or observed.get("projection_only") is not True
            or observed.get("gate_experiment_started") is not False):
        errors.append("scale or execution claim differs")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "summary_json_sha256": sha256(args.summary_json),
        "report_md_sha256": sha256(args.report_md),
        "bindings": {
            "corrected_stage1_timing_audit_sha256": sha256(args.stage1_audit),
            "corrected_stage2_timing_audit_sha256": sha256(args.stage2_audit),
            "plan_sha256": sha256(args.plan),
            "reporter_sha256": sha256(args.reporter),
            "auditor_sha256": sha256(Path(__file__).resolve()),
        },
        "paper_scale_executed": False,
        "projection_only": True,
        "gate_experiment_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage1_audit", "stage2_audit", "plan", "reporter",
                 "summary_json", "report_md", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args = argparse.Namespace(**{
        key: value.expanduser().resolve() for key, value in vars(args).items()
    })
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
