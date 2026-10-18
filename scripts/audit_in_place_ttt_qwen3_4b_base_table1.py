#!/usr/bin/env python3
"""Independently recompute the corrected-Base seven-context conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts import summarize_in_place_ttt_qwen3_4b_base_table1 as reporter
except ModuleNotFoundError:  # Direct script execution.
    import summarize_in_place_ttt_qwen3_4b_base_table1 as reporter


PROTOCOL = "in_place_ttt_qwen3_4b_base_table1_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.summary_json, args.report_md, args.corrected_matrix_audit,
        args.corrected_formal_256k_audit,
        args.corrected_primary_resource_audit,
        args.corrected_256k_resource_audit, args.plan, args.reporter,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing required files: {missing}"]}
    errors: list[str] = []
    if args.reporter.resolve() != Path(reporter.__file__).resolve():
        errors.append("reporter path differs from imported implementation")
    try:
        recomputed = reporter.summarize(
            matrix_root=args.matrix_root,
            corrected_matrix_audit=args.corrected_matrix_audit,
            formal_256k_root=args.formal_256k_root,
            corrected_formal_256k_audit=args.corrected_formal_256k_audit,
            corrected_primary_resource_audit=(
                args.corrected_primary_resource_audit
            ),
            corrected_256k_resource_audit=args.corrected_256k_resource_audit,
            plan=args.plan,
        )
    except Exception as exc:  # Preserve the independent audit artifact.
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"independent recomputation failed: {exc}"]}
    observed = json.loads(args.summary_json.read_text(encoding="utf-8"))
    if observed != recomputed:
        errors.append("saved Table 1 JSON differs from independent recomputation")
    expected_markdown = reporter.render_markdown(recomputed)
    if args.report_md.read_text(encoding="utf-8") != expected_markdown:
        errors.append("saved Table 1 Markdown differs from recomputation")
    if (observed.get("protocol") != reporter.PROTOCOL
            or observed.get("completed") is not True
            or observed.get("contexts")
            != [suffix for _, suffix in reporter.CONTEXTS]
            or observed.get("baseline_evidence_complete") is not True
            or observed.get("gate_experiment_started") is not False
            or observed.get("claim_boundary")
            != "public_data_fineweb_edu_longcrawl64_matched_replication_"
               "not_private_paper_corpus_or_scale"):
        errors.append("Table 1 completion or claim boundary differs")
    bootstrap = observed.get("paired_task_bootstrap", {})
    if (bootstrap.get("seed") != reporter.BOOTSTRAP_SEED
            or bootstrap.get("resamples") != reporter.BOOTSTRAP_REPS
            or bootstrap.get("unit") != "paired_ruler_task"):
        errors.append("registered bootstrap configuration differs")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "model_revision": reporter.MODEL_REVISION,
        "summary_json_sha256": sha256(args.summary_json),
        "report_md_sha256": sha256(args.report_md),
        "bindings": {
            "corrected_matrix_audit_sha256": sha256(
                args.corrected_matrix_audit
            ),
            "corrected_formal_256k_audit_sha256": sha256(
                args.corrected_formal_256k_audit
            ),
            "corrected_primary_resource_audit_sha256": sha256(
                args.corrected_primary_resource_audit
            ),
            "corrected_256k_resource_audit_sha256": sha256(
                args.corrected_256k_resource_audit
            ),
            "plan_sha256": sha256(args.plan),
            "reporter_sha256": sha256(args.reporter),
            "auditor_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": observed.get("claim_boundary"),
        "baseline_evidence_complete": not errors,
        "gate_experiment_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "matrix_root", "corrected_matrix_audit", "formal_256k_root",
        "corrected_formal_256k_audit", "corrected_primary_resource_audit",
        "corrected_256k_resource_audit", "plan", "reporter", "summary_json",
        "report_md", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
