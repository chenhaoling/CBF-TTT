"""Recompute the frozen V10 failure diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v10_three_source import (
    PLAN, PROTOCOL, add_arguments, diagnose,
)


AUDIT_PROTOCOL = "qwen_utility_record_controller_v10_failure_diagnostic_audit"


def close(left, right, tolerance=1e-10):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0,
                            abs_tol=tolerance)
    return left == right


def audit(args) -> dict:
    path = Path(args.diagnostic)
    observed = json.loads(path.read_text())
    reconstructed = diagnose(args)
    observed_fixed = {key: value for key, value in observed.items()
                      if key != "seconds"}
    reconstructed_fixed = {key: value for key, value in reconstructed.items()
                           if key != "seconds"}
    checks = {
        "protocol_plan_and_integrity_locked": (
            observed.get("protocol") == PROTOCOL
            and observed.get("plan") == PLAN
            and observed.get("passed") is True
        ),
        "complete_diagnostic_reconstructed": close(
            observed_fixed, reconstructed_fixed
        ),
        "diagnosis_reconstructed": (
            observed.get("diagnosis") == reconstructed.get("diagnosis")
        ),
        "no_refit_or_new_selection_outputs": (
            observed.get("next_stage") == "close_v10_constructed_corpus_branch"
        ),
        "heldout_sources_sealed": all(
            not observed[name] for name in (
                "templama_test_loaded", "lama_confirmation_loaded",
                "longmemeval_loaded", "locomo_loaded",
            )
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed, "checks": checks,
        "diagnostic_sha256": sha256(path),
        "diagnosis": observed.get("diagnosis"),
        "next_stage": observed.get("next_stage") if passed else "stop",
        "terminal_status": (
            "v10_failure_diagnostic_audit_passed" if passed
            else "v10_failure_diagnostic_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
