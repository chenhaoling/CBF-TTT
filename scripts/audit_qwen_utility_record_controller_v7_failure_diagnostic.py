"""Recompute and audit the V7 cross-source failure diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import (
    PROTOCOL,
    add_arguments,
    diagnose,
)


AUDIT_PROTOCOL = "qwen_utility_record_controller_v7_failure_diagnostic_audit"


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
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
    return left == right


def audit(args) -> dict:
    path = Path(args.diagnostic)
    observed = json.loads(path.read_text())
    reconstructed = diagnose(args)
    # Runtime is intentionally excluded from deterministic equivalence.
    observed_without_time = {k: v for k, v in observed.items() if k != "seconds"}
    reconstructed_without_time = {
        k: v for k, v in reconstructed.items() if k != "seconds"
    }
    checks = {
        "protocol_and_plan_locked": (
            observed.get("protocol") == PROTOCOL
            and observed.get("plan")
            == "QWEN_UTILITY_RECORD_CONTROLLER_V7_FAILURE_DIAGNOSTIC_PLAN.md"
        ),
        "complete_summary_reconstructed": close(
            observed_without_time, reconstructed_without_time
        ),
        "diagnosis_reconstructed": (
            observed.get("diagnosis") == reconstructed.get("diagnosis")
        ),
        "heldout_sources_sealed": all(
            not observed[name] for name in (
                "templama_test_loaded", "lama_confirmation_loaded",
                "longmemeval_loaded", "locomo_loaded",
            )
        ),
        "diagnostic_integrity_passed": observed.get("passed") is True,
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": passed,
        "checks": checks,
        "diagnostic_sha256": sha256(path),
        "diagnosis": observed.get("diagnosis"),
        "next_stage": observed.get("next_stage") if passed else "stop",
        "terminal_status": (
            "v7_failure_diagnostic_audit_passed" if passed
            else "v7_failure_diagnostic_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--v7-training-root", required=True)
    parser.add_argument("--v7-audit", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
