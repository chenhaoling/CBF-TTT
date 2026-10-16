"""Independently rerun and audit the deterministic V11 failure diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.diagnose_qwen_utility_record_controller_v11_relation_transfer import (
    AUDIT_PROTOCOL, add_arguments, diagnose,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); observed = json.loads(Path(args.diagnostic).read_text())
    expected = diagnose(args)
    checks = {
        "diagnostic_exactly_reconstructed": observed == expected,
        "registered_conclusion": observed.get("conclusion")
            == "constructed_corpus_effect_supported_robust_loss_not_supported",
        "all_tests_still_sealed": not any(observed[key] for key in (
            "templama_development_loaded", "lama_development_loaded",
            "internal_test_loaded", "external_test_loaded",
            "longmemeval_loaded", "locomo_loaded")),
    }
    result = {
        "protocol": AUDIT_PROTOCOL, "passed": all(checks.values()),
        "checks": checks, "diagnostic_sha256": sha256(Path(args.diagnostic)),
        "terminal_status": "v11_failure_diagnostic_audit_passed"
                           if all(checks.values()) else "v11_failure_diagnostic_audit_failed",
    }
    write_json(Path(args.output), result); print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
