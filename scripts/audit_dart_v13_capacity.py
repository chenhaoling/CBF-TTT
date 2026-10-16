"""Independently reconstruct the DART V13 capacity decision and file hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.inspect_dart_v13_capacity import inspect
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-episodes", type=Path, action="append", required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed = json.loads((args.capacity_root / "capacity.json").read_text())
    reconstructed, rows, materials, removed = inspect(
        args.source_root, args.exclude_episodes, observed["seed"],
    )
    checks = {
        "capacity_decision_reconstructed": all(
            observed.get(key) == reconstructed.get(key)
            for key in (
                "protocol", "passed", "checks", "seed", "source_commit",
                "source_manifest_sha256", "train_sha256", "source_summary",
                "prior_corpus_exclusions", "counts", "thresholds",
                "target_candidates_ranked", "selected_target_relations",
                "background_relations", "allocation_error", "capacity",
                "model_loaded", "sealed_benchmarks_loaded", "next_stage",
                "terminal_status",
            )
        ),
        "eligible_rows_exact": read_jsonl(args.capacity_root / "eligible_rows.jsonl") == rows,
        "materials_exact": read_jsonl(args.capacity_root / "materials.jsonl") == materials,
        "exclusions_exact": read_jsonl(args.capacity_root / "exclusions.jsonl") == removed,
        "registered_file_hashes_match": all(
            sha256(args.capacity_root / name) == digest
            for name, digest in observed["files_sha256"].items()
        ),
        "no_model_or_sealed_benchmark_access": not observed["model_loaded"]
            and not observed["sealed_benchmarks_loaded"],
    }
    audit = {
        "protocol": "qwen_utility_record_controller_v13_dart_capacity_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "capacity_passed": observed["passed"],
        "terminal_status": "v13_dart_capacity_audit_passed"
            if all(checks.values()) else "v13_dart_capacity_audit_failed",
    }
    write_json(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
