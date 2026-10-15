"""Independent audit of V1-controller features on TempLAMA development."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from scripts.audit_qwen_utility_record_controller_v1_features import (
    close,
    deterministic_features_are_valid,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
)
from tasks.extract_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL,
    EXPECTED_LABELS,
    PROTOCOL,
)
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES,
    teacher_targets,
)
from tasks.label_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(feature_root: Path, data_root: Path, label_root: Path) -> dict:
    summary = json.loads((feature_root / "summary.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    label_audit = json.loads(
        (label_root / "development_labels_audit.json").read_text()
    )
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    labels = []; rows = []; resources = []
    for shard in (0, 1):
        label_path = label_root / f"labels_shard{shard}.jsonl"
        feature_path = feature_root / f"features_shard{shard}.jsonl"
        resource = json.loads((feature_root / f"resource_shard{shard}.json").read_text())
        labels.extend(load_jsonl(label_path)); part = load_jsonl(feature_path)
        if resource["rows_sha256"] != sha256(feature_path) or resource["boundaries"] != len(part):
            raise ValueError("development feature shard hash/count mismatch")
        if resource["label_shard_sha256"] != sha256(label_path):
            raise ValueError("development feature references another label shard")
        rows.extend(part); resources.append(resource)
    labels_by_id = {(row["episode_id"], row["boundary_index"]): row for row in labels}
    rows_by_id = {(row["episode_id"], row["boundary_index"]): row for row in rows}
    coverage = set(labels_by_id) == set(rows_by_id) and len(rows_by_id) == len(rows)
    targets_valid = True; deterministic_valid = True; schema_valid = True; finite = True
    for identity, row in rows_by_id.items():
        label = labels_by_id[identity]
        targets_valid &= row["targets"] == teacher_targets(label)
        targets_valid &= row["candidate_record_ids"] == label["candidate_record_ids"]
        schema_valid &= tuple(row["feature_names"]) == FEATURE_NAMES
        schema_valid &= all(len(values) == len(FEATURE_NAMES) for values in row["features"])
        finite &= all(math.isfinite(value) for values in row["features"] for value in values)
        deterministic_valid &= deterministic_features_are_valid(
            episode_by_id[row["episode_id"]], label, row
        )
    deletes = [value for row in rows for value in row["targets"]["delete_targets"]]
    risks = [value for row in rows for value in row["targets"]["risk_targets"]]
    checks = {
        "complete_one_to_one_coverage": coverage and len(rows) == EXPECTED_LABELS,
        "exact_binary_teachers_and_risks": targets_valid,
        "fixed_numeric_schema": schema_valid and finite,
        "deterministic_and_context_features": deterministic_valid,
        "valid_binary_targets": (
            bool(deletes) and bool(risks)
            and set(deletes) <= {0, 1} and set(risks) <= {0, 1}
        ),
        "locked_model_and_inputs": all(
            data_audit.get("protocol") == DATA_AUDIT_PROTOCOL
            and data_audit.get("passed") is True
            and label_audit.get("protocol") == LABEL_AUDIT_PROTOCOL
            and label_audit.get("passed") is True
            and resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["data_manifest_sha256"] == sha256(data_root / "manifest.json")
            and resource["data_audit_sha256"]
            == sha256(data_root / "data_audit.json")
            and resource["label_summary_sha256"] == sha256(label_root / "summary.json")
            and resource["label_audit_sha256"]
            == sha256(label_root / "development_labels_audit.json")
            for resource in resources
        ),
        "summary_reconstruction": (
            summary["protocol"] == PROTOCOL and summary["boundaries"] == len(rows)
            and summary["record_examples"] == len(deletes)
            and close(summary["delete_fraction"], statistics.fmean(deletes))
            and close(summary["risk_fraction"], statistics.fmean(risks))
            and summary["safe_binary_actions"] == dict(Counter(
                row["targets"]["safe_binary_action"] for row in rows
            ))
            and summary["delete_values"] == sorted(set(deletes))
            and summary["risk_values"] == sorted(set(risks))
        ),
        "heldouts_sealed": all(
            not resource["templama_test_loaded"]
            and not resource["v1_confirmation_loaded"]
            and not resource["external_benchmarks_loaded"]
            for resource in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "boundaries": len(rows),
        "record_examples": len(deletes),
        "feature_dimension": len(FEATURE_NAMES),
        "delete_fraction": statistics.fmean(deletes),
        "risk_fraction": statistics.fmean(risks),
        "feature_summary_sha256": sha256(feature_root / "summary.json"),
        "templama_development_label_audit_sha256": sha256(
            label_root / "development_labels_audit.json"
        ),
        "next_stage": "templama_development_controller_evaluation" if passed else "stop",
        "terminal_status": (
            "templama_development_features_audit_passed" if passed
            else "templama_development_features_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.feature_root, args.data_root, args.label_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
