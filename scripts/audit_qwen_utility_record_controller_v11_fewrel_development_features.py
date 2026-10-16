"""Independently audit V11 FewRel development 66-feature rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from scripts.audit_qwen_utility_record_controller_v1_features import (
    close, deterministic_features_are_valid,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v11_fewrel_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL, EXPECTED_RELATIONS,
)
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES, teacher_targets,
)
from tasks.extract_qwen_utility_record_controller_v11_fewrel_development_features import (
    AUDIT_PROTOCOL, EXPECTED_LABELS, PROTOCOL,
)
from tasks.label_qwen_utility_record_controller_v11_fewrel_development import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(feature_root: Path, data_root: Path, data_audit_path: Path,
          label_root: Path) -> dict:
    summary = json.loads((feature_root / "summary.json").read_text())
    data_audit = json.loads(data_audit_path.read_text())
    label_audit_path = label_root / "formal_audit.json"
    label_audit = json.loads(label_audit_path.read_text())
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    labels, rows, resources = [], [], []
    for shard in (0, 1):
        label_path = label_root / f"labels_shard{shard}.jsonl"
        feature_path = feature_root / f"features_shard{shard}.jsonl"
        resource = json.loads(
            (feature_root / f"resource_shard{shard}.json").read_text()
        )
        labels.extend(load_jsonl(label_path)); part = load_jsonl(feature_path)
        if resource["rows_sha256"] != sha256(feature_path) \
                or resource["boundaries"] != len(part):
            raise ValueError("V11 FewRel development feature shard hash/count mismatch")
        if resource["label_shard_sha256"] != sha256(label_path):
            raise ValueError("V11 FewRel development feature references another label shard")
        rows.extend(part); resources.append(resource)
    labels_by_id = {(row["episode_id"], row["boundary_index"]): row
                    for row in labels}
    rows_by_id = {(row["episode_id"], row["boundary_index"]): row
                  for row in rows}
    coverage = set(labels_by_id) == set(rows_by_id) \
        and len(rows_by_id) == len(rows)
    targets_ok = deterministic_ok = schema_ok = finite = True
    for identity, row in rows_by_id.items():
        label = labels_by_id[identity]
        targets_ok &= row["targets"] == teacher_targets(label)
        targets_ok &= row["candidate_record_ids"] == label["candidate_record_ids"]
        schema_ok &= tuple(row["feature_names"]) == FEATURE_NAMES
        schema_ok &= all(len(values) == len(FEATURE_NAMES)
                         for values in row["features"])
        finite &= all(math.isfinite(value) for values in row["features"]
                      for value in values)
        deterministic_ok &= deterministic_features_are_valid(
            episode_by_id[row["episode_id"]], label, row
        )
    deletes = [value for row in rows
               for value in row["targets"]["delete_targets"]]
    risks = [value for row in rows for value in row["targets"]["risk_targets"]]
    relations = Counter(row["audit"]["target_relation"] for row in rows)
    checks = {
        "complete_one_to_one_coverage": coverage and len(rows) == EXPECTED_LABELS,
        "exact_binary_teachers_and_risks": bool(targets_ok),
        "fixed_66_numeric_schema": bool(schema_ok and finite),
        "deterministic_and_context_features": bool(deterministic_ok),
        "both_target_classes_present": (
            set(deletes) == {0, 1} and set(risks) == {0, 1}
        ),
        "complete_relation_coverage": (
            len(relations) == EXPECTED_RELATIONS
            and set(relations.values()) == {84}
        ),
        "locked_plain_model_and_inputs": all(
            data_audit.get("protocol") == DATA_AUDIT_PROTOCOL
            and data_audit.get("passed") is True
            and label_audit.get("protocol") == LABEL_AUDIT_PROTOCOL
            and label_audit.get("passed") is True
            and label_audit.get("release_passed") is True
            and resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["data_manifest_sha256"]
            == sha256(data_root / "manifest.json")
            and resource["data_audit_sha256"] == sha256(data_audit_path)
            and resource["label_summary_sha256"]
            == sha256(label_root / "summary.json")
            and resource["label_audit_sha256"] == sha256(label_audit_path)
            for resource in resources
        ),
        "summary_reconstructed": (
            summary["protocol"] == PROTOCOL
            and summary["boundaries"] == len(rows)
            and summary["record_examples"] == len(deletes)
            and summary["feature_dimension"] == len(FEATURE_NAMES)
            and close(summary["delete_fraction"], statistics.fmean(deletes))
            and close(summary["risk_fraction"], statistics.fmean(risks))
            and summary["safe_binary_actions"] == dict(Counter(
                row["targets"]["safe_binary_action"] for row in rows
            ))
            and summary["relation_boundary_counts"]
            == dict(sorted(relations.items()))
        ),
        "heldout_sources_sealed": all(
            not resource["development_labels_loaded"]
            and not resource["development_model_scores_loaded"]
            and not resource["heldout_benchmarks_loaded"]
            and not resource["internal_test_loaded"]
            and not resource["external_test_loaded"]
            for resource in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed, "checks": checks,
        "boundaries": len(rows), "record_examples": len(deletes),
        "feature_dimension": len(FEATURE_NAMES),
        "delete_fraction": statistics.fmean(deletes),
        "risk_fraction": statistics.fmean(risks),
        "feature_summary_sha256": sha256(feature_root / "summary.json"),
        "formal_label_audit_sha256": sha256(label_audit_path),
        "next_stage": (
            "v11_relation_transfer_training" if passed else "stop"
        ),
        "terminal_status": (
            "v11_development_features_audit_passed"
            if passed else "v11_development_features_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.feature_root, args.data_root, args.data_audit,
                   args.label_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
