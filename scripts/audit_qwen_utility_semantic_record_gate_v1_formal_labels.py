"""Independent row-level audit of formal utility counterfactual labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_semantic_record_gate_v1_pilot import (
    bootstrap,
    close,
    expected_actions,
    independent_label,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
    FORMAL_GROUPS,
    PROTOCOL as DATA_PROTOCOL,
)
from tasks.label_qwen_utility_semantic_record_gate_v1_formal import (
    EXPECTED_LABELS,
    PRIMARY_ACTION_SPACE,
    PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_formal_labels_audit"


def audit(label_root: Path, data_root: Path) -> dict:
    summary_path = label_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    manifest_path = data_root / "manifest.json"
    data_audit_path = data_root / "formal_data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    episodes = load_jsonl(data_root / "formal_train_episodes.jsonl")
    expected_identities = {
        (episode["episode_id"], boundary["boundary_index"])
        for episode in episodes for boundary in episode["boundaries"]
    }
    rows, resources = [], []
    for shard in (0, 1):
        row_path = label_root / f"labels_shard{shard}.jsonl"
        resource_path = label_root / f"resource_shard{shard}.json"
        part = load_jsonl(row_path)
        resource = json.loads(resource_path.read_text())
        if resource["rows_sha256"] != sha256(row_path) or resource["labels"] != len(part):
            raise ValueError("formal raw shard hash/count mismatch")
        rows.extend(part); resources.append(resource)

    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    action_valid = True; labels_valid = True; finite = True
    for row in rows:
        expected = expected_actions(row["candidate_count"])
        if set(row["actions"]) != set(expected):
            action_valid = False
            continue
        for name, action in row["actions"].items():
            action_valid &= action["spec"] == expected[name]
            numeric = [action[key] for key in (
                "queries", "correct", "total_answer_nll", "unrelated_queries",
                "unrelated_correct", "mean_memory_mass", "selected_null",
            )]
            finite &= all(math.isfinite(float(value)) for value in numeric)
        label = independent_label(row)
        labels_valid &= label == row["safe_oracle_action"]
        labels_valid &= row["actions"][label]["unrelated_correct"] >= (
            row["actions"][row["retain_action"]]["unrelated_correct"]
        )

    total_queries = sum(row["actions"][row["retain_action"]]["queries"] for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"] for row in rows)
    oracle_accuracy = oracle_correct / total_queries
    retain_accuracy = retain_correct / total_queries
    interval = bootstrap(rows)
    deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"] for row in rows
    ]
    half_rows = [row for row in rows if row["safe_oracle_contains_half"]]
    half_unique = [row for row in half_rows
                   if row["half_vs_endpoint_correct_gain"] > 0
                   or (row["half_vs_endpoint_correct_gain"] == 0
                       and row["half_vs_endpoint_nll_gain"] >= .01)]
    mechanism_actions = {
        mechanism: len({row["safe_oracle_action"] for row in rows
                        if row["mechanism"] == mechanism})
        for mechanism in sorted({row["mechanism"] for row in rows})
    }
    checks = {
        "formal_data_locked_and_audited": (
            manifest.get("protocol") == DATA_PROTOCOL
            and data_audit.get("protocol") == DATA_AUDIT_PROTOCOL
            and data_audit.get("passed") is True
            and all(resource["data_manifest_sha256"] == sha256(manifest_path)
                    and resource["data_audit_sha256"] == sha256(data_audit_path)
                    for resource in resources)
        ),
        "complete_unique_rows": (
            len(rows) == EXPECTED_LABELS
            and identities == expected_identities
            and len(identities) == len(rows)
            and len({row["source_group"] for row in rows}) == FORMAL_GROUPS
        ),
        "actions_and_metrics": action_valid and finite,
        "independent_safe_labels": labels_valid,
        "aggregate_accuracy_and_interval": (
            close(summary["safe_oracle_accuracy"], oracle_accuracy)
            and close(summary["retain_all_accuracy"], retain_accuracy)
            and close(summary["safe_oracle_gain"], oracle_accuracy - retain_accuracy)
            and all(close(a, b) for a, b in zip(
                summary["safe_oracle_gain_group_bootstrap_95ci"], interval
            ))
        ),
        "barrier_reconstruction": (
            min(deltas) >= 0
            and summary["added_unrelated_errors"] == 0
            and summary["additional_unrelated_correct"] == sum(deltas)
            and summary["unrelated_barrier_rowwise_passed"] is True
        ),
        "half_ablation_frozen": (
            summary["half_strength"]["selected_labels"] == len(half_rows)
            and summary["half_strength"]["unique_margin_labels"] == len(half_unique)
            and summary["half_strength"]["primary_action_space"] == PRIMARY_ACTION_SPACE
            and summary["half_strength"]["decision_source"] == "sealed_resource_pilot"
        ),
        "mechanism_action_support": min(mechanism_actions.values()) >= 2,
        "resource_sanity": (
            max(resource["overall_peak_reserved_gib"] for resource in resources) < 24
            and close(summary["resource"]["two_gpu_critical_seconds"],
                      max(resource["total_seconds"] for resource in resources))
        ),
        "frozen_plain_qwen": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["value_scale"] == 8.0 for resource in resources
        ),
        "gate_and_sealing": (
            summary["protocol"] == PROTOCOL
            and summary["partition"] == "train"
            and summary["gate"]["passed"] is True
            and all(summary["gate"]["checks"].values())
            and all(not resource["development_loaded"]
                    and not resource["confirmation_loaded"]
                    and not resource["external_benchmarks_loaded"]
                    for resource in resources)
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "rows": len(rows),
        "future_queries": total_queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": sum(max(0, -delta) for delta in deltas),
        "additional_unrelated_correct": sum(max(0, delta) for delta in deltas),
        "safe_oracle_actions": dict(Counter(row["safe_oracle_action"] for row in rows)),
        "summary_sha256": sha256(summary_path),
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(data_audit_path),
        "next_stage": "controller_feature_construction" if passed else "stop",
        "terminal_status": (
            "formal_labels_audit_passed" if passed else "formal_labels_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.label_root, args.data_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
