"""Independent row and aggregate audit of the TempLAMA V2 resource pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
    PROTOCOL as DATA_PROTOCOL,
)
from tasks.pilot_qwen_utility_record_controller_v2_templama import (
    AUDIT_PROTOCOL,
    EXPECTED_LABELS,
    PROTOCOL,
    bootstrap_group_gain,
    half_strength_decision,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


ALPHAS = (0.0, 0.5, 1.0)


def expected_actions(size: int) -> dict[str, dict]:
    rows = {}
    for values in itertools.product(ALPHAS, repeat=size):
        name = "grid_" + "_".join(
            "05" if value == .5 else str(int(value)) for value in values
        )
        rows[name] = {"name": name, "kind": "grid", "alphas": list(values)}
    rows["global_clear"] = {"name": "global_clear", "kind": "global_clear", "alphas": []}
    return rows


def independent_label(row: dict) -> str:
    baseline = row["actions"][row["retain_action"]]
    eligible = [
        (name, value) for name, value in row["actions"].items()
        if value["unrelated_correct"] >= baseline["unrelated_correct"]
    ]
    return max(eligible, key=lambda item: (
        item[1]["correct"], -item[1]["total_answer_nll"],
        -item[1]["mean_memory_mass"], sum(item[1]["spec"]["alphas"]), item[0],
    ))[0]


def close(left, right, tolerance=1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def audit(root: Path, data_root: Path) -> dict:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    data_manifest_path, data_audit_path = (
        data_root / "manifest.json", data_root / "data_audit.json"
    )
    data_manifest = json.loads(data_manifest_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    expected_identities = {
        (episode["episode_id"], boundary["boundary_index"])
        for episode in episodes for boundary in episode["boundaries"]
    }
    rows = []; resources = []; action_valid = True; label_valid = True; finite = True
    for shard in (0, 1):
        row_path = root / f"labels_shard{shard}.jsonl"
        part = load_jsonl(row_path)
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        if resource["rows_sha256"] != sha256(row_path) or resource["labels"] != len(part):
            raise ValueError("TempLAMA pilot shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
    for row in rows:
        expected = expected_actions(row["candidate_count"])
        action_valid &= row["protocol"] == PROTOCOL and set(row["actions"]) == set(expected)
        for name, value in row["actions"].items():
            action_valid &= value["spec"] == expected[name]
            finite &= all(math.isfinite(float(value[key])) for key in (
                "queries", "correct", "total_answer_nll", "unrelated_queries",
                "unrelated_correct", "mean_memory_mass", "selected_null",
            ))
        label_valid &= row["safe_oracle_action"] == independent_label(row)

    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    queries = sum(row["actions"][row["retain_action"]]["queries"] for row in rows)
    oracle_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in rows)
    retain_correct = sum(row["actions"][row["retain_action"]]["correct"] for row in rows)
    oracle_accuracy, retain_accuracy = oracle_correct / queries, retain_correct / queries
    interval = bootstrap_group_gain(rows)
    unrelated_deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"] for row in rows
    ]
    half = half_strength_decision(rows)
    mechanism_actions = {
        mechanism: len({row["safe_oracle_action"] for row in rows
                        if row["mechanism"] == mechanism})
        for mechanism in {row["mechanism"] for row in rows}
    }
    checks = {
        "data_audit_and_provenance": (
            data_manifest.get("protocol") == DATA_PROTOCOL
            and data_audit.get("protocol") == DATA_AUDIT_PROTOCOL
            and data_audit.get("passed") is True
            and all(resource["data_manifest_sha256"] == sha256(data_manifest_path)
                    and resource["data_audit_sha256"] == sha256(data_audit_path)
                    for resource in resources)
        ),
        "complete_unique_rows": (
            len(rows) == EXPECTED_LABELS and identities == expected_identities
        ),
        "actions_metrics_and_safe_labels": action_valid and label_valid and finite,
        "aggregate_accuracy_and_interval": (
            close(summary["safe_oracle_accuracy"], oracle_accuracy)
            and close(summary["retain_all_accuracy"], retain_accuracy)
            and close(summary["safe_oracle_gain"], oracle_accuracy - retain_accuracy)
            and all(close(left, right) for left, right in zip(
                summary["safe_oracle_gain_group_bootstrap_95ci"], interval
            ))
        ),
        "barrier_reconstruction": (
            min(unrelated_deltas) >= 0
            and summary["added_unrelated_errors"] == 0
            and summary["additional_unrelated_correct"]
            == sum(max(0, delta) for delta in unrelated_deltas)
        ),
        "half_strength_decision": summary["half_strength"] == half,
        "mechanism_action_support_reconstructed": (
            summary["gate"]["checks"]["every_mechanism_has_two_actions"]
            == all(value >= 2 for value in mechanism_actions.values())
        ),
        "resource_accounting": (
            summary["resource"]["two_gpu_critical_seconds"]
            == max(resource["total_seconds"] for resource in resources)
            and summary["resource"]["peak_reserved_gib"]
            == max(resource["overall_peak_reserved_gib"] for resource in resources)
        ),
        "frozen_plain_qwen": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["value_scale"] == 8.0 for resource in resources
        ),
        "heldouts_sealed": (
            summary["v1_confirmation_loaded"] is False
            and summary["development_loaded"] is False
            and summary["test_loaded"] is False
            and summary["external_benchmarks_loaded"] is False
            and all(not resource["v1_confirmation_loaded"]
                    and not resource["development_loaded"]
                    and not resource["test_loaded"]
                    and not resource["external_benchmarks_loaded"]
                    for resource in resources)
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "scientific_gate_passed": summary["gate"]["passed"],
        "decision": summary["next_stage"],
        "rows": len(rows),
        "future_queries": queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": sum(max(0, -delta) for delta in unrelated_deltas),
        "summary_sha256": sha256(summary_path),
        "data_manifest_sha256": sha256(data_manifest_path),
        "data_audit_sha256": sha256(data_audit_path),
        "next_stage": summary["next_stage"] if passed else "stop_audit_failure",
        "terminal_status": (
            "templama_resource_pilot_audit_passed" if passed
            else "templama_resource_pilot_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
