"""Independent structural and target audit for controller V1 features."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from tasks.extract_qwen_utility_record_controller_v1_features import (
    BASE_FEATURE_NAMES,
    EXPECTED_LABELS,
    FEATURE_NAMES,
    PROTOCOL,
    teacher_targets,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


AUDIT_PROTOCOL = "qwen_utility_record_controller_v1_features_audit"
TOLERANCE = 1e-6


def close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def deterministic_features_are_valid(episode: dict, label: dict, feature_row: dict) -> bool:
    event_index = label["event_index"]
    boundary = episode["boundaries"][label["boundary_index"]]
    incoming = episode["events"][event_index]["record"]
    prior_events = episode["events"][:event_index]
    prior_by_id = {event["record"]["record_id"]: (index, event["record"])
                   for index, event in enumerate(prior_events)}
    candidates = [prior_by_id[identity] for identity in label["candidate_record_ids"]]
    memory_size = len(prior_events)
    base_rows = [row[:len(BASE_FEATURE_NAMES)] for row in feature_row["features"]]
    relation_scores = [row[4] for row in base_rows]
    ranked = sorted(range(len(base_rows)), key=lambda index: (-relation_scores[index], index))
    ranks = {index: rank + 1 for rank, index in enumerate(ranked)}
    for slot, ((candidate_index, record), values) in enumerate(zip(candidates, base_rows)):
        subject = normalize(record["input"]["subject"])
        value = normalize(record["input"]["value"])
        incoming_subject = normalize(incoming["input"]["subject"])
        incoming_value = normalize(incoming["input"]["value"])
        statement = normalize(record["input"]["statement"])
        incoming_statement = normalize(incoming["input"]["statement"])
        competitors = [score for index, score in enumerate(relation_scores) if index != slot]
        exact_repetitions = sum(
            normalize(event["record"]["input"]["statement"]) == statement
            for event in prior_events
        )
        subject_repetitions = sum(
            normalize(event["record"]["input"]["subject"]) == subject
            for event in prior_events
        )
        subject_value_repetitions = sum(
            normalize(event["record"]["input"]["subject"]) == subject
            and normalize(event["record"]["input"]["value"]) == value
            for event in prior_events
        )
        prior_accesses = sum(
            any(normalize(query["input"]["subject"]) == subject
                for query in previous["queries"])
            for previous in episode["boundaries"]
            if previous["event_index"] < event_index
        )
        expected = {
            0: float(subject == incoming_subject),
            1: float(value == incoming_value),
            2: float(subject == incoming_subject and value != incoming_value),
            3: float(statement == incoming_statement),
            8: relation_scores[slot] - (max(competitors) if competitors else 0.0),
            9: ranks[slot] / len(candidates),
            11: min(1.0, (event_index - candidate_index) / 63.0),
            12: (event_index - candidate_index) / memory_size,
            13: exact_repetitions / memory_size,
            14: subject_repetitions / memory_size,
            15: subject_value_repetitions / memory_size,
            16: min(1.0, memory_size / 64.0),
            17: min(1.0, prior_accesses / 3.0),
            18: 1.0,
            19: 1.0,
            20: float(boundary["reset"]),
            21: float(record["audit"]["semantic_key"] in boundary["revoke_keys"]),
        }
        if any(not close(values[index], expected_value)
               for index, expected_value in expected.items()):
            return False
        if any(not -1.0001 <= values[index] <= 1.0001 for index in (4, 5, 6, 7)):
            return False
    means = [statistics.fmean(row[index] for row in base_rows)
             for index in range(len(BASE_FEATURE_NAMES))]
    maxima = [max(row[index] for row in base_rows)
              for index in range(len(BASE_FEATURE_NAMES))]
    for row in feature_row["features"]:
        if any(not close(a, b) for a, b in zip(
            row[len(BASE_FEATURE_NAMES):2 * len(BASE_FEATURE_NAMES)], means
        )):
            return False
        if any(not close(a, b) for a, b in zip(
            row[2 * len(BASE_FEATURE_NAMES):], maxima
        )):
            return False
    return True


def audit(feature_root: Path, data_root: Path, label_root: Path) -> dict:
    summary = json.loads((feature_root / "summary.json").read_text())
    episodes = load_jsonl(data_root / "formal_train_episodes.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    labels = []
    feature_rows = []
    resources = []
    for shard in (0, 1):
        label_path = label_root / f"labels_shard{shard}.jsonl"
        feature_path = feature_root / f"features_shard{shard}.jsonl"
        resource_path = feature_root / f"resource_shard{shard}.json"
        labels.extend(load_jsonl(label_path))
        part = load_jsonl(feature_path)
        resource = json.loads(resource_path.read_text())
        if resource["rows_sha256"] != sha256(feature_path) or resource["boundaries"] != len(part):
            raise ValueError("feature shard hash/count mismatch")
        if resource["label_shard_sha256"] != sha256(label_path):
            raise ValueError("feature shard references another label shard")
        feature_rows.extend(part); resources.append(resource)
    label_by_id = {(row["episode_id"], row["boundary_index"]): row for row in labels}
    feature_by_id = {(row["episode_id"], row["boundary_index"]): row for row in feature_rows}
    coverage = set(label_by_id) == set(feature_by_id) and len(feature_by_id) == len(feature_rows)
    targets_valid = True; deterministic_valid = True; schema_valid = True
    finite = True
    for identity, row in feature_by_id.items():
        label = label_by_id[identity]
        episode = episode_by_id[row["episode_id"]]
        targets_valid &= row["targets"] == teacher_targets(label)
        targets_valid &= row["candidate_record_ids"] == label["candidate_record_ids"]
        schema_valid &= tuple(row["feature_names"]) == FEATURE_NAMES
        schema_valid &= len(row["features"]) == label["candidate_count"]
        schema_valid &= all(len(values) == len(FEATURE_NAMES) for values in row["features"])
        finite &= all(math.isfinite(value) for values in row["features"] for value in values)
        deterministic_valid &= deterministic_features_are_valid(episode, label, row)

    deletes = [value for row in feature_rows for value in row["targets"]["delete_targets"]]
    risks = [value for row in feature_rows for value in row["targets"]["risk_targets"]]
    checks = {
        "complete_one_to_one_coverage": coverage and len(feature_rows) == EXPECTED_LABELS,
        "exact_binary_teachers_and_risks": targets_valid,
        "fixed_numeric_schema": schema_valid and finite,
        "deterministic_and_context_features": deterministic_valid,
        "binary_class_support": set(deletes) == {0, 1} and set(risks) == {0, 1},
        "no_forbidden_tensor_fields": all(
            isinstance(value, (int, float))
            for row in feature_rows for values in row["features"] for value in values
        ),
        "locked_model_and_inputs": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["data_manifest_sha256"] == sha256(data_root / "manifest.json")
            and resource["data_audit_sha256"] == sha256(data_root / "formal_data_audit.json")
            and resource["label_summary_sha256"] == sha256(label_root / "summary.json")
            and resource["label_audit_sha256"]
            == sha256(label_root / "formal_labels_audit.json")
            for resource in resources
        ),
        "summary_reconstruction": (
            summary["protocol"] == PROTOCOL
            and summary["boundaries"] == len(feature_rows)
            and summary["record_examples"] == len(deletes)
            and close(summary["delete_fraction"], statistics.fmean(deletes))
            and close(summary["risk_fraction"], statistics.fmean(risks))
            and summary["safe_binary_actions"] == dict(Counter(
                row["targets"]["safe_binary_action"] for row in feature_rows
            ))
        ),
        "heldouts_sealed": all(
            not resource["development_loaded"] and not resource["confirmation_loaded"]
            and not resource["external_benchmarks_loaded"] for resource in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "boundaries": len(feature_rows),
        "record_examples": len(deletes),
        "feature_dimension": len(FEATURE_NAMES),
        "delete_fraction": statistics.fmean(deletes),
        "risk_fraction": statistics.fmean(risks),
        "feature_summary_sha256": sha256(feature_root / "summary.json"),
        "formal_label_audit_sha256": sha256(label_root / "formal_labels_audit.json"),
        "next_stage": "controller_training" if passed else "stop",
        "terminal_status": (
            "controller_features_audit_passed" if passed else "controller_features_audit_failed"
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
