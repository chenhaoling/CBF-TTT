"""Independent row and aggregate audit of TempLAMA V4 formal-train binary labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import statistics

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v4_templama_train import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
    GROUPS,
    PROTOCOL as DATA_PROTOCOL,
)
from tasks.label_qwen_utility_record_controller_v4_templama_train import (
    AUDIT_PROTOCOL,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED_LABELS,
    PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def expected_actions(size: int) -> dict[str, dict]:
    output = {}
    for values in itertools.product((0.0, 1.0), repeat=size):
        name = "grid_" + "_".join(str(int(value)) for value in values)
        output[name] = {
            "name": name, "kind": "grid", "alphas": list(values),
        }
    output["global_clear"] = {
        "name": "global_clear", "kind": "global_clear", "alphas": [],
    }
    return output


def independent_label(row: dict) -> str:
    baseline = row["actions"][row["retain_action"]]
    eligible = [
        (name, value) for name, value in row["actions"].items()
        if value["unrelated_correct"] >= baseline["unrelated_correct"]
    ]
    if not eligible:
        raise ValueError("no independently safe binary action")
    return max(eligible, key=lambda item: (
        item[1]["correct"],
        -item[1]["total_answer_nll"],
        -item[1]["mean_memory_mass"],
        sum(item[1]["spec"]["alphas"]),
        item[0],
    ))[0]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def bootstrap(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    gains = {
        group: sum(
            row["actions"][row["safe_oracle_action"]]["correct"]
            - row["actions"][row["retain_action"]]["correct"]
            for row in values
        ) / sum(
            row["actions"][row["retain_action"]]["queries"]
            for row in values
        )
        for group, values in grouped.items()
    }
    groups = sorted(gains)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(gains[group] for group in sample))
    return [percentile(draws, .025), percentile(draws, .975)]


def close(left, right, tolerance=1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def audit(label_root: Path, data_root: Path) -> dict:
    summary = json.loads((label_root / "summary.json").read_text())
    manifest = json.loads((data_root / "manifest.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    episodes = load_jsonl(data_root / "train_episodes.jsonl")
    expected_ids = {
        (episode["episode_id"], boundary["boundary_index"])
        for episode in episodes for boundary in episode["boundaries"]
    }
    expected_relations = {
        episode["episode_id"]: episode["audit"]["target_relation"]
        for episode in episodes
    }
    rows, resources = [], []
    for shard in (0, 1):
        path = label_root / f"labels_shard{shard}.jsonl"
        resource = json.loads(
            (label_root / f"resource_shard{shard}.json").read_text()
        )
        part = load_jsonl(path)
        if resource["rows_sha256"] != sha256(path) \
                or resource["labels"] != len(part):
            raise ValueError("TempLAMA raw shard hash/count mismatch")
        rows.extend(part)
        resources.append(resource)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    actions_valid = True
    labels_valid = True
    finite = True
    relations_valid = True
    query_roles_valid = True
    for row in rows:
        expected = expected_actions(row["candidate_count"])
        actions_valid &= set(row["actions"]) == set(expected)
        if set(row["actions"]) != set(expected):
            continue
        for name, action in row["actions"].items():
            actions_valid &= action["spec"] == expected[name]
            finite &= all(math.isfinite(float(action[key])) for key in (
                "queries", "correct", "total_answer_nll", "unrelated_queries",
                "unrelated_correct", "mean_memory_mass", "selected_null",
            ))
            roles = action.get("per_query_role", {})
            query_roles_valid &= set(roles) == {
                "anchor", "current_exact", "current_rephrase",
                "retained_1", "retained_2",
            }
            query_roles_valid &= (
                sum(value["queries"] for value in roles.values())
                == action["queries"]
                and sum(value["correct"] for value in roles.values())
                == action["correct"]
                and abs(sum(value["total_answer_nll"] for value in roles.values())
                        - action["total_answer_nll"]) <= 1e-9
            )
        labels_valid &= independent_label(row) == row["safe_oracle_action"]
        labels_valid &= not row["safe_oracle_contains_half"]
        relations_valid &= (
            row["target_relation"] == expected_relations[row["episode_id"]]
        )
    total_queries = sum(
        row["actions"][row["retain_action"]]["queries"] for row in rows
    )
    oracle_correct = sum(
        row["actions"][row["safe_oracle_action"]]["correct"] for row in rows
    )
    retain_correct = sum(
        row["actions"][row["retain_action"]]["correct"] for row in rows
    )
    oracle_accuracy = oracle_correct / total_queries
    retain_accuracy = retain_correct / total_queries
    interval = bootstrap(rows)
    deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"]
        for row in rows
    ]
    checks = {
        "formal_train_data_locked": (
            manifest["protocol"] == DATA_PROTOCOL
            and data_audit["protocol"] == DATA_AUDIT_PROTOCOL
            and data_audit["passed"]
            and all(
                resource["data_manifest_sha256"]
                == sha256(data_root / "manifest.json")
                and resource["data_audit_sha256"]
                == sha256(data_root / "data_audit.json")
                for resource in resources
            )
        ),
        "complete_unique_rows": (
            len(rows) == EXPECTED_LABELS
            and identities == expected_ids
            and len({row["source_group"] for row in rows}) == GROUPS
        ),
        "binary_actions_and_finite_metrics": actions_valid and finite,
        "query_role_aggregates": query_roles_valid,
        "independent_safe_labels": labels_valid,
        "relation_provenance": relations_valid,
        "aggregate_reconstruction": (
            close(summary["safe_oracle_accuracy"], oracle_accuracy)
            and close(summary["retain_all_accuracy"], retain_accuracy)
            and close(
                summary["safe_oracle_gain"],
                oracle_accuracy - retain_accuracy,
            )
            and all(close(left, right) for left, right in zip(
                summary["safe_oracle_gain_group_bootstrap_95ci"], interval
            ))
        ),
        "barrier_reconstruction": (
            min(deltas) >= 0
            and summary["added_unrelated_errors"] == 0
            and summary["additional_unrelated_correct"]
            == sum(max(0, value) for value in deltas)
        ),
        "frozen_plain_qwen": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["value_scale"] == 8.0
            and resource["action_alphas"] == [0.0, 1.0]
            for resource in resources
        ),
        "gate_and_sealing": (
            summary["protocol"] == PROTOCOL
            and summary["passed"]
            and all(summary["gate"]["checks"].values())
            and all(
                not resource["templama_validation_loaded"]
                and not resource["templama_test_loaded"]
                and not resource["v1_confirmation_loaded"]
                and not resource["external_benchmarks_loaded"]
                for resource in resources
            )
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
        "added_unrelated_errors": sum(max(0, -value) for value in deltas),
        "additional_unrelated_correct": sum(max(0, value) for value in deltas),
        "summary_sha256": sha256(label_root / "summary.json"),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_root / "data_audit.json"),
        "next_stage": (
            "templama_v4_train_feature_construction" if passed else "stop"
        ),
        "terminal_status": (
            "templama_v4_train_labels_audit_passed" if passed
            else "templama_v4_train_labels_audit_failed"
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
