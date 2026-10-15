"""Independent row-level audit of the utility semantic-record resource pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import statistics

from scripts.inspect_lama_pararel_utility_source import sha256


PROTOCOL = "qwen_utility_semantic_record_gate_v1_resource_pilot"
AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_resource_pilot_audit"
ALPHAS = (0.0, 0.5, 1.0)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610167


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_actions(size: int) -> dict[str, dict]:
    rows = {}
    for values in itertools.product(ALPHAS, repeat=size):
        name = "grid_" + "_".join("05" if value == .5 else str(int(value))
                                    for value in values)
        rows[name] = {"name": name, "kind": "grid", "alphas": list(values)}
    rows["global_clear"] = {"name": "global_clear", "kind": "global_clear", "alphas": []}
    return rows


def independent_label(row: dict) -> str:
    baseline = row["actions"][row["retain_action"]]
    eligible = []
    for name, value in row["actions"].items():
        if value["unrelated_correct"] >= baseline["unrelated_correct"]:
            eligible.append((name, value))
    if not eligible:
        raise ValueError("no independently safe action")
    return max(eligible, key=lambda item: (
        item[1]["correct"], -item[1]["total_answer_nll"],
        -item[1]["mean_memory_mass"], sum(item[1]["spec"]["alphas"]), item[0],
    ))[0]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def bootstrap(rows: list[dict]) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    gains = {}
    for group, values in grouped.items():
        numerator = sum(
            row["actions"][row["safe_oracle_action"]]["correct"]
            - row["actions"][row["retain_action"]]["correct"] for row in values
        )
        denominator = sum(row["actions"][row["retain_action"]]["queries"] for row in values)
        gains[group] = numerator / denominator
    groups = sorted(gains); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(gains[group] for group in sample))
    return [percentile(draws, .025), percentile(draws, .975)]


def close(left, right, tolerance=1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def audit(original_root: Path, recovery_root: Path, data_root: Path) -> dict:
    summary = json.loads((recovery_root / "summary.json").read_text())
    original_summary = json.loads((original_root / "summary.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    rows, resources = [], []
    shard_hashes = {}
    for shard in (0, 1):
        row_path = original_root / f"labels_shard{shard}.jsonl"
        resource_path = original_root / f"resource_shard{shard}.json"
        resource = json.loads(resource_path.read_text()); part = load_jsonl(row_path)
        if resource["rows_sha256"] != sha256(row_path) or resource["labels"] != len(part):
            raise ValueError("raw shard hash/count mismatch")
        rows.extend(part); resources.append(resource)
        shard_hashes[row_path.name] = sha256(row_path)
        shard_hashes[resource_path.name] = sha256(resource_path)

    data_rows = load_jsonl(data_root / "pilot_episodes.jsonl")
    expected_identities = {
        (episode["episode_id"], boundary["boundary_index"])
        for episode in data_rows for boundary in episode["boundaries"]
    }
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    action_valid, labels_valid, finite = True, True, True
    for row in rows:
        expected = expected_actions(row["candidate_count"])
        if set(row["actions"]) != set(expected):
            action_valid = False; continue
        for name, action in row["actions"].items():
            if action["spec"] != expected[name]:
                action_valid = False
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
    oracle_accuracy, retain_accuracy = oracle_correct / total_queries, retain_correct / total_queries
    interval = bootstrap(rows)
    unrelated_deltas = [
        row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
        - row["actions"][row["retain_action"]]["unrelated_correct"] for row in rows
    ]
    half_rows = [row for row in rows if row["safe_oracle_contains_half"]]
    half_unique = [row for row in half_rows
                   if row["half_vs_endpoint_correct_gain"] > 0
                   or (row["half_vs_endpoint_correct_gain"] == 0
                       and row["half_vs_endpoint_nll_gain"] >= .01)]
    per_mechanism_actions = {
        mechanism: len({row["safe_oracle_action"] for row in rows
                        if row["mechanism"] == mechanism})
        for mechanism in sorted({row["mechanism"] for row in rows})
    }
    recovery_hash_lines = (recovery_root / "input_hashes.sha256").read_text().splitlines()
    recovery_hashes = {Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0]
                       for line in recovery_hash_lines}
    recovery_inputs_match = all(
        recovery_hashes.get(name) == digest for name, digest in shard_hashes.items()
    ) and recovery_hashes.get("summary.json") == sha256(original_root / "summary.json")

    checks = {
        "data_audit_and_provenance": (
            data_manifest.get("protocol") == "qwen_utility_semantic_record_gate_v1_pilot_data"
            and data_audit.get("passed") is True
            and all(resource["data_manifest_sha256"] == sha256(data_root / "manifest.json")
                    and resource["data_audit_sha256"] == sha256(data_root / "data_audit.json")
                    for resource in resources)
        ),
        "complete_unique_rows": len(rows) == 336 and identities == expected_identities,
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
            min(unrelated_deltas) >= 0
            and summary["added_unrelated_errors"] == 0
            and summary["additional_unrelated_correct"] == sum(unrelated_deltas) == 134
            and summary["unrelated_barrier_rowwise_passed"] is True
        ),
        "half_strength_decision": (
            summary["half_strength"]["selected_labels"] == len(half_rows)
            and summary["half_strength"]["unique_margin_labels"] == len(half_unique)
            and summary["half_strength"]["primary_action_space"]
            == "binary_with_half_ablation"
        ),
        "mechanism_action_support": min(per_mechanism_actions.values()) >= 2,
        "resource_and_size_selection": (
            max(resource["overall_peak_reserved_gib"] for resource in resources) < 24
            and summary["resource"]["selected_formal_groups"] == 512
            and summary["resource"]["two_gpu_critical_seconds"]
            == max(resource["total_seconds"] for resource in resources)
        ),
        "summary_recovery_provenance": (
            original_summary["gate"]["passed"] is False
            and original_summary["added_unrelated_errors"] == -134
            and recovery_inputs_match
            and (recovery_root / "terminal_status.txt").read_text().strip()
            == "completed_summary_only_recovery"
            and (recovery_root / "runner_exit_code.txt").read_text().strip() == "0"
        ),
        "gate_and_sealing": (
            summary["gate"]["passed"] is True
            and all(summary["gate"]["checks"].values())
            and all(not resource["development_loaded"]
                    and not resource["confirmation_loaded"]
                    and not resource["external_benchmarks_loaded"]
                    for resource in resources)
        ),
        "frozen_plain_qwen": all(
            resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["value_scale"] == 8.0 for resource in resources
        ),
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": len(rows),
        "future_queries": total_queries,
        "safe_oracle_accuracy": oracle_accuracy,
        "retain_all_accuracy": retain_accuracy,
        "safe_oracle_gain": oracle_accuracy - retain_accuracy,
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "added_unrelated_errors": sum(max(0, -delta) for delta in unrelated_deltas),
        "additional_unrelated_correct": sum(max(0, delta) for delta in unrelated_deltas),
        "summary_sha256": sha256(recovery_root / "summary.json"),
        "original_summary_sha256": sha256(original_root / "summary.json"),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_root / "data_audit.json"),
        "next_stage": "formal_label_construction" if all(checks.values()) else "stop",
        "terminal_status": "pilot_audit_passed" if all(checks.values()) else "pilot_audit_failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.original_root, args.recovery_root, args.data_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
