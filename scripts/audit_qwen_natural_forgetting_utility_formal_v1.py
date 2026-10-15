"""Independent audit of the full calibration forgetting-utility corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts.audit_qwen_natural_forgetting_utility_pilot_v1 import (
    action_consistent,
    bootstrap,
    oracle,
)
from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_formal_v1 import PROTOCOL
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    EXPECTED,
    UNIQUE_ACTIONS,
    registered_grid,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PLAN = "QWEN_NATURAL_FORGETTING_UTILITY_FORMAL_V1_PLAN.md"


def audit(root, repo, data_root, v5_root, context_root, scope_root):
    root, repo, data_root, v5_root, context_root, scope_root = map(
        Path, (root, repo, data_root, v5_root, context_root, scope_root)
    )
    summary = json.loads((root / "summary.json").read_text())
    labels, resources = [], []
    shard_checks = True
    for shard in (0, 1):
        label_path = root / f"labels_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        part = [json.loads(line) for line in label_path.read_text().splitlines() if line]
        shard_checks &= resource["protocol"] == PROTOCOL
        shard_checks &= resource["labels_sha256"] == sha(label_path)
        shard_checks &= resource["labels"] == len(part)
        shard_checks &= resource["sealed_inputs"] == EXPECTED
        shard_checks &= not resource["development_loaded"] and not resource["test_loaded"]
        shard_checks &= not resource["external_benchmarks_loaded"]
        labels.extend(part); resources.append(resource)

    _, sessions = load(data_root, "calibration")
    expected = {(session["id"], step) for session in sessions
                for step in range(len(session["events"]))}
    observed = {(row["session_id"], row["event_step"]) for row in labels}
    raw_consistency = True
    for row in labels:
        raw_consistency &= row["protocol"] == PROTOCOL
        raw_consistency &= set(row["actions"]) == set(UNIQUE_ACTIONS)
        raw_consistency &= all(action_consistent(action) for action in row["actions"].values())
        safe = oracle(row["actions"], True); unconstrained = oracle(row["actions"], False)
        raw_consistency &= row["safe_oracle_action"] == safe
        raw_consistency &= row["unconstrained_oracle_action"] == unconstrained
        raw_consistency &= row["safe_oracle_nonretain"] == int(safe != "retain_all")
        raw_consistency &= row["safe_oracle_gain"] == (
            row["actions"][safe]["candidate_accuracy"]
            - row["actions"]["retain_all"]["candidate_accuracy"])

    safe_correct = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in labels)
    retain_correct = sum(row["actions"]["retain_all"]["correct"] for row in labels)
    queries = sum(row["actions"]["retain_all"]["queries"] for row in labels)
    gain = (safe_correct - retain_correct) / queries
    ci, session_gains = bootstrap(labels)
    selected = Counter(row["safe_oracle_action"] for row in labels)
    unrelated_errors = sum(max(
        0, row["actions"]["retain_all"]["unrelated_correct"]
        - row["actions"][row["safe_oracle_action"]]["unrelated_correct"]
    ) for row in labels)
    seconds = [row["label_seconds"] for row in labels]
    max_allocated = max(row["peak_allocated_gib"] for row in labels)
    aggregate_checks = {
        "exactly_65_sessions": len({row["session_id"] for row in labels}) == 65,
        "exactly_774_unique_labels": len(labels) == 774 and len(observed) == 774,
        "complete_unique_and_registered_grid": all(
            set(row["actions"]) == set(UNIQUE_ACTIONS) for row in labels
        ) and len(registered_grid()) == 15,
        "safe_oracle_gain_ci_lower_above_zero": ci[0] > 0,
        "nonretain_rate_at_least_point_10":
            statistics.fmean(row["safe_oracle_nonretain"] for row in labels) >= .10,
        "retain_all_at_least_50_labels": selected["retain_all"] >= 50,
        "scope_top1_a0_at_least_50_labels": selected["scope_top1_a0"] >= 50,
        "at_least_five_selected_actions": len(selected) >= 5,
        "no_action_above_point_90": max(selected.values()) / len(labels) <= .90,
        "zero_added_unrelated_errors": unrelated_errors == 0,
        "median_label_seconds_at_most_120": statistics.median(seconds) <= 120,
        "peak_allocated_gib_at_most_28": max_allocated <= 28,
    }
    aggregate_consistency = all((
        summary["sessions"] == len({row["session_id"] for row in labels}),
        summary["labels"] == len(labels),
        summary["registered_grid"] == registered_grid(),
        summary["unique_actions"] == list(UNIQUE_ACTIONS),
        summary["label_distribution"] == dict(sorted(selected.items())),
        summary["safe_oracle_nonretain_rate"]
            == statistics.fmean(row["safe_oracle_nonretain"] for row in labels),
        summary["retain_candidate_accuracy"] == retain_correct / queries,
        summary["safe_oracle_candidate_accuracy"] == safe_correct / queries,
        summary["safe_oracle_gain"] == gain,
        summary["safe_oracle_gain_session_bootstrap_95ci"] == ci,
        summary["session_gains"] == session_gains,
        summary["added_unrelated_errors"] == unrelated_errors,
        summary["gate"]["checks"] == aggregate_checks,
        summary["gate"]["controller_training_authorized"] == all(aggregate_checks.values()),
    ))
    input_hashes = {
        "data_manifest": sha(data_root / "manifest.json"),
        "v5_checkpoint": sha(v5_root / "result" / "constrained_scope_gate.pt"),
        "transforms": sha(context_root / "transforms.pt"),
        "scope": sha(scope_root / "scope_gates.pt"),
    }
    terminal = ("completed_qwen_natural_forgetting_utility_formal_v1_train"
                if summary["gate"]["controller_training_authorized"] else
                "completed_qwen_natural_forgetting_utility_formal_v1_stop")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip() == terminal
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "complete_calibration_sampling": observed == expected and len(observed) == len(labels),
        "shard_hashes_and_seals": shard_checks,
        "raw_action_and_oracle_consistency": raw_consistency,
        "independent_aggregate_and_gate": aggregate_consistency,
        "sealed_input_hashes": input_hashes == EXPECTED == summary["sealed_inputs"],
        "plan_hash": summary["plan_sha256"] == sha(repo / PLAN),
        "no_heldout_release": summary["calibration_scored"]
            and not summary["development_loaded"] and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": {
                "labels": len(labels), "sessions": len({row["session_id"] for row in labels}),
                "label_distribution": dict(sorted(selected.items())),
                "retain_candidate_accuracy": retain_correct / queries,
                "safe_oracle_candidate_accuracy": safe_correct / queries,
                "safe_oracle_gain": gain, "bootstrap_95ci": ci,
                "added_unrelated_errors": unrelated_errors,
                "aggregate_checks": aggregate_checks,
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "data_root", "v5_root", "context_root",
                 "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.repo, args.data_root, args.v5_root,
                   args.context_root, args.scope_root)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
