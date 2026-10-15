"""Independent integrity and aggregate audit for forgetting-utility pilot V1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    ACTION_ORDER,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EXPECTED,
    PROTOCOL,
    UNIQUE_ACTIONS,
    registered_grid,
    select_pilot_sessions,
    selected_decision_steps,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def oracle(actions, constrained):
    baseline = actions["retain_all"]
    candidates = list(UNIQUE_ACTIONS)
    if constrained:
        candidates = [name for name in candidates
                      if actions[name]["unrelated_correct"] >= baseline["unrelated_correct"]]
    return max(candidates, key=lambda name: (
        actions[name]["correct"], -actions[name]["total_answer_nll"],
        -actions[name]["mean_memory_mass"], -ACTION_ORDER[name],
    ))


def action_consistent(action):
    outcomes = action["outcomes"]
    related = [row for row in outcomes if row["related"]]
    unrelated = [row for row in outcomes if not row["related"]]
    total_nll = sum(row["answer_nll"] for row in outcomes)
    return all((
        action["queries"] == len(outcomes),
        action["correct"] == sum(row["correct"] for row in outcomes),
        action["candidate_accuracy"] == statistics.fmean(row["correct"] for row in outcomes),
        action["total_answer_nll"] == total_nll,
        action["mean_answer_nll"] == total_nll / len(outcomes),
        action["related_queries"] == len(related),
        action["related_correct"] == sum(row["correct"] for row in related),
        action["unrelated_queries"] == len(unrelated),
        action["unrelated_correct"] == sum(row["correct"] for row in unrelated),
        action["stale_reads"] == sum(row["stale"] for row in outcomes),
        action["wrong_chain_reads"] == sum(row["wrong_chain"] for row in outcomes),
        action["defers"] == sum(row["defer"] for row in outcomes),
    ))


def bootstrap(labels):
    values = {}
    for session_id in sorted({row["session_id"] for row in labels}):
        part = [row for row in labels if row["session_id"] == session_id]
        safe = sum(row["actions"][row["safe_oracle_action"]]["correct"] for row in part)
        retain = sum(row["actions"]["retain_all"]["correct"] for row in part)
        queries = sum(row["actions"]["retain_all"]["queries"] for row in part)
        values[session_id] = (safe - retain) / queries
    groups = sorted(values); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    return [quantile(draws, .025), quantile(draws, .975)], values


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
        shard_checks &= resource["labels_sha256"] == sha(label_path)
        shard_checks &= resource["labels"] == len(part)
        shard_checks &= resource["sealed_inputs"] == EXPECTED
        shard_checks &= not resource["development_loaded"] and not resource["test_loaded"]
        shard_checks &= not resource["external_benchmarks_loaded"]
        labels.extend(part); resources.append(resource)

    _, sessions = load(data_root, "calibration")
    pilot = select_pilot_sessions(sessions)
    expected = {(session["id"], step) for session in pilot
                for step in selected_decision_steps(session)}
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
        raw_consistency &= row["label_seconds"] >= 0
        raw_consistency &= row["qwen_cache_hits"] >= 0 and row["qwen_cache_misses"] >= 0

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
        "at_least_90_labels": len(labels) >= 90,
        "complete_unique_and_registered_grid": all(
            set(row["actions"]) == set(UNIQUE_ACTIONS) for row in labels
        ) and len(registered_grid()) == 15,
        "nonretain_rate_at_least_point_10":
            statistics.fmean(row["safe_oracle_nonretain"] for row in labels) >= .10,
        "safe_oracle_gain_at_least_point_01": gain >= .01,
        "zero_added_unrelated_errors": unrelated_errors == 0,
        "at_least_three_selected_actions": len(selected) >= 3,
        "median_label_seconds_at_most_120": statistics.median(seconds) <= 120,
        "peak_allocated_gib_at_most_28": max_allocated <= 28,
    }
    aggregate_consistency = all((
        summary["labels"] == len(labels),
        summary["sessions"] == len({row["session_id"] for row in labels}),
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
        summary["gate"]["scale_to_formal"] == all(aggregate_checks.values()),
        summary["gate"]["pilot_evidence_ready"] == (ci[0] > 0),
    ))
    input_hashes = {
        "data_manifest": sha(data_root / "manifest.json"),
        "v5_checkpoint": sha(v5_root / "result" / "constrained_scope_gate.pt"),
        "transforms": sha(context_root / "transforms.pt"),
        "scope": sha(scope_root / "scope_gates.pt"),
    }
    terminal = ("completed_qwen_natural_forgetting_utility_pilot_v1_scale"
                if summary["gate"]["scale_to_formal"] else
                "completed_qwen_natural_forgetting_utility_pilot_v1_stop")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip() == terminal
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "calibration_only_sampling": observed == expected and len(observed) == len(labels),
        "shard_hashes_and_seals": shard_checks,
        "raw_action_and_oracle_consistency": raw_consistency,
        "independent_aggregate_and_gate": aggregate_consistency,
        "sealed_input_hashes": input_hashes == EXPECTED == summary["sealed_inputs"],
        "plan_hash": summary["plan_sha256"]
            == sha(repo / "QWEN_NATURAL_FORGETTING_UTILITY_PILOT_V1_PLAN.md"),
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
