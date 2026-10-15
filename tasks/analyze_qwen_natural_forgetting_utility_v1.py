"""Score-free diagnosis of counterfactual forgetting-utility labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import UNIQUE_ACTIONS
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOLS = {
    "qwen_natural_forgetting_utility_pilot_v1",
    "qwen_natural_forgetting_utility_formal_v1",
}


def load_labels(root):
    root = Path(root); summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] not in PROTOCOLS:
        raise ValueError("unexpected forgetting-utility protocol")
    rows = []
    for path in sorted(root.glob("labels_shard*.jsonl")):
        if summary["label_files_sha256"].get(path.name) != sha(path):
            raise ValueError("label shard hash mismatch")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if len(rows) != summary["labels"]:
        raise ValueError("label count mismatch")
    return summary, rows


def policy_metrics(rows, selector):
    chosen = [(row, selector(row)) for row in rows]
    correct = sum(row["actions"][action]["correct"] for row, action in chosen)
    queries = sum(row["actions"][action]["queries"] for row, action in chosen)
    nll = sum(row["actions"][action]["total_answer_nll"] for row, action in chosen)
    unrelated_harm = sum(max(
        0, row["actions"]["retain_all"]["unrelated_correct"]
        - row["actions"][action]["unrelated_correct"]
    ) for row, action in chosen)
    return {"candidate_accuracy": correct / queries, "mean_answer_nll": nll / queries,
            "unrelated_errors_added": unrelated_harm,
            "action_distribution": dict(sorted(Counter(action for _, action in chosen).items()))}


def source_rule(row):
    return "scope_top1_a0" if row["desired_source_action"] == "local" else "retain_all"


def diagnose(root):
    summary, rows = load_labels(root)
    policies = {name: policy_metrics(rows, lambda row, action=name: action)
                for name in UNIQUE_ACTIONS}
    policies.update({
        "source_action_rule": policy_metrics(rows, source_rule),
        "safe_oracle": policy_metrics(rows, lambda row: row["safe_oracle_action"]),
        "unconstrained_oracle": policy_metrics(
            rows, lambda row: row["unconstrained_oracle_action"]),
    })
    by_kind = {}
    for kind in sorted({row["event_kind"] for row in rows}):
        part = [row for row in rows if row["event_kind"] == kind]
        by_kind[kind] = {
            "labels": len(part),
            "safe_actions": dict(sorted(Counter(
                row["safe_oracle_action"] for row in part).items())),
            "nonretain_rate": statistics.fmean(row["safe_oracle_nonretain"] for row in part),
            "mean_label_accuracy_gain": statistics.fmean(row["safe_oracle_gain"] for row in part),
            "safe_policy": policy_metrics(part, lambda row: row["safe_oracle_action"]),
            "retain_policy": policy_metrics(part, lambda row: "retain_all"),
            "source_action_rule": policy_metrics(part, source_rule),
        }
    by_memory_size = {}
    for size in sorted({row["memory_size_before"] for row in rows}):
        part = [row for row in rows if row["memory_size_before"] == size]
        by_memory_size[str(size)] = {
            "labels": len(part),
            "safe_actions": dict(sorted(Counter(
                row["safe_oracle_action"] for row in part).items())),
            "nonretain_rate": statistics.fmean(row["safe_oracle_nonretain"] for row in part),
            "mean_label_accuracy_gain": statistics.fmean(row["safe_oracle_gain"] for row in part),
        }
    same_chain = Counter(); action_alpha = Counter(); scope_family = Counter()
    for row in rows:
        action = row["safe_oracle_action"]
        if action == "retain_all":
            action_alpha["1.0"] += 1; scope_family["retain"] += 1
            continue
        meta = row["actions"][action]["intervention"]
        action_alpha[str(meta["alpha"])] += 1
        scope_family[meta["scope"]] += 1
        affected = meta["affected"]
        same_chain["has_same_chain"] += int(any(
            item["chain_id"] == row["chain_id"] for item in affected))
        same_chain["lacks_same_chain"] += int(not any(
            item["chain_id"] == row["chain_id"] for item in affected))
    positive = [row["safe_oracle_gain"] for row in rows]
    candidate_ties = 0; margins = []; single_feasible = 0
    for row in rows:
        feasible = [action for action in UNIQUE_ACTIONS
                    if row["actions"][action]["unrelated_correct"]
                    >= row["actions"]["retain_all"]["unrelated_correct"]]
        accuracies = sorted((row["actions"][action]["candidate_accuracy"]
                             for action in feasible), reverse=True)
        if len(accuracies) == 1:
            single_feasible += 1
        else:
            margins.append(accuracies[0] - accuracies[1])
            candidate_ties += int(accuracies[0] == accuracies[1])
    supports = Counter(row["safe_oracle_action"] for row in rows)
    return {
        "protocol": summary["protocol"] + "_diagnosis",
        "labels": len(rows), "sessions": summary["sessions"],
        "policies": policies, "by_event_kind": by_kind,
        "by_memory_size": by_memory_size,
        "safe_action_support": dict(sorted(supports.items())),
        "classes_with_at_least_20_labels": sorted(
            action for action, count in supports.items() if count >= 20),
        "alpha_distribution": dict(sorted(action_alpha.items())),
        "scope_family_distribution": dict(sorted(scope_family.items())),
        "nonretain_affected_chain_match": dict(same_chain),
        "barrier_binding_labels": sum(
            row["safe_oracle_action"] != row["unconstrained_oracle_action"] for row in rows),
        "labels_with_positive_accuracy_gain": sum(value > 0 for value in positive),
        "labels_with_zero_accuracy_gain": sum(value == 0 for value in positive),
        "mean_label_accuracy_gain": statistics.fmean(positive),
        "single_feasible_action_labels": single_feasible,
        "candidate_objective_tie_rate_among_multi_feasible":
            candidate_ties / (len(rows) - single_feasible),
        "mean_best_second_candidate_margin": statistics.fmean(margins),
        "source_summary_sha256": sha(Path(root) / "summary.json"),
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = diagnose(args.root)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
