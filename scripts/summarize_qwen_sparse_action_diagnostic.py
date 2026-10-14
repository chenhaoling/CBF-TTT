"""Describe endpoint versus interior actions without changing any release gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


SEEDS = (2301, 2302, 2303)
POLICY = "grid_classifier"


def _action(values):
    action = tuple(float(value) for value in values)
    if len(action) != 3 or any(value not in (0.0, 0.5, 1.0) for value in action):
        raise ValueError(f"invalid grid action: {action}")
    return action


def summarize_rows(rows):
    buckets = {
        "endpoint": {"decisions": 0, "matches": 0, "queries": 0, "correct": 0,
                     "retention_mae": []},
        "interior": {"decisions": 0, "matches": 0, "queries": 0, "correct": 0,
                     "retention_mae": []},
    }
    oracle_counts, predicted_counts = Counter(), Counter()
    for episode in rows:
        if episode.get("policy") != POLICY or episode.get("reader") != "component_top1":
            raise ValueError("unexpected policy or reader")
        decisions, queries = episode["decisions"], episode["queries"]
        if len(queries) != 3 * len(decisions):
            raise ValueError("each decision must have exactly three queries")
        for index, decision in enumerate(decisions):
            oracle = _action(decision["oracle_retention"])
            predicted = _action(decision["retention"])
            kind = "interior" if 0.5 in oracle else "endpoint"
            values = buckets[kind]
            values["decisions"] += 1
            values["matches"] += int(predicted == oracle)
            values["retention_mae"].append(float(decision["retention_mae"]))
            selected = queries[3 * index : 3 * index + 3]
            values["queries"] += len(selected)
            values["correct"] += sum(int(query["correct"]) for query in selected)
            oracle_counts[str(oracle)] += 1
            predicted_counts[str(predicted)] += 1
    if not all(values["decisions"] for values in buckets.values()):
        raise ValueError("both endpoint and interior decisions are required")
    total = sum(values["decisions"] for values in buckets.values())
    return {
        "decisions": total,
        "oracle_interior_fraction": buckets["interior"]["decisions"] / total,
        "predicted_interior_fraction": (
            sum(count for action, count in predicted_counts.items() if "0.5" in action)
            / total
        ),
        "buckets": {
            kind: {
                "decisions": values["decisions"],
                "action_accuracy": values["matches"] / values["decisions"],
                "query_accuracy": values["correct"] / values["queries"],
                "retention_mae": statistics.fmean(values["retention_mae"]),
            }
            for kind, values in buckets.items()
        },
        "oracle_label_counts": dict(oracle_counts),
        "predicted_label_counts": dict(predicted_counts),
    }


def combine(per_seed):
    combined = {"endpoint": Counter(), "interior": Counter()}
    total, oracle_interior, predicted_interior = 0, 0.0, 0.0
    for summary in per_seed.values():
        count = summary["decisions"]
        total += count
        oracle_interior += count * summary["oracle_interior_fraction"]
        predicted_interior += count * summary["predicted_interior_fraction"]
        for kind in combined:
            values = summary["buckets"][kind]
            decisions = values["decisions"]
            combined[kind]["decisions"] += decisions
            combined[kind]["matches"] += decisions * values["action_accuracy"]
            combined[kind]["queries"] += 3 * decisions
            combined[kind]["correct"] += 3 * decisions * values["query_accuracy"]
            combined[kind]["mae_sum"] += decisions * values["retention_mae"]
    return {
        "decisions": total,
        "oracle_interior_fraction": oracle_interior / total,
        "predicted_interior_fraction": predicted_interior / total,
        "buckets": {
            kind: {
                "decisions": values["decisions"],
                "action_accuracy": values["matches"] / values["decisions"],
                "query_accuracy": values["correct"] / values["queries"],
                "retention_mae": values["mae_sum"] / values["decisions"],
            }
            for kind, values in combined.items()
        },
    }


def run(root, output):
    root = Path(root)
    per_seed, hashes = {}, {}
    for seed in SEEDS:
        path = root / f"dev_seed{seed}_{POLICY}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        per_seed[str(seed)] = summarize_rows(rows)
        hashes[path.name] = sha(path)
    result = {
        "protocol": "qwen_sparse_action_diagnostic_v1",
        "status": "posthoc_development_description_not_a_release_gate",
        "policy": POLICY,
        "seeds": list(SEEDS),
        "source_sha256": hashes,
        "per_seed": per_seed,
        "macro": combine(per_seed),
        "terminal_status": "completed_qwen_sparse_action_diagnostic",
    }
    write_json(Path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
