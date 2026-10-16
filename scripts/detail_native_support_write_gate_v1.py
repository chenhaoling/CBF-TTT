"""Create descriptive (non-gating) pilot tables for the V1 report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows)


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * len(values)))]


def describe(rows):
    actions = ["reject", "admit"]
    result = {
        "labels": len(rows),
        "positive": sum(row["B"] > 1e-4 for row in rows),
        "negative": sum(row["B"] < -1e-4 for row in rows),
        "ties": sum(abs(row["B"]) <= 1e-4 for row in rows),
        "mean_B": mean(rows, "B"),
        "median_B": statistics.median(row["B"] for row in rows),
        "J": {"never_write": mean(rows, "J0"), "always_write": mean(rows, "J1")},
        "metrics": {},
    }
    for action in actions:
        result["metrics"][action] = {
            key: statistics.fmean(row["actions"][action][key] for row in rows)
            for key in ("accuracy", "protected_accuracy", "protected_nll", "eos_rate")
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "data/manifest.json").read_text())
    reverse = {value: key for key, value in manifest["mechanism_hashes"].items()}
    paths = sorted(args.root.glob("pilot_shard*_labels.jsonl"))
    shard_rows = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                  for path in paths]
    rows = [row for part in shard_rows for row in part]
    if len(rows) != 144:
        raise ValueError(f"expected 144 labels, got {len(rows)}")
    by_mechanism = {}
    for mechanism in manifest["mechanism_hashes"]:
        selected = [row for row in rows if reverse[row["mechanism_hash"]] == mechanism]
        by_mechanism[mechanism] = describe(selected)
    times = [row["resource"]["seconds"] for row in rows]
    result = {
        "protocol": manifest["protocol"] + "_descriptive_detail",
        "overall": describe(rows),
        "per_mechanism": by_mechanism,
        "resource": {
            "mean_label_seconds": statistics.fmean(times),
            "median_label_seconds": statistics.median(times),
            "p95_label_seconds": percentile(times, .95),
            "max_label_seconds": max(times),
            "aggregate_label_seconds": sum(times),
            "two_gpu_label_critical_seconds": max(
                sum(row["resource"]["seconds"] for row in part) for part in shard_rows
            ),
            "peak_allocated_gib": max(row["resource"]["peak_allocated_gib"] for row in rows),
            "peak_reserved_gib": max(row["resource"]["peak_reserved_gib"] for row in rows),
        },
        "safe_oracle_actions": dict(Counter(
            "admit" if row["J1"] < row["J0"]
            and row["actions"]["admit"]["protected_accuracy"]
            >= row["actions"]["reject"]["protected_accuracy"] else "reject"
            for row in rows
        )),
        "descriptive_only": True,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
