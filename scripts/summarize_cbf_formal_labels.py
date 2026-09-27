"""Validate completed scenario shards and counterfactual labels, then write a manifest."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "dev", "test")
GRID = [0.0, 0.5, 1.0]


def summarize(directory: Path, shards: int = 2, boundary: int = 2) -> dict:
    if shards < 1 or boundary < 1:
        raise ValueError("shards and boundary must be positive")
    expected = {}
    split_by_group = {}
    shard_by_group = {}
    for split in SPLITS:
        for shard in range(shards):
            path = directory / "shards" / f"{split}_shard{shard}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    scenario_id, group = row["id"], row["group_id"]
                    if row["split"] != split or scenario_id in expected:
                        raise ValueError(f"invalid split or duplicate scenario ID {scenario_id}")
                    if group in split_by_group and split_by_group[group] != split:
                        raise ValueError(f"group {group} appears in multiple splits")
                    if group in shard_by_group and shard_by_group[group] != shard:
                        raise ValueError(f"group {group} appears in multiple shards")
                    split_by_group[group] = split
                    shard_by_group[group] = shard
                    expected[scenario_id] = (split, shard, group, row["regime"])

    seen = set()
    coefficient_counts = {split: Counter() for split in SPLITS}
    regime_counts = defaultdict(Counter)
    times, allocated, reserved, gains = [], [], [], []
    update_rule = None
    for split in SPLITS:
        for shard in range(shards):
            path = directory / f"{split}_shard{shard}_labels.jsonl"
            summary_path = Path(str(path) + ".summary.json")
            if not summary_path.is_file():
                raise ValueError(f"collector did not finish {path}")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_rule = row.get("update_rule", "forget")
                    if row_rule not in ("forget", "write") or (update_rule is not None and row_rule != update_rule):
                        raise ValueError("labels must share one known update rule")
                    update_rule = row_rule
                    label_key = "write_gate_star" if row_rule == "write" else "alpha_star"
                    scenario_id = row["id"]
                    expected_entry = expected.get(scenario_id)
                    if scenario_id in seen or expected_entry is None or expected_entry[:3] != (
                        split, shard, row["group_id"]
                    ):
                        raise ValueError(f"duplicate or misplaced label {scenario_id}")
                    if row["split"] != split or row["boundary"] != boundary or row["grid"] != GRID:
                        raise ValueError(f"invalid boundary, grid, or split for {scenario_id}")
                    losses = row["losses"]
                    if len(losses) != 3 or any(not math.isfinite(x) for x in losses):
                        raise ValueError(f"nonfinite or incomplete losses for {scenario_id}")
                    if len(row["benefits"]) != 3 or any(not math.isfinite(x) for x in row["benefits"]):
                        raise ValueError(f"nonfinite or incomplete benefits for {scenario_id}")
                    if row.get(label_key) not in GRID or row["label_time_s"] <= 0:
                        raise ValueError(f"invalid coefficient or elapsed time for {scenario_id}")
                    baseline_index = -1 if row_rule == "write" else 0
                    if any(abs(gain - (losses[baseline_index] - loss)) > 1e-5
                           for gain, loss in zip(row["benefits"], losses)):
                        raise ValueError(f"benefit reference does not match {row_rule} baseline for {scenario_id}")
                    seen.add(scenario_id)
                    coefficient_counts[split][str(row[label_key])] += 1
                    regime_counts[expected_entry[3]][str(row[label_key])] += 1
                    times.append(row["label_time_s"])
                    allocated.append(row["peak_allocated_gib"])
                    reserved.append(row["peak_reserved_gib"])
                    gains.append(row["benefits"][GRID.index(row[label_key])])
    if seen != set(expected):
        raise ValueError(f"missing {len(set(expected) - seen)} scenario labels")
    result = {
        "scenarios_and_labels": len(seen),
        "groups_by_split": {split: sum(value == split for value in split_by_group.values()) for split in SPLITS},
        "update_rule": update_rule,
        ("gate_counts_by_split" if update_rule == "write" else "alpha_counts_by_split"):
            {split: dict(coefficient_counts[split]) for split in SPLITS},
        ("gate_counts_by_regime" if update_rule == "write" else "alpha_counts_by_regime"):
            {regime: dict(counts) for regime, counts in sorted(regime_counts.items())},
        "mean_label_time_s": statistics.mean(times),
        "max_label_time_s": max(times),
        "max_peak_allocated_gib": max(allocated),
        "max_peak_reserved_gib": max(reserved),
        ("mean_best_vs_always_write_gain" if update_rule == "write" else "mean_best_vs_alpha0_gain"):
            statistics.mean(gains),
    }
    (directory / "formal_label_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--boundary", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.shards, args.boundary), indent=2))


if __name__ == "__main__":
    main()
