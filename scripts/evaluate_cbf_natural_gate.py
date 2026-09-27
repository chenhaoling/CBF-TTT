"""Evaluate the predeclared source-group gate for natural continuation labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REGIMES = frozenset(("both_relevant", "old_only", "new_only", "neither_relevant"))
TARGET_REGIMES = frozenset(("both_relevant", "new_only"))


def evaluate(paths: list[Path], min_gain: float = 0.005,
             min_positive_groups: int = 3) -> dict:
    if not paths or min_gain < 0 or min_positive_groups < 1:
        raise ValueError("paths, nonnegative gain, and positive group threshold required")
    by_group = defaultdict(dict)
    best = Counter()
    gains = defaultdict(list)
    positive_groups = set()
    count = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "joint_natural_v1" or row["regime"] not in REGIMES:
                raise ValueError("expected joint_natural_v1 labels with four natural regimes")
            group, regime = row["group_id"], row["regime"]
            if regime in by_group[group]:
                raise ValueError(f"duplicate regime {group}/{regime}")
            by_group[group][regime] = row
            corners = row["corner_losses"]
            if any(not math.isfinite(corners[key]) for key in ("00", "01", "10", "11")):
                raise ValueError("corner loss must be finite")
            best[min(("00", "01", "10", "11"), key=lambda key: corners[key])] += 1
            gain = corners["00"] - corners["11"]
            gains[regime].append(gain)
            if regime in TARGET_REGIMES and gain > min_gain:
                positive_groups.add(group)
            count += 1
    if not by_group or any(set(rows) != REGIMES for rows in by_group.values()):
        raise ValueError("each source group must contain all four regimes")
    return {
        "protocol": "joint_natural_v1",
        "labels": count,
        "source_groups": len(by_group),
        "minimum_11_gain_nll": min_gain,
        "required_positive_groups": min_positive_groups,
        "positive_groups": len(positive_groups),
        "positive_groups_by_regime": {
            regime: sum(value > min_gain for value in gains[regime])
            for regime in sorted(TARGET_REGIMES)
        },
        "mean_11_gain_vs_00_by_regime": {
            regime: statistics.mean(gains[regime]) for regime in sorted(REGIMES)
        },
        "best_corner_counts": {key: best[key] for key in ("00", "01", "10", "11")},
        "passed": len(positive_groups) >= min_positive_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--min-positive-groups", type=int, default=3)
    args = parser.parse_args()
    result = evaluate(args.labels, args.min_gain, args.min_positive_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
