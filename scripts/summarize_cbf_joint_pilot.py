"""Summarize joint retention/write labels without loading feature tensors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CORNERS = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))


def summarize(paths: list[Path], flat_tolerance: float = 1e-4) -> dict:
    if not paths or flat_tolerance < 0:
        raise ValueError("label paths are required and tolerance must be nonnegative")
    seen = set()
    best_corners = Counter()
    regimes = defaultdict(Counter)
    label_times, allocated, reserved, interactions, ranges, interior_gains = [], [], [], [], [], []
    flat = interior_better = 0
    groups = set()
    grids = set()
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["id"], row["boundary"])
                if key in seen or row["protocol"] != "joint_v1":
                    raise ValueError(f"duplicate or incompatible joint label {key}")
                seen.add(key)
                groups.add(row["group_id"])
                actions = [tuple(action) for action in row["actions"]]
                losses = row["losses"]
                if len(actions) != len(losses) or len(set(actions)) != len(actions):
                    raise ValueError(f"invalid action/loss table for {key}")
                if any(not math.isfinite(value) for value in losses):
                    raise ValueError(f"nonfinite loss for {key}")
                if not all(corner in actions for corner in CORNERS):
                    raise ValueError(f"missing a corner action for {key}")
                grids.add(tuple(row["grid"]))
                corner_losses = [losses[actions.index(corner)] for corner in CORNERS]
                best_index = min(range(4), key=lambda index: corner_losses[index])
                action_name = "".join(str(int(value)) for value in CORNERS[best_index])
                best_corners[action_name] += 1
                regimes[row["regime"]][action_name] += 1
                spread = max(corner_losses) - min(corner_losses)
                ranges.append(spread)
                flat += spread <= flat_tolerance
                advantage = min(corner_losses) - min(losses)
                interior_gains.append(advantage)
                interior_better += advantage > flat_tolerance
                interactions.append(row["interaction"])
                label_times.append(row["label_time_s"])
                if row["peak_allocated_gib"] is not None:
                    allocated.append(row["peak_allocated_gib"])
                    reserved.append(row["peak_reserved_gib"])
    if not seen:
        raise ValueError("no joint labels found")
    if len(grids) != 1:
        raise ValueError("label files use different joint grids")
    count = len(seen)
    return {
        "labels": count, "source_groups": len(groups), "grid": list(next(iter(grids))),
        "flat_tolerance": flat_tolerance, "flat_fraction": flat / count,
        "best_corner_counts": dict(best_corners),
        "best_corner_by_regime": {name: dict(counts) for name, counts in sorted(regimes.items())},
        "mean_corner_loss_spread": statistics.mean(ranges),
        "mean_interaction": statistics.mean(interactions),
        "mean_abs_interaction": statistics.mean(abs(value) for value in interactions),
        "interior_better_fraction": interior_better / count,
        "mean_interior_gain": statistics.mean(interior_gains),
        "mean_label_time_s": statistics.mean(label_times),
        "p95_label_time_s": sorted(label_times)[math.ceil(0.95 * count) - 1],
        "max_label_time_s": max(label_times),
        "max_peak_allocated_gib": max(allocated) if allocated else None,
        "max_peak_reserved_gib": max(reserved) if reserved else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flat-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    result = summarize(args.labels, args.flat_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
