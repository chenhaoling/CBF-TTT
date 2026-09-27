"""Aggregate gap mechanism traces without exposing per-scenario observations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


TRAJECTORIES = tuple(
    f"{action}/{policy}"
    for action in ("00", "01", "10", "11")
    for policy in ("normal_11", "freeze_10")
)


def summarize(path: Path) -> dict:
    by_trajectory = defaultdict(lambda: defaultdict(list))
    memory_by_trajectory = defaultdict(lambda: defaultdict(list))
    times, reserved, regimes = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["protocol"] != "gap_diagnostic_v1" or set(row["trajectories"]) != set(TRAJECTORIES):
            raise ValueError("expected complete gap_diagnostic_v1 trajectories")
        regimes.append(row["regime"])
        times.append(row["diagnostic_time_s"])
        if row["peak_reserved_gib"] is not None:
            reserved.append(row["peak_reserved_gib"])
        for name in TRAJECTORIES:
            points = row["trajectories"][name]
            if len(points) != row["gap_chunks"] + 1:
                raise ValueError("incomplete gap trajectory")
            for step, point in enumerate(points):
                if point["gap_chunks"] != step:
                    raise ValueError("nonsequential gap trajectory")
                by_trajectory[name][step].append(point["mean_nll"])
                memory_by_trajectory[name][step].append(point["memory_ratio"])
    if not regimes:
        raise ValueError("no gap diagnostics found")
    return {
        "protocol": "gap_diagnostic_v1",
        "scenarios": len(regimes),
        "regime_counts": {name: regimes.count(name) for name in sorted(set(regimes))},
        "mean_nll_by_trajectory": {
            name: [statistics.mean(by_trajectory[name][step])
                   for step in sorted(by_trajectory[name])]
            for name in TRAJECTORIES
        },
        "mean_memory_ratio_by_trajectory": {
            name: [statistics.mean(memory_by_trajectory[name][step])
                   for step in sorted(memory_by_trajectory[name])]
            for name in TRAJECTORIES
        },
        "mean_diagnostic_time_s": statistics.mean(times),
        "max_peak_reserved_gib": max(reserved) if reserved else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
