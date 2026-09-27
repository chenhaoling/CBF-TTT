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
    corner_values = defaultdict(list)
    group_rows = defaultdict(list)
    energy_a, energy_c = [], []
    protocols = set()
    gap_corner_values = defaultdict(lambda: defaultdict(list))
    gap_best_corners = defaultdict(Counter)
    gap_regime_corners = defaultdict(lambda: defaultdict(Counter))
    query_corner_values = defaultdict(lambda: defaultdict(list))
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
                if key in seen or row["protocol"] not in (
                    "joint_v1", "joint_v2", "joint_natural_v1", "joint_postcutoff_v1"
                ):
                    raise ValueError(f"duplicate or incompatible joint label {key}")
                seen.add(key)
                protocols.add(row["protocol"])
                groups.add(row["group_id"])
                group_rows[row["group_id"]].append(row)
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
                if row["protocol"] in ("joint_v2", "joint_natural_v1", "joint_postcutoff_v1"):
                    expected_futures = 2 if row["protocol"] == "joint_v2" else 1
                    if len(row["future_meta"]) != expected_futures or any(
                        len(action_futures) != expected_futures for action_futures in row["losses_by_future"]
                    ):
                        raise ValueError(f"invalid joint future table for {key}")
                    for future_index, future in enumerate(row["future_meta"]):
                        gap = str(future["gap_chunks"])
                        gap_losses = [row["losses_by_future"][actions.index(corner)][future_index]
                                      for corner in CORNERS]
                        gap_best = min(range(4), key=lambda index: gap_losses[index])
                        gap_best_name = "".join(str(int(value)) for value in CORNERS[gap_best])
                        gap_best_corners[gap][gap_best_name] += 1
                        gap_regime_corners[gap][row["regime"]][gap_best_name] += 1
                        for corner, loss in zip(CORNERS, gap_losses):
                            name = "".join(str(int(value)) for value in corner)
                            gap_corner_values[gap][name].append(loss)
                            query_losses = row["query_losses_by_future"][actions.index(corner)][future_index]
                            if len(query_losses) != len(future["query_kinds"]):
                                raise ValueError(f"invalid joint_v2 query table for {key}")
                            for kind, query_loss in zip(future["query_kinds"], query_losses):
                                query_corner_values[f"gap_{gap}/{kind}"][name].append(query_loss)
                for corner, loss in zip(CORNERS, corner_losses):
                    corner_values["".join(str(int(value)) for value in corner)].append(loss)
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
                energy_a.append(row["energy_terms"]["A"])
                energy_c.append(row["energy_terms"]["C"])
                if row["peak_allocated_gib"] is not None:
                    allocated.append(row["peak_allocated_gib"])
                    reserved.append(row["peak_reserved_gib"])
    if not seen:
        raise ValueError("no joint labels found")
    if len(grids) != 1:
        raise ValueError("label files use different joint grids")
    if len(protocols) != 1:
        raise ValueError("label files mix joint protocols")
    count = len(seen)
    group_differences = [
        statistics.mean(row["corner_losses"]["11"] - row["corner_losses"]["00"] for row in rows)
        for rows in group_rows.values()
    ]
    return {
        "labels": count, "source_groups": len(groups), "grid": list(next(iter(grids))),
        "protocol": next(iter(protocols)),
        "flat_tolerance": flat_tolerance, "flat_fraction": flat / count,
        "best_corner_counts": dict(best_corners),
        "best_corner_by_regime": {name: dict(counts) for name, counts in sorted(regimes.items())},
        "mean_corner_losses": {name: statistics.mean(values) for name, values in sorted(corner_values.items())},
        "best_corner_counts_by_gap": {gap: dict(counts) for gap, counts in sorted(gap_best_corners.items())},
        "best_corner_by_regime_and_gap": {
            gap: {name: dict(counts) for name, counts in sorted(by_regime.items())}
            for gap, by_regime in sorted(gap_regime_corners.items())
        },
        "mean_corner_losses_by_gap": {
            gap: {name: statistics.mean(values) for name, values in sorted(by_corner.items())}
            for gap, by_corner in sorted(gap_corner_values.items())
        },
        "mean_corner_losses_by_query_kind": {
            kind: {name: statistics.mean(values) for name, values in sorted(by_corner.items())}
            for kind, by_corner in sorted(query_corner_values.items())
        },
        "mean_group_11_minus_00": statistics.mean(group_differences),
        "sd_group_11_minus_00": statistics.stdev(group_differences) if len(group_differences) > 1 else None,
        "groups_00_better_than_11": sum(value > 0 for value in group_differences),
        "mean_corner_loss_spread": statistics.mean(ranges),
        "mean_interaction": statistics.mean(interactions),
        "mean_abs_interaction": statistics.mean(abs(value) for value in interactions),
        "interior_better_fraction": interior_better / count,
        "mean_interior_gain": statistics.mean(interior_gains),
        "max_interior_gain": max(interior_gains),
        "mean_energy_A": statistics.mean(energy_a),
        "mean_energy_C": statistics.mean(energy_c),
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
