"""Aggregate unbounded TTT update-scale probes by task regime and source group."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(path: Path, threshold: float = 0.005) -> dict:
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("no scale diagnostics found")
    scales = rows[0]["scales"]
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("scale grid must contain 0 and 1")
    by_group = defaultdict(set)
    by_regime = defaultdict(list)
    seen_ids = set()
    for row in rows:
        if row["protocol"] != "update_scale_diagnostic_v1" or row["scales"] != scales or (
            len(row["losses"]) != len(scales) or row["id"] in seen_ids or
            any(not math.isfinite(value) for value in row["losses"])
        ):
            raise ValueError("incompatible, duplicate, or nonfinite scale trajectory")
        seen_ids.add(row["id"])
        by_group[row["group_id"]].add(row["regime"])
        by_regime[row["regime"]].append(row)
    if set(by_regime) != {"new_only", "old_only"} or any(
        regimes != {"new_only", "old_only"} for regimes in by_group.values()
    ):
        raise ValueError("each source group needs new_only and old_only trajectories")
    zero = scales.index(0.0)
    positive = [index for index, scale in enumerate(scales) if scale > 0]
    negative = [index for index, scale in enumerate(scales) if scale < 0]
    if not positive or not negative:
        raise ValueError("diagnostic needs positive and negative scales")
    results = {}
    for regime, values in sorted(by_regime.items()):
        best = Counter(str(scales[min(range(len(scales)), key=lambda index: row["losses"][index])])
                       for row in values)
        positive_gains = [row["losses"][zero] - min(row["losses"][index] for index in positive)
                          for row in values]
        negative_gains = [row["losses"][zero] - min(row["losses"][index] for index in negative)
                          for row in values]
        results[regime] = {
            "scenarios": len(values),
            "mean_nll_by_scale": {str(scale): statistics.mean(row["losses"][index] for row in values)
                                  for index, scale in enumerate(scales)},
            "best_scale_counts": dict(best),
            "positive_scale_benefit_groups": sum(gain > threshold for gain in positive_gains),
            "negative_scale_benefit_groups": sum(gain > threshold for gain in negative_gains),
            "mean_best_positive_gain_vs_zero": statistics.mean(positive_gains),
            "mean_best_negative_gain_vs_zero": statistics.mean(negative_gains),
            "mean_candidate_ratio": statistics.mean(row["candidate_ratio"] for row in values),
            "mean_chunk_surprise": statistics.mean(row["chunk_surprise"] for row in values),
        }
    peaks = [row["peak_reserved_gib"] for row in rows if row["peak_reserved_gib"] is not None]
    return {"protocol": "update_scale_diagnostic_v1", "scenarios": len(rows),
            "source_groups": len(by_group), "scales": scales, "threshold_nll": threshold,
            "by_regime": results,
            "mean_time_s": statistics.mean(row["time_s"] for row in rows),
            "max_peak_reserved_gib": max(peaks) if peaks else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    result = summarize(args.input, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
