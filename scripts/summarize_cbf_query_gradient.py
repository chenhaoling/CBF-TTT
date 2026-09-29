"""Summarize answer-gradient alignment without publishing per-title diagnostic rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(paths: list[Path], expected_groups: int = 8, threshold: float = 0.005) -> dict:
    if not paths or expected_groups < 1 or threshold < 0:
        raise ValueError("paths, positive group count, and nonnegative threshold required")
    rows_by_group = defaultdict(dict)
    seen = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "query_gradient_diagnostic_v1" or row["split"] != "train":
                raise ValueError("expected train query-gradient diagnostic rows")
            group, regime = row["group_id"], row["regime"]
            if row["id"] in seen or regime in rows_by_group[group] or regime not in ("new_only", "old_only"):
                raise ValueError("duplicate or invalid source-group regime")
            seen.add(row["id"])
            numeric = [row["base_loss"], row["gradient_loss"], row["candidate_norm"],
                       row["gradient_norm"], row["gradient_dot_candidate"],
                       row["gradient_candidate_cosine"], row["time_s"]]
            numeric += list(row["raw_losses"].values()) + list(row["oracle_losses"].values())
            if any(not math.isfinite(value) for value in numeric):
                raise ValueError("nonfinite query-gradient diagnostic")
            if row["candidate_norm"] <= 0 or row["gradient_norm"] <= 0 or (
                abs(row["gradient_candidate_cosine"]) > 1.00001
            ):
                raise ValueError("invalid gradient/candidate norm or cosine")
            for key in ("0.25", "1.0"):
                if key not in row["raw_losses"] or key not in row["oracle_losses"]:
                    raise ValueError("both perturbation scales required")
            rows_by_group[group][regime] = row
    if len(rows_by_group) != expected_groups or any(set(rows) != {"new_only", "old_only"}
                                                     for rows in rows_by_group.values()):
        raise ValueError("expected complete paired train source groups")

    by_regime = {}
    all_rows = [row for rows in rows_by_group.values() for row in rows.values()]
    for regime in ("new_only", "old_only"):
        subset = [rows[regime] for rows in rows_by_group.values()]
        by_regime[regime] = {
            "groups": len(subset),
            "mean_gradient_candidate_cosine": statistics.mean(row["gradient_candidate_cosine"] for row in subset),
            "positive_directional_derivative_groups": sum(row["gradient_dot_candidate"] > 0 for row in subset),
            "mean_raw_gain_at_0.25": statistics.mean(row["base_loss"] - row["raw_losses"]["0.25"]
                                                 for row in subset),
            "mean_raw_gain_at_1": statistics.mean(row["base_loss"] - row["raw_losses"]["1.0"]
                                              for row in subset),
            "mean_oracle_gain_at_0.25": statistics.mean(row["base_loss"] - row["oracle_losses"]["0.25"]
                                                    for row in subset),
            "mean_oracle_gain_at_1": statistics.mean(row["base_loss"] - row["oracle_losses"]["1.0"]
                                                 for row in subset),
            "oracle_gain_over_threshold_groups": sum(
                row["base_loss"] - min(row["oracle_losses"].values()) > threshold for row in subset
            ),
        }
    allocated = [row["peak_allocated_gib"] for row in all_rows if row["peak_allocated_gib"] is not None]
    reserved = [row["peak_reserved_gib"] for row in all_rows if row["peak_reserved_gib"] is not None]
    new = by_regime["new_only"]
    supports_misalignment = (new["oracle_gain_over_threshold_groups"] >= 3 and
                             max(new["mean_oracle_gain_at_0.25"], new["mean_oracle_gain_at_1"]) > 0 and
                             new["positive_directional_derivative_groups"] > expected_groups / 2)
    return {
        "protocol": "query_gradient_diagnostic_v1", "labels": len(all_rows),
        "source_groups": len(rows_by_group), "threshold_nll": threshold,
        "by_regime": by_regime,
        "supports_candidate_misalignment_hypothesis": supports_misalignment,
        "mean_time_s": statistics.mean(row["time_s"] for row in all_rows),
        "max_time_s": max(row["time_s"] for row in all_rows),
        "max_peak_allocated_gib": max(allocated) if allocated else None,
        "max_peak_reserved_gib": max(reserved) if reserved else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-groups", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    result = summarize(args.input, args.expected_groups, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
