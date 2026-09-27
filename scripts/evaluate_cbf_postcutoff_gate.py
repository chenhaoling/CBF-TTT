"""Evaluate isolated write and retention effects in post-cutoff document labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REGIMES = frozenset(("both_relevant", "old_only", "new_only", "neither_relevant"))
CORNERS = ("00", "01", "10", "11")


def evaluate(paths: list[Path], threshold: float = 0.005, expected_groups: int = 12,
             min_groups: int = 3) -> dict:
    if not paths or threshold < 0 or expected_groups < 1 or min_groups < 1:
        raise ValueError("paths and positive group counts with nonnegative threshold required")
    by_group = defaultdict(dict)
    splits = {}
    best = Counter()
    effects = defaultdict(lambda: defaultdict(list))
    seen_ids = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "joint_postcutoff_v1" or row["regime"] not in REGIMES:
                raise ValueError("expected joint_postcutoff_v1 labels")
            group, regime = row["group_id"], row["regime"]
            if row["id"] in seen_ids or regime in by_group[group]:
                raise ValueError("duplicate ID or group/regime")
            seen_ids.add(row["id"])
            if group in splits and splits[group] != row["split"]:
                raise ValueError("source group crosses splits")
            splits[group] = row["split"]
            corners = row["corner_losses"]
            if any(key not in corners or not math.isfinite(corners[key]) for key in CORNERS):
                raise ValueError("four finite corner losses required")
            by_group[group][regime] = row
            best[min(CORNERS, key=lambda key: corners[key])] += 1
            effects[regime]["write"].append(corners["10"] - corners["11"])
            effects[regime]["keep"].append(corners["00"] - corners["10"])
    if len(by_group) != expected_groups or any(set(rows) != REGIMES for rows in by_group.values()):
        raise ValueError("expected complete four-regime source groups")
    useful_write = {group for group, rows in by_group.items()
                    if rows["new_only"]["corner_losses"]["10"] -
                    rows["new_only"]["corner_losses"]["11"] > threshold}
    harmful_write = {group for group, rows in by_group.items()
                     if any(rows[regime]["corner_losses"]["11"] -
                            rows[regime]["corner_losses"]["10"] > threshold
                            for regime in ("old_only", "neither_relevant"))}
    useful_keep = {group for group, rows in by_group.items()
                   if rows["old_only"]["corner_losses"]["00"] -
                   rows["old_only"]["corner_losses"]["10"] > threshold}
    useful_clear = {group for group, rows in by_group.items()
                    if rows["neither_relevant"]["corner_losses"]["10"] -
                    rows["neither_relevant"]["corner_losses"]["00"] > threshold}
    mean_write_new = statistics.mean(effects["new_only"]["write"])
    passed_write = (len(useful_write) >= min_groups and len(harmful_write) >= min_groups
                    and mean_write_new > 0)
    passed_retention = len(useful_keep) >= min_groups and len(useful_clear) >= min_groups
    return {
        "protocol": "joint_postcutoff_v1", "labels": len(seen_ids),
        "source_groups": len(by_group), "threshold_nll": threshold,
        "required_groups_per_direction": min_groups,
        "useful_write_groups": len(useful_write),
        "harmful_write_groups": len(harmful_write),
        "useful_keep_groups": len(useful_keep),
        "useful_clear_groups": len(useful_clear),
        "mean_write_gain_new_only": mean_write_new,
        "mean_effect_by_regime": {
            regime: {kind: statistics.mean(values) for kind, values in effects[regime].items()}
            for regime in sorted(REGIMES)
        },
        "best_corner_counts": {key: best[key] for key in CORNERS},
        "passed_write_gate": passed_write,
        "passed_retention_gate": passed_retention,
        "passed_joint_gate": passed_write and passed_retention,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--expected-groups", type=int, default=12)
    parser.add_argument("--min-groups", type=int, default=3)
    args = parser.parse_args()
    result = evaluate(args.labels, args.threshold, args.expected_groups, args.min_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
