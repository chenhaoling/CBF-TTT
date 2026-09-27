"""Apply the preregistered source-group decision rules to joint_v2 pilot labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CORNERS = {"00": (0.0, 0.0), "01": (0.0, 1.0),
           "10": (1.0, 0.0), "11": (1.0, 1.0)}


def evaluate(paths: list[Path], threshold: float = 0.005) -> dict:
    if not paths or threshold <= 0:
        raise ValueError("label paths and a positive threshold are required")
    seen = set()
    by_regime = defaultdict(list)
    meaningful_winners = defaultdict(set)
    interior_count = 0
    total = 0
    max_reserved = 0.0
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["id"], row["boundary"])
                if key in seen or row["protocol"] != "joint_v2":
                    raise ValueError(f"duplicate or incompatible v2 label {key}")
                seen.add(key)
                actions = [tuple(action) for action in row["actions"]]
                if len(actions) != 9 or any(corner not in actions for corner in CORNERS.values()):
                    raise ValueError(f"invalid v2 grid for {key}")
                futures = row["future_meta"]
                gaps = [future["gap_chunks"] for future in futures]
                if len(gaps) != 2 or gaps[0] != 0 or gaps[1] < 1:
                    raise ValueError(f"invalid short/long futures for {key}")
                long_losses = [losses[1] for losses in row["losses_by_future"]]
                if len(long_losses) != len(actions):
                    raise ValueError(f"invalid action future table for {key}")
                corners = {name: long_losses[actions.index(action)] for name, action in CORNERS.items()}
                write_gain = min(corners["00"], corners["10"]) - min(corners["01"], corners["11"])
                retain_gain = min(corners["00"], corners["01"]) - min(corners["10"], corners["11"])
                by_regime[row["regime"]].append({"group_id": row["group_id"],
                                                 "write_gain": write_gain,
                                                 "retain_gain": retain_gain})
                order = sorted(corners, key=corners.get)
                if corners[order[1]] - corners[order[0]] > threshold:
                    meaningful_winners[order[0]].add(row["group_id"])
                interior_count += min(corners.values()) - min(long_losses) > threshold
                max_reserved = max(max_reserved, row["peak_reserved_gib"] or 0.0)
                total += 1
    if not seen:
        raise ValueError("no v2 labels found")
    required = (
        "old_relevant_new_informative", "old_relevant_new_noise",
        "old_conflict_new_correction", "old_conflict_new_noise",
    )
    if any(len(by_regime[name]) != total // 4 for name in required) or total % 4:
        raise ValueError("v2 pilot lacks balanced four-case source groups")
    group_sets = [{row["group_id"] for row in by_regime[name]} for name in required]
    if any(len(groups) != total // 4 or groups != group_sets[0] for groups in group_sets):
        raise ValueError("v2 pilot source groups do not match across cases")
    write_counts = {name: sum(row["write_gain"] > threshold for row in by_regime[name])
                    for name in required}
    skip_counts = {name: sum(row["write_gain"] < -threshold for row in by_regime[name])
                   for name in required}
    retain_counts = {name: sum(row["retain_gain"] > threshold for row in by_regime[name])
                     for name in required}
    winner_counts = {name: len(meaningful_winners[name]) for name in CORNERS}
    interior_fraction = interior_count / total
    checks = {
        "new_informative_write": write_counts["old_relevant_new_informative"] >= 3,
        "correction_write": write_counts["old_conflict_new_correction"] >= 3,
        "redundant_or_noise_skip": skip_counts["old_relevant_new_noise"] >= 3,
        "obsolete_and_noise_skip": skip_counts["old_conflict_new_noise"] >= 3,
        "old_relevant_retention": max(retain_counts["old_relevant_new_informative"],
                                      retain_counts["old_relevant_new_noise"]) >= 3,
        "multiple_meaningful_corners": sum(count >= 3 for count in winner_counts.values()) >= 2,
        "interior_not_dominant": interior_fraction <= 0.2,
        "memory_headroom": max_reserved < 28.0,
    }
    return {
        "protocol": "joint_v2", "labels": total, "source_groups": total // 4,
        "meaningful_nll_threshold": threshold,
        "write_gain_groups_by_regime": write_counts,
        "skip_write_groups_by_regime": skip_counts,
        "retain_old_groups_by_regime": retain_counts,
        "meaningful_corner_winner_groups": winner_counts,
        "substantive_interior_fraction": interior_fraction,
        "max_peak_reserved_gib": max_reserved,
        "checks": checks, "passed_before_repeat_check": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    result = evaluate(args.labels, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
