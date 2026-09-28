"""Apply the preregistered source-group gate to memory-only title-recall labels."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REGIMES = frozenset(("both_relevant", "old_only", "new_only", "neither_relevant"))
CORNERS = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))


def evaluate(paths: list[Path], threshold: float = 0.005, expected_groups: int = 12,
             min_groups: int = 3) -> dict:
    if not paths or threshold < 0 or expected_groups < 1 or min_groups < 1:
        raise ValueError("paths and positive group counts with nonnegative threshold required")
    by_group = defaultdict(dict)
    splits = {}
    seen = set()
    effects = {condition: defaultdict(lambda: defaultdict(list))
               for condition in ("kv_intact", "memory_only")}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "joint_title_recall_v1" or row["regime"] not in REGIMES:
                raise ValueError("expected title recall joint labels")
            group, regime = row["group_id"], row["regime"]
            if row["id"] in seen or regime in by_group[group] or row["boundary"] != 2:
                raise ValueError("duplicate ID, group/regime, or wrong decision boundary")
            seen.add(row["id"])
            if group in splits and splits[group] != row["split"]:
                raise ValueError("source group crosses splits")
            splits[group] = row["split"]
            meta = row["future_meta"]
            if len(meta) != 2 or [item.get("reset_kv") for item in meta] != [False, True]:
                raise ValueError("KV-intact/memory-only future order required")
            actions = [tuple(action) for action in row["actions"]]
            if len(actions) != len(set(actions)) or not all(corner in actions for corner in CORNERS):
                raise ValueError("four unique corners required")
            if len(row["losses_by_future"]) != len(actions):
                raise ValueError("action/future loss mismatch")
            corners_by_condition = {}
            for future_index, condition in enumerate(("kv_intact", "memory_only")):
                corner_losses = {}
                for corner in CORNERS:
                    name = "".join(str(int(value)) for value in corner)
                    losses = row["losses_by_future"][actions.index(corner)]
                    if len(losses) != 2 or not math.isfinite(losses[future_index]):
                        raise ValueError("finite two-future corner losses required")
                    corner_losses[name] = losses[future_index]
                corners_by_condition[condition] = corner_losses
                effects[condition][regime]["write"].append(corner_losses["10"] - corner_losses["11"])
                effects[condition][regime]["keep"].append(corner_losses["00"] - corner_losses["10"])
            by_group[group][regime] = corners_by_condition
    if len(by_group) != expected_groups or any(set(rows) != REGIMES for rows in by_group.values()):
        raise ValueError("expected complete four-regime source groups")
    memory = {group: {regime: values["memory_only"] for regime, values in rows.items()}
              for group, rows in by_group.items()}
    useful_write = sum(rows["new_only"]["10"] - rows["new_only"]["11"] > threshold
                       for rows in memory.values())
    harmful_write = sum(any(rows[regime]["11"] - rows[regime]["10"] > threshold
                            for regime in ("old_only", "neither_relevant"))
                        for rows in memory.values())
    useful_keep = sum(rows["old_only"]["00"] - rows["old_only"]["10"] > threshold
                      for rows in memory.values())
    useful_clear = sum(rows["neither_relevant"]["10"] - rows["neither_relevant"]["00"] > threshold
                       for rows in memory.values())
    mean_write = statistics.mean(effects["memory_only"]["new_only"]["write"])
    passed_write = useful_write >= min_groups and harmful_write >= min_groups and mean_write > 0
    passed_retention = useful_keep >= min_groups and useful_clear >= min_groups
    return {
        "protocol": "joint_title_recall_v1", "labels": len(seen),
        "source_groups": len(memory), "threshold_nll": threshold,
        "required_groups_per_direction": min_groups,
        "useful_write_groups": useful_write, "harmful_write_groups": harmful_write,
        "useful_keep_groups": useful_keep, "useful_clear_groups": useful_clear,
        "mean_write_gain_new_only_memory_only": mean_write,
        "mean_effect_by_condition_and_regime": {
            condition: {regime: {kind: statistics.mean(values) for kind, values in kinds.items()}
                        for regime, kinds in sorted(regimes.items())}
            for condition, regimes in effects.items()
        },
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
