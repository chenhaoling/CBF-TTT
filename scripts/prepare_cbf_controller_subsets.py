"""Create nested, source-group-safe label subsets for a controller learning curve."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


def prepare_subsets(train_paths: list[Path], dev_paths: list[Path], output_dir: Path,
                    group_counts: list[int], seed: int = 42) -> dict:
    counts = sorted(set(group_counts))
    if not counts or counts[0] < 1:
        raise ValueError("group counts must be positive")

    train_groups, dev_groups = set(), set()
    update_rules = set()
    for split, paths, groups in (("train", train_paths, train_groups), ("dev", dev_paths, dev_groups)):
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row["split"] != split:
                        raise ValueError(f"unexpected split in {path}: {row['split']}")
                    update_rules.add(row.get("update_rule", "forget"))
                    groups.add(row["group_id"])
    if len(update_rules) != 1 or not update_rules <= {"forget", "write"}:
        raise ValueError("train and dev labels must share one known update rule")
    update_rule = update_rules.pop()
    label_key = "write_gate_star" if update_rule == "write" else "alpha_star"
    if train_groups & dev_groups:
        raise ValueError("train and dev share source groups")
    if counts[-1] > len(train_groups):
        raise ValueError(f"requested {counts[-1]} groups but only {len(train_groups)} exist")

    ordered = sorted(train_groups)
    random.Random(seed).shuffle(ordered)
    selected = {size: set(ordered[:size]) for size in counts}
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    coefficient_counts = {size: Counter() for size in counts}
    with ExitStack() as stack:
        sinks = {
            size: stack.enter_context((output_dir / f"train_groups_{size}.jsonl").open("w", encoding="utf-8"))
            for size in counts
        }
        for path in train_paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    group = row["group_id"]
                    for size in counts:
                        if group in selected[size]:
                            sinks[size].write(line if line.endswith("\n") else line + "\n")
                            label_counts[size] += 1
                            coefficient_counts[size][str(row[label_key])] += 1

    result = {
        "seed": seed, "train_source_groups": len(train_groups), "dev_source_groups": len(dev_groups),
        "update_rule": update_rule,
        "subsets": {
            str(size): {
                "source_groups": size,
                "labels": label_counts[size],
                ("gate_counts" if update_rule == "write" else "alpha_counts"):
                    dict(coefficient_counts[size]),
                "file": str(output_dir / f"train_groups_{size}.jsonl"),
            }
            for size in counts
        },
    }
    (output_dir / "subset_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", nargs="+", required=True, type=Path)
    parser.add_argument("--dev-labels", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-counts", nargs="+", type=int, default=(250, 500, 1000))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_subsets(args.train_labels, args.dev_labels, args.output_dir,
                                     args.group_counts, args.seed), indent=2))


if __name__ == "__main__":
    main()
