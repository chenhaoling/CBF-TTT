"""Split CBF scenarios evenly by source group for independent GPU workers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path


SPLITS = ("train", "dev", "test")


def shard_scenarios(source: Path, output_dir: Path, shards: int) -> dict:
    if shards < 1:
        raise ValueError("--shards must be positive")

    groups_by_split = defaultdict(set)
    split_by_group = {}
    scenario_ids = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split, group, scenario_id = row["split"], row["group_id"], row["id"]
            if split not in SPLITS:
                raise ValueError(f"unknown split {split!r} at line {line_number}")
            if not group or not scenario_id:
                raise ValueError(f"empty group or scenario ID at line {line_number}")
            if group in split_by_group and split_by_group[group] != split:
                raise ValueError(f"group {group!r} occurs in multiple splits")
            if scenario_id in scenario_ids:
                raise ValueError(f"duplicate scenario ID {scenario_id!r}")
            split_by_group[group] = split
            groups_by_split[split].add(group)
            scenario_ids.add(scenario_id)

    if not scenario_ids:
        raise ValueError("input contains no scenarios")

    # Sorting makes the assignment independent of input line order and keeps
    # each split's group count balanced to within one group per shard.
    assignment = {}
    for split in SPLITS:
        for index, group in enumerate(sorted(groups_by_split[split])):
            assignment[group] = index % shards
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {split: [0] * shards for split in SPLITS}
    paths = {(split, shard): output_dir / f"{split}_shard{shard}.jsonl"
             for split in SPLITS for shard in range(shards)}
    temporary = {key: path.with_suffix(path.suffix + ".incomplete") for key, path in paths.items()}
    try:
        with ExitStack() as stack:
            sinks = {key: stack.enter_context(path.open("w", encoding="utf-8"))
                     for key, path in temporary.items()}
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    split = row["split"]
                    shard = assignment[row["group_id"]]
                    sinks[(split, shard)].write(line if line.endswith("\n") else line + "\n")
                    counts[split][shard] += 1
        for key, path in paths.items():
            temporary[key].replace(path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    manifest = {
        "source": str(source), "shards": shards, "scenarios": len(scenario_ids),
        "splits": {
            split: {
                "groups": len(groups_by_split[split]),
                "scenarios_per_shard": counts[split],
                "groups_per_shard": [sum(assignment[group] == shard for group in groups_by_split[split])
                                     for shard in range(shards)],
            }
            for split in SPLITS
        },
    }
    if sum(sum(values["scenarios_per_shard"]) for values in manifest["splits"].values()) != len(scenario_ids):
        raise RuntimeError("shard counts do not match source")
    (output_dir / "shard_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(shard_scenarios(args.input, args.output_dir, args.shards)))


if __name__ == "__main__":
    main()
