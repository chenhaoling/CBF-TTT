"""Independent invariant audit for the V3 source-locked corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks.build_native_layerwise_write_gate_v3 import (
    GROUP_PREFIX,
    MECHANISM_SOURCE,
    PROTOCOL,
    SOURCE_GROUPS,
    SPLIT_PER_MECHANISM,
    stable_rank,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contains_forbidden_online_field(value) -> bool:
    forbidden = {"text", "statement", "answer", "subject", "relation", "value", "mechanism", "future", "oracle"}
    if isinstance(value, dict):
        return any(any(word in str(key).lower() for word in forbidden)
                   or contains_forbidden_online_field(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_online_field(item) for item in value)
    return isinstance(value, str) and len(value.split()) > 8


def audit(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    source_manifest = json.loads((root / "semantic_source_manifest.json").read_text())
    episodes = load_jsonl(root / "semantic_source_episodes.jsonl")
    semantic = load_jsonl(root / "semantic_audit.jsonl")
    model = {
        split: load_jsonl(root / f"{split}_model_inputs.jsonl")
        for split in SPLIT_PER_MECHANISM
    }
    checks = {}
    checks["protocol_and_plan"] = manifest["protocol"] == PROTOCOL and len(manifest["plan_sha256"]) == 64
    checks["file_hashes"] = all(digest(root / name) == value for name, value in manifest["files_sha256"].items())
    checks["registered_counts"] = (
        manifest["source_groups"] == SOURCE_GROUPS
        and {name: len(rows) for name, rows in model.items()}
        == {name: value * len(MECHANISM_SOURCE) for name, value in SPLIT_PER_MECHANISM.items()}
        and len(episodes) == SOURCE_GROUPS * 2
        and len(semantic) == SOURCE_GROUPS
    )
    episode_groups = defaultdict(list)
    for row in episodes:
        episode_groups[row["source_group"]].append(row)
    checks["twins_and_boundaries"] = (
        len(episode_groups) == SOURCE_GROUPS
        and all({row["twin"] for row in rows} == {0, 1} for rows in episode_groups.values())
        and all(len(row["boundaries"]) == 3 for row in episodes)
    )
    seen_facts, seen_subjects = set(), set()
    disjoint = True
    for rows in episode_groups.values():
        facts = set(rows[0]["audit"]["material_source_fact_hashes"])
        subjects = set(rows[0]["audit"]["material_source_subject_norms"])
        disjoint &= not (seen_facts & facts) and not (seen_subjects & subjects)
        disjoint &= all(set(row["audit"]["material_source_fact_hashes"]) == facts
                        and set(row["audit"]["material_source_subject_norms"]) == subjects
                        for row in rows)
        seen_facts.update(facts); seen_subjects.update(subjects)
    checks["source_world_disjointness"] = bool(disjoint)
    all_model = [row for rows in model.values() for row in rows]
    checks["model_ids_unique"] = len({row["group_id"] for row in all_model}) == SOURCE_GROUPS
    checks["online_plaintext_absent"] = not any(contains_forbidden_online_field(row) for row in all_model)
    by_split_mechanism = {
        split: Counter(row["mechanism"] for row in semantic if row["split"] == split)
        for split in model
    }
    checks["balanced_splits"] = all(
        set(counts.values()) == {SPLIT_PER_MECHANISM[split]} and len(counts) == len(MECHANISM_SOURCE)
        for split, counts in by_split_mechanism.items()
    )
    assigned = manifest["assignment"]
    expected_order = True
    for source_mechanism in MECHANISM_SOURCE.values():
        values = sorted(
            (group for group, rows in episode_groups.items() if rows[0]["mechanism"] == source_mechanism),
            key=stable_rank,
        )
        cursor = 0
        for split, count in SPLIT_PER_MECHANISM.items():
            expected_order &= all(assigned[group] == split for group in values[cursor:cursor + count])
            cursor += count
    checks["deterministic_assignment"] = bool(expected_order)
    checks["pilot_exclusion_and_model_free"] = (
        source_manifest["pilot_overlap"] == {"source_fact_hashes": 0, "subjects": 0}
        and not manifest["model_outputs_loaded"] and not manifest["sealed_test_scored"]
    )
    checks["semantic_separation"] = (
        manifest["semantic_online_separation"]
        and {row["group_id"] for row in semantic} == {row["group_id"] for row in all_model}
        and all("mechanism_hash" not in row for row in all_model)
    )
    result = {
        "protocol": PROTOCOL + "_data_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {name: len(rows) for name, rows in model.items()},
        "manifest_sha256": digest(root / "manifest.json"),
        "source_manifest_sha256": digest(root / "semantic_source_manifest.json"),
        "test_opened": False,
    }
    (root / "data_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
