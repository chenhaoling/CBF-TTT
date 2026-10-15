"""Independent reconstruction audit for TempLAMA V2 pilot episodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    AUDIT_PROTOCOL,
    GROUPS_PER_RELATION,
    PARAPHRASES,
    PROTOCOL,
    build,
    validate,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(data_root: Path, source_root: Path, inspection_path: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "pilot_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = load_jsonl(episode_path)
    expected_episodes, expected_manifest = build(source_root, inspection_path)
    observed_without_hashes = {
        key: value for key, value in manifest.items() if key != "files_sha256"
    }
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    target_rows = [row for row in episodes if row["twin"] == 0]
    target_relations = Counter(row["audit"]["target_relation"] for row in target_rows)
    mechanisms = Counter(row["mechanism"] for row in episodes)
    lengths = Counter(row["length"] for row in episodes)
    checks = {
        "source_and_inspection_locked": (
            manifest["source_manifest_sha256"]
            == sha256(source_root / "metadata" / "manifest.json")
            and manifest["source_inspection_sha256"] == sha256(inspection_path)
        ),
        "manifest_and_file_hashes": (
            manifest.get("protocol") == PROTOCOL
            and manifest["files_sha256"] == {
                "pilot_episodes.jsonl": sha256(episode_path)
            }
        ),
        "exact_builder_reconstruction": (
            episodes == expected_episodes
            and observed_without_hashes == expected_manifest
        ),
        "registered_size_and_balance": (
            len(groups) == 72 and len(episodes) == 144
            and len(target_relations) == len(PARAPHRASES) == 9
            and set(target_relations.values()) == {GROUPS_PER_RELATION}
            and max(mechanisms.values()) - min(mechanisms.values()) <= 2
            and set(lengths.values()) == {36}
        ),
        "twins_and_sparse_boundaries": (
            all({row["twin"] for row in values} == {0, 1}
                and len(values) == 2 for values in groups.values())
            and all(len(row["boundaries"]) == 3 for row in episodes)
        ),
        "real_temporal_transition_per_group": (
            len({row["audit"]["templama_transition"]["subject_id"]
                 for row in target_rows}) == 72
            and all(
                row["audit"]["templama_transition"]["old_year"]
                < row["audit"]["templama_transition"]["new_year"]
                and row["audit"]["templama_transition"]["old_value_id"]
                != row["audit"]["templama_transition"]["new_value_id"]
                and row["audit"]["source_partition"] == "train"
                for row in episodes
            )
        ),
        "query_contract": all(
            len(boundary["queries"]) == 5
            and all(len(query["candidates"]) == 6
                    and query["answer"] in query["candidates"]
                    for query in boundary["queries"])
            for episode in episodes for boundary in episode["boundaries"]
        ),
        "ternary_grid_and_incoming_write_frozen": (
            manifest["alpha_grid"] == [0.0, 0.5, 1.0]
            and manifest["current_write_gate"] == 1.0
        ),
        "heldouts_and_forbidden_sources_sealed": (
            manifest["v1_confirmation_loaded"] is False
            and manifest["development_episodes_generated"] is False
            and manifest["test_episodes_generated"] is False
            and manifest["forbidden_sources_loaded"] is False
            and manifest["model_outputs_loaded"] is False
        ),
    }
    try:
        validate(episodes, manifest)
        checks["builder_invariants"] = True
    except (AssertionError, ValueError, KeyError):
        checks["builder_invariants"] = False
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "source_groups": len(groups),
        "episodes": len(episodes),
        "boundaries": sum(len(row["boundaries"]) for row in episodes),
        "queries": sum(len(boundary["queries"]) for row in episodes
                       for boundary in row["boundaries"]),
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path),
        "source_inspection_sha256": sha256(inspection_path),
        "next_stage": "templama_resource_pilot_labels" if passed else "stop",
        "terminal_status": (
            "templama_pilot_data_audit_passed" if passed
            else "templama_pilot_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.source_root, args.inspection)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
