"""Independent reconstruction audit for the once-opened V6 test corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    BACKGROUND_HELDOUT_GROUPS,
    GROUPS,
)
from tasks.build_qwen_utility_record_controller_v6_templama_test import (
    AUDIT_PROTOCOL,
    PROTOCOL,
    build,
    validate,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import BOUNDARIES, TWINS
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(data_root: Path, source_root: Path, inspection_path: Path,
          v6_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "test_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = load_jsonl(episode_path)
    expected_episodes, expected_manifest = build(
        source_root, inspection_path, v6_root
    )
    observed_without_hashes = {
        key: value for key, value in manifest.items() if key != "files_sha256"
    }
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    target_rows = [row for row in episodes if row["twin"] == 0]
    target_subjects = {
        row["audit"]["templama_transition"]["subject_id"] for row in target_rows
    }
    background_occurrences = Counter(
        subject for row in target_rows
        for subject in row["audit"]["background_subject_ids"]
    )
    heldout = [row for row in target_rows if row["audit"]["background_heldout"]]
    target_relations = Counter(row["audit"]["target_relation"] for row in target_rows)
    checks = {
        "source_inspection_and_v6_release_locked": (
            manifest["source_manifest_sha256"]
            == sha256(source_root / "metadata" / "manifest.json")
            and manifest["source_inspection_sha256"] == sha256(inspection_path)
            and manifest["v6_development_result_sha256"]
            == sha256(v6_root / "result.json")
            and manifest["v6_development_audit_sha256"]
            == sha256(v6_root / "result_audit.json")
        ),
        "manifest_and_file_hashes": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("files_sha256") == {
                "test_episodes.jsonl": sha256(episode_path)
            }
        ),
        "exact_builder_reconstruction": (
            episodes == expected_episodes
            and observed_without_hashes == expected_manifest
        ),
        "registered_size_and_allocation": (
            len(groups) == GROUPS
            and len(episodes) == GROUPS * TWINS
            and len(target_subjects) == GROUPS
            and dict(target_relations) == manifest["relation_quotas"]
            and len(target_relations) == 9
            and all(value >= 8 for value in target_relations.values())
        ),
        "twins_and_sparse_boundaries": all(
            len(values) == TWINS
            and {row["twin"] for row in values} == {0, 1}
            and all(len(row["boundaries"]) == BOUNDARIES for row in values)
            for values in groups.values()
        ),
        "unique_real_test_targets": all(
            row["audit"]["source_partition"] == "test"
            and row["audit"]["templama_transition"]["old_year"]
            < row["audit"]["templama_transition"]["new_year"]
            and row["audit"]["templama_transition"]["old_value_id"]
            != row["audit"]["templama_transition"]["new_value_id"]
            for row in episodes
        ),
        "background_bank_and_heldout": (
            len(heldout) == BACKGROUND_HELDOUT_GROUPS
            and len({row["audit"]["target_relation"] for row in heldout})
            == BACKGROUND_HELDOUT_GROUPS
            and all(
                len(set(row["audit"]["background_subject_ids"])) == 6
                and not (set(row["audit"]["background_subject_ids"])
                         & target_subjects)
                for row in target_rows
            )
            and all(
                all(background_occurrences[subject] == 1
                    for subject in row["audit"]["background_subject_ids"])
                for row in heldout
            )
        ),
        "query_contract": all(
            len(boundary["queries"]) == 5
            and all(len(query["candidates"]) == 6
                    and query["answer"] in query["candidates"]
                    for query in boundary["queries"])
            for episode in episodes for boundary in episode["boundaries"]
        ),
        "binary_action_and_write_gate_frozen": (
            manifest["primary_action_space"]
            == "binary_endpoints_plus_global_clear"
            and manifest["current_write_gate"] == 1.0
        ),
        "only_release_audit_opened_before_test": (
            manifest["model_outputs_loaded"] is False
            and manifest["development_release_audit_loaded"] is True
            and manifest["development_examples_loaded"] is False
            and manifest["v1_confirmation_loaded"] is False
            and manifest["external_benchmarks_loaded"] is False
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
        "queries": sum(
            len(boundary["queries"])
            for row in episodes for boundary in row["boundaries"]
        ),
        "relation_quotas": dict(sorted(target_relations.items())),
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path),
        "next_stage": "templama_v6_test_binary_labels" if passed else "stop",
        "terminal_status": (
            "templama_v6_test_data_audit_passed"
            if passed else "templama_v6_test_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.data_root, args.source_root, args.inspection, args.v6_root)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
