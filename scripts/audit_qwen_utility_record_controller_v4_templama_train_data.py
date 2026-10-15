"""Independent exact-reconstruction audit of the V4 TempLAMA train corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v4_templama_train import (
    AUDIT_PROTOCOL,
    EXPECTED_QUOTAS,
    GROUPS,
    MINIMUM_PER_RELATION,
    PROTOCOL,
    build,
    validate,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(data_root: Path, source_root: Path, inspection_path: Path,
          pilot_root: Path, v3_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "train_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = load_jsonl(episode_path)
    expected_episodes, expected_manifest = build(
        source_root, inspection_path, pilot_root, v3_root
    )
    observed_without_hashes = {
        key: value for key, value in manifest.items() if key != "files_sha256"
    }
    target_rows = [row for row in episodes if row["twin"] == 0]
    target_subjects = {
        row["audit"]["templama_transition"]["subject_id"]
        for row in target_rows
    }
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    relations = Counter(row["audit"]["target_relation"] for row in target_rows)
    mechanisms = Counter(row["mechanism"] for row in episodes)
    lengths = Counter(row["length"] for row in episodes)
    background_occurrences = Counter(
        subject for row in target_rows
        for subject in row["audit"]["background_subject_ids"]
    )
    checks = {
        "upstream_artifacts_locked": (
            manifest["source_manifest_sha256"]
            == sha256(source_root / "metadata" / "manifest.json")
            and manifest["source_inspection_sha256"] == sha256(inspection_path)
            and manifest["pilot_audit_sha256"]
            == sha256(pilot_root / "pilot_audit.json")
            and manifest["v3_result_audit_sha256"]
            == sha256(v3_root / "result_audit.json")
        ),
        "manifest_and_file_hashes": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("files_sha256") == {
                "train_episodes.jsonl": sha256(episode_path)
            }
        ),
        "exact_builder_reconstruction": (
            episodes == expected_episodes
            and observed_without_hashes == expected_manifest
        ),
        "registered_size_and_relation_allocation": (
            len(groups) == GROUPS
            and len(episodes) == GROUPS * TWINS
            and dict(relations) == EXPECTED_QUOTAS
            and all(value >= MINIMUM_PER_RELATION
                    for value in relations.values())
        ),
        "twins_mechanisms_lengths_and_sparse_boundaries": (
            all(len(rows) == TWINS
                and {row["twin"] for row in rows} == set(range(TWINS))
                for rows in groups.values())
            and set(mechanisms) == set(MECHANISMS)
            and max(mechanisms.values()) - min(mechanisms.values()) <= TWINS
            and set(lengths) == set(LENGTHS)
            and set(lengths.values()) == {GROUPS * TWINS // len(LENGTHS)}
            and all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
        ),
        "unique_real_train_targets": (
            len(target_subjects) == GROUPS
            and all(
                row["audit"]["source_partition"] == "train"
                and row["audit"]["templama_transition"]["old_year"]
                < row["audit"]["templama_transition"]["new_year"]
                and row["audit"]["templama_transition"]["old_value_id"]
                != row["audit"]["templama_transition"]["new_value_id"]
                for row in episodes
            )
        ),
        "background_target_isolation": all(
            len(row["audit"]["background_subject_ids"]) == 6
            and len(set(row["audit"]["background_subject_ids"])) == 6
            and not set(row["audit"]["background_subject_ids"]) & target_subjects
            and row["audit"]["anchor_alternate_subject_id"]
            not in target_subjects
            for row in target_rows
        ),
        "query_contract": all(
            len(boundary["queries"]) == 5
            and all(len(query["candidates"]) == 6
                    and query["answer"] in query["candidates"]
                    for query in boundary["queries"])
            for episode in episodes for boundary in episode["boundaries"]
        ),
        "registered_binary_action_space": (
            manifest["primary_action_space"]
            == "binary_endpoints_plus_global_clear"
            and manifest["current_write_gate"] == 1.0
        ),
        "validation_test_and_external_evidence_sealed": (
            manifest["model_outputs_loaded"] is False
            and manifest["validation_episodes_loaded"] is False
            and manifest["test_episodes_generated"] is False
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
        "queries": sum(len(boundary["queries"]) for row in episodes
                       for boundary in row["boundaries"]),
        "relation_quotas": dict(sorted(relations.items())),
        "background_bank": {
            "unique_subjects": len(background_occurrences),
            "reused_subjects": sum(
                count > 1 for count in background_occurrences.values()
            ),
            "maximum_group_occurrences": max(background_occurrences.values()),
        },
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path),
        "next_stage": "templama_v4_train_binary_labels" if passed else "stop",
        "terminal_status": (
            "templama_v4_train_data_audit_passed" if passed
            else "templama_v4_train_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.data_root, args.source_root, args.inspection,
        args.pilot_root, args.v3_root,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
