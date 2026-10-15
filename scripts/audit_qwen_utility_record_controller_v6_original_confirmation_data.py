"""Independent exact audit of V6 original-source confirmation episodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from scripts.audit_qwen_utility_semantic_record_gate_v1_data import (
    input_fields_are_clean,
    query_structure_is_valid,
    twin_structure_is_valid,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v6_original_confirmation import (
    AUDIT_PROTOCOL,
    GROUPS,
    MIN_RELATION_SUBJECTS,
    PROTOCOL,
    build,
    validate,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
    write_jsonl,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def provenance_is_unique(episodes: list[dict]) -> bool:
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    seen_hashes, seen_subjects = set(), set()
    for rows in groups.values():
        material_hashes = set(rows[0]["audit"]["material_source_fact_hashes"])
        input_hashes = set(rows[0]["audit"]["input_source_fact_hashes"])
        input_subjects = set(rows[0]["audit"]["input_source_subject_norms"])
        if len(material_hashes) != 12 or len(input_hashes) != 7 or len(input_subjects) != 7:
            return False
        if input_hashes & seen_hashes or input_subjects & seen_subjects:
            return False
        if any(set(row["audit"]["material_source_fact_hashes"]) != material_hashes
               or set(row["audit"]["input_source_fact_hashes"]) != input_hashes
               or set(row["audit"]["input_source_subject_norms"]) != input_subjects
               for row in rows[1:]):
            return False
        seen_hashes.update(input_hashes); seen_subjects.update(input_subjects)
    return True


def audit(source_root: Path, inspection_path: Path, release_root: Path,
          data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "confirmation_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = load_jsonl(episode_path)
    expected_episodes, expected_manifest = build(
        source_root, inspection_path, release_root
    )
    validate(expected_episodes, expected_manifest)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory) / episode_path.name
        write_jsonl(rebuilt, expected_episodes)
        rebuilt_hash = sha256(rebuilt)
    expected_manifest["files_sha256"] = {episode_path.name: rebuilt_hash}
    counts = {
        "source_groups": len({row["source_group"] for row in episodes}),
        "episodes": len(episodes),
        "boundaries": sum(len(row["boundaries"]) for row in episodes),
        "queries": sum(len(boundary["queries"]) for row in episodes
                       for boundary in row["boundaries"]),
        "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
        "lengths": dict(Counter(str(row["length"]) for row in episodes)),
        "target_relations": dict(Counter(
            row["audit"]["target_relation"] for row in episodes if row["twin"] == 0
        )),
    }
    relation_sets = {name: set(values) for name, values in manifest["splits"].items()}
    eligibility = manifest["confirmation_relation_eligibility"]
    included_relations = set(eligibility["included_relations"])
    mechanism_counts = list(counts["mechanisms"].values())
    target_counts = list(counts["target_relations"].values())
    checks = {
        "protocol_and_locked_inputs": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("source_inspection_sha256") == sha256(inspection_path)
            and manifest.get("release_result_sha256")
            == sha256(release_root / "result.json")
            and manifest.get("release_audit_sha256")
            == sha256(release_root / "result_audit.json")
        ),
        "exact_row_reconstruction": episodes == expected_episodes,
        "episode_file_hash": (
            sha256(episode_path) == manifest["files_sha256"][episode_path.name]
            == rebuilt_hash
        ),
        "manifest_reconstruction": manifest == expected_manifest,
        "registered_counts": (
            counts["source_groups"] == GROUPS
            and counts["episodes"] == GROUPS * TWINS
            and counts["boundaries"] == GROUPS * TWINS * BOUNDARIES
            and counts["queries"] == GROUPS * TWINS * BOUNDARIES * 5
        ),
        "balanced_design": (
            max(mechanism_counts) - min(mechanism_counts) <= TWINS
            and set(counts["lengths"].values()) == {GROUPS * TWINS // len(LENGTHS)}
            and max(target_counts) - min(target_counts) <= 1
            and set(counts["mechanisms"]) == set(MECHANISMS)
        ),
        "relation_partition_isolation": (
            not (relation_sets["confirmation"] & relation_sets["train"])
            and not (relation_sets["confirmation"] & relation_sets["development"])
            and all(set(row["audit"]["relation_ids"]) <= included_relations
                    for row in episodes)
        ),
        "capacity_eligibility_reconstructed": (
            included_relations == set(counts["target_relations"])
            and len(included_relations) == 6
            and eligibility["excluded_relations"] == ["P530"]
            and all(
                eligibility["subject_counts"][relation] >= MIN_RELATION_SUBJECTS
                for relation in included_relations
            )
            and all(
                eligibility["subject_counts"][relation] < MIN_RELATION_SUBJECTS
                for relation in eligibility["excluded_relations"]
            )
        ),
        "group_provenance_unique": provenance_is_unique(episodes),
        "input_audit_separation": input_fields_are_clean(episodes),
        "twin_structure": twin_structure_is_valid(episodes),
        "causal_queries": query_structure_is_valid(episodes),
        "confirmation_opened_external_sealed": (
            not manifest["development_episodes_generated"]
            and manifest["confirmation_episodes_generated"]
            and not manifest["forbidden_sources_loaded"]
            and not manifest["model_outputs_loaded"]
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "counts": counts,
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path),
        "release_audit_sha256": sha256(release_root / "result_audit.json"),
        "confirmation_episodes_generated": True,
        "model_outputs_loaded": False,
        "next_stage": "original_confirmation_label_construction" if passed else "stop",
        "terminal_status": (
            "original_confirmation_data_audit_passed" if passed
            else "original_confirmation_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_root, args.inspection, args.release_root, args.data_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
