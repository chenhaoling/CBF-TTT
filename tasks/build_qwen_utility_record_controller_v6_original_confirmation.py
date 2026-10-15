"""Build the sealed V6 LAMA/ParaRel original-source confirmation corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v6_templama_test import (
    AUDIT_PROTOCOL as RELEASE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
    build_episode,
    cross_partition_subjects,
    load_facts,
    load_patterns,
    read_json,
    write_json,
    write_jsonl,
)
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
    choose_formal_material,
    input_source_rows,
    material_rows,
)


PROTOCOL = "qwen_utility_record_controller_v6_original_confirmation_data"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v6_original_confirmation_data_audit"
INSPECTION_PROTOCOL = "qwen_utility_semantic_record_source_inspection_v1"
GROUPS = 64
GROUP_PREFIX = "usr-v6-original-confirmation"


def build(
    source_root: Path,
    inspection_path: Path,
    release_root: Path,
) -> tuple[list[dict], dict]:
    inspection = read_json(inspection_path)
    release_result_path = release_root / "result.json"
    release_audit_path = release_root / "result_audit.json"
    release_result = read_json(release_result_path)
    release_audit = read_json(release_audit_path)
    if inspection.get("protocol") != INSPECTION_PROTOCOL or not inspection.get("passed"):
        raise ValueError("source inspection did not pass")
    if release_result.get("protocol") != "qwen_utility_record_controller_v6_templama_test_confirmation":
        raise ValueError("unexpected V6 release result")
    if (release_audit.get("protocol") != RELEASE_AUDIT_PROTOCOL
            or not release_audit.get("passed")
            or not release_audit.get("confirmation_passed")):
        raise ValueError("V6 TempLAMA confirmation did not release original source")
    if release_audit.get("result_sha256") != sha256(release_result_path):
        raise ValueError("V6 release result changed after audit")

    splits = inspection["splits"]
    relations = {relation for values in splits.values() for relation in values}
    patterns = load_patterns(source_root, relations)
    facts = load_facts(source_root, relations)
    removed_subjects = cross_partition_subjects(facts, splits)
    facts = {
        relation: [row for row in rows if row["subject_norm"] not in removed_subjects]
        for relation, rows in facts.items()
    }
    confirmation_relations = splits["confirmation"]
    forbidden_hashes: set[str] = set()
    forbidden_subjects: set[str] = set()
    episodes = []
    for group_index in range(GROUPS):
        mechanism = MECHANISMS[group_index % len(MECHANISMS)]
        length = LENGTHS[group_index % len(LENGTHS)]
        material = choose_formal_material(
            group_index, confirmation_relations, facts, patterns,
            forbidden_hashes, forbidden_subjects,
            strict_same_relation_value=False,
        )
        provenance = material_rows(material)
        input_provenance = input_source_rows(material)
        for twin in range(TWINS):
            episode = build_episode(
                group_index, mechanism, length, twin, material,
                group_prefix=GROUP_PREFIX,
            )
            episode["audit"]["material_source_fact_hashes"] = sorted(
                row["source_fact_hash"] for row in provenance
            )
            episode["audit"]["material_source_subject_norms"] = sorted(
                row["subject_norm"] for row in provenance
            )
            episode["audit"]["input_source_fact_hashes"] = sorted(
                row["source_fact_hash"] for row in input_provenance
            )
            episode["audit"]["input_source_subject_norms"] = sorted(
                row["subject_norm"] for row in input_provenance
            )
            episodes.append(episode)

    input_hashes = {value for row in episodes
                    for value in row["audit"]["input_source_fact_hashes"]}
    input_subjects = {value for row in episodes
                      for value in row["audit"]["input_source_subject_norms"]}
    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V6_ORIGINAL_SOURCE_CONFIRMATION_PLAN.md",
        "source_manifest_sha256": inspection["source_manifest_sha256"],
        "source_inspection_sha256": sha256(inspection_path),
        "release_result_sha256": sha256(release_result_path),
        "release_audit_sha256": sha256(release_audit_path),
        "counts": {
            "source_groups": GROUPS,
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
            "target_relations": dict(Counter(
                row["audit"]["target_relation"] for row in episodes if row["twin"] == 0
            )),
            "input_source_facts": len(input_hashes),
            "input_source_subjects": len(input_subjects),
            "cross_partition_subjects_removed": len(removed_subjects),
        },
        "splits": splits,
        "partition": "lama_pararel_confirmation",
        "alpha_grid": [0.0, 0.5, 1.0],
        "primary_action_space": "binary_with_half_ablation",
        "current_write_gate": 1.0,
        "model_outputs_loaded": False,
        "development_episodes_generated": False,
        "confirmation_episodes_generated": True,
        "forbidden_sources_loaded": False,
        "terminal_status": "original_confirmation_data_built_pending_independent_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    assert manifest["counts"]["source_groups"] == GROUPS
    assert len(episodes) == GROUPS * TWINS
    assert manifest["counts"]["boundaries"] == len(episodes) * BOUNDARIES
    assert manifest["counts"]["queries"] == len(episodes) * BOUNDARIES * 5
    assert set(manifest["counts"]["lengths"].values()) == {GROUPS * TWINS // 4}
    assert max(manifest["counts"]["mechanisms"].values()) - min(
        manifest["counts"]["mechanisms"].values()
    ) <= TWINS
    assert max(manifest["counts"]["target_relations"].values()) - min(
        manifest["counts"]["target_relations"].values()
    ) <= 1
    # Seven confirmation relations yield five stable other-relation rows, so
    # seven unique source subjects enter each group.
    assert manifest["counts"]["input_source_facts"] == GROUPS * 7
    assert manifest["counts"]["input_source_subjects"] == GROUPS * 7
    assert all(set(row["audit"]["relation_ids"]) <= set(manifest["splits"]["confirmation"])
               for row in episodes)
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    assert len(groups) == GROUPS
    assert all({row["twin"] for row in values} == {0, 1} for values in groups.values())
    assert not manifest["model_outputs_loaded"]
    assert not manifest["development_episodes_generated"]
    assert manifest["confirmation_episodes_generated"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(args.source_root, args.inspection, args.release_root)
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "confirmation_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
