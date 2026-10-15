"""Construct the once-opened 256-group TempLAMA V6 test corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v6_fact_key_development import (
    AUDIT_PROTOCOL as V6_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from scripts.inspect_templama_utility_source import PROTOCOL as INSPECTION_PROTOCOL
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    BACKGROUND_HELDOUT_GROUPS,
    GROUPS,
    choose_materials,
)
from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    eligible_trajectories,
    source_inventory,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
    build_episode,
    write_json,
    write_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v6_templama_test_data"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v6_templama_test_data_audit"
GROUP_PREFIX = "usr-v6-templama-test"


def build(source_root: Path, inspection_path: Path,
          v6_root: Path) -> tuple[list[dict], dict]:
    inspection = json.loads(inspection_path.read_text())
    v6_audit_path = v6_root / "result_audit.json"
    v6_result_path = v6_root / "result.json"
    v6_audit = json.loads(v6_audit_path.read_text())
    if inspection.get("protocol") != INSPECTION_PROTOCOL or not inspection.get("passed"):
        raise ValueError("TempLAMA source inspection did not pass")
    if (v6_audit.get("protocol") != V6_AUDIT_PROTOCOL
            or not v6_audit.get("passed")
            or not v6_audit.get("release_passed")
            or v6_audit.get("next_stage") != "templama_v6_test_construction"
            or v6_audit.get("result_sha256") != sha256(v6_result_path)):
        raise ValueError("V6 development audit did not release TempLAMA test")
    _, inventories = source_inventory(source_root)
    expected = inspection["post_decontamination"]["test"]
    if len(eligible_trajectories(inventories["test"])) \
            != expected["eligible_trajectories"]:
        raise ValueError("TempLAMA test eligibility reconstruction mismatch")
    materials, quotas = choose_materials(inventories["test"])
    episodes = []
    for group_index, material in materials:
        mechanism = MECHANISMS[group_index % len(MECHANISMS)]
        length = LENGTHS[group_index % len(LENGTHS)]
        for twin in range(TWINS):
            episode = build_episode(
                group_index, mechanism, length, twin, material,
                group_prefix=GROUP_PREFIX,
            )
            episode["audit"]["templama_transition"] = material[
                "temporal_transition"
            ]
            episode["audit"]["source_partition"] = "test"
            episode["audit"]["background_subject_ids"] = material[
                "background_subject_ids"
            ]
            episode["audit"]["anchor_alternate_subject_id"] = material[
                "anchor_alternate_subject_id"
            ]
            episode["audit"]["background_heldout"] = material[
                "background_heldout"
            ]
            episodes.append(episode)
    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V6_FACT_KEY_SAFETY_PLAN.md",
        "constructor": (
            "frozen_v2_episode_grammar_and_hamilton_allocator_on_test_inventory"
        ),
        "source_manifest_sha256": sha256(
            source_root / "metadata" / "manifest.json"
        ),
        "source_inspection_sha256": sha256(inspection_path),
        "v6_development_result_sha256": sha256(v6_result_path),
        "v6_development_audit_sha256": sha256(v6_audit_path),
        "source_partition": "test",
        "relation_quotas": quotas,
        "counts": {
            "source_groups": GROUPS,
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(
                len(boundary["queries"])
                for row in episodes for boundary in row["boundaries"]
            ),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
            "target_relations": dict(Counter(
                row["audit"]["target_relation"]
                for row in episodes if row["twin"] == 0
            )),
            "target_subjects": len({
                row["audit"]["templama_transition"]["subject_id"]
                for row in episodes if row["twin"] == 0
            }),
            "background_subjects": len({
                subject for row in episodes if row["twin"] == 0
                for subject in row["audit"]["background_subject_ids"]
            }),
            "background_heldout_groups": sum(
                row["audit"]["background_heldout"]
                for row in episodes if row["twin"] == 0
            ),
        },
        "primary_action_space": "binary_endpoints_plus_global_clear",
        "current_write_gate": 1.0,
        "model_outputs_loaded": False,
        "development_release_audit_loaded": True,
        "development_examples_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "terminal_status": "templama_v6_test_data_built_pending_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    assert manifest["counts"]["source_groups"] == GROUPS
    assert len(episodes) == GROUPS * TWINS
    assert manifest["counts"]["boundaries"] == len(episodes) * BOUNDARIES == 1536
    assert manifest["counts"]["queries"] == len(episodes) * BOUNDARIES * 5 == 7680
    assert manifest["counts"]["target_subjects"] == GROUPS
    assert manifest["relation_quotas"] == manifest["counts"]["target_relations"]
    assert sum(manifest["relation_quotas"].values()) == GROUPS
    assert set(manifest["counts"]["lengths"].values()) == {128}
    assert max(manifest["counts"]["mechanisms"].values()) \
        - min(manifest["counts"]["mechanisms"].values()) <= TWINS
    assert all(len(row["events"]) == row["length"] for row in episodes)
    assert all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
    assert all(
        len(query["candidates"]) == 6 and query["answer"] in query["candidates"]
        for row in episodes for boundary in row["boundaries"]
        for query in boundary["queries"]
    )
    groups = defaultdict(list)
    for row in episodes:
        groups[row["source_group"]].append(row)
    assert len(groups) == GROUPS
    assert all({row["twin"] for row in values} == {0, 1}
               for values in groups.values())
    target_subjects = {
        row["audit"]["templama_transition"]["subject_id"] for row in episodes
    }
    assert all(
        len(set(row["audit"]["background_subject_ids"])) == 6
        and not (set(row["audit"]["background_subject_ids"]) & target_subjects)
        and row["audit"]["source_partition"] == "test"
        for row in episodes
    )
    target_rows = [row for row in episodes if row["twin"] == 0]
    background_counts = Counter(
        subject for row in target_rows
        for subject in row["audit"]["background_subject_ids"]
    )
    heldout_rows = [row for row in target_rows if row["audit"]["background_heldout"]]
    assert len(heldout_rows) == BACKGROUND_HELDOUT_GROUPS
    assert len({row["audit"]["target_relation"] for row in heldout_rows}) \
        == BACKGROUND_HELDOUT_GROUPS
    assert all(
        all(background_counts[subject] == 1
            for subject in row["audit"]["background_subject_ids"])
        for row in heldout_rows
    )
    assert not manifest["model_outputs_loaded"]
    assert manifest["development_release_audit_loaded"]
    assert not manifest["development_examples_loaded"]
    assert not manifest["v1_confirmation_loaded"]
    assert not manifest["external_benchmarks_loaded"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(args.source_root, args.inspection, args.v6_root)
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "test_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
