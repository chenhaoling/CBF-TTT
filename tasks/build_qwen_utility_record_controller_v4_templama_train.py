"""Build the 512-group official TempLAMA train corpus for V4."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from scripts.inspect_templama_utility_source import PROTOCOL as INSPECTION_PROTOCOL
from tasks.evaluate_qwen_utility_record_controller_v3_templama_development import (
    AUDIT_PROTOCOL as V3_RESULT_AUDIT_PROTOCOL,
)
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    hamilton_quotas,
)
from tasks.build_qwen_utility_record_controller_v2_templama_pilot import (
    PARAPHRASES,
    eligible_trajectories,
    source_inventory,
    temporal_transition,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
    build_episode,
    stable_rank,
    write_json,
    write_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v4_templama_train_data"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v4_templama_train_data_audit"
GROUPS = 512
MINIMUM_PER_RELATION = 16
SEED = 202610174
GROUP_PREFIX = "usr-v4-templama-train"
EXPECTED_QUOTAS = {
    "P102": 75, "P108": 87, "P127": 31, "P286": 60, "P39": 64,
    "P488": 47, "P54": 70, "P6": 54, "P69": 24,
}


def ordered_keys(keys, *namespace):
    return sorted(keys, key=lambda key: (
        stable_rank(SEED, *namespace, *key), key
    ))


def choose_background(keys, eligible, group_index, role, used_subjects,
                      used_values):
    for key in ordered_keys(keys, "background", group_index, role):
        row = eligible[key][0][1]
        if key[0] in used_subjects or row["value_norm"] in used_values:
            continue
        return key, row
    raise ValueError(f"no V4 background candidate for {group_index}:{role}")


def choose_materials(trajectories: dict) -> tuple[list[tuple[int, dict]], dict]:
    eligible = eligible_trajectories(trajectories)
    capacities = Counter(key[1] for key in eligible)
    quotas = hamilton_quotas(
        dict(capacities), total=GROUPS, minimum=MINIMUM_PER_RELATION
    )
    if quotas != EXPECTED_QUOTAS:
        raise ValueError(f"V4 quota mismatch: {quotas}")
    targets = []
    target_subjects = set()
    for relation in sorted(quotas, key=lambda value: (
        capacities[value] - quotas[value], value
    )):
        candidates = ordered_keys(
            [key for key in eligible if key[1] == relation],
            "target", relation,
        )
        selected = [key for key in candidates if key[0] not in target_subjects]
        selected = selected[:quotas[relation]]
        if len(selected) != quotas[relation]:
            raise ValueError(f"V4 target quota not met for {relation}")
        targets.extend(selected)
        target_subjects.update(key[0] for key in selected)
    targets = sorted(targets, key=lambda key: (
        stable_rank(SEED, "target-order", *key), key
    ))
    if len(targets) != GROUPS or len(target_subjects) != GROUPS:
        raise ValueError("V4 target subjects are not unique")
    background_keys = [key for key in eligible if key[0] not in target_subjects]
    materials = []
    for group_index, target_key in enumerate(targets):
        target_relation = target_key[1]
        old, current = temporal_transition(eligible[target_key])
        used_subjects = {target_key[0]}
        used_values = {old["value_norm"], current["value_norm"]}
        same_key, same_other = choose_background(
            [key for key in background_keys if key[1] == target_relation],
            eligible, group_index, "same-relation", used_subjects, used_values,
        )
        used_subjects.add(same_key[0]); used_values.add(same_other["value_norm"])
        background_ids = [same_key[0]]
        other_relations = sorted(
            (relation for relation in PARAPHRASES if relation != target_relation),
            key=lambda relation: (
                stable_rank(SEED, "background-relation", group_index, relation),
                relation,
            ),
        )[:5]
        stable = []
        for index, relation in enumerate(other_relations):
            key, row = choose_background(
                [candidate for candidate in background_keys
                 if candidate[1] == relation],
                eligible, group_index, f"stable-{index}",
                used_subjects, used_values,
            )
            stable.append(row); background_ids.append(key[0])
            used_subjects.add(key[0]); used_values.add(row["value_norm"])
        anchor_relation = stable[0]["relation_id"]
        anchor_alternate = None; anchor_source = None
        for key in ordered_keys(
            [candidate for candidate in background_keys
             if candidate[1] == anchor_relation],
            "anchor-alternate", group_index,
        ):
            for _, row in eligible[key]:
                if row["value_norm"] not in used_values:
                    anchor_source, anchor_alternate = key[0], row
                    break
            if anchor_alternate is not None:
                break
        if anchor_alternate is None:
            raise ValueError(f"no V4 anchor alternate for {group_index}")
        materials.append((group_index, {
            "target": old,
            "versions": [current, current, current, current],
            "same_relation_other": same_other,
            "stable": stable,
            "anchor_alternate": anchor_alternate,
            "patterns": {
                relation: list(PARAPHRASES[relation])
                for relation in {target_relation,
                                 *(row["relation_id"] for row in stable)}
            },
            "temporal_transition": {
                "subject_id": target_key[0],
                "relation_id": target_relation,
                "old_year": old["year"],
                "new_year": current["year"],
                "old_value_id": old["value_id"],
                "new_value_id": current["value_id"],
            },
            "background_subject_ids": sorted(background_ids),
            "anchor_alternate_subject_id": anchor_source,
        }))
    return materials, quotas


def build(source_root: Path, inspection_path: Path, pilot_root: Path,
          v3_root: Path) -> tuple[list[dict], dict]:
    inspection = json.loads(inspection_path.read_text())
    pilot_audit_path = pilot_root / "pilot_audit.json"
    pilot_audit = json.loads(pilot_audit_path.read_text())
    v3_audit_path = v3_root / "result_audit.json"
    v3_audit = json.loads(v3_audit_path.read_text())
    if (inspection.get("protocol") != INSPECTION_PROTOCOL
            or not inspection.get("passed")):
        raise ValueError("TempLAMA source inspection did not pass")
    if not pilot_audit.get("passed"):
        raise ValueError("TempLAMA resource pilot audit did not pass")
    if (v3_audit.get("protocol") != V3_RESULT_AUDIT_PROTOCOL
            or not v3_audit.get("passed") or v3_audit.get("release_passed")
            or v3_audit.get("next_stage") != "stop_v3_without_opening_test"):
        raise ValueError("V3 failed-release branch transition not verified")
    _, inventories = source_inventory(source_root)
    expected = inspection["post_decontamination"]["train"]
    if len(eligible_trajectories(inventories["train"])) \
            != expected["eligible_trajectories"]:
        raise ValueError("TempLAMA train eligibility reconstruction mismatch")
    materials, quotas = choose_materials(inventories["train"])
    episodes = []
    for group_index, material in materials:
        mechanism = MECHANISMS[group_index % len(MECHANISMS)]
        length = LENGTHS[group_index % len(LENGTHS)]
        for twin in range(TWINS):
            episode = build_episode(
                group_index, mechanism, length, twin, material,
                group_prefix=GROUP_PREFIX,
            )
            episode["audit"]["templama_transition"] = material["temporal_transition"]
            episode["audit"]["source_partition"] = "train"
            episode["audit"]["background_subject_ids"] = material[
                "background_subject_ids"
            ]
            episode["audit"]["anchor_alternate_subject_id"] = material[
                "anchor_alternate_subject_id"
            ]
            episodes.append(episode)
    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V4_DOMAIN_TRAINING_PLAN.md",
        "source_manifest_sha256": sha256(source_root / "metadata" / "manifest.json"),
        "source_inspection_sha256": sha256(inspection_path),
        "pilot_audit_sha256": sha256(pilot_audit_path),
        "v3_result_audit_sha256": sha256(v3_audit_path),
        "seed": SEED,
        "partition": "templama_train_formal",
        "relation_quotas": quotas,
        "counts": {
            "source_groups": GROUPS,
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
            "target_relations": dict(Counter(
                row["audit"]["target_relation"] for row in episodes
                if row["twin"] == 0
            )),
            "target_subjects": len({
                row["audit"]["templama_transition"]["subject_id"]
                for row in episodes if row["twin"] == 0
            }),
            "background_subjects": len({
                subject for row in episodes if row["twin"] == 0
                for subject in row["audit"]["background_subject_ids"]
            }),
        },
        "primary_action_space": "binary_endpoints_plus_global_clear",
        "current_write_gate": 1.0,
        "model_outputs_loaded": False,
        "validation_episodes_loaded": False,
        "test_episodes_generated": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "terminal_status": "templama_v4_train_data_built_pending_independent_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    assert len(episodes) == GROUPS * TWINS
    assert manifest["counts"]["source_groups"] == GROUPS
    assert manifest["counts"]["boundaries"] == GROUPS * TWINS * BOUNDARIES
    assert manifest["counts"]["queries"] == GROUPS * TWINS * BOUNDARIES * 5
    assert manifest["counts"]["target_subjects"] == GROUPS
    assert manifest["relation_quotas"] == EXPECTED_QUOTAS
    assert manifest["counts"]["target_relations"] == EXPECTED_QUOTAS
    assert set(manifest["counts"]["lengths"].values()) == {256}
    assert max(manifest["counts"]["mechanisms"].values()) \
        - min(manifest["counts"]["mechanisms"].values()) <= TWINS
    groups = defaultdict(list)
    for row in episodes:
        groups[row["source_group"]].append(row)
    assert len(groups) == GROUPS
    assert all({row["twin"] for row in values} == {0, 1}
               for values in groups.values())
    targets = {row["audit"]["templama_transition"]["subject_id"]
               for row in episodes}
    assert all(
        row["audit"]["source_partition"] == "train"
        and len(set(row["audit"]["background_subject_ids"])) == 6
        and not set(row["audit"]["background_subject_ids"]) & targets
        and row["audit"]["templama_transition"]["old_year"]
        < row["audit"]["templama_transition"]["new_year"]
        and row["audit"]["templama_transition"]["old_value_id"]
        != row["audit"]["templama_transition"]["new_value_id"]
        and len(row["events"]) == row["length"]
        and len(row["boundaries"]) == BOUNDARIES
        for row in episodes
    )
    assert all(len(boundary["queries"]) == 5
               and all(len(query["candidates"]) == 6
                       and query["answer"] in query["candidates"]
                       for query in boundary["queries"])
               for row in episodes for boundary in row["boundaries"])
    assert not manifest["model_outputs_loaded"]
    assert not manifest["validation_episodes_loaded"]
    assert not manifest["test_episodes_generated"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(
        args.source_root, args.inspection, args.pilot_root, args.v3_root
    )
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "train_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
