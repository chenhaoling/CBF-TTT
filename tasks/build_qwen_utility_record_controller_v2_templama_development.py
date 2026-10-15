"""Build the sealed 256-group TempLAMA validation corpus for V2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from scripts.inspect_templama_utility_source import PROTOCOL as INSPECTION_PROTOCOL
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
from tasks.pilot_qwen_utility_record_controller_v2_templama import (
    AUDIT_PROTOCOL as PILOT_AUDIT_PROTOCOL,
    PROTOCOL as PILOT_PROTOCOL,
)


PROTOCOL = "qwen_utility_record_controller_v2_templama_development_data"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v2_templama_development_data_audit"
GROUPS = 256
MINIMUM_PER_RELATION = 8
BACKGROUND_HELDOUT_GROUPS = len(PARAPHRASES)
SEED = 202610171
GROUP_PREFIX = "usr-v2-templama-development"


def development_ordered_keys(keys, *namespace) -> list[tuple[str, str]]:
    """Order source keys with the frozen development seed.

    The pilot has its own sampling seed.  Keeping a local ordering function
    makes the validation construction independent of the pilot sampling
    implementation while remaining exactly reconstructable by the auditor.
    """
    return sorted(keys, key=lambda key: (
        stable_rank(SEED, *namespace, *key), key
    ))


def hamilton_quotas(capacities: dict[str, int], total: int = GROUPS,
                    minimum: int = MINIMUM_PER_RELATION) -> dict[str, int]:
    if not capacities or any(value < minimum for value in capacities.values()):
        raise ValueError("relation capacity below registered minimum")
    remaining = total - minimum * len(capacities)
    residual = {key: value - minimum for key, value in capacities.items()}
    residual_total = sum(residual.values())
    if remaining < 0 or residual_total < remaining:
        raise ValueError("insufficient total capacity")
    exact = {key: remaining * value / residual_total for key, value in residual.items()}
    quota = {key: minimum + math.floor(value) for key, value in exact.items()}
    leftover = total - sum(quota.values())
    order = sorted(capacities, key=lambda key: (
        -(exact[key] - math.floor(exact[key])), key
    ))
    for key in order[:leftover]:
        quota[key] += 1
    if sum(quota.values()) != total or any(quota[key] > capacities[key] for key in quota):
        raise ValueError("invalid Hamilton allocation")
    return quota


def choose_background(key_candidates, eligible, group_index, role, used_subjects,
                      used_values):
    for key in development_ordered_keys(
        key_candidates, "development", group_index, role
    ):
        row = eligible[key][0][1]
        if key[0] in used_subjects or row["value_norm"] in used_values:
            continue
        return key, row
    raise ValueError(f"no background candidate for {group_index}:{role}")


def choose_materials(trajectories: dict) -> tuple[list[tuple[int, dict]], dict]:
    eligible = eligible_trajectories(trajectories)
    capacities = Counter(key[1] for key in eligible)
    quotas = hamilton_quotas(dict(capacities))
    target_keys = []
    target_subjects = set()
    # Allocate the most capacity-constrained relations first so that a subject
    # occurring under multiple relations cannot consume a scarce slot later.
    relation_order = sorted(quotas, key=lambda relation: (
        capacities[relation] - quotas[relation], relation
    ))
    for relation in relation_order:
        candidates = [key for key in eligible if key[1] == relation]
        selected = [
            key for key in development_ordered_keys(
                candidates, "development-target", relation
            )
            if key[0] not in target_subjects
        ][:quotas[relation]]
        if len(selected) != quotas[relation]:
            raise ValueError("target quota not met")
        target_keys.extend(selected)
        target_subjects.update(key[0] for key in selected)
    target_keys = sorted(target_keys, key=lambda key: (
        stable_rank(SEED, "target-order", *key), key
    ))
    background_keys = [key for key in eligible if key[0] not in target_subjects]
    if len(target_subjects) != GROUPS:
        raise ValueError("development target subjects are not unique")

    heldout_group_indices = {
        next(index for index, key in enumerate(target_keys) if key[1] == relation)
        for relation in sorted(quotas)
    }

    def construct_material(group_index, target_key, allowed_background_keys,
                           background_heldout):
        target_relation = target_key[1]
        old, current = temporal_transition(eligible[target_key])
        used_subjects = {target_key[0]}
        used_values = {old["value_norm"], current["value_norm"]}
        background_ids = []
        same_key, same_other = choose_background(
            [key for key in allowed_background_keys if key[1] == target_relation],
            eligible, group_index, "same-relation", used_subjects, used_values,
        )
        used_subjects.add(same_key[0]); used_values.add(same_other["value_norm"])
        background_ids.append(same_key[0])

        stable = []
        if background_heldout:
            selected_relations = set()
            for index in range(5):
                relation_choices = []
                for relation in PARAPHRASES:
                    if relation == target_relation or relation in selected_relations:
                        continue
                    candidates = [
                        key for key in allowed_background_keys
                        if key[1] == relation and key[0] not in used_subjects
                    ]
                    try:
                        chosen_key, row = choose_background(
                            candidates, eligible, group_index, f"stable-{index}",
                            used_subjects, used_values,
                        )
                    except ValueError:
                        continue
                    relation_choices.append((
                        relation, chosen_key, row, len(candidates)
                    ))
                if not relation_choices:
                    raise ValueError(f"no stable relation for {group_index}:{index}")
                # Globally unique held-out allocation consumes abundant
                # relations first, preserving scarce same-relation supply for
                # the later one-per-relation groups.
                relation, chosen_key, row, _ = min(
                    relation_choices,
                    key=lambda value: (
                        -value[3],
                        stable_rank(
                            SEED, "heldout-background-relation", group_index,
                            index, value[0],
                        ),
                        value[0],
                    ),
                )
                stable.append(row); background_ids.append(chosen_key[0])
                used_subjects.add(chosen_key[0]); used_values.add(row["value_norm"])
                selected_relations.add(relation)
        else:
            other_relations = sorted(
                (relation for relation in PARAPHRASES
                 if relation != target_relation),
                key=lambda relation: (
                    stable_rank(
                        SEED, "background-relation", group_index, relation
                    ),
                    relation,
                ),
            )[:5]
            for index, relation in enumerate(other_relations):
                chosen_key, row = choose_background(
                    [key for key in allowed_background_keys
                     if key[1] == relation],
                    eligible, group_index, f"stable-{index}",
                    used_subjects, used_values,
                )
                stable.append(row); background_ids.append(chosen_key[0])
                used_subjects.add(chosen_key[0]); used_values.add(row["value_norm"])

        anchor_relation = stable[0]["relation_id"]
        anchor_alternate = None; anchor_source = None
        for key in development_ordered_keys(
            [key for key in allowed_background_keys if key[1] == anchor_relation],
            "development", group_index, "anchor-alternate",
        ):
            for _, row in eligible[key]:
                if row["value_norm"] not in used_values:
                    anchor_source, anchor_alternate = key[0], row; break
            if anchor_alternate is not None:
                break
        if anchor_alternate is None:
            raise ValueError("no development anchor alternate")
        material = {
            "target": old,
            "versions": [current, current, current, current],
            "same_relation_other": same_other,
            "stable": stable,
            "anchor_alternate": anchor_alternate,
            "patterns": {
                relation: list(PARAPHRASES[relation])
                for relation in {target_relation, *(row["relation_id"] for row in stable)}
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
            "background_heldout": background_heldout,
        }
        return material

    # Reserve one target group per relation with six background subjects that
    # never occur in any other group.  This makes the preregistered
    # background-held-out sensitivity estimable rather than empty.
    materials_by_index = {}
    reserved_heldout_subjects = set()
    for group_index in sorted(heldout_group_indices):
        target_key = target_keys[group_index]
        allowed = [key for key in background_keys
                   if key[0] not in reserved_heldout_subjects]
        material = construct_material(group_index, target_key, allowed, True)
        materials_by_index[group_index] = material
        reserved_heldout_subjects.update(material["background_subject_ids"])
    regular_background_keys = [
        key for key in background_keys
        if key[0] not in reserved_heldout_subjects
    ]
    for group_index, target_key in enumerate(target_keys):
        if group_index in heldout_group_indices:
            continue
        materials_by_index[group_index] = construct_material(
            group_index, target_key, regular_background_keys, False
        )
    materials = [(index, materials_by_index[index]) for index in range(GROUPS)]
    return materials, quotas


def build(source_root: Path, inspection_path: Path, pilot_root: Path) -> tuple[list[dict], dict]:
    inspection = json.loads(inspection_path.read_text())
    pilot_summary_path, pilot_audit_path = pilot_root / "summary.json", pilot_root / "pilot_audit.json"
    pilot_summary = json.loads(pilot_summary_path.read_text())
    pilot_audit = json.loads(pilot_audit_path.read_text())
    if (inspection.get("protocol") != INSPECTION_PROTOCOL
            or not inspection.get("passed")):
        raise ValueError("TempLAMA source inspection did not pass")
    if (pilot_summary.get("protocol") != PILOT_PROTOCOL
            or not pilot_summary.get("gate", {}).get("passed")
            or pilot_summary.get("next_stage")
            != "templama_replication_development_construction"):
        raise ValueError("TempLAMA resource pilot did not release development")
    if pilot_audit.get("protocol") != PILOT_AUDIT_PROTOCOL or not pilot_audit.get("passed"):
        raise ValueError("TempLAMA resource pilot audit did not pass")
    _, inventories = source_inventory(source_root)
    expected_val = inspection["post_decontamination"]["val"]
    if len(eligible_trajectories(inventories["val"])) \
            != expected_val["eligible_trajectories"]:
        raise ValueError("TempLAMA validation eligibility reconstruction mismatch")
    materials, quotas = choose_materials(inventories["val"])
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
            episode["audit"]["source_partition"] = "val"
            episode["audit"]["background_subject_ids"] = material["background_subject_ids"]
            episode["audit"]["anchor_alternate_subject_id"] = (
                material["anchor_alternate_subject_id"]
            )
            episode["audit"]["background_heldout"] = material["background_heldout"]
            episodes.append(episode)
    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V2_CROSS_SOURCE_PLAN.md",
        "clarification": "QWEN_UTILITY_RECORD_CONTROLLER_V2_CAPACITY_CLARIFICATION.md",
        "background_heldout_clarification": (
            "QWEN_UTILITY_RECORD_CONTROLLER_V2_BACKGROUND_HELDOUT_CLARIFICATION.md"
        ),
        "source_manifest_sha256": sha256(source_root / "metadata" / "manifest.json"),
        "source_inspection_sha256": sha256(inspection_path),
        "pilot_summary_sha256": sha256(pilot_summary_path),
        "pilot_audit_sha256": sha256(pilot_audit_path),
        "seed": SEED,
        "partition": "validation_development",
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
                row["audit"]["target_relation"] for row in episodes if row["twin"] == 0
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
        "test_episodes_generated": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "terminal_status": "templama_development_data_built_pending_independent_audit",
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
    assert max(manifest["counts"]["mechanisms"].values()) - min(
        manifest["counts"]["mechanisms"].values()
    ) <= TWINS
    assert all(len(row["events"]) == row["length"] for row in episodes)
    assert all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
    assert all(len(query["candidates"]) == 6 and query["answer"] in query["candidates"]
               for row in episodes for boundary in row["boundaries"]
               for query in boundary["queries"])
    groups = defaultdict(list)
    for row in episodes:
        groups[row["source_group"]].append(row)
    assert len(groups) == GROUPS
    assert all({row["twin"] for row in values} == {0, 1}
               for values in groups.values())
    target_subjects = {
        row["audit"]["templama_transition"]["subject_id"]
        for row in episodes
    }
    assert all(
        len(set(row["audit"]["background_subject_ids"])) == 6
        and not (set(row["audit"]["background_subject_ids"]) & target_subjects)
        and row["audit"]["source_partition"] == "val"
        for row in episodes
    )
    target_rows = [row for row in episodes if row["twin"] == 0]
    background_counts = Counter(
        subject for row in target_rows
        for subject in row["audit"]["background_subject_ids"]
    )
    heldout_rows = [row for row in target_rows
                    if row["audit"]["background_heldout"]]
    assert manifest["counts"]["background_heldout_groups"] == BACKGROUND_HELDOUT_GROUPS
    assert len(heldout_rows) == BACKGROUND_HELDOUT_GROUPS
    assert (len({row["audit"]["target_relation"] for row in heldout_rows})
            == BACKGROUND_HELDOUT_GROUPS)
    assert all(
        all(background_counts[subject] == 1
            for subject in row["audit"]["background_subject_ids"])
        for row in heldout_rows
    )
    assert not manifest["model_outputs_loaded"]
    assert not manifest["test_episodes_generated"]
    assert not manifest["v1_confirmation_loaded"]
    assert not manifest["external_benchmarks_loaded"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(
        args.source_root, args.inspection, args.pilot_root
    )
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "development_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
