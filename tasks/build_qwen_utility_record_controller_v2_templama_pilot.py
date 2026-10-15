"""Build the model-free TempLAMA cross-source resource-pilot episodes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from scripts.inspect_templama_utility_source import (
    IDENTITY,
    PROTOCOL as INSPECTION_PROTOCOL,
    SPLITS,
    choose_owner,
    extract_subject,
    load_rows,
    load_templates,
    singleton_answer,
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


PROTOCOL = "qwen_utility_record_controller_v2_templama_pilot_data"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v2_templama_pilot_data_audit"
SEED = 202610169
GROUPS_PER_RELATION = 8
GROUP_PREFIX = "usr-v2-templama-pilot"

PARAPHRASES = {
    "P54": ("[X] plays for [Y].", "[X] is a player for [Y].",
             "[Y] is the sports team of [X]."),
    "P39": ("[X] holds the position of [Y].", "[X]'s position is [Y].",
             "[Y] is the position held by [X]."),
    "P108": ("[X] works for [Y].", "[X] is employed by [Y].",
              "[Y] is the employer of [X]."),
    "P102": ("[X] is a member of the [Y].", "[X] belongs to the [Y].",
              "The political party of [X] is the [Y]."),
    "P286": ("[Y] is the head coach of [X].", "[X] is coached by [Y].",
              "The head coach for [X] is [Y]."),
    "P69": ("[X] attended [Y].", "[X] was educated at [Y].",
             "[Y] is an institution attended by [X]."),
    "P488": ("[Y] is the chair of [X].", "[X] is chaired by [Y].",
              "The chairperson of [X] is [Y]."),
    "P6": ("[Y] is the head of the government of [X].",
            "The government of [X] is headed by [Y].",
            "[X]'s head of government is [Y]."),
    "P127": ("[X] is owned by [Y].", "The owner of [X] is [Y].",
              "[Y] owns [X]."),
}


def answer_value(row: dict) -> tuple[str, str, str] | None:
    singleton = singleton_answer(row)
    if singleton is None:
        return None
    answer_id, value_norm = singleton
    answer = next(
        value for value in row["answer"]
        if value.get("wikidata_id") == answer_id
        and normalize(value.get("name", "")) == value_norm
    )
    value = re.sub(r"\s+", " ", answer["name"].strip())
    return answer_id, value, value_norm


def source_inventory(source_root: Path) -> tuple[dict, dict[str, dict]]:
    template_path = (
        source_root / "sources" / "google-research-language"
        / "language" / "templama" / "templates.csv"
    )
    templates = load_templates(template_path)
    raw = {}; subjects_by_split = {}; surfaces_by_subject_by_split = {}
    for split in SPLITS:
        trajectories = defaultdict(dict); subjects = set()
        surfaces_by_subject = defaultdict(set)
        for row in load_rows(source_root / "downloads" / f"{split}.json"):
            identity = IDENTITY.fullmatch(str(row.get("id", "")))
            if identity is None:
                continue
            subject_id, relation, year_text = identity.groups()
            if relation not in templates or relation != row.get("relation"):
                continue
            subject = extract_subject(row.get("query", ""), templates[relation]["template"])
            if subject is None:
                continue
            subjects.add(subject_id)
            answer = answer_value(row)
            if answer is None:
                continue
            answer_id, value, value_norm = answer
            year = int(year_text)
            if year in trajectories[(subject_id, relation)]:
                continue
            source_row = {
                "relation_id": relation,
                "subject_id": subject_id,
                "subject": subject,
                "subject_norm": normalize(subject),
                "value_id": answer_id,
                "value": value,
                "value_norm": value_norm,
                "year": year,
                "source_uuid": row["id"],
                "source_fact_hash": stable_rank(
                    "templama", split, subject_id, relation, year, answer_id
                ),
            }
            trajectories[(subject_id, relation)][year] = source_row
            surfaces_by_subject[subject_id].add(
                normalize(row["query"].replace("_X_", value))
            )
        raw[split] = trajectories
        subjects_by_split[split] = subjects
        surfaces_by_subject_by_split[split] = surfaces_by_subject

    owners = defaultdict(set)
    for split in SPLITS:
        for subject in subjects_by_split[split]:
            owners[subject].add(split)
    assigned = {subject: choose_owner(subject, values)
                for subject, values in owners.items()}
    owned_surfaces = {
        split: {
            surface
            for subject, values in surfaces_by_subject_by_split[split].items()
            if assigned[subject] == split
            for surface in values
        }
        for split in SPLITS
    }
    collisions = set()
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            collisions.update(owned_surfaces[left] & owned_surfaces[right])
    collision_subjects = {
        split: {
            subject for subject, values in surfaces_by_subject_by_split[split].items()
            if assigned[subject] == split and set(values) & collisions
        }
        for split in SPLITS
    }
    filtered = {}
    for split in SPLITS:
        filtered[split] = {
            key: sorted(by_year.items())
            for key, by_year in raw[split].items()
            if assigned[key[0]] == split and key[0] not in collision_subjects[split]
        }
    return templates, filtered


def temporal_transition(ordered: list[tuple[int, dict]]) -> tuple[dict, dict] | None:
    for (_, left), (_, right) in zip(ordered, ordered[1:]):
        if left["value_id"] != right["value_id"]:
            return left, right
    return None


def eligible_trajectories(trajectories: dict) -> dict:
    return {
        key: ordered for key, ordered in trajectories.items()
        if temporal_transition(ordered) is not None
    }


def ordered_keys(keys, *namespace) -> list[tuple[str, str]]:
    return sorted(keys, key=lambda key: (
        stable_rank(SEED, *namespace, *key), key
    ))


def choose_pilot_materials(trajectories: dict) -> list[tuple[int, str, dict]]:
    eligible = eligible_trajectories(trajectories)
    relations = sorted(PARAPHRASES)
    by_relation = {
        relation: [key for key in eligible if key[1] == relation]
        for relation in relations
    }
    target_keys = []
    for relation in relations:
        ordered = ordered_keys(by_relation[relation], "targets", relation)
        if len(ordered) < GROUPS_PER_RELATION:
            raise ValueError(f"relation {relation} lacks pilot trajectories")
        target_keys.extend((relation, key) for key in ordered[:GROUPS_PER_RELATION])

    forbidden_subjects = {key[0] for _, key in target_keys}
    materials = []
    all_keys = set(eligible)
    for group_index, (target_relation, target_key) in enumerate(target_keys):
        target_ordered = eligible[target_key]
        transition = temporal_transition(target_ordered)
        if transition is None:
            raise AssertionError("selected target lost temporal transition")
        old, current = transition
        used_values = {old["value_norm"], current["value_norm"]}

        same_candidates = [
            key for key in by_relation[target_relation]
            if key != target_key and key[0] not in forbidden_subjects
        ]
        same_key = next((key for key in ordered_keys(
            same_candidates, "same-relation", group_index
        ) if eligible[key][0][1]["value_norm"] not in used_values), None)
        if same_key is None:
            raise ValueError("no same-relation other-subject trajectory")
        same_other = eligible[same_key][0][1]
        forbidden_subjects.add(same_key[0]); used_values.add(same_other["value_norm"])

        relation_supply = Counter(
            key[1] for key in all_keys if key[0] not in forbidden_subjects
            and key[1] != target_relation
        )
        other_relations = sorted(relation_supply, key=lambda relation: (
            -relation_supply[relation],
            stable_rank(SEED, "stable-rel", group_index, relation), relation,
        ))[:5]
        if len(other_relations) < 5:
            raise ValueError("fewer than five stable relations remain")
        stable = []
        for index, relation in enumerate(other_relations):
            candidates = [
                key for key in all_keys
                if key[1] == relation and key[0] not in forbidden_subjects
            ]
            chosen_key = next((key for key in ordered_keys(
                candidates, "stable", group_index, index
            ) if eligible[key][0][1]["value_norm"] not in used_values), None)
            if chosen_key is None:
                raise ValueError("no distinct stable trajectory")
            row = eligible[chosen_key][0][1]
            stable.append(row); forbidden_subjects.add(chosen_key[0])
            used_values.add(row["value_norm"])

        anchor_relation = stable[0]["relation_id"]
        anchor_alternate = None
        for key in ordered_keys(
            [key for key in all_keys if key[1] == anchor_relation],
            "anchor-alternate", group_index,
        ):
            for _, row in eligible[key]:
                if row["value_norm"] not in used_values:
                    anchor_alternate = row; break
            if anchor_alternate is not None:
                break
        if anchor_alternate is None:
            raise ValueError("no distinct anchor alternate")

        material = {
            "target": old,
            # The first update is the observed temporal transition.  Repeated
            # later boundaries paraphrase the same current value and therefore
            # do not fabricate unobserved temporal states.
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
        }
        materials.append((group_index, target_relation, material))
    return materials


def build(source_root: Path, inspection_path: Path) -> tuple[list[dict], dict]:
    inspection = json.loads(inspection_path.read_text())
    if inspection.get("protocol") != INSPECTION_PROTOCOL or not inspection.get("passed"):
        raise ValueError("TempLAMA source inspection did not pass")
    templates, inventories = source_inventory(source_root)
    train_eligible = eligible_trajectories(inventories["train"])
    expected = inspection["post_decontamination"]["train"]
    if len(train_eligible) != expected["eligible_trajectories"]:
        raise ValueError("TempLAMA train eligibility reconstruction mismatch")
    materials = choose_pilot_materials(inventories["train"])
    episodes = []
    for group_index, target_relation, material in materials:
        mechanism = MECHANISMS[group_index % len(MECHANISMS)]
        length = LENGTHS[(group_index // GROUPS_PER_RELATION + group_index) % len(LENGTHS)]
        for twin in range(TWINS):
            episode = build_episode(
                group_index, mechanism, length, twin, material,
                group_prefix=GROUP_PREFIX,
            )
            episode["audit"]["templama_transition"] = material["temporal_transition"]
            episode["audit"]["source_partition"] = "train"
            episode["audit"]["official_template"] = templates[target_relation]["template"]
            episodes.append(episode)

    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V2_CROSS_SOURCE_PLAN.md",
        "source_manifest_sha256": sha256(source_root / "metadata" / "manifest.json"),
        "source_inspection_sha256": sha256(inspection_path),
        "source_repository_commit": inspection["repository_commit"],
        "seed": SEED,
        "partition": "train_resource_pilot",
        "counts": {
            "relations": len(PARAPHRASES),
            "source_groups": len(materials),
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "target_relations": dict(Counter(
                row["audit"]["target_relation"] for row in episodes if row["twin"] == 0
            )),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
        },
        "alpha_grid": [0.0, 0.5, 1.0],
        "current_write_gate": 1.0,
        "nontrain_content_used_only_for_decontamination": True,
        "model_outputs_loaded": False,
        "v1_confirmation_loaded": False,
        "development_episodes_generated": False,
        "test_episodes_generated": False,
        "forbidden_sources_loaded": False,
        "terminal_status": "templama_pilot_data_built_pending_independent_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    expected_groups = len(PARAPHRASES) * GROUPS_PER_RELATION
    assert manifest["counts"]["source_groups"] == expected_groups == 72
    assert len(episodes) == expected_groups * TWINS
    assert manifest["counts"]["boundaries"] == len(episodes) * BOUNDARIES
    assert manifest["counts"]["queries"] == len(episodes) * BOUNDARIES * 5
    assert set(manifest["counts"]["target_relations"].values()) == {GROUPS_PER_RELATION}
    assert max(manifest["counts"]["mechanisms"].values()) - min(
        manifest["counts"]["mechanisms"].values()
    ) <= TWINS
    assert set(manifest["counts"]["lengths"].values()) == {36}
    assert all(len(row["events"]) == row["length"] for row in episodes)
    assert all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
    assert all(len(query["candidates"]) == 6 and query["answer"] in query["candidates"]
               for row in episodes for boundary in row["boundaries"]
               for query in boundary["queries"])
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    assert len(groups) == expected_groups
    assert all({row["twin"] for row in values} == {0, 1}
               for values in groups.values())
    target_subjects = {
        row["audit"]["templama_transition"]["subject_id"]
        for row in episodes if row["twin"] == 0
    }
    assert len(target_subjects) == expected_groups
    assert all(
        row["audit"]["templama_transition"]["old_value_id"]
        != row["audit"]["templama_transition"]["new_value_id"]
        and row["audit"]["templama_transition"]["old_year"]
        < row["audit"]["templama_transition"]["new_year"]
        for row in episodes
    )
    assert not manifest["model_outputs_loaded"]
    assert not manifest["v1_confirmation_loaded"]
    assert not manifest["development_episodes_generated"]
    assert not manifest["test_episodes_generated"]
    assert not manifest["forbidden_sources_loaded"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(args.source_root, args.inspection)
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "pilot_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
