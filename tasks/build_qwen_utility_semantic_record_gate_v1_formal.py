"""Build the sealed 512-group formal training corpus for utility forgetting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    GROUPS_PER_MECHANISM,
    LENGTHS,
    MECHANISMS,
    TWINS,
    build_episode,
    choose_group_material,
    cross_partition_subjects,
    load_facts,
    load_patterns,
    read_json,
    stable_rank,
    validate as validate_pilot,
    write_json,
    write_jsonl,
)


PROTOCOL = "qwen_utility_semantic_record_gate_v1_formal_train_data"
AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_formal_train_data_audit"
PILOT_DATA_PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data"
PILOT_DATA_AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data_audit"
PILOT_LABEL_AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_resource_pilot_audit"
INSPECTION_PROTOCOL = "qwen_utility_semantic_record_source_inspection_v1"
FORMAL_GROUPS = 512
FORMAL_SEED = 202610161
GROUP_PREFIX = "usr-v1-formal"


def material_rows(material: dict) -> list[dict]:
    return [
        material["target"],
        *material["versions"],
        material["same_relation_other"],
        *material["stable"],
        material["anchor_alternate"],
    ]


def input_source_rows(material: dict) -> list[dict]:
    """Rows whose subjects are rendered in the episode input.

    Version and anchor-alternate rows donate values to another subject and are
    therefore provenance rows rather than independent input subjects.
    """

    return [
        material["target"],
        material["same_relation_other"],
        *material["stable"],
    ]


def ordered_candidates(rows: list[dict], namespace: str) -> list[dict]:
    return sorted(rows, key=lambda row: (
        stable_rank(FORMAL_SEED, namespace, row["source_fact_hash"]),
        row["source_fact_hash"],
    ))


def first_eligible(
    rows: list[dict],
    namespace: str,
    forbidden_fact_hashes: set[str],
    forbidden_subjects: set[str],
    forbidden_values: set[str] | None = None,
) -> dict:
    forbidden_values = forbidden_values or set()
    for row in ordered_candidates(rows, namespace):
        if row["source_fact_hash"] in forbidden_fact_hashes:
            continue
        if row["subject_norm"] in forbidden_subjects:
            continue
        if row["value_norm"] in forbidden_values:
            continue
        return row
    raise ValueError(f"no eligible fact for {namespace}")


def choose_formal_material(
    group_index: int,
    train_relations: list[str],
    facts: dict[str, list[dict]],
    patterns: dict[str, list[str]],
    forbidden_fact_hashes: set[str],
    forbidden_subjects: set[str],
    strict_same_relation_value: bool = True,
) -> dict:
    """Select one group and mutate the two forbidden sets after success."""

    target_relation = train_relations[group_index % len(train_relations)]
    local_hashes = set(forbidden_fact_hashes)
    local_subjects = set(forbidden_subjects)
    chosen: list[dict] = []
    reserved: list[dict] = []

    def take(relation: str, role: str, forbidden_values=(), reserve_input=False) -> dict:
        row = first_eligible(
            facts[relation], f"formal:{group_index}:{role}:{relation}",
            local_hashes, local_subjects, set(forbidden_values),
        )
        chosen.append(row)
        local_hashes.add(row["source_fact_hash"])
        local_subjects.add(row["subject_norm"])
        if reserve_input:
            reserved.append(row)
        return row

    target = take(target_relation, "target", reserve_input=True)
    versions = []
    used_values = {target["value_norm"]}
    for index in range(4):
        row = take(target_relation, f"version-{index}", used_values)
        versions.append(row)
        used_values.add(row["value_norm"])
    same_relation_other = take(
        target_relation,
        "same-relation-other",
        used_values if strict_same_relation_value else {target["value_norm"]},
        reserve_input=True,
    )
    used_values.add(same_relation_other["value_norm"])

    other_relations = sorted(
        (relation for relation in train_relations if relation != target_relation),
        key=lambda relation: (
            stable_rank(FORMAL_SEED, "formal-other-relation", group_index, relation), relation
        ),
    )[:5]
    stable = []
    for index, relation in enumerate(other_relations):
        row = take(relation, f"stable-{index}", used_values, reserve_input=True)
        stable.append(row)
        used_values.add(row["value_norm"])
    anchor_alternate = take(
        stable[0]["relation_id"], "anchor-alternate", used_values
    )

    forbidden_fact_hashes.update(row["source_fact_hash"] for row in reserved)
    forbidden_subjects.update(row["subject_norm"] for row in reserved)
    return {
        "target": target,
        "versions": versions,
        "same_relation_other": same_relation_other,
        "stable": stable,
        "anchor_alternate": anchor_alternate,
        "patterns": {
            relation: patterns[relation]
            for relation in [target_relation, *other_relations]
        },
    }


def reconstruct_pilot_exclusions(
    train_relations: list[str],
    facts: dict[str, list[dict]],
    patterns: dict[str, list[str]],
) -> tuple[set[str], set[str]]:
    fact_hashes: set[str] = set()
    subjects: set[str] = set()
    pilot_groups = len(MECHANISMS) * GROUPS_PER_MECHANISM
    for group_index in range(pilot_groups):
        material = choose_group_material(group_index, train_relations, facts, patterns)
        for row in material_rows(material):
            fact_hashes.add(row["source_fact_hash"])
            subjects.add(row["subject_norm"])
    return fact_hashes, subjects


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_prerequisites(
    pilot_data_root: Path, pilot_label_audit_path: Path
) -> tuple[dict, dict, dict, list[dict]]:
    manifest = read_json(pilot_data_root / "manifest.json")
    data_audit = read_json(pilot_data_root / "data_audit.json")
    label_audit = read_json(pilot_label_audit_path)
    episodes = read_jsonl(pilot_data_root / "pilot_episodes.jsonl")
    if manifest.get("protocol") != PILOT_DATA_PROTOCOL:
        raise ValueError("unexpected pilot data protocol")
    if data_audit.get("protocol") != PILOT_DATA_AUDIT_PROTOCOL or not data_audit.get("passed"):
        raise ValueError("pilot data audit did not pass")
    if data_audit.get("manifest_sha256") != sha256(pilot_data_root / "manifest.json"):
        raise ValueError("pilot manifest changed after data audit")
    if (manifest.get("files_sha256", {}).get("pilot_episodes.jsonl")
            != sha256(pilot_data_root / "pilot_episodes.jsonl")):
        raise ValueError("pilot episode file changed after manifest")
    if data_audit.get("episodes_sha256") != sha256(pilot_data_root / "pilot_episodes.jsonl"):
        raise ValueError("pilot episode file changed after data audit")
    if label_audit.get("protocol") != PILOT_LABEL_AUDIT_PROTOCOL or not label_audit.get("passed"):
        raise ValueError("pilot label audit did not pass")
    if label_audit.get("data_manifest_sha256") != sha256(pilot_data_root / "manifest.json"):
        raise ValueError("pilot label audit references another corpus")
    if label_audit.get("data_audit_sha256") != sha256(pilot_data_root / "data_audit.json"):
        raise ValueError("pilot label audit references another data audit")
    if label_audit.get("next_stage") != "formal_label_construction":
        raise ValueError("pilot did not release formal construction")
    validate_pilot(episodes, manifest)
    return manifest, data_audit, label_audit, episodes


def build(
    source_root: Path,
    inspection_path: Path,
    pilot_data_root: Path,
    pilot_label_audit_path: Path,
) -> tuple[list[dict], dict]:
    inspection = read_json(inspection_path)
    if inspection.get("protocol") != INSPECTION_PROTOCOL or not inspection.get("passed"):
        raise ValueError("source inspection did not pass")
    pilot_manifest, data_audit, label_audit, pilot_episodes = validate_prerequisites(
        pilot_data_root, pilot_label_audit_path
    )
    splits = inspection["splits"]
    relations = {relation for values in splits.values() for relation in values}
    patterns = load_patterns(source_root, relations)
    facts = load_facts(source_root, relations)
    removed_subjects = cross_partition_subjects(facts, splits)
    facts = {
        relation: [row for row in rows if row["subject_norm"] not in removed_subjects]
        for relation, rows in facts.items()
    }
    train_relations = splits["train"]
    forbidden_hashes, forbidden_subjects = reconstruct_pilot_exclusions(
        train_relations, facts, patterns
    )
    pilot_input_subjects = {
        normalize(event["record"]["input"]["subject"])
        for episode in pilot_episodes for event in episode["events"]
    }
    forbidden_subjects.update(pilot_input_subjects)
    pilot_forbidden_hashes = set(forbidden_hashes)
    pilot_forbidden_subjects = set(forbidden_subjects)

    episodes = []
    for group_index in range(FORMAL_GROUPS):
        mechanism = MECHANISMS[group_index % len(MECHANISMS)]
        length = LENGTHS[group_index % len(LENGTHS)]
        material = choose_formal_material(
            group_index, train_relations, facts, patterns,
            forbidden_hashes, forbidden_subjects,
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

    formal_hashes = {
        value for episode in episodes
        for value in episode["audit"]["material_source_fact_hashes"]
    }
    formal_subjects = {
        value for episode in episodes
        for value in episode["audit"]["material_source_subject_norms"]
    }
    input_hashes = {
        value for episode in episodes
        for value in episode["audit"]["input_source_fact_hashes"]
    }
    input_subjects = {
        value for episode in episodes
        for value in episode["audit"]["input_source_subject_norms"]
    }
    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_SUPERVISED_SEMANTIC_RECORD_GATE_V1_PLAN.md",
        "source_manifest_sha256": inspection["source_manifest_sha256"],
        "source_inspection_sha256": sha256(inspection_path),
        "source_commits": inspection["source_commits"],
        "pilot_data_manifest_sha256": sha256(pilot_data_root / "manifest.json"),
        "pilot_data_audit_sha256": sha256(pilot_data_root / "data_audit.json"),
        "pilot_label_audit_sha256": sha256(pilot_label_audit_path),
        "pilot_gate_passed": label_audit["passed"],
        "seed": FORMAL_SEED,
        "counts": {
            "source_groups": FORMAL_GROUPS,
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
            "target_relations": dict(Counter(
                row["audit"]["target_relation"] for row in episodes if row["twin"] == 0
            )),
            "material_source_facts": len(formal_hashes),
            "material_source_subjects": len(formal_subjects),
            "input_source_facts": len(input_hashes),
            "input_source_subjects": len(input_subjects),
            "pilot_source_facts_excluded": len(pilot_forbidden_hashes),
            "pilot_subjects_excluded": len(pilot_forbidden_subjects),
            "cross_partition_subjects_removed": len(removed_subjects),
        },
        "splits": splits,
        "partition": "train",
        "alpha_grid": [0.0, 0.5, 1.0],
        "primary_action_space": "binary_with_half_ablation",
        "current_write_gate": 1.0,
        "pilot_overlap": {
            "source_fact_hashes": len(formal_hashes & pilot_forbidden_hashes),
            "subjects": len(formal_subjects & pilot_forbidden_subjects),
        },
        "model_outputs_loaded": False,
        "development_episodes_generated": False,
        "confirmation_episodes_generated": False,
        "forbidden_sources_loaded": False,
        "terminal_status": "formal_train_data_built_pending_independent_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    assert manifest["counts"]["source_groups"] == FORMAL_GROUPS
    assert len(episodes) == FORMAL_GROUPS * TWINS
    assert manifest["counts"]["boundaries"] == len(episodes) * BOUNDARIES
    assert manifest["counts"]["queries"] == len(episodes) * BOUNDARIES * 5
    assert max(manifest["counts"]["mechanisms"].values()) - min(
        manifest["counts"]["mechanisms"].values()
    ) <= TWINS
    assert set(manifest["counts"]["lengths"].values()) == {FORMAL_GROUPS * TWINS // 4}
    assert max(manifest["counts"]["target_relations"].values()) - min(
        manifest["counts"]["target_relations"].values()
    ) <= 1
    assert manifest["counts"]["input_source_facts"] == FORMAL_GROUPS * 7
    assert manifest["counts"]["input_source_subjects"] == FORMAL_GROUPS * 7
    assert manifest["pilot_overlap"] == {"source_fact_hashes": 0, "subjects": 0}
    assert all(len(row["events"]) == row["length"] for row in episodes)
    assert all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
    assert all(len(query["candidates"]) == 6 and query["answer"] in query["candidates"]
               for row in episodes for boundary in row["boundaries"]
               for query in boundary["queries"])
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    assert len(groups) == FORMAL_GROUPS
    assert all({row["twin"] for row in values} == {0, 1} for values in groups.values())
    assert not manifest["model_outputs_loaded"]
    assert not manifest["development_episodes_generated"]
    assert not manifest["confirmation_episodes_generated"]
    assert not manifest["forbidden_sources_loaded"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-data-root", type=Path, required=True)
    parser.add_argument("--pilot-label-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    episodes, manifest = build(
        args.source_root, args.inspection, args.pilot_data_root, args.pilot_label_audit
    )
    validate(episodes, manifest)
    args.output_root.mkdir(parents=True)
    episode_path = args.output_root / "formal_train_episodes.jsonl"
    write_jsonl(episode_path, episodes)
    manifest["files_sha256"] = {episode_path.name: sha256(episode_path)}
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
