"""Independent exact audit of the 512-group formal utility corpus."""

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
from scripts.inspect_lama_pararel_utility_source import normalize, sha256
from tasks.build_qwen_utility_semantic_record_gate_v1_formal import (
    AUDIT_PROTOCOL,
    FORMAL_GROUPS,
    PROTOCOL,
    build,
    read_jsonl,
    validate,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    LENGTHS,
    MECHANISMS,
    TWINS,
    write_jsonl,
)


def provenance_is_disjoint(episodes: list[dict], pilot_episodes: list[dict]) -> bool:
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    seen_hashes, seen_subjects = set(), set()
    for rows in groups.values():
        hashes = set(rows[0]["audit"]["material_source_fact_hashes"])
        subjects = set(rows[0]["audit"]["material_source_subject_norms"])
        if len(hashes) != 12 or len(subjects) != 12:
            return False
        if any(set(row["audit"]["material_source_fact_hashes"]) != hashes
               or set(row["audit"]["material_source_subject_norms"]) != subjects
               for row in rows[1:]):
            return False
        if hashes & seen_hashes or subjects & seen_subjects:
            return False
        seen_hashes.update(hashes); seen_subjects.update(subjects)
    pilot_subjects = {
        normalize(event["record"]["input"]["subject"])
        for episode in pilot_episodes for event in episode["events"]
    }
    return not (seen_subjects & pilot_subjects)


def inference_surfaces_are_train_only(episodes: list[dict], train_relations: set[str]) -> bool:
    return all(
        set(episode["audit"]["relation_ids"]) <= train_relations
        and episode["audit"]["target_relation"] in train_relations
        for episode in episodes
    )


def audit(
    source_root: Path,
    inspection_path: Path,
    pilot_data_root: Path,
    pilot_label_audit_path: Path,
    data_root: Path,
) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "formal_train_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = read_jsonl(episode_path)
    pilot_episodes = read_jsonl(pilot_data_root / "pilot_episodes.jsonl")
    expected_episodes, expected_manifest = build(
        source_root, inspection_path, pilot_data_root, pilot_label_audit_path
    )
    validate(expected_episodes, expected_manifest)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt = Path(directory) / "formal_train_episodes.jsonl"
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
    relations_disjoint = not (
        relation_sets["train"] & relation_sets["development"]
        or relation_sets["train"] & relation_sets["confirmation"]
        or relation_sets["development"] & relation_sets["confirmation"]
    )
    mechanism_values = list(counts["mechanisms"].values())
    target_values = list(counts["target_relations"].values())
    checks = {
        "protocol_and_locked_inputs": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("source_inspection_sha256") == sha256(inspection_path)
            and manifest.get("pilot_data_manifest_sha256")
            == sha256(pilot_data_root / "manifest.json")
            and manifest.get("pilot_data_audit_sha256")
            == sha256(pilot_data_root / "data_audit.json")
            and manifest.get("pilot_label_audit_sha256") == sha256(pilot_label_audit_path)
            and manifest.get("pilot_gate_passed") is True
        ),
        "exact_row_reconstruction": episodes == expected_episodes,
        "episode_file_hash": (
            sha256(episode_path) == manifest.get("files_sha256", {}).get(episode_path.name)
            == rebuilt_hash
        ),
        "manifest_reconstruction": manifest == expected_manifest,
        "registered_counts": (
            counts["source_groups"] == FORMAL_GROUPS
            and counts["episodes"] == FORMAL_GROUPS * TWINS
            and counts["boundaries"] == FORMAL_GROUPS * TWINS * BOUNDARIES
            and counts["queries"] == FORMAL_GROUPS * TWINS * BOUNDARIES * 5
        ),
        "balanced_design": (
            max(mechanism_values) - min(mechanism_values) <= TWINS
            and set(counts["lengths"].values()) == {FORMAL_GROUPS * TWINS // len(LENGTHS)}
            and max(target_values) - min(target_values) <= 1
            and set(counts["mechanisms"]) == set(MECHANISMS)
        ),
        "relation_partition_isolation": (
            relations_disjoint
            and inference_surfaces_are_train_only(episodes, relation_sets["train"])
        ),
        "pilot_and_group_disjointness": (
            manifest.get("pilot_overlap") == {"source_fact_hashes": 0, "subjects": 0}
            and provenance_is_disjoint(episodes, pilot_episodes)
        ),
        "input_audit_separation": input_fields_are_clean(episodes),
        "twin_structure": twin_structure_is_valid(episodes),
        "causal_queries": query_structure_is_valid(episodes),
        "sealed_and_model_free": (
            manifest.get("partition") == "train"
            and not manifest.get("model_outputs_loaded")
            and not manifest.get("development_episodes_generated")
            and not manifest.get("confirmation_episodes_generated")
            and not manifest.get("forbidden_sources_loaded")
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
        "source_inspection_sha256": sha256(inspection_path),
        "pilot_data_manifest_sha256": sha256(pilot_data_root / "manifest.json"),
        "pilot_label_audit_sha256": sha256(pilot_label_audit_path),
        "development_episodes_generated": False,
        "confirmation_episodes_generated": False,
        "model_outputs_loaded": False,
        "next_stage": "formal_label_construction" if passed else "stop",
        "terminal_status": (
            "formal_train_data_audit_passed" if passed
            else "formal_train_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-data-root", type=Path, required=True)
    parser.add_argument("--pilot-label-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.source_root, args.inspection, args.pilot_data_root,
        args.pilot_label_audit, args.data_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
