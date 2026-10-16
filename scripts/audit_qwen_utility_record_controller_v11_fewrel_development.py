"""Independently reconstruct and audit the V11 FewRel development corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import normalize
from tasks.build_qwen_utility_record_controller_v11_fewrel_development import (
    AUDIT_PROTOCOL, EXPECTED_RELATIONS, GROUP_PREFIX, PROTOCOL,
    SOURCE_PARTITION, build_corpus,
)
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    GROUPS_PER_MECHANISM, GROUPS_PER_RELATION,
)
from tasks.build_qwen_utility_record_controller_v11_fewrel_development import (
    MIN_ROWS, MIN_SUBJECTS, MIN_TEMPLATES, MIN_VALUES,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES, MECHANISMS, TWINS, write_json,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(data_root: Path, source_root: Path, exclusion_paths: list[Path]) -> dict:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = ("development_episodes.jsonl", "materials.jsonl", "exclusions.jsonl")
    observed = {name: load_jsonl(data_root / name) for name in names}
    expected_episodes, expected_materials, expected_removed, expected_manifest = \
        build_corpus(source_root, exclusion_paths, manifest["seed"])
    expected = {
        "development_episodes.jsonl": expected_episodes,
        "materials.jsonl": expected_materials,
        "exclusions.jsonl": expected_removed,
    }
    file_hashes = {name: sha256(data_root / name) for name in names}
    episodes, materials = observed["development_episodes.jsonl"], observed["materials.jsonl"]
    relations = Counter(row["target_relation"] for row in materials)
    mechanisms = Counter(row["mechanism"] for row in materials)
    groups = defaultdict(list)
    for row in episodes:
        groups[row["source_group"]].append(row)
    sources = [value for row in materials for value in row["material_source_row_sha256"]]
    pairs = [tuple(value) for row in materials for value in row["material_entity_pairs"]]
    natural_queries = all(
        normalize(boundary["queries"][0]["input"]["query"])
        != normalize(boundary["queries"][1]["input"]["query"])
        and boundary["queries"][0]["answer"] == boundary["queries"][1]["answer"]
        and all(query["answer"] in query["candidates"] and len(query["candidates"]) == 6
                for query in boundary["queries"])
        for episode in episodes for boundary in episode["boundaries"]
    )
    checks = {
        "protocol_and_registered_parameters": (
            manifest["protocol"] == PROTOCOL
            and manifest["eligibility_thresholds"] == {
                "rows": MIN_ROWS, "subjects": MIN_SUBJECTS,
                "values": MIN_VALUES, "templates": MIN_TEMPLATES,
            }
            and manifest["alpha_grid"] == [0.0, 0.5, 1.0]
            and manifest["candidate_memory_cap"] == 3
        ),
        "development_relations_only_and_tests_sealed": (
            len(relations) == EXPECTED_RELATIONS
            and set(relations) == set(manifest["partitions"]["development"])
            and not (set(relations) & set(manifest["partitions"]["train"]))
            and not manifest["internal_test_loaded"]
            and not manifest["external_test_loaded"]
            and all(row["audit"]["source_partition"] == SOURCE_PARTITION
                    for row in episodes)
        ),
        "training_relations_are_background_only": (
            set(manifest["background_relations"]) == set(manifest["partitions"]["train"])
            and all(row["target_relation"] in manifest["partitions"]["development"]
                    for row in materials)
            and all(record["relation_id"] in manifest["partitions"]["train"]
                    for row in materials for record in row["stable"])
        ),
        "all_relations_eligible": (
            len(manifest["capacity"]) == EXPECTED_RELATIONS
            and all(value["eligible"] for value in manifest["capacity"].values())
        ),
        "deterministic_full_reconstruction": all(
            observed[name] == expected[name] for name in names
        ),
        "manifest_reconstruction": all(
            manifest[key] == expected_manifest[key]
            for key in expected_manifest if key != "files_sha256"
        ),
        "file_hashes_recomputed": manifest["files_sha256"] == file_hashes,
        "source_rows_and_pairs_unique": (
            len(sources) == len(set(sources)) and len(pairs) == len(set(pairs))
        ),
        "balanced_relations_and_mechanisms": (
            set(relations.values()) == {GROUPS_PER_RELATION}
            and set(mechanisms) == set(MECHANISMS)
            and set(mechanisms.values()) == {EXPECTED_RELATIONS * GROUPS_PER_MECHANISM}
        ),
        "twin_boundary_query_structure": (
            len(groups) == EXPECTED_RELATIONS * GROUPS_PER_RELATION
            and all(len(values) == TWINS and {row["twin"] for row in values} == set(range(TWINS))
                    and all(row["source_group"].startswith(GROUP_PREFIX) for row in values)
                    for values in groups.values())
            and all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
            and natural_queries
        ),
        "expected_scale": (
            len(materials) == 112 and len(episodes) == 224
            and sum(len(row["boundaries"]) for row in episodes) == 672
            and sum(len(boundary["queries"]) for row in episodes
                    for boundary in row["boundaries"]) == 3360
        ),
        "no_model_or_test_access": (
            not manifest["model_outputs_loaded"]
            and not manifest["development_labels_loaded"]
            and not manifest["development_model_scores_loaded"]
        ),
        "v10_donor_hashes_used_as_exclusions": any(
            value["episodes"] == 1344
            for value in manifest["exclusions"]["files"].values()
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed, "checks": checks,
        "counts": manifest["counts"], "manifest_sha256": sha256(manifest_path),
        "files_sha256": file_hashes,
        "next_stage": "v11_development_qwen_labels" if passed else "stop",
        "terminal_status": "v11_development_corpus_audit_passed" if passed else "v11_development_corpus_audit_failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-episodes", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.source_root, args.exclusion_episodes)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
