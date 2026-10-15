"""Independently reconstruct and audit the V10 FewRel corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    AUDIT_PROTOCOL, BOUNDARIES, EXPECTED_TRAIN_RELATIONS,
    GROUPS_PER_MECHANISM, GROUPS_PER_RELATION, MECHANISMS, MIN_ROWS,
    MIN_SUBJECTS, MIN_TEMPLATES, MIN_VALUES, PILOT_RELATIONS, PROTOCOL,
    TWINS, build_corpus, normalize,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def audit(data_root: Path, source_root: Path,
          development_paths: list[Path]) -> dict:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    observed = {
        name: load_jsonl(data_root / name)
        for name in ("formal_episodes.jsonl", "pilot_episodes.jsonl",
                     "materials.jsonl", "exclusions.jsonl")
    }
    expected_formal, expected_pilot, expected_materials, expected_exclusions, \
        expected_manifest = build_corpus(
            source_root, development_paths, manifest["seed"]
        )
    expected = {
        "formal_episodes.jsonl": expected_formal,
        "pilot_episodes.jsonl": expected_pilot,
        "materials.jsonl": expected_materials,
        "exclusions.jsonl": expected_exclusions,
    }
    file_hashes = {name: sha256(data_root / name) for name in observed}
    sources = [value for row in observed["materials.jsonl"]
               for value in row["material_source_row_sha256"]]
    pairs = [tuple(value) for row in observed["materials.jsonl"]
             for value in row["material_entity_pairs"]]
    groups = defaultdict(list)
    for row in observed["formal_episodes.jsonl"]:
        groups[row["source_group"]].append(row)
    mechanisms = Counter(row["mechanism"] for row in observed["materials.jsonl"])
    relations = Counter(row["target_relation"] for row in observed["materials.jsonl"])
    pilot_relations = {
        row["audit"]["target_source_relation"]
        for row in observed["pilot_episodes.jsonl"]
    }
    exact_rephrase = all(
        normalize(boundary["queries"][0]["input"]["query"])
        != normalize(boundary["queries"][1]["input"]["query"])
        and boundary["queries"][0]["answer"] == boundary["queries"][1]["answer"]
        and all(query["answer"] in query["candidates"]
                and len(query["candidates"]) == 6
                for query in boundary["queries"])
        for episode in observed["formal_episodes.jsonl"]
        for boundary in episode["boundaries"]
    )
    counts = manifest["counts"]
    checks = {
        "protocol_seed_and_registered_parameters": (
            manifest["protocol"] == PROTOCOL
            and manifest["seed"] == 202610181
            and manifest["eligibility_thresholds"] == {
                "rows": MIN_ROWS, "subjects": MIN_SUBJECTS,
                "values": MIN_VALUES, "templates": MIN_TEMPLATES,
            }
            and manifest["alpha_grid"] == [0.0, 0.5, 1.0]
            and manifest["candidate_memory_cap"] == 3
        ),
        "official_train_partition_only": (
            len(manifest["partitions"]["train"]) == EXPECTED_TRAIN_RELATIONS
            and set(relations) == set(manifest["partitions"]["train"])
            and all(row["audit"]["source_partition"] == "fewrel_train_only"
                    for row in observed["formal_episodes.jsonl"])
        ),
        "all_registered_relations_eligible": (
            len(manifest["capacity"]) == EXPECTED_TRAIN_RELATIONS
            and all(value["eligible"] for value in manifest["capacity"].values())
        ),
        "deterministic_full_reconstruction": all(
            observed[name] == expected[name] for name in observed
        ),
        "manifest_reconstruction": all(
            manifest[key] == expected_manifest[key]
            for key in expected_manifest if key != "files_sha256"
        ),
        "file_hashes_recomputed": manifest["files_sha256"] == file_hashes,
        "source_rows_unique_across_groups": len(sources) == len(set(sources)),
        "entity_pairs_unique_across_groups": len(pairs) == len(set(pairs)),
        "balanced_relation_and_mechanism_coverage": (
            set(relations.values()) == {GROUPS_PER_RELATION}
            and set(mechanisms) == set(MECHANISMS)
            and set(mechanisms.values())
            == {EXPECTED_TRAIN_RELATIONS * GROUPS_PER_MECHANISM}
        ),
        "episode_twin_boundary_query_structure": (
            len(groups) == EXPECTED_TRAIN_RELATIONS * GROUPS_PER_RELATION
            and all(len(values) == TWINS
                    and {row["twin"] for row in values} == set(range(TWINS))
                    for values in groups.values())
            and all(len(row["boundaries"]) == BOUNDARIES
                    for row in observed["formal_episodes.jsonl"])
        ),
        "natural_exact_and_rephrase_queries": exact_rephrase,
        "pilot_is_two_complete_relations": (
            pilot_relations == set(manifest["pilot_relations"])
            and len(pilot_relations) == PILOT_RELATIONS
            and counts["pilot_source_groups"]
            == PILOT_RELATIONS * GROUPS_PER_RELATION
            and counts["pilot_episodes"]
            == PILOT_RELATIONS * GROUPS_PER_RELATION * TWINS
        ),
        "manifest_counts_match_rows": (
            counts["source_groups"] == len(observed["materials.jsonl"])
            and counts["episodes"] == len(observed["formal_episodes.jsonl"])
            and counts["pilot_episodes"] == len(observed["pilot_episodes.jsonl"])
            and counts["exclusions"] == len(observed["exclusions.jsonl"])
            and counts["material_source_rows"] == len(set(sources))
            and counts["material_entity_pairs"] == len(set(pairs))
        ),
        "development_used_for_exclusions_only": (
            manifest["development_exclusions"]["labels_or_model_outputs_used"] is False
            and manifest["development_labels_loaded"] is False
            and manifest["development_model_scores_loaded"] is False
        ),
        "no_model_or_sealed_benchmark_access": (
            manifest["model_outputs_loaded"] is False
            and manifest["sealed_benchmarks_loaded"] is False
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": passed,
        "checks": checks,
        "counts": counts,
        "manifest_sha256": sha256(manifest_path),
        "files_sha256": file_hashes,
        "next_stage": "v10_28_group_qwen_resource_pilot" if passed else "stop",
        "terminal_status": (
            "v10_fewrel_corpus_audit_passed"
            if passed else "v10_fewrel_corpus_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--development-episodes", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.source_root, args.development_episodes)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
