"""Build the preregistered V11 unseen-relation FewRel development corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_fewrel_selective_forgetting import (
    SOURCE_COMMIT, relation_partitions, verify_source,
)
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    GROUPS_PER_MECHANISM, GROUPS_PER_RELATION, MIN_ROWS, MIN_SUBJECTS,
    MIN_TEMPLATES, MIN_VALUES, capacity, choose_materials,
    development_exclusions, filter_rows, material_episode, row_from_source,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES, MECHANISMS, TWINS, write_json, write_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v11_fewrel_development_corpus"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V11_RELATION_TRANSFER_PLAN.md"
SEED = 202610191
PARTITION_SEED = 2311
EXPECTED_RELATIONS = 8
GROUP_PREFIX = "usr-v11-fewrel-development"
SOURCE_PARTITION = "fewrel_development_only"


def build_corpus(source_root: Path, exclusion_paths: list[Path], seed: int = SEED):
    source_hashes = verify_source(source_root)
    train = json.loads((source_root / "data/train_wiki.json").read_text())
    external = json.loads((source_root / "data/val_wiki.json").read_text())
    partitions = relation_partitions(train, external, seed=PARTITION_SEED)
    relations = sorted(partitions["development"])
    if len(relations) != EXPECTED_RELATIONS:
        raise ValueError("unexpected V11 FewRel development relation count")

    raw, source_removed = [], []
    raw_source_rows = 0
    for relation in relations:
        for index, source in enumerate(train[relation]):
            raw_source_rows += 1
            try:
                raw.append(row_from_source(relation, index, source))
            except ValueError:
                from tasks.build_fewrel_selective_forgetting import canonical_sha
                source_removed.append({
                    "relation_id": relation,
                    "source_index": index,
                    "source_row_sha256": canonical_sha(source),
                    "reason": "multiple_or_invalid_entity_mentions",
                })
    exclusions, exclusion_summary = development_exclusions(exclusion_paths)
    retained, removed = filter_rows(raw, exclusions)
    removed = source_removed + removed
    capacities = capacity(retained, relations)
    if not all(value["eligible"] for value in capacities.values()):
        failed = [key for key, value in capacities.items() if not value["eligible"]]
        raise ValueError(f"V11 relation capacity shortfall: {failed}")

    materials = choose_materials(retained, relations, seed)
    episodes = [
        material_episode(
            material, twin, group_prefix=GROUP_PREFIX,
            source_partition=SOURCE_PARTITION,
        )
        for material in materials for twin in range(TWINS)
    ]
    manifest = {
        "protocol": PROTOCOL, "plan": PLAN, "seed": seed,
        "partition_seed": PARTITION_SEED,
        "source": {"repository_commit": SOURCE_COMMIT,
                   "files_sha256": source_hashes},
        "partitions": partitions,
        "selected_partition": "development",
        "selected_relations": relations,
        "exclusions": exclusion_summary,
        "eligibility_thresholds": {
            "rows": MIN_ROWS, "subjects": MIN_SUBJECTS,
            "values": MIN_VALUES, "templates": MIN_TEMPLATES,
        },
        "capacity": capacities,
        "counts": {
            "raw_rows": raw_source_rows, "parsed_rows": len(raw),
            "retained_rows": len(retained), "exclusions": len(removed),
            "exclusions_by_reason": dict(sorted(Counter(
                row["reason"] for row in removed).items())),
            "eligible_relations": sum(value["eligible"] for value in capacities.values()),
            "source_groups": len(materials), "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "mechanisms": dict(sorted(Counter(
                row["mechanism"] for row in materials).items())),
            "relations": dict(sorted(Counter(
                row["target_relation"] for row in materials).items())),
            "material_source_rows": len({value for row in materials
                                         for value in row["material_source_row_sha256"]}),
            "material_entity_pairs": len({tuple(value) for row in materials
                                           for value in row["material_entity_pairs"]}),
        },
        "alpha_grid": [0.0, 0.5, 1.0], "candidate_memory_cap": 3,
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "internal_test_loaded": False, "external_test_loaded": False,
        "model_outputs_loaded": False,
    }
    return episodes, materials, removed, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-episodes", type=Path, action="append",
                        required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    episodes, materials, removed, manifest = build_corpus(
        args.source_root, args.exclusion_episodes, args.seed
    )
    args.output_root.mkdir(parents=True)
    write_jsonl(args.output_root / "development_episodes.jsonl", episodes)
    write_jsonl(args.output_root / "materials.jsonl", materials)
    write_jsonl(args.output_root / "exclusions.jsonl", removed)
    manifest["files_sha256"] = {
        name: sha256(args.output_root / name)
        for name in ("development_episodes.jsonl", "materials.jsonl",
                     "exclusions.jsonl")
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
