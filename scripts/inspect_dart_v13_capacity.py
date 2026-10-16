"""Run the registered model-free DART V13 feasibility allocation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    choose_materials,
    development_exclusions,
    filter_rows,
    normalize,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    stable_rank,
    write_json,
    write_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v13_dart_capacity"
SOURCE_COMMIT = "bb644b12ec1d0264f96a31145534b9a94f6ff5f5"
SOURCE_MANIFEST_SHA256 = "dc242d6873951b774d36c02783b546a265f2d645f5c236c9d51a176c187ad558"
TRAIN_SHA256 = "92c8594979c05f508f5739047079ec2ffe5a244e58bfa2b50a9cb8b9c65f5a2b"
SEED = 202610211
TARGET_RELATIONS = 8
TARGET_MIN_PAIRS = 140
TARGET_MIN_TEMPLATES = 7
BACKGROUND_MIN_PAIRS = 5
BACKGROUND_MIN_TEMPLATES = 4
HUMAN_SOURCES = {
    "WikiTableQuestions_lily",
    "WikiSQL_lily",
    "WikiTableQuestions_mturk",
}
EXCLUDED_SOURCES = {"WikiSQL_decl_sents", "webnlg", "e2e"}


def display(value: str) -> str:
    return " ".join(str(value).replace("_", " ").strip().strip('"').split())


def normalized_entity(value: str) -> str:
    return normalize(display(value))


def aligned_pattern(text: str, subject: str, value: str) -> str | None:
    """Replace exactly one nonoverlapping literal subject/value occurrence."""
    subject_text, value_text = display(subject), display(value)
    if not subject_text or not value_text or normalize(subject_text) == normalize(value_text):
        return None
    subject_matches = list(re.finditer(re.escape(subject_text), text, re.IGNORECASE))
    value_matches = list(re.finditer(re.escape(value_text), text, re.IGNORECASE))
    if len(subject_matches) != 1 or len(value_matches) != 1:
        return None
    spans = [
        (subject_matches[0].start(), subject_matches[0].end(), "[X]"),
        (value_matches[0].start(), value_matches[0].end(), "[Y]"),
    ]
    spans.sort()
    if spans[0][1] > spans[1][0]:
        return None
    output, cursor = [], 0
    for start, end, slot in spans:
        output.extend((text[cursor:start], slot)); cursor = end
    output.append(text[cursor:])
    pattern = " ".join("".join(output).split())
    if pattern.count("[X]") != 1 or pattern.count("[Y]") != 1:
        return None
    return pattern


def canonical_row_hash(entry: dict, entry_index: int, annotation_index: int) -> str:
    payload = json.dumps({
        "entry_index": entry_index,
        "annotation_index": annotation_index,
        "entry": entry,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_train(train_path: Path) -> tuple[list[dict], dict]:
    entries = json.loads(train_path.read_text())
    if not isinstance(entries, list):
        raise ValueError("DART train JSON must contain a list")
    raw_rows, exclusions = [], Counter()
    sources, triple_lengths = Counter(), Counter()
    for entry_index, entry in enumerate(entries):
        triples = entry.get("tripleset")
        annotations = entry.get("annotations")
        if not isinstance(triples, list) or not isinstance(annotations, list):
            exclusions["invalid_entry_schema"] += 1
            continue
        triple_lengths[str(len(triples))] += 1
        for annotation in annotations:
            sources[str(annotation.get("source"))] += 1
        if len(triples) != 1:
            exclusions["not_exactly_one_triple"] += len(annotations) or 1
            continue
        triple = triples[0]
        if not isinstance(triple, list) or len(triple) != 3:
            exclusions["invalid_triple_schema"] += len(annotations) or 1
            continue
        subject, relation, value = map(str, triple)
        if not display(relation) or not normalized_entity(subject) \
                or not normalized_entity(value) \
                or normalized_entity(subject) == normalized_entity(value):
            exclusions["invalid_triple_values"] += len(annotations) or 1
            continue
        for annotation_index, annotation in enumerate(annotations):
            source = str(annotation.get("source", ""))
            if source not in HUMAN_SOURCES:
                exclusions["excluded_or_unknown_annotation_source"] += 1
                continue
            text = " ".join(str(annotation.get("text", "")).split())
            pattern = aligned_pattern(text, subject, value)
            if pattern is None:
                exclusions["not_exact_literal_xy_alignment"] += 1
                continue
            source_hash = canonical_row_hash(entry, entry_index, annotation_index)
            raw_rows.append({
                "relation_id": display(relation),
                "source_index": entry_index,
                "annotation_index": annotation_index,
                "annotation_source": source,
                "source_row_sha256": source_hash,
                "source_fact_hash": source_hash,
                "head_entity_id": normalized_entity(subject),
                "tail_entity_id": normalized_entity(value),
                "entity_pair": [normalized_entity(subject), normalized_entity(value)],
                "subject": display(subject),
                "subject_norm": normalized_entity(subject),
                "value": display(value),
                "value_norm": normalized_entity(value),
                "pattern": pattern,
                "pattern_norm": normalize(pattern),
                "statement": text,
            })
    summary = {
        "entries": len(entries),
        "triple_lengths": dict(sorted(triple_lengths.items())),
        "annotation_sources": dict(sorted(sources.items())),
        "raw_aligned_rows": len(raw_rows),
        "parse_exclusions": dict(sorted(exclusions.items())),
    }
    return raw_rows, summary


def deduplicate_pairs(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["relation_id"], *row["entity_pair"])].append(row)
    kept, removed = [], []
    for key, values in sorted(grouped.items()):
        ranked = sorted(values, key=lambda row: (
            stable_rank(seed, "dart-pair", *key, row["source_row_sha256"]),
            row["source_row_sha256"],
        ))
        kept.append(ranked[0])
        removed.extend({
            "relation_id": row["relation_id"],
            "source_index": row["source_index"],
            "source_row_sha256": row["source_row_sha256"],
            "reason": "duplicate_relation_entity_pair",
        } for row in ranked[1:])
    return kept, removed


def relation_capacity(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["relation_id"]].append(row)
    output = {}
    for relation, values in sorted(grouped.items()):
        output[relation] = {
            "pairs": len(values),
            "subjects": len({row["subject_norm"] for row in values}),
            "values": len({row["value_norm"] for row in values}),
            "templates": len({row["pattern_norm"] for row in values}),
            "annotation_sources": dict(sorted(Counter(
                row["annotation_source"] for row in values).items())),
        }
    return output


def inspect(source_root: Path, exclusion_paths: list[Path], seed: int = SEED) \
        -> tuple[dict, list[dict], list[dict], list[dict]]:
    manifest_path = source_root / "source_manifest.json"
    train_path = source_root / "repository/data/v1.1.1/dart-v1.1.1-full-train.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != SOURCE_MANIFEST_SHA256 \
            or manifest.get("repository_commit") != SOURCE_COMMIT \
            or manifest.get("json_or_xml_parsed") is not False \
            or sha256(train_path) != TRAIN_SHA256:
        raise ValueError("DART source identity differs from V13 registration")

    raw_rows, source_summary = parse_train(train_path)
    pair_rows, pair_removed = deduplicate_pairs(raw_rows, seed)
    prior_keys, prior_summary = development_exclusions(exclusion_paths)
    retained, overlap_removed = filter_rows(pair_rows, prior_keys)
    removed = pair_removed + overlap_removed
    capacities = relation_capacity(retained)
    target_candidates = [
        relation for relation, row in capacities.items()
        if row["pairs"] >= TARGET_MIN_PAIRS
        and row["templates"] >= TARGET_MIN_TEMPLATES
    ]
    target_candidates.sort(key=lambda relation: (
        stable_rank(seed, "dart-target", relation), relation,
    ))
    selected = target_candidates[:TARGET_RELATIONS]
    backgrounds = sorted(
        relation for relation, row in capacities.items()
        if relation not in selected
        and row["pairs"] >= BACKGROUND_MIN_PAIRS
        and row["templates"] >= BACKGROUND_MIN_TEMPLATES
    )
    materials, allocation_error = [], None
    if len(selected) == TARGET_RELATIONS:
        try:
            materials = choose_materials(
                retained, selected, seed=seed, background_relations=backgrounds,
            )
        except ValueError as error:
            allocation_error = str(error)
    else:
        allocation_error = (
            f"only {len(selected)} relations meet the frozen target threshold"
        )

    material_hashes = [value for row in materials
                       for value in row["material_source_row_sha256"]]
    material_pairs = [tuple(value) for row in materials
                      for value in row["material_entity_pairs"]]
    checks = {
        "source_commit_manifest_and_train_hash_locked": True,
        "only_registered_train_json_parsed": True,
        "human_sources_only": all(
            row["annotation_source"] in HUMAN_SOURCES for row in retained
        ),
        "excluded_sources_absent": all(
            row["annotation_source"] not in EXCLUDED_SOURCES for row in retained
        ),
        "eight_frozen_ranked_target_relations": len(selected) == TARGET_RELATIONS,
        "complete_allocator_succeeded": len(materials) == TARGET_RELATIONS * 14,
        "globally_unique_material_rows": bool(materials)
            and len(material_hashes) == len(set(material_hashes)),
        "globally_unique_material_pairs": bool(materials)
            and len(material_pairs) == len(set(material_pairs)),
        "no_model_or_sealed_benchmark_access": True,
    }
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "train_sha256": TRAIN_SHA256,
        "source_summary": source_summary,
        "prior_corpus_exclusions": prior_summary,
        "counts": {
            "raw_aligned_rows": len(raw_rows),
            "deduplicated_pair_rows": len(pair_rows),
            "retained_rows": len(retained),
            "removed_rows": len(removed),
            "relations": len(capacities),
            "target_candidates": len(target_candidates),
            "background_relations": len(backgrounds),
            "allocated_groups": len(materials),
            "material_source_rows": len(set(material_hashes)),
            "material_entity_pairs": len(set(material_pairs)),
        },
        "thresholds": {
            "target_pairs": TARGET_MIN_PAIRS,
            "target_templates": TARGET_MIN_TEMPLATES,
            "background_pairs": BACKGROUND_MIN_PAIRS,
            "background_templates": BACKGROUND_MIN_TEMPLATES,
        },
        "target_candidates_ranked": target_candidates,
        "selected_target_relations": selected,
        "background_relations": backgrounds,
        "allocation_error": allocation_error,
        "capacity": capacities,
        "model_loaded": False,
        "sealed_benchmarks_loaded": False,
        "next_stage": "v13_dart_corpus_construction" if all(checks.values())
                      else "stop_v13_dart_capacity_failure",
        "terminal_status": "v13_dart_capacity_passed" if all(checks.values())
                           else "v13_dart_capacity_failed",
    }
    return result, retained, materials, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclude-episodes", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    result, rows, materials, removed = inspect(
        args.source_root, args.exclude_episodes, args.seed,
    )
    args.output_root.mkdir(parents=True)
    write_jsonl(args.output_root / "eligible_rows.jsonl", rows)
    write_jsonl(args.output_root / "materials.jsonl", materials)
    write_jsonl(args.output_root / "exclusions.jsonl", removed)
    result["files_sha256"] = {
        name: sha256(args.output_root / name)
        for name in ("eligible_rows.jsonl", "materials.jsonl", "exclusions.jsonl")
    }
    write_json(args.output_root / "capacity.json", result)
    print(json.dumps({
        "passed": result["passed"],
        "counts": result["counts"],
        "selected_target_relations": result["selected_target_relations"],
        "allocation_error": result["allocation_error"],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
