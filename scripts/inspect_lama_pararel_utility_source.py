"""Inspect only aggregate eligibility for the frozen LAMA/ParaRel protocol."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile


PROTOCOL = "qwen_utility_semantic_record_source_inspection_v1"
SOURCE_PROTOCOL = "qwen_utility_semantic_record_sources_v1"
SPLIT_SEED = "261016:utility-semantic-record:relation"
MIN_PATTERNS = 4
MIN_FACTS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(" .?!:;")


def relation_rank(relation: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{relation}".encode()).hexdigest()


def split_relations(relations: list[str]) -> dict[str, list[str]]:
    ordered = sorted(relations, key=lambda value: (relation_rank(value), value))
    train_end = int(len(ordered) * 0.6)
    development_end = train_end + int(len(ordered) * 0.2)
    return {
        "train": ordered[:train_end],
        "development": ordered[train_end:development_end],
        "confirmation": ordered[development_end:],
    }


def load_patterns(pattern_root: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    patterns: dict[str, list[str]] = {}
    rejected = Counter()
    for path in sorted(pattern_root.glob("*.jsonl")):
        relation = path.stem
        valid = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pattern = row.get("pattern")
            if not isinstance(pattern, str):
                rejected["missing_pattern"] += 1
                continue
            compact = re.sub(r"\s+", " ", pattern.strip())
            if compact.count("[X]") != 1 or compact.count("[Y]") != 1:
                rejected["invalid_slots"] += 1
                continue
            valid.add(compact)
        patterns[relation] = sorted(valid)
    return patterns, dict(rejected)


def load_trex_facts(archive: Path) -> tuple[dict[str, set[tuple[str, str]]], dict[str, set[str]], dict[str, int]]:
    facts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    subjects: dict[str, set[str]] = defaultdict(set)
    rejected = Counter()
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(
            name for name in bundle.namelist()
            if name.startswith("data/TREx/") and name.endswith(".jsonl")
        )
        for name in names:
            relation_from_path = Path(name).stem
            with bundle.open(name) as source:
                for raw in source:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    relation = row.get("predicate_id")
                    subject = row.get("sub_label")
                    value = row.get("obj_label")
                    if relation != relation_from_path:
                        rejected["relation_path_mismatch"] += 1
                        continue
                    if not isinstance(subject, str) or not isinstance(value, str):
                        rejected["missing_text"] += 1
                        continue
                    subject_norm, value_norm = normalize(subject), normalize(value)
                    if not subject_norm or not value_norm:
                        rejected["blank_text"] += 1
                        continue
                    if subject_norm == value_norm:
                        rejected["identical_subject_value"] += 1
                        continue
                    before = len(facts[relation])
                    facts[relation].add((subject_norm, value_norm))
                    if len(facts[relation]) == before:
                        rejected["duplicate_fact"] += 1
                    subjects[relation].add(subject_norm)
    return dict(facts), dict(subjects), dict(rejected)


def inspect(source_root: Path) -> dict:
    source_manifest_path = source_root / "metadata" / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("unexpected source acquisition protocol")
    archive = source_root / "downloads" / "lama_data.zip"
    expected_archive = source_manifest["sources"]["lama_data"]
    if archive.stat().st_size != expected_archive["bytes"]:
        raise ValueError("LAMA archive byte count changed")
    if sha256(archive) != expected_archive["sha256"]:
        raise ValueError("LAMA archive hash changed")

    pattern_root = source_root / "sources" / "pararel" / "data" / "pattern_data" / "graphs_json"
    patterns, pattern_rejected = load_patterns(pattern_root)
    facts, subjects, fact_rejected = load_trex_facts(archive)
    intersection = sorted(set(patterns) & set(facts))
    eligible = sorted(
        relation for relation in intersection
        if len(patterns[relation]) >= MIN_PATTERNS and len(facts[relation]) >= MIN_FACTS
    )
    splits = split_relations(eligible)
    relation_to_split = {
        relation: split for split, relations in splits.items() for relation in relations
    }
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for relation in eligible:
        for subject in subjects[relation]:
            subject_splits[subject].add(relation_to_split[relation])
    cross_partition_subjects = {
        subject for subject, owners in subject_splits.items() if len(owners) > 1
    }
    retained_counts = {
        relation: sum(subject not in cross_partition_subjects for subject, _ in facts[relation])
        for relation in eligible
    }
    details = {
        relation: {
            "patterns": len(patterns[relation]),
            "facts": len(facts[relation]),
            "subjects": len(subjects[relation]),
            "post_cross_partition_subject_filter_facts": retained_counts[relation],
            "split": relation_to_split[relation],
        }
        for relation in eligible
    }
    return {
        "protocol": PROTOCOL,
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_commits": {
            "lama": source_manifest["sources"]["lama"]["commit"],
            "pararel": source_manifest["sources"]["pararel"]["commit"],
        },
        "source_archive": expected_archive,
        "thresholds": {"min_patterns": MIN_PATTERNS, "min_facts": MIN_FACTS},
        "counts": {
            "trex_relations": len(facts),
            "pararel_relations": len(patterns),
            "intersection_relations": len(intersection),
            "eligible_relations": len(eligible),
            "eligible_facts": sum(len(facts[relation]) for relation in eligible),
            "eligible_patterns": sum(len(patterns[relation]) for relation in eligible),
            "cross_partition_subjects": len(cross_partition_subjects),
            "post_subject_filter_facts": sum(retained_counts.values()),
        },
        "split_counts": {split: len(relations) for split, relations in splits.items()},
        "splits": splits,
        "relations": details,
        "rejected": {"patterns": pattern_rejected, "facts": fact_rejected},
        "gates": {
            "at_least_20_relations": len(eligible) >= 20,
            "at_least_4_development_relations": len(splits["development"]) >= 4,
            "at_least_4_confirmation_relations": len(splits["confirmation"]) >= 4,
            "every_relation_has_post_filter_facts": all(value > 0 for value in retained_counts.values()),
        },
        "content_written": False,
        "forbidden_sources_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.source_root)
    result["passed"] = all(result["gates"].values())
    result["terminal_status"] = (
        "source_inspection_passed" if result["passed"] else "source_inspection_failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "counts": result["counts"],
        "split_counts": result["split_counts"],
        "gates": result["gates"],
        "passed": result["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
