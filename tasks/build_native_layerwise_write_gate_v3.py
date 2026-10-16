"""Build the source-disjoint 48/384/96/96 corpus registered for V3.

Plain semantic material is written separately from tokenized model inputs.  A
scoring job receives one split file and cannot use the semantic audit file.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks import build_qwen_utility_semantic_record_gate_v1_formal as source_builder
from tasks.build_native_support_write_gate_v1 import (
    MECHANISM_SOURCE,
    TOKENIZER_PATTERNS,
    convert_episode,
    sha256,
    write_jsonl,
)


PROTOCOL = "native_layerwise_fastweight_admission_v3"
SOURCE_PROTOCOL = PROTOCOL + "_semantic_source"
PLAN = "NATIVE_LAYERWISE_WRITE_GATE_V3_PLAN.md"
SEED = 2026101701
GROUP_PREFIX = "layerwise-v3"
SOURCE_GROUPS = 624
SPLIT_PER_MECHANISM = {
    "confirmation": 8,
    "train": 64,
    "development": 16,
    "sealed_test": 16,
}
SOURCE_MECHANISMS = tuple(MECHANISM_SOURCE.values())


def stable_rank(source_group: str) -> str:
    return hashlib.sha256(f"{SEED}|{source_group}".encode()).hexdigest()


def opaque_group_id(mechanism: str, source_group: str) -> str:
    value = hashlib.sha256(f"{SEED}|{mechanism}|{source_group}".encode()).hexdigest()[:24]
    return f"{GROUP_PREFIX}-{value}"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_corpus(args) -> tuple[list[dict], dict]:
    # These module globals are read at call time.  Changing them here creates a
    # new opt-in corpus without changing the existing V1 builder defaults.
    original = (
        source_builder.FORMAL_GROUPS,
        source_builder.FORMAL_SEED,
        source_builder.GROUP_PREFIX,
        source_builder.MECHANISMS,
    )
    try:
        source_builder.FORMAL_GROUPS = SOURCE_GROUPS
        source_builder.FORMAL_SEED = SEED
        source_builder.GROUP_PREFIX = f"{GROUP_PREFIX}-source"
        source_builder.MECHANISMS = SOURCE_MECHANISMS
        inspection = source_builder.read_json(args.inspection)
        if inspection.get("protocol") != source_builder.INSPECTION_PROTOCOL or not inspection.get("passed"):
            raise ValueError("source inspection did not pass")
        pilot_manifest, data_audit, label_audit, pilot_episodes = source_builder.validate_prerequisites(
            args.pilot_data_root, args.pilot_label_audit
        )
        splits = inspection["splits"]
        relations = {relation for values in splits.values() for relation in values}
        patterns = source_builder.load_patterns(args.source_root, relations)
        facts = source_builder.load_facts(args.source_root, relations)
        removed_subjects = source_builder.cross_partition_subjects(facts, splits)
        facts = {
            relation: [row for row in rows if row["subject_norm"] not in removed_subjects]
            for relation, rows in facts.items()
        }
        train_relations = splits["train"]
        forbidden_hashes, forbidden_subjects = source_builder.reconstruct_pilot_exclusions(
            train_relations, facts, patterns
        )
        forbidden_subjects.update(
            source_builder.normalize(event["record"]["input"]["subject"])
            for episode in pilot_episodes for event in episode["events"]
        )
        pilot_forbidden_hashes = set(forbidden_hashes)
        pilot_forbidden_subjects = set(forbidden_subjects)
        episodes = []
        target_counts = Counter()
        for group_index in range(SOURCE_GROUPS):
            mechanism = SOURCE_MECHANISMS[group_index % len(SOURCE_MECHANISMS)]
            length = source_builder.LENGTHS[group_index % len(source_builder.LENGTHS)]
            def available(relation: str) -> int:
                return sum(row["source_fact_hash"] not in forbidden_hashes
                           and row["subject_norm"] not in forbidden_subjects
                           for row in facts[relation])
            candidates = sorted(
                train_relations,
                key=lambda relation: (
                    -available(relation), target_counts[relation],
                    source_builder.stable_rank(SEED, "v3-target", group_index, relation), relation,
                ),
            )
            material = None
            failures = []
            for target_relation in candidates:
                try:
                    material = source_builder.choose_formal_material(
                        group_index, train_relations, facts, patterns,
                        forbidden_hashes, forbidden_subjects,
                        target_relation_override=target_relation,
                        capacity_aware=True,
                    )
                except ValueError as error:
                    failures.append(str(error))
                    continue
                target_counts[target_relation] += 1
                break
            if material is None:
                raise ValueError(
                    f"capacity-aware allocation stopped at group {group_index}: {failures[:3]}"
                )
            provenance = source_builder.material_rows(material)
            # V3's registered isolation is stronger than the earlier corpus:
            # donor facts and donor subjects are also one-use across groups.
            forbidden_hashes.update(row["source_fact_hash"] for row in provenance)
            forbidden_subjects.update(row["subject_norm"] for row in provenance)
            input_provenance = source_builder.input_source_rows(material)
            for twin in range(source_builder.TWINS):
                episode = source_builder.build_episode(
                    group_index, mechanism, length, twin, material,
                    group_prefix=f"{GROUP_PREFIX}-source",
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
        manifest = {
            "protocol": SOURCE_PROTOCOL,
            "plan": PLAN,
            "source_manifest_sha256": inspection["source_manifest_sha256"],
            "source_inspection_sha256": sha256(args.inspection),
            "source_commits": inspection["source_commits"],
            "pilot_data_manifest_sha256": sha256(args.pilot_data_root / "manifest.json"),
            "pilot_data_audit_sha256": sha256(args.pilot_data_root / "data_audit.json"),
            "pilot_label_audit_sha256": sha256(args.pilot_label_audit),
            "seed": SEED,
            "counts": {
                "source_groups": SOURCE_GROUPS,
                "episodes": len(episodes),
                "boundaries": sum(len(row["boundaries"]) for row in episodes),
                "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
                "target_relations": dict(target_counts),
            },
            "splits": splits,
            "partition": "train",
            "pilot_overlap": {
                "source_fact_hashes": len(forbidden_hashes & pilot_forbidden_hashes),
                "subjects": len(forbidden_subjects & pilot_forbidden_subjects),
            },
            "cross_partition_subjects_removed": len(removed_subjects),
            "model_outputs_loaded": False,
            "sealed_test_scored": False,
        }
        # The forbidden sets contain their initial exclusions by construction;
        # overlap is measured on generated provenance below instead.
        generated_hashes = {value for row in episodes for value in row["audit"]["material_source_fact_hashes"]}
        generated_subjects = {value for row in episodes for value in row["audit"]["material_source_subject_norms"]}
        manifest["pilot_overlap"] = {
            "source_fact_hashes": len(generated_hashes & pilot_forbidden_hashes),
            "subjects": len(generated_subjects & pilot_forbidden_subjects),
        }
    finally:
        (
            source_builder.FORMAL_GROUPS,
            source_builder.FORMAL_SEED,
            source_builder.GROUP_PREFIX,
            source_builder.MECHANISMS,
        ) = original
    manifest["protocol"] = SOURCE_PROTOCOL
    manifest["plan"] = PLAN
    return episodes, manifest


def assign_splits(episodes: list[dict]) -> tuple[dict[str, list[tuple[str, dict]]], dict]:
    twin_zero = [row for row in episodes if row["twin"] == 0]
    by_source = defaultdict(list)
    for row in twin_zero:
        by_source[row["mechanism"]].append(row)
    expected = set(SOURCE_MECHANISMS)
    if set(by_source) != expected:
        raise ValueError("source mechanism set changed")
    result = {name: [] for name in SPLIT_PER_MECHANISM}
    assignment = {}
    for mechanism, source_mechanism in MECHANISM_SOURCE.items():
        candidates = sorted(by_source[source_mechanism], key=lambda row: stable_rank(row["source_group"]))
        if len(candidates) != sum(SPLIT_PER_MECHANISM.values()):
            raise ValueError(f"{source_mechanism}: expected 104 groups, got {len(candidates)}")
        cursor = 0
        for split, count in SPLIT_PER_MECHANISM.items():
            for episode in candidates[cursor:cursor + count]:
                result[split].append((mechanism, episode))
                assignment[episode["source_group"]] = split
            cursor += count
    if len(assignment) != SOURCE_GROUPS:
        raise ValueError("split assignment is not exhaustive")
    return result, assignment


def materialize(args) -> dict:
    from transformers import AutoTokenizer

    output = args.output_root
    if output.exists():
        raise FileExistsError(output)
    plan = args.plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    episodes, source_manifest = _source_corpus(args)
    split_rows, assignment = assign_splits(episodes)
    tokenizer_root = Path(args.tokenizer)
    tokenizer_files = sorted({
        path for pattern in TOKENIZER_PATTERNS
        for path in tokenizer_root.glob(pattern) if path.is_file()
    })
    if not tokenizer_files:
        raise ValueError("local tokenizer identity files are required")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    output.mkdir(parents=True)
    semantic_path = output / "semantic_source_episodes.jsonl"
    write_jsonl(semantic_path, episodes)
    source_manifest["files_sha256"] = {semantic_path.name: sha256(semantic_path)}
    _write_json(output / "semantic_source_manifest.json", source_manifest)

    files, audits, model_counts = {}, [], {}
    for split, values in split_rows.items():
        models = []
        for mechanism, episode in values:
            model, audit = convert_episode(
                tokenizer,
                episode,
                mechanism,
                protocol=PROTOCOL,
                group_prefix=GROUP_PREFIX,
                group_id_override=opaque_group_id(mechanism, episode["source_group"]),
            )
            # A mechanism hash would still reveal the six-way source class to a
            # learner.  Keep the mechanism only in the offline semantic audit.
            model.pop("mechanism_hash", None)
            audit["split"] = split
            models.append(model); audits.append(audit)
        path = output / f"{split}_model_inputs.jsonl"
        write_jsonl(path, models)
        files[path.name] = sha256(path)
        model_counts[split] = len(models)
    semantic_audit_path = output / "semantic_audit.jsonl"
    write_jsonl(semantic_audit_path, audits)
    files[semantic_audit_path.name] = sha256(semantic_audit_path)
    files[semantic_path.name] = sha256(semantic_path)
    files["semantic_source_manifest.json"] = sha256(output / "semantic_source_manifest.json")

    confirm = split_rows["confirmation"]
    smoke = []
    for mechanism in MECHANISM_SOURCE:
        candidates = [episode for name, episode in confirm if name == mechanism]
        smoke.extend(
            opaque_group_id(mechanism, row["source_group"]) for row in candidates[:2]
        )
    material_groups = defaultdict(set)
    material_subjects = defaultdict(set)
    for row in episodes:
        group = row["source_group"]
        material_groups[group].update(row["audit"]["material_source_fact_hashes"])
        material_subjects[group].update(row["audit"]["material_source_subject_norms"])
    manifest = {
        "protocol": PROTOCOL,
        "stage": "corpus",
        "plan_sha256": sha256(plan),
        "seed": SEED,
        "source_groups": SOURCE_GROUPS,
        "boundaries_per_group": 3,
        "split_groups": model_counts,
        "split_per_mechanism": SPLIT_PER_MECHANISM,
        "mechanisms": list(MECHANISM_SOURCE),
        "source_mechanisms": dict(MECHANISM_SOURCE),
        "assignment": assignment,
        "smoke_groups": smoke,
        "tokenizer": str(tokenizer_root),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files_sha256": {path.name: sha256(path) for path in tokenizer_files},
        "semantic_online_separation": True,
        "online_plaintext_fields": [],
        "material_fact_hashes": len(set().union(*material_groups.values())),
        "material_subjects": len(set().union(*material_subjects.values())),
        "source_manifest_sha256": sha256(output / "semantic_source_manifest.json"),
        "files_sha256": files,
        "model_outputs_loaded": False,
        "sealed_test_scored": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--pilot-data-root", type=Path, required=True)
    parser.add_argument("--pilot-label-audit", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--plan", type=Path, default=Path(PLAN))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
