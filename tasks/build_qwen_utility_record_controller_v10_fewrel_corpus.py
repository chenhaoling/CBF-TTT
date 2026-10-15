"""Build the V10 train-only natural FewRel counterfactual corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_fewrel_selective_forgetting import (
    SOURCE_COMMIT, SOURCE_HASHES, canonical_sha, masked_surface,
    relation_partitions, verify_source,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES, LENGTHS, MECHANISMS, TWINS, build_episode, render,
    stable_rank, write_json, write_jsonl,
)


PROTOCOL = "qwen_utility_record_controller_v10_fewrel_corpus"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v10_fewrel_corpus_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V10_FEWREL_CORPUS_PLAN.md"
SEED = 202610181
GROUPS_PER_MECHANISM = 2
GROUPS_PER_RELATION = GROUPS_PER_MECHANISM * len(MECHANISMS)
PILOT_RELATIONS = 2
MIN_ROWS = 28
MIN_SUBJECTS = 20
MIN_VALUES = 20
MIN_TEMPLATES = 7
PATTERNS_PER_GROUP_RELATION = 4
EXPECTED_TRAIN_RELATIONS = 48
GROUP_PREFIX = "usr-v10-fewrel-train"


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def source_pattern(row: dict) -> str:
    """Return a natural FewRel sentence with exactly one X/Y slot each."""
    surface = masked_surface(row)
    if surface.count("[SUBJECT]") != 1 or surface.count("[VALUE]") != 1:
        raise ValueError("V10 requires one head and one tail mention")
    return surface.replace("[SUBJECT]", "[X]").replace("[VALUE]", "[Y]")


def row_from_source(relation: str, index: int, row: dict) -> dict:
    pattern = source_pattern(row)
    subject = " ".join(str(row["h"][0]).split())
    value = " ".join(str(row["t"][0]).split())
    source_hash = canonical_sha(row)
    return {
        "relation_id": relation,
        "source_index": index,
        "source_row_sha256": source_hash,
        "source_fact_hash": source_hash,
        "head_entity_id": str(row["h"][1]),
        "tail_entity_id": str(row["t"][1]),
        "entity_pair": [str(row["h"][1]), str(row["t"][1])],
        "subject": subject,
        "subject_norm": normalize(subject),
        "value": value,
        "value_norm": normalize(value),
        "pattern": pattern,
        "pattern_norm": normalize(pattern),
        "statement": render(pattern, subject, value),
    }


def _walk_hashes(value) -> set[str]:
    output = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_row_sha256" and isinstance(item, str):
                output.add(item)
            else:
                output.update(_walk_hashes(item))
    elif isinstance(value, list):
        for item in value:
            output.update(_walk_hashes(item))
    return output


def mask_text(text: str, subject: str, value: str | None) -> str:
    output = str(text)
    if subject:
        output = output.replace(subject, "[X]", 1)
    if value:
        output = output.replace(value, "[Y]", 1)
    output = output.replace("____", "[Y]")
    output = re.sub(r"^Complete the effective session fact:\s*", "", output)
    return normalize(output)


def development_exclusions(paths: list[Path]) -> tuple[dict[str, set[str]], dict]:
    """Read development text only to produce exclusion hashes/keys."""
    values = {"source_rows": set(), "statements": set(), "templates": set()}
    files = {}
    for path in paths:
        files[str(path)] = sha256(path)
        count = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            episode = json.loads(line); count += 1
            values["source_rows"].update(_walk_hashes(episode.get("audit", {})))
            for event in episode.get("events", []):
                record = event.get("record", {})
                payload = record.get("input", {})
                statement = payload.get("statement")
                if isinstance(statement, str):
                    values["statements"].add(normalize(statement))
                    values["templates"].add(mask_text(
                        statement, str(payload.get("subject", "")),
                        str(payload.get("value", "")),
                    ))
            for boundary in episode.get("boundaries", []):
                for query in boundary.get("queries", []):
                    payload = query.get("input", {})
                    text = payload.get("query")
                    if isinstance(text, str):
                        values["templates"].add(mask_text(
                            text, str(payload.get("subject", "")), None,
                        ))
        files[str(path)] = {"sha256": files[str(path)], "episodes": count}
    summary = {
        "files": files,
        "source_row_hashes": len(values["source_rows"]),
        "normalized_statements": len(values["statements"]),
        "masked_templates": len(values["templates"]),
        "labels_or_model_outputs_used": False,
    }
    return values, summary


def filter_rows(rows: list[dict], exclusions: dict[str, set[str]]) \
        -> tuple[list[dict], list[dict]]:
    kept, removed = [], []
    seen_hashes, seen_statements = set(), set()
    for row in rows:
        reason = None
        if row["source_row_sha256"] in exclusions["source_rows"]:
            reason = "development_source_row_overlap"
        elif normalize(row["statement"]) in exclusions["statements"]:
            reason = "development_statement_overlap"
        elif row["pattern_norm"] in exclusions["templates"]:
            reason = "development_masked_template_overlap"
        elif row["source_row_sha256"] in seen_hashes:
            reason = "duplicate_source_row"
        elif (row["relation_id"], normalize(row["statement"])) in seen_statements:
            reason = "duplicate_relation_statement"
        if reason:
            removed.append({
                "relation_id": row["relation_id"],
                "source_index": row["source_index"],
                "source_row_sha256": row["source_row_sha256"],
                "reason": reason,
            })
        else:
            kept.append(row)
            seen_hashes.add(row["source_row_sha256"])
            seen_statements.add((row["relation_id"], normalize(row["statement"])))
    return kept, removed


def capacity(rows: list[dict], relations: list[str]) -> dict[str, dict]:
    by_relation = defaultdict(list)
    for row in rows:
        by_relation[row["relation_id"]].append(row)
    return {
        relation: {
            "rows": len(by_relation[relation]),
            "subjects": len({row["subject_norm"] for row in by_relation[relation]}),
            "values": len({row["value_norm"] for row in by_relation[relation]}),
            "templates": len({row["pattern_norm"] for row in by_relation[relation]}),
            "eligible": (
                len(by_relation[relation]) >= MIN_ROWS
                and len({row["subject_norm"] for row in by_relation[relation]}) >= MIN_SUBJECTS
                and len({row["value_norm"] for row in by_relation[relation]}) >= MIN_VALUES
                and len({row["pattern_norm"] for row in by_relation[relation]}) >= MIN_TEMPLATES
            ),
        }
        for relation in relations
    }


class Allocator:
    def __init__(self, rows: list[dict], seed: int):
        self.seed = seed
        self.by_relation = defaultdict(list)
        for row in rows:
            self.by_relation[row["relation_id"]].append(row)
        self.used_rows = set()
        self.used_pairs = set()

    def take(self, relation: str, namespace: str, predicate=lambda row: True) -> dict:
        candidates = [
            row for row in self.by_relation[relation]
            if row["source_row_sha256"] not in self.used_rows
            and tuple(row["entity_pair"]) not in self.used_pairs
            and predicate(row)
        ]
        if not candidates:
            raise ValueError(f"capacity exhausted for {relation}:{namespace}")
        row = min(candidates, key=lambda value: (
            stable_rank(self.seed, namespace, value["source_row_sha256"]),
            value["source_row_sha256"],
        ))
        self.used_rows.add(row["source_row_sha256"])
        self.used_pairs.add(tuple(row["entity_pair"]))
        return row


def choose_materials(rows: list[dict], relations: list[str], seed: int = SEED,
                     mechanisms=MECHANISMS,
                     repetitions: int = GROUPS_PER_MECHANISM) -> list[dict]:
    allocator = Allocator(rows, seed)
    materials = []
    for relation_index, relation in enumerate(relations):
        for mechanism_index, mechanism in enumerate(mechanisms):
            for repetition in range(repetitions):
                group_index = len(materials)
                prefix = f"g{group_index}:{relation}:{mechanism}:{repetition}"
                target = allocator.take(relation, prefix + ":target")
                used_values = {target["value_norm"]}
                versions = []
                for index in range(4):
                    row = allocator.take(
                        relation, prefix + f":version:{index}",
                        lambda value, used=used_values: value["value_norm"] not in used,
                    )
                    versions.append(row); used_values.add(row["value_norm"])
                same_other = allocator.take(
                    relation, prefix + ":same-other",
                    lambda value: value["subject_norm"] != target["subject_norm"]
                    and value["value_norm"] not in used_values,
                )
                used_subjects = {target["subject_norm"], same_other["subject_norm"]}
                used_values.add(same_other["value_norm"])

                other_relations = [value for value in relations if value != relation]
                other_relations.sort(key=lambda value: (
                    stable_rank(seed, "stable-relation", relation_index,
                                mechanism_index, repetition, value), value,
                ))
                stable = []
                for role, other_relation in enumerate(other_relations):
                    try:
                        row = allocator.take(
                            other_relation, prefix + f":stable:{role}",
                            lambda value, subjects=used_subjects, values=used_values:
                            value["subject_norm"] not in subjects
                            and value["value_norm"] not in values,
                        )
                    except ValueError:
                        continue
                    stable.append(row)
                    used_subjects.add(row["subject_norm"])
                    used_values.add(row["value_norm"])
                    if len(stable) == 5:
                        break
                if len(stable) != 5:
                    raise ValueError(f"insufficient stable relations for {prefix}")
                anchor_alternate = allocator.take(
                    stable[0]["relation_id"], prefix + ":anchor-alternate",
                    lambda value: value["value_norm"] not in used_values,
                )
                used_values.add(anchor_alternate["value_norm"])

                involved = [relation, *(row["relation_id"] for row in stable)]
                patterns, pattern_sources = {}, []
                for involved_relation in involved:
                    local_patterns = set()
                    selected = []
                    for index in range(PATTERNS_PER_GROUP_RELATION):
                        donor = allocator.take(
                            involved_relation, prefix + f":pattern:{involved_relation}:{index}",
                            lambda value, local=local_patterns:
                            value["pattern_norm"] not in local,
                        )
                        local_patterns.add(donor["pattern_norm"])
                        selected.append(donor["pattern"])
                        pattern_sources.append(donor)
                    patterns[involved_relation] = selected

                factual = [target, *versions, same_other, *stable, anchor_alternate]
                sources = factual + pattern_sources
                materials.append({
                    "group_index": group_index,
                    "target_relation": relation,
                    "mechanism": mechanism,
                    "repetition": repetition,
                    "target": target,
                    "versions": versions,
                    "same_relation_other": same_other,
                    "stable": stable,
                    "anchor_alternate": anchor_alternate,
                    "patterns": patterns,
                    "material_source_row_sha256": [
                        row["source_row_sha256"] for row in sources
                    ],
                    "material_entity_pairs": [row["entity_pair"] for row in sources],
                })
    return materials


def naturalize_current_queries(episode: dict, patterns: dict[str, list[str]]) -> None:
    for boundary in episode["boundaries"]:
        incoming = episode["events"][boundary["event_index"]]["record"]
        subject, value = incoming["input"]["subject"], incoming["input"]["value"]
        exact = incoming["input"]["statement"].replace(value, "____", 1)
        relation = incoming["audit"]["relation_id"]
        alternatives = [render(pattern, subject, None) for pattern in patterns[relation]]
        rephrase = next((value for value in alternatives
                         if normalize(value) != normalize(exact)), None)
        if rephrase is None:
            raise ValueError("natural exact/rephrase query collision")
        for query, text in zip(boundary["queries"][:2], (exact, rephrase)):
            query["input"]["query"] = "Complete the effective session fact: " + text
            query["input"]["subject"] = subject
            query["answer"] = value
            query["candidates"] = list(dict.fromkeys(
                [value, *query["candidates"]]
            ))[:6]
            if len(query["candidates"]) != 6:
                raise ValueError("naturalized query lost six candidates")


def material_episode(material: dict, twin: int) -> dict:
    group_index = material["group_index"]
    payload = {key: material[key] for key in (
        "target", "versions", "same_relation_other", "stable",
        "anchor_alternate", "patterns",
    )}
    episode = build_episode(
        group_index, material["mechanism"],
        LENGTHS[group_index % len(LENGTHS)], twin, payload,
        group_prefix=GROUP_PREFIX,
    )
    naturalize_current_queries(episode, material["patterns"])
    episode["audit"].update({
        "source_partition": "fewrel_train_only",
        "target_source_relation": material["target_relation"],
        "mechanism_repetition": material["repetition"],
        "material_source_row_sha256": material["material_source_row_sha256"],
        "material_entity_pairs": material["material_entity_pairs"],
        "natural_template_entity_substitution": True,
    })
    return episode


def build_corpus(source_root: Path, development_paths: list[Path], seed: int = SEED):
    source_hashes = verify_source(source_root)
    train = json.loads((source_root / "data/train_wiki.json").read_text())
    external = json.loads((source_root / "data/val_wiki.json").read_text())
    partitions = relation_partitions(train, external, seed=2311)
    relations = sorted(partitions["train"])
    if len(relations) != EXPECTED_TRAIN_RELATIONS:
        raise ValueError("unexpected V10 FewRel train relation count")
    raw, source_removed = [], []
    raw_source_rows = 0
    for relation in relations:
        for index, source in enumerate(train[relation]):
            raw_source_rows += 1
            try:
                raw.append(row_from_source(relation, index, source))
            except ValueError:
                source_removed.append({
                    "relation_id": relation,
                    "source_index": index,
                    "source_row_sha256": canonical_sha(source),
                    "reason": "multiple_or_invalid_entity_mentions",
                })
    dev_keys, dev_summary = development_exclusions(development_paths)
    retained, removed = filter_rows(raw, dev_keys)
    removed = source_removed + removed
    capacities = capacity(retained, relations)
    if not all(value["eligible"] for value in capacities.values()):
        failed = [key for key, value in capacities.items() if not value["eligible"]]
        raise ValueError(f"V10 relation capacity shortfall: {failed}")
    materials = choose_materials(retained, relations, seed)
    formal = [material_episode(material, twin)
              for material in materials for twin in range(TWINS)]
    pilot_relation_set = set(relations[:PILOT_RELATIONS])
    pilot_group_ids = {
        f"{GROUP_PREFIX}-g{row['group_index']:03d}"
        for row in materials if row["target_relation"] in pilot_relation_set
    }
    pilot = [row for row in formal if row["source_group"] in pilot_group_ids]
    manifest = {
        "protocol": PROTOCOL, "plan": PLAN, "seed": seed,
        "source": {"repository_commit": SOURCE_COMMIT,
                   "files_sha256": source_hashes},
        "partitions": partitions,
        "development_exclusions": dev_summary,
        "eligibility_thresholds": {
            "rows": MIN_ROWS, "subjects": MIN_SUBJECTS,
            "values": MIN_VALUES, "templates": MIN_TEMPLATES,
        },
        "capacity": capacities,
        "pilot_relations": sorted(pilot_relation_set),
        "counts": {
            "raw_rows": raw_source_rows, "parsed_rows": len(raw),
            "retained_rows": len(retained),
            "exclusions": len(removed),
            "exclusions_by_reason": dict(sorted(Counter(
                row["reason"] for row in removed).items())),
            "eligible_relations": sum(value["eligible"] for value in capacities.values()),
            "source_groups": len(materials), "episodes": len(formal),
            "boundaries": sum(len(row["boundaries"]) for row in formal),
            "queries": sum(len(boundary["queries"]) for row in formal
                           for boundary in row["boundaries"]),
            "pilot_source_groups": len(pilot_group_ids),
            "pilot_episodes": len(pilot),
            "pilot_boundaries": sum(len(row["boundaries"]) for row in pilot),
            "mechanisms": dict(sorted(Counter(
                row["mechanism"] for row in materials).items())),
            "relations": dict(sorted(Counter(
                row["target_relation"] for row in materials).items())),
            "material_source_rows": len({value for row in materials
                                         for value in row["material_source_row_sha256"]}),
            "material_entity_pairs": len({tuple(value) for row in materials
                                           for value in row["material_entity_pairs"]}),
        },
        "alpha_grid": [0.0, 0.5, 1.0],
        "candidate_memory_cap": 3,
        "natural_template_entity_substitution": True,
        "model_outputs_loaded": False,
        "development_labels_loaded": False,
        "development_model_scores_loaded": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "v10_fewrel_corpus_built_pending_independent_audit",
    }
    return formal, pilot, materials, removed, manifest


def validate(formal: list[dict], pilot: list[dict], materials: list[dict],
             manifest: dict) -> None:
    expected_groups = EXPECTED_TRAIN_RELATIONS * GROUPS_PER_RELATION
    assert len(materials) == expected_groups
    assert len(formal) == expected_groups * TWINS
    assert len(pilot) == PILOT_RELATIONS * GROUPS_PER_RELATION * TWINS
    assert all(len(row["boundaries"]) == BOUNDARIES for row in formal)
    assert all(len(boundary["queries"]) == 5 for row in formal
               for boundary in row["boundaries"])
    assert all(boundary["queries"][0]["input"]["query"]
               != boundary["queries"][1]["input"]["query"]
               for row in formal for boundary in row["boundaries"])
    sources = [value for row in materials
               for value in row["material_source_row_sha256"]]
    pairs = [tuple(value) for row in materials for value in row["material_entity_pairs"]]
    assert len(sources) == len(set(sources))
    assert len(pairs) == len(set(pairs))
    assert manifest["counts"]["source_groups"] == expected_groups
    assert not manifest["model_outputs_loaded"]
    assert not manifest["sealed_benchmarks_loaded"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--development-episodes", type=Path, action="append",
                        required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    formal, pilot, materials, exclusions, manifest = build_corpus(
        args.source_root, args.development_episodes, args.seed,
    )
    validate(formal, pilot, materials, manifest)
    args.output_root.mkdir(parents=True)
    files = {
        "formal_episodes.jsonl": formal,
        "pilot_episodes.jsonl": pilot,
        "materials.jsonl": materials,
        "exclusions.jsonl": exclusions,
    }
    for name, rows in files.items():
        write_jsonl(args.output_root / name, rows)
    manifest["files_sha256"] = {
        name: sha256(args.output_root / name) for name in files
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
