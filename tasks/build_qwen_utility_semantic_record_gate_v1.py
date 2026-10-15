"""Build the model-free LAMA/ParaRel utility-label resource pilot corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile

from scripts.inspect_lama_pararel_utility_source import normalize, sha256


PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data"
INSPECTION_PROTOCOL = "qwen_utility_semantic_record_source_inspection_v1"
SEED = 20261016
GROUPS_PER_MECHANISM = 8
TWINS = 2
BOUNDARIES = 3
LENGTHS = (8, 16, 32, 64)
MECHANISMS = (
    "semantic_overwrite",
    "semantic_duplicate",
    "same_subject_different_relation",
    "different_subject_same_relation",
    "stale_repetition",
    "explicit_reset_revoke",
    "mixed_lifecycle",
)


def stable_rank(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True) + "\n")


def load_patterns(source_root: Path, relations: set[str]) -> dict[str, list[str]]:
    root = source_root / "sources" / "pararel" / "data" / "pattern_data" / "graphs_json"
    result = {}
    for relation in sorted(relations):
        values = set()
        for line in (root / f"{relation}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            pattern = json.loads(line).get("pattern", "")
            compact = re.sub(r"\s+", " ", pattern.strip())
            if compact.count("[X]") == 1 and compact.count("[Y]") == 1:
                values.add(compact)
        result[relation] = sorted(values)
    return result


def load_facts(source_root: Path, relations: set[str]) -> dict[str, list[dict]]:
    archive = source_root / "downloads" / "lama_data.zip"
    result = defaultdict(dict)
    with zipfile.ZipFile(archive) as bundle:
        for relation in sorted(relations):
            with bundle.open(f"data/TREx/{relation}.jsonl") as source:
                for raw in source:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    subject, value = row.get("sub_label"), row.get("obj_label")
                    if not isinstance(subject, str) or not isinstance(value, str):
                        continue
                    subject_norm, value_norm = normalize(subject), normalize(value)
                    if not subject_norm or not value_norm or subject_norm == value_norm:
                        continue
                    key = (subject_norm, value_norm)
                    candidate = {
                        "relation_id": relation,
                        "subject": re.sub(r"\s+", " ", subject.strip()),
                        "subject_norm": subject_norm,
                        "value": re.sub(r"\s+", " ", value.strip()),
                        "value_norm": value_norm,
                        "source_uuid": row.get("uuid", ""),
                        "source_fact_hash": stable_rank(
                            "fact", relation, subject_norm, value_norm
                        ),
                    }
                    previous = result[relation].get(key)
                    if previous is None or candidate["source_uuid"] < previous["source_uuid"]:
                        result[relation][key] = candidate
    return {
        relation: sorted(values.values(), key=lambda row: (
            stable_rank("fact-order", relation, row["source_fact_hash"]),
            row["source_fact_hash"],
        ))
        for relation, values in result.items()
    }


def cross_partition_subjects(facts: dict[str, list[dict]], splits: dict[str, list[str]]) -> set[str]:
    relation_to_split = {
        relation: split for split, relations in splits.items() for relation in relations
    }
    owners = defaultdict(set)
    for relation, rows in facts.items():
        for row in rows:
            owners[row["subject_norm"]].add(relation_to_split[relation])
    return {subject for subject, values in owners.items() if len(values) > 1}


def render(pattern: str, subject: str, value: str | None = None) -> str:
    text = pattern.replace("[X]", subject)
    text = text.replace("[Y]", value if value is not None else "____")
    return re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"\s+", " ", text)).strip()


def semantic_key(subject_norm: str, relation: str) -> str:
    return stable_rank("semantic-key", subject_norm, relation)[:24]


def make_record(
    group_id: str,
    index: int,
    relation: str,
    subject: str,
    subject_norm: str,
    value: str,
    value_norm: str,
    pattern: str,
    role: str,
) -> dict:
    return {
        "record_id": f"{group_id}-r{index:03d}",
        "input": {
            "statement": render(pattern, subject, value),
            "subject": subject,
            "value": value,
        },
        "audit": {
            "relation_id": relation,
            "subject_norm": subject_norm,
            "value_norm": value_norm,
            "semantic_key": semantic_key(subject_norm, relation),
            "pattern_hash": stable_rank("pattern", relation, pattern),
            "role": role,
        },
    }


def choose_distinct_values(rows: list[dict], count: int, namespace: str, excluded=()) -> list[dict]:
    excluded_values = set(excluded)
    ordered = sorted(rows, key=lambda row: (
        stable_rank(namespace, row["source_fact_hash"]), row["source_fact_hash"]
    ))
    result, values = [], set(excluded_values)
    for row in ordered:
        if row["value_norm"] in values:
            continue
        result.append(row)
        values.add(row["value_norm"])
        if len(result) == count:
            return result
    raise ValueError(f"{namespace} lacks {count} distinct values")


def choose_group_material(
    group_index: int,
    train_relations: list[str],
    facts: dict[str, list[dict]],
    patterns: dict[str, list[str]],
) -> dict:
    target_relation = train_relations[group_index % len(train_relations)]
    target_rows = facts[target_relation]
    target = min(target_rows, key=lambda row: (
        stable_rank("target", group_index, row["source_fact_hash"]),
        row["source_fact_hash"],
    ))
    versions = choose_distinct_values(
        target_rows, 4, f"versions:{group_index}", excluded=(target["value_norm"],)
    )
    same_relation_other = min(
        (row for row in target_rows if row["subject_norm"] != target["subject_norm"]
         and row["value_norm"] not in {item["value_norm"] for item in versions}),
        key=lambda row: (stable_rank("same-relation", group_index, row["source_fact_hash"]),
                         row["source_fact_hash"]),
    )
    other_relations = sorted(
        (relation for relation in train_relations if relation != target_relation),
        key=lambda relation: (stable_rank("other-relation", group_index, relation), relation),
    )[:5]
    stable = []
    used_subjects = {target["subject_norm"], same_relation_other["subject_norm"]}
    used_values = {target["value_norm"], same_relation_other["value_norm"],
                   *(item["value_norm"] for item in versions)}
    for relation in other_relations:
        candidates = [row for row in facts[relation]
                      if row["subject_norm"] not in used_subjects
                      and row["value_norm"] not in used_values]
        chosen = min(candidates, key=lambda row: (
            stable_rank("stable", group_index, relation, row["source_fact_hash"]),
            row["source_fact_hash"],
        ))
        stable.append(chosen)
        used_subjects.add(chosen["subject_norm"]); used_values.add(chosen["value_norm"])
    anchor_alternate = choose_distinct_values(
        facts[stable[0]["relation_id"]], 1, f"anchor-twin:{group_index}",
        excluded=(stable[0]["value_norm"], *used_values),
    )[0]
    return {
        "target": target,
        "versions": versions,
        "same_relation_other": same_relation_other,
        "stable": stable,
        "anchor_alternate": anchor_alternate,
        "patterns": {relation: patterns[relation] for relation in [target_relation, *other_relations]},
    }


def make_queries(
    state: dict[str, dict],
    history: dict[str, dict],
    current: dict,
    candidates: list[str],
    patterns,
    target_key: str,
    protected_keys: dict[str, str],
) -> list[dict]:
    current_audit = current["audit"]
    current_relation = current_audit["relation_id"]
    current_key = current_audit["semantic_key"]
    current_patterns = patterns[current_relation]

    def query_for(key: str, pattern_index: int, role: str) -> dict:
        active_record = state.get(key)
        record = active_record if active_record is not None else history[key]
        answer = record["input"]["value"] if active_record is not None else "unknown"
        relation = record["audit"]["relation_id"]
        relation_patterns = patterns[relation]
        question = "Complete the effective session fact: " + render(
            relation_patterns[pattern_index % len(relation_patterns)],
            record["input"]["subject"], None,
        )
        row_candidates = list(dict.fromkeys([answer, *candidates]))[:6]
        if len(row_candidates) < 6:
            raise ValueError("query has fewer than six candidates")
        return {
            "role": role,
            "input": {"query": question, "subject": record["input"]["subject"]},
            "audit": {"semantic_key": key},
            "answer": answer,
            "candidates": row_candidates,
        }

    if current_key != target_key:
        retained_keys = [target_key, protected_keys["retained_1"]]
    else:
        retained_keys = [protected_keys["retained_1"], protected_keys["retained_2"]]
    return [
        query_for(current_key, 0, "current_exact"),
        query_for(current_key, 1, "current_rephrase"),
        query_for(retained_keys[0], 0, "retained_1"),
        query_for(retained_keys[1], 1, "retained_2"),
        query_for(protected_keys["anchor"], 2, "anchor"),
    ]


def replay(
    events: list[dict],
    boundary_indices: set[int],
    candidate_pool: list[str],
    patterns,
    target_key: str,
    protected_keys: dict[str, str],
):
    state, history, boundaries = {}, {}, []
    for index, event in enumerate(events):
        if event["reset"]:
            state.clear()
        for key in event["revoke_keys"]:
            state.pop(key, None)
        record = event["record"]
        state[record["audit"]["semantic_key"]] = record
        history[record["audit"]["semantic_key"]] = record
        if index in boundary_indices:
            boundaries.append({
                "boundary_index": len(boundaries),
                "event_index": index,
                "incoming_record_id": record["record_id"],
                "reset": event["reset"],
                "revoke_keys": event["revoke_keys"],
                "active_state": {
                    key: value["record_id"] for key, value in sorted(state.items())
                },
                "queries": make_queries(
                    state, history, record, candidate_pool, patterns,
                    target_key, protected_keys,
                ),
            })
    return boundaries


def build_episode(
    group_index: int,
    mechanism: str,
    length: int,
    twin: int,
    material: dict,
    group_prefix: str = "usr-v1",
) -> dict:
    group_id = f"{group_prefix}-g{group_index:03d}-t{twin}"
    target, versions = material["target"], material["versions"]
    same_other, stable = material["same_relation_other"], material["stable"]
    patterns = material["patterns"]
    target_relation = target["relation_id"]
    records, events = [], []

    def record(row, relation=None, subject=None, subject_norm=None, value=None,
               value_norm=None, pattern_index=0, role="filler"):
        relation = relation or row["relation_id"]
        subject = subject or row["subject"]
        subject_norm = subject_norm or row["subject_norm"]
        value = value or row["value"]
        value_norm = value_norm or row["value_norm"]
        item = make_record(
            group_id, len(records), relation, subject, subject_norm, value, value_norm,
            patterns[relation][pattern_index % len(patterns[relation])], role,
        )
        records.append(item)
        return item

    def append(item, reset=False, revoke=()):
        events.append({"event_index": len(events), "record": item, "reset": reset,
                       "revoke_keys": list(revoke)})

    anchor = stable[0] if twin == 0 else {
        **stable[0],
        "value": material["anchor_alternate"]["value"],
        "value_norm": material["anchor_alternate"]["value_norm"],
    }
    initial = record(target, pattern_index=0, role="target_initial")
    append(initial)
    anchor_record = record(anchor, pattern_index=0, role="protected_anchor")
    retained_1_record = record(stable[1], pattern_index=0, role="retained_1")
    retained_2_record = record(stable[2], pattern_index=0, role="retained_2")
    append(anchor_record)
    append(retained_1_record)
    append(retained_2_record)
    append(record(same_other, pattern_index=0, role="same_relation_other_subject"))

    target_key = initial["audit"]["semantic_key"]
    alternate_relation = stable[3]["relation_id"]
    alternate_key = semantic_key(target["subject_norm"], alternate_relation)
    same_other_key = semantic_key(same_other["subject_norm"], target_relation)

    def target_version(version_index, role):
        row = versions[version_index]
        return record(row, relation=target_relation, subject=target["subject"],
                      subject_norm=target["subject_norm"], pattern_index=version_index + 1,
                      role=role)

    if mechanism == "semantic_overwrite":
        boundary_records = [target_version(i, "semantic_overwrite") for i in range(3)]
        boundary_ops = [(False, ()), (False, ()), (False, ())]
    elif mechanism == "semantic_duplicate":
        boundary_records = [
            record(target, pattern_index=i + 1, role="semantic_duplicate") for i in range(3)
        ]
        boundary_ops = [(False, ()), (False, ()), (False, ())]
    elif mechanism == "same_subject_different_relation":
        row = stable[3]
        boundary_records = [
            record(row, subject=target["subject"], subject_norm=target["subject_norm"],
                   pattern_index=i, role="same_subject_different_relation")
            for i in range(3)
        ]
        boundary_ops = [(False, ()), (False, ()), (False, ())]
    elif mechanism == "different_subject_same_relation":
        boundary_records = [
            record(same_other, pattern_index=i + 1, role="different_subject_same_relation")
            for i in range(3)
        ]
        boundary_ops = [(False, ()), (False, ()), (False, ())]
    elif mechanism == "stale_repetition":
        boundary_records = [
            record(target, pattern_index=1, role="stale_repeat"),
            target_version(0, "semantic_overwrite"),
            target_version(1, "semantic_overwrite"),
        ]
        boundary_ops = [(False, ()), (False, ()), (False, ())]
    elif mechanism == "explicit_reset_revoke":
        boundary_records = [target_version(i, "lifecycle_update") for i in range(3)]
        boundary_ops = [(False, (target_key,)), (True, ()), (False, (target_key,))]
    elif mechanism == "mixed_lifecycle":
        boundary_records = [
            target_version(0, "semantic_overwrite"),
            target_version(0, "semantic_duplicate"),
            target_version(1, "post_reset_update"),
        ]
        boundary_ops = [(False, ()), (False, ()), (True, ())]
    else:
        raise ValueError(f"unknown mechanism {mechanism}")

    boundary_positions = [5, max(6, length // 2), length - 1]
    if len(set(boundary_positions)) != BOUNDARIES:
        raise ValueError("boundary positions are not unique")
    filler_cursor = 0
    for position in range(5, length):
        if position in boundary_positions:
            boundary = boundary_positions.index(position)
            reset, revoke = boundary_ops[boundary]
            append(boundary_records[boundary], reset=reset, revoke=revoke)
        else:
            row = stable[1 + filler_cursor % (len(stable) - 1)]
            append(record(row, pattern_index=filler_cursor + 1, role="stable_filler"))
            filler_cursor += 1
    if len(events) != length:
        raise AssertionError("episode length mismatch")

    all_values = ["unknown", target["value"], *(row["value"] for row in versions),
                  *(row["value"] for row in stable), same_other["value"]]
    candidate_pool = []
    for value in sorted(set(all_values), key=lambda value: (stable_rank(group_id, value), value)):
        candidate_pool.append(value)
    protected_keys = {
        "anchor": anchor_record["audit"]["semantic_key"],
        "retained_1": retained_1_record["audit"]["semantic_key"],
        "retained_2": retained_2_record["audit"]["semantic_key"],
    }
    boundaries = replay(
        events, set(boundary_positions), candidate_pool, patterns,
        target_key, protected_keys,
    )
    if len(boundaries) != BOUNDARIES:
        raise AssertionError("boundary count mismatch")
    return {
        "episode_id": group_id,
        "source_group": f"{group_prefix}-g{group_index:03d}",
        "twin": twin,
        "mechanism": mechanism,
        "length": length,
        "events": events,
        "boundaries": boundaries,
        "audit": {
            "target_relation": target_relation,
                "target_key": target_key,
                "protected_keys": protected_keys,
            "alternate_key": alternate_key,
            "same_relation_other_key": same_other_key,
            "relation_ids": sorted(patterns),
        },
    }


def build(source_root: Path, inspection_path: Path) -> tuple[list[dict], dict]:
    inspection = read_json(inspection_path)
    if inspection.get("protocol") != INSPECTION_PROTOCOL or not inspection.get("passed"):
        raise ValueError("source inspection did not pass")
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
    episodes = []
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        for within in range(GROUPS_PER_MECHANISM):
            group_index = mechanism_index * GROUPS_PER_MECHANISM + within
            material = choose_group_material(group_index, train_relations, facts, patterns)
            length = LENGTHS[(within + mechanism_index) % len(LENGTHS)]
            for twin in range(TWINS):
                episodes.append(build_episode(group_index, mechanism, length, twin, material))

    manifest = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_SUPERVISED_SEMANTIC_RECORD_GATE_V1_PLAN.md",
        "source_manifest_sha256": inspection["source_manifest_sha256"],
        "source_inspection_sha256": sha256(inspection_path),
        "source_commits": inspection["source_commits"],
        "seed": SEED,
        "counts": {
            "source_groups": len({row["source_group"] for row in episodes}),
            "episodes": len(episodes),
            "boundaries": sum(len(row["boundaries"]) for row in episodes),
            "queries": sum(len(boundary["queries"]) for row in episodes
                           for boundary in row["boundaries"]),
            "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
            "lengths": dict(Counter(str(row["length"]) for row in episodes)),
            "relations": {key: len(value) for key, value in splits.items()},
            "cross_partition_subjects_removed": len(removed_subjects),
        },
        "splits": splits,
        "alpha_grid": [0.0, 0.5, 1.0],
        "current_write_gate": 1.0,
        "model_outputs_loaded": False,
        "development_episodes_generated": False,
        "confirmation_episodes_generated": False,
        "forbidden_sources_loaded": False,
        "terminal_status": "pilot_data_built_pending_independent_audit",
    }
    return episodes, manifest


def validate(episodes: list[dict], manifest: dict) -> None:
    expected_groups = len(MECHANISMS) * GROUPS_PER_MECHANISM
    assert manifest["counts"]["source_groups"] == expected_groups
    assert len(episodes) == expected_groups * TWINS
    assert manifest["counts"]["boundaries"] == len(episodes) * BOUNDARIES
    assert all(len(row["events"]) == row["length"] for row in episodes)
    assert all(len(row["boundaries"]) == BOUNDARIES for row in episodes)
    assert all(len(boundary["queries"]) == 5 for row in episodes
               for boundary in row["boundaries"])
    assert all(query["answer"] in query["candidates"] and len(query["candidates"]) == 6
               for row in episodes for boundary in row["boundaries"]
               for query in boundary["queries"])
    groups = defaultdict(list)
    for row in episodes:
        groups[row["source_group"]].append(row)
    assert all({item["twin"] for item in rows} == {0, 1} for rows in groups.values())
    assert not manifest["model_outputs_loaded"]
    assert not manifest["development_episodes_generated"]
    assert not manifest["confirmation_episodes_generated"]


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
