"""Build the frozen post-V19 natural semantic-conflict headroom corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "post_v19_natural_semantic_headroom_v1"
INVENTORY_PROTOCOL = "post_v20_semantic_controller_source_inventory_v1"
INVENTORY_MANIFEST_SHA256 = (
    "df3002f4913663ef829b865b3d1502305b712f5a96c401568c413a9fa2862836"
)
SEMANTIC_ROWS_SHA256 = (
    "222f7c51d384a9e26da93ecfd0dda91dd0add148c9901e5765f2fe4458614a99"
)
SEED = 2101
FOLDS = 5
PILOT_PER_FOLD = 16
FORMAL_PER_FOLD = 64
TOTAL_PER_FOLD = PILOT_PER_FOLD + FORMAL_PER_FOLD
LENGTHS = (8, 16, 32, 64)
MECHANISMS = (
    "single_conflict",
    "stale_repetition",
    "useful_repetition",
    "relation_distractor",
)
FIXED_POLICIES = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
QUERY_KINDS = (
    "update_old_surface",
    "update_current_surface",
    "retain_1",
    "retain_2",
    "locality",
)


def load_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def normalized_surface(value):
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(" .?!:;")


def rank(seed, namespace, *values):
    text = ":".join((str(seed), namespace, *(str(value) for value in values)))
    return hashlib.sha256(text.encode()).hexdigest()


def eligible(row):
    surfaces, values = row.get("surfaces", []), row.get("values", [])
    return (
        row.get("source") in {"counterfact", "mquake"}
        and row.get("outer_fold") in range(FOLDS)
        and bool(row.get("subject_normalized"))
        and bool(row.get("relation_id"))
        and len(surfaces) >= 2
        and len(values) >= 2
        and all(isinstance(value, str) and value.strip() and "\n" not in value
                and len(value) <= 96 for value in values[:2])
        and values[0].strip() != values[1].strip()
        and all(isinstance(value, str) and value.strip() and "\n" not in value
                and len(value) <= 640 for value in surfaces[:2])
        and normalized_surface(surfaces[0]) != normalized_surface(surfaces[1])
    )


def allocate_fold(rows, fold, seed=SEED):
    candidates = [row for row in rows if row["outer_fold"] == fold and eligible(row)]
    relation_counts, source_counts = Counter(), Counter()
    selected, used_subjects = [], set()
    while len(selected) < TOTAL_PER_FOLD:
        available = [row for row in candidates
                     if row["subject_normalized"] not in used_subjects]
        if not available:
            raise ValueError(f"fold {fold} lacks {TOTAL_PER_FOLD} eligible subjects")
        choice = min(
            available,
            key=lambda row: (
                relation_counts[row["relation_id"]],
                source_counts[row["source"]],
                rank(seed, "target", row["source_row_sha256"]),
            ),
        )
        selected.append(choice)
        used_subjects.add(choice["subject_normalized"])
        relation_counts[choice["relation_id"]] += 1
        source_counts[choice["source"]] += 1
    pilot = [row for index, row in enumerate(selected) if index % 5 == 0]
    formal = [row for index, row in enumerate(selected) if index % 5 != 0]
    if len(pilot) != PILOT_PER_FOLD or len(formal) != FORMAL_PER_FOLD:
        raise RuntimeError("deterministic pilot/formal interleave is malformed")
    return pilot, formal


def allocate(rows, seed=SEED):
    splits = {"pilot": [], "formal": []}
    for fold in range(FOLDS):
        pilot, formal = allocate_fold(rows, fold, seed)
        splits["pilot"].extend(pilot)
        splits["formal"].extend(formal)
    return splits


def donor_rows(target, pool, seed=SEED):
    candidates = [row for row in pool
                  if row["outer_fold"] == target["outer_fold"]
                  and row["subject_normalized"] != target["subject_normalized"]
                  and eligible(row)]
    same = sorted((row for row in candidates
                   if row["relation_id"] == target["relation_id"]),
                  key=lambda row: rank(seed, "same-donor", target["source_row_sha256"],
                                       row["source_row_sha256"]))
    different = sorted((row for row in candidates
                        if row["relation_id"] != target["relation_id"]),
                       key=lambda row: rank(seed, "different-donor",
                                            target["source_row_sha256"],
                                            row["source_row_sha256"]))
    if not same:
        raise ValueError(f"no same-relation other-subject donor for {target['row_id']}")
    ordered = same[:1] + different
    result, values, subjects = [], set(target["values"][:2]), set()
    for row in ordered:
        value = row["values"][0].strip()
        if value in values or row["subject_normalized"] in subjects:
            continue
        result.append(row)
        values.add(value)
        subjects.add(row["subject_normalized"])
        if len(result) == 6:
            break
    if len(result) < 6:
        raise ValueError(f"insufficient distinct donors for {target['row_id']}")
    return result


def render_fact(surface, value):
    surface = surface.strip()
    if surface.endswith("?"):
        return f'The recorded answer to "{surface}" is {value}.'
    punctuation = "" if surface.endswith((" ", "'", '"')) else " "
    return f"{surface}{punctuation}{value}."


def render_query(surface):
    surface = surface.strip()
    if surface.endswith("?"):
        return surface
    return f"Complete the effective updated fact: {surface}"


def make_session(split, episode_index, session_index, row, surface_index, value, role):
    surfaces = row["surfaces"]
    surface = surfaces[surface_index % len(surfaces)]
    return {
        "session_id": f"p19n-{split}-{episode_index:03d}-s{session_index:03d}",
        "session_index": session_index,
        "source_row_sha256": row["source_row_sha256"],
        "source": row["source"],
        "subject_normalized": row["subject_normalized"],
        "relation_id": row["relation_id"],
        "surface": surface,
        "key_id": normalized_surface(surface),
        "value": value,
        "role": role,
        "statement": render_fact(surface, value),
    }


def query_specs(target, donors, locality_entity, locality_value):
    current, old = target["values"][1].strip(), target["values"][0].strip()
    donor_values = [row["values"][0].strip() for row in donors]
    candidates = list(dict.fromkeys([current, old, *donor_values]))[:6]
    if len(candidates) < 5:
        raise ValueError("natural episode has fewer than five distinct candidates")
    retain_candidates = list(dict.fromkeys(donor_values + [old, current]))[:6]
    locality_candidates = [locality_value, current, old, *donor_values[:3]]
    return [
        {"kind": "update_old_surface", "statement": render_query(target["surfaces"][0]),
         "answer": current, "candidates": candidates},
        {"kind": "update_current_surface", "statement": render_query(target["surfaces"][1]),
         "answer": current, "candidates": candidates},
        {"kind": "retain_1", "statement": render_query(donors[1]["surfaces"][0]),
         "answer": donor_values[1], "candidates": retain_candidates},
        {"kind": "retain_2", "statement": render_query(donors[2]["surfaces"][0]),
         "answer": donor_values[2], "candidates": retain_candidates},
        {"kind": "locality",
         "statement": (f"The independent checksum for {locality_entity} is "
                       f"{locality_value}. What is its checksum?"),
         "answer": locality_value,
         "candidates": list(dict.fromkeys(locality_candidates))},
    ]


def build_episode(target, pool, split, episode_index, seed=SEED):
    donors = donor_rows(target, pool, seed)
    fold = target["outer_fold"]
    mechanism = MECHANISMS[(episode_index + fold) % len(MECHANISMS)]
    length = LENGTHS[(episode_index * 3 + fold) % len(LENGTHS)]
    sessions = []

    def append(row, surface_index, value, role):
        sessions.append(make_session(split, episode_index, len(sessions), row,
                                     surface_index, value, role))

    append(target, 0, target["values"][0].strip(), "stale_target")
    append(donors[1], 0, donors[1]["values"][0].strip(), "retain_1")
    append(donors[2], 0, donors[2]["values"][0].strip(), "retain_2")
    append(donors[0], 0, donors[0]["values"][0].strip(), "same_relation_other_subject")
    donor_cursor = 3
    while len(sessions) < length - 1:
        step = len(sessions)
        if mechanism == "stale_repetition" and step % 2 == 0:
            append(target, 0, target["values"][0].strip(), "stale_repeat")
        elif mechanism == "useful_repetition" and step % 2 == 0:
            append(donors[1], step, donors[1]["values"][0].strip(), "useful_repeat")
        elif mechanism == "relation_distractor" and step % 2 == 0:
            append(donors[0], step, donors[0]["values"][0].strip(),
                   "same_relation_other_subject")
        else:
            donor = donors[donor_cursor % len(donors)]
            append(donor, step, donor["values"][0].strip(), "stable_filler")
            donor_cursor += 1
    append(target, 1, target["values"][1].strip(), "current_update")
    current_id = sessions[-1]["session_id"]
    locality_entity = "REF-" + rank(seed, "locality", target["source_row_sha256"])[:10]
    locality_value = "CHECK-" + rank(seed, "locality-value",
                                      target["source_row_sha256"])[:8].upper()
    episode_id = rank(seed, "episode", split, target["source_row_sha256"])[:20]
    return {
        "episode_id": episode_id,
        "split": split,
        "row_index": episode_index,
        "source": target["source"],
        "source_split": target["source_split"],
        "source_row_sha256": target["source_row_sha256"],
        "subject": hashlib.sha256(target["subject_normalized"].encode()).hexdigest(),
        "subject_normalized": target["subject_normalized"],
        "relation_id": target["relation_id"],
        "fold": fold,
        "length": length,
        "mechanism": mechanism,
        "old_value": target["values"][0].strip(),
        "current_value": target["values"][1].strip(),
        "old_surface": target["surfaces"][0],
        "current_surface": target["surfaces"][1],
        "current_session_id": current_id,
        "locality_entity": locality_entity,
        "locality_value": locality_value,
        "donor_source_row_sha256": [row["source_row_sha256"] for row in donors],
        "sessions": sessions,
        "queries": query_specs(target, donors, locality_entity, locality_value),
    }


def build_states(episode):
    identifiers = [row["session_id"] for row in episode["sessions"]]
    current = episode["current_session_id"]
    current_session = episode["sessions"][-1]
    exact_removed = [row["session_id"] for row in episode["sessions"][:-1]
                     if row["subject_normalized"] == episode["subject_normalized"]
                     and row["key_id"] == current_session["key_id"]
                     and row["value"] != episode["current_value"]]
    semantic_removed = [row["session_id"] for row in episode["sessions"][:-1]
                        if row["subject_normalized"] == episode["subject_normalized"]
                        and row["relation_id"] == episode["relation_id"]
                        and row["value"] != episode["current_value"]]
    if exact_removed or not semantic_removed or current in semantic_removed:
        raise ValueError("natural exact/semantic intervention invariant failed")
    fixed = {
        "no_memory": [],
        "retain_all": identifiers,
        "global_reset": [current],
        "last_1": [current],
        "last_2": identifiers[-2:],
    }
    return {
        "episode_id": episode["episode_id"],
        "row_index": episode["row_index"],
        "fixed": fixed,
        "special": {
            "exact_conflict": [value for value in identifiers
                               if value not in set(exact_removed)],
            "semantic_oracle": [value for value in identifiers
                                if value not in set(semantic_removed)],
        },
        "exact_removed": exact_removed,
        "semantic_removed": semantic_removed,
    }


def verify_inventory(inventory_root):
    root = Path(inventory_root)
    manifest_path, rows_path = root / "manifest.json", root / "semantic_rows.jsonl"
    audit_path = root.parent / "audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    checks = (
        manifest.get("protocol") == INVENTORY_PROTOCOL,
        sha(manifest_path) == INVENTORY_MANIFEST_SHA256,
        sha(rows_path) == SEMANTIC_ROWS_SHA256,
        manifest.get("files_sha256", {}).get("semantic_rows.jsonl")
            == SEMANTIC_ROWS_SHA256,
        manifest.get("checks", {}).get("relation_disjoint") is True,
        manifest.get("checks", {}).get("subject_disjoint") is True,
        manifest.get("model_outcomes_loaded") is False,
        manifest.get("locomo_loaded") is False,
        manifest.get("longmemeval_loaded") is False,
        audit.get("passed") is True,
        audit.get("manifest_sha256") == INVENTORY_MANIFEST_SHA256,
    )
    if not all(checks):
        raise ValueError("natural headroom construction requires the audited inventory")
    return rows_path, {
        "inventory_manifest_sha256": sha(manifest_path),
        "inventory_audit_sha256": sha(audit_path),
        "semantic_rows_sha256": sha(rows_path),
    }


def split_counts(episodes, states):
    return {
        "rows": len(episodes),
        "by_fold": {str(fold): sum(row["fold"] == fold for row in episodes)
                    for fold in range(FOLDS)},
        "by_source": dict(sorted(Counter(row["source"] for row in episodes).items())),
        "by_relation": dict(sorted(Counter(row["relation_id"] for row in episodes).items())),
        "by_length": {str(length): sum(row["length"] == length for row in episodes)
                      for length in LENGTHS},
        "by_mechanism": {name: sum(row["mechanism"] == name for row in episodes)
                         for name in MECHANISMS},
        "semantic_deletions": sum(len(row["semantic_removed"]) for row in states),
        "exact_deletions": sum(len(row["exact_removed"]) for row in states),
    }


def build(output, inventory_root, seed=SEED):
    output = Path(output)
    rows_path, dependencies = verify_inventory(inventory_root)
    rows = load_jsonl(rows_path)
    selected = allocate(rows, seed)
    output.mkdir(parents=True, exist_ok=False)
    file_meta = {}
    split_manifests = {}
    for split in ("pilot", "formal"):
        pool = selected[split]
        episodes = [build_episode(row, pool, split, index, seed)
                    for index, row in enumerate(pool)]
        states = [build_states(row) for row in episodes]
        episode_path, state_path = output / f"{split}.jsonl", output / f"{split}_states.jsonl"
        episode_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                        for row in episodes))
        state_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                      for row in states))
        file_meta[episode_path.name], file_meta[state_path.name] = sha(episode_path), sha(state_path)
        split_manifests[split] = split_counts(episodes, states)
    pilot_hashes = {row["source_row_sha256"] for row in selected["pilot"]}
    formal_hashes = {row["source_row_sha256"] for row in selected["formal"]}
    pilot_subjects = {row["subject_normalized"] for row in selected["pilot"]}
    formal_subjects = {row["subject_normalized"] for row in selected["formal"]}
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "folds": FOLDS,
        "lengths": list(LENGTHS),
        "mechanisms": list(MECHANISMS),
        "fixed_policies": list(FIXED_POLICIES),
        "query_kinds": list(QUERY_KINDS),
        "splits": split_manifests,
        "files_sha256": file_meta,
        "dependencies": dependencies,
        "checks": {
            "pilot_formal_source_rows_disjoint": not (pilot_hashes & formal_hashes),
            "pilot_formal_subjects_disjoint": not (pilot_subjects & formal_subjects),
            "current_update_always_retained": True,
            "exact_rule_relation_id_free": True,
            "semantic_oracle_query_and_answer_free": True,
        },
        "pilot_gate": {
            "subject_macro_joint_gain_over_exact_at_least": 0.05,
            "update_accuracy_at_least": 0.85,
            "retained_fact_accuracy_at_least": 0.90,
            "locality_accuracy_at_least": 0.97,
            "no_negative_fold": True,
        },
        "formal_gate": {
            "subject_macro_joint_gain_over_exact_at_least": 0.10,
            "subject_and_relation_cluster_ci_lower_above": 0.0,
            "update_and_retained_fact_accuracy_at_least": 0.90,
            "locality_and_unrelated_retention_at_least": 0.97,
            "no_negative_fold_or_source": True,
        },
        "model_scored": False,
        "controller_trained": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "frozen_post_v19_natural_data_before_model_scoring",
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError("natural pilot/formal separation failed")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
