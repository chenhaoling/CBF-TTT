"""Build the frozen 320-row V18 controlled-interference formal corpus.

The formal split is released only by an independently audited, admitted V18
Qwen pilot.  Its entities, value range, and surface templates are disjoint
from the pilot.  Subjects and template groups are nested in five outer folds
so every formal comparison can select a fixed baseline without seeing the
held-out subjects or templates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tasks.build_controlled_interference_v18 import (
    FIXED_POLICIES, LENGTHS, MECHANISMS, QUERY_KINDS, build_states, opaque,
    opaque_value,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_formal"
PILOT_PROTOCOL = "controlled_memory_interference_v18_qwen_pilot"
SEED = 1811
ROWS = 320
SUBJECTS = 80
TEMPLATE_GROUPS = 20
FOLDS = 5
VALUE_OFFSET = 4096

DOMAINS = ("registry", "ledger", "catalog", "inventory", "reference")
RELATIONS = ("field", "slot", "entry", "channel")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_pilot_release(pilot_root, pilot_data_root):
    pilot_root, pilot_data_root = Path(pilot_root), Path(pilot_data_root)
    summary_path, audit_path = pilot_root / "summary.json", pilot_root / "audit.json"
    summary, audit = read_json(summary_path), read_json(audit_path)
    pilot_manifest_path = pilot_data_root / "manifest.json"
    pilot_data_audit_path = pilot_data_root.parent / "audit.json"
    if not pilot_data_audit_path.exists():
        pilot_data_audit_path = pilot_data_root / "audit.json"
    pilot_manifest, pilot_data_audit = (read_json(pilot_manifest_path),
                                        read_json(pilot_data_audit_path))
    checks = (
        summary.get("protocol") == PILOT_PROTOCOL,
        summary.get("corpus_admission_gate", {}).get("passed") is True,
        summary.get("terminal_status") == "pilot_corpus_admitted",
        summary.get("formal_generated") is False,
        summary.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("recomputed_corpus_admission_gate", {}).get("passed") is True,
        audit.get("formal_generated") is False,
        audit.get("external_sets_loaded") is False,
        summary.get("dependencies", {}).get("manifest_sha256") == sha(pilot_manifest_path),
        summary.get("dependencies", {}).get("data_audit_sha256")
            == sha(pilot_data_audit_path),
        pilot_manifest.get("protocol") == "controlled_memory_interference_v18_pilot",
        pilot_data_audit.get("audit_passed") is True,
        pilot_data_audit.get("manifest_sha256") == sha(pilot_manifest_path),
        (pilot_root / "runner_exit_code.txt").read_text().strip() == "0",
        (pilot_root / "terminal_status.txt").read_text().strip()
            == "completed_controlled_interference_v18_qwen_pilot",
    )
    if not all(checks):
        raise ValueError("formal V18 construction requires an audited admitted pilot")
    return {
        "pilot_summary_sha256": sha(summary_path),
        "pilot_audit_sha256": sha(audit_path),
        "pilot_raw_rows_sha256": summary["raw_rows_sha256"],
        "pilot_data_manifest_sha256": sha(pilot_manifest_path),
        "pilot_data_audit_sha256": sha(pilot_data_audit_path),
        "pilot_data_sha256": sha(pilot_data_root / "pilot.jsonl"),
        "pilot_states_sha256": sha(pilot_data_root / "pilot_states.jsonl"),
    }


def formal_fact_statement(entity, key, value, template_group, variant):
    domain = DOMAINS[template_group % len(DOMAINS)]
    relation = RELATIONS[template_group // len(DOMAINS)]
    if variant == 0:
        return f"The {domain} lists {value} for {entity} at {relation} {key}."
    if variant == 1:
        return f"Under {entity}, {relation} {key} in the {domain} carries {value}."
    raise ValueError(variant)


def formal_query_specs(entity, keys, initial, current, locality_entity,
                       locality_value, template_group):
    domain = DOMAINS[template_group % len(DOMAINS)]
    relation = RELATIONS[template_group // len(DOMAINS)]
    target_candidates = [initial[0], current[0]] + [current[index] for index in (3, 4, 5)]
    all_candidates = list(dict.fromkeys(current))
    locality_candidates = [locality_value] + current[:5]
    return [
        {"kind": "update_old_surface",
         "statement": f"Which code is associated with {entity}'s {relation} {keys[0]}?",
         "answer": current[0], "candidates": target_candidates},
        {"kind": "update_current_surface",
         "statement": f"Read the current {domain} value at {relation} {keys[0]} for {entity}.",
         "answer": current[0], "candidates": target_candidates},
        {"kind": "retain_1",
         "statement": f"Return the value kept at {entity}'s {relation} {keys[1]}.",
         "answer": current[1], "candidates": all_candidates},
        {"kind": "retain_2",
         "statement": f"In the {domain}, what occupies {relation} {keys[2]} for {entity}?",
         "answer": current[2], "candidates": all_candidates},
        {"kind": "locality",
         "statement": (f"A public reference says {locality_entity} maps to {locality_value}. "
                       f"What code maps to {locality_entity}?"),
         "answer": locality_value, "candidates": locality_candidates},
    ]


def build_formal_episode(index, seed=SEED):
    if not 0 <= index < ROWS:
        raise ValueError(index)
    subject_index, within_subject = divmod(index, len(MECHANISMS))
    mechanism = MECHANISMS[within_subject]
    length = LENGTHS[(subject_index + within_subject) % len(LENGTHS)]
    template_group_index = subject_index % TEMPLATE_GROUPS
    fold = template_group_index % FOLDS
    template_group = f"formal_templates_{template_group_index:02d}"
    rng = random.Random(seed * 100_000 + index)
    entity = f"ENT-F-{opaque('formal-entity', subject_index, 8)}"
    locality_entity = f"REF-F-{opaque('formal-locality-entity', index, 8)}"
    keys = [f"K{opaque(f'formal-key-{index}', key_index, 7)}" for key_index in range(6)]
    if mechanism == "irrelevant_near_duplicate":
        keys[3] = keys[0] + "A"
    initial = [opaque_value(VALUE_OFFSET + index * 8 + key_index)
               for key_index in range(6)]
    current = list(initial)
    current[0] = opaque_value(VALUE_OFFSET + index * 8 + 6)
    locality_value = opaque_value(VALUE_OFFSET + index * 8 + 7)

    facts = [(key_index, initial[key_index], "initial") for key_index in range(6)]
    for filler in range(length - 7):
        if mechanism == "stale_repetition" and filler % 2 == 0:
            key_index, role = 0, "stale_repeat"
        elif mechanism == "irrelevant_near_duplicate" and filler % 2 == 0:
            key_index, role = 3, "near_duplicate"
        elif mechanism == "useful_repetition" and filler % 2 == 0:
            key_index, role = 1, "useful_repeat"
        else:
            key_index, role = 3 + filler % 3, "stable_filler"
        facts.append((key_index, initial[key_index], role))
    facts.append((0, current[0], "current_update"))
    if len(facts) != length:
        raise RuntimeError("formal V18 session count mismatch")

    sessions = []
    for session_index, (key_index, value, role) in enumerate(facts):
        variant = rng.randrange(2)
        sessions.append({
            "session_id": f"v18f-{index:03d}-s{session_index:03d}",
            "session_index": session_index,
            "key_id": keys[key_index],
            "key_index": key_index,
            "value": value,
            "role": role,
            "statement": formal_fact_statement(
                entity, keys[key_index], value, template_group_index, variant),
            "template_id": f"formal_fact_{template_group_index:02d}_{variant}",
        })
    episode_id = hashlib.sha256(
        f"v18-formal:{seed}:{index}:{subject_index}:{length}:{mechanism}".encode()
    ).hexdigest()[:20]
    return {
        "episode_id": episode_id,
        "split": "formal",
        "row_index": index,
        "subject": hashlib.sha256(entity.encode()).hexdigest(),
        "subject_index": subject_index,
        "entity": entity,
        "template_group": template_group,
        "template_group_index": template_group_index,
        "fold": fold,
        "length": length,
        "mechanism": mechanism,
        "query_step": length - 1,
        "keys": keys,
        "initial_values": initial,
        "current_values": current,
        "target_key_id": keys[0],
        "primary_old_value": initial[0],
        "primary_current_value": current[0],
        "current_session_id": sessions[-1]["session_id"],
        "locality_entity": locality_entity,
        "locality_value": locality_value,
        "sessions": sessions,
        "queries": formal_query_specs(
            entity, keys, initial, current, locality_entity, locality_value,
            template_group_index),
    }


def build(output, pilot_root, pilot_data_root, seed=SEED):
    output = Path(output)
    dependencies = verify_pilot_release(pilot_root, pilot_data_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = [build_formal_episode(index, seed) for index in range(ROWS)]
    states = [build_states(episode) for episode in episodes]
    data_path, state_path = output / "formal.jsonl", output / "formal_states.jsonl"
    with data_path.open("x") as sink:
        sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes))
    with state_path.open("x") as sink:
        sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in states))
    counts = {
        "rows": len(episodes),
        "subjects": len({row["subject"] for row in episodes}),
        "template_groups": len({row["template_group"] for row in episodes}),
        "folds": len({row["fold"] for row in episodes}),
        "by_length": {str(length): sum(row["length"] == length for row in episodes)
                      for length in LENGTHS},
        "by_mechanism": {name: sum(row["mechanism"] == name for row in episodes)
                         for name in MECHANISMS},
        "by_fold": {str(fold): sum(row["fold"] == fold for row in episodes)
                    for fold in range(FOLDS)},
        "single_deletion_states": sum(len(row["delete_one"]) for row in states),
        "key_deletion_states": sum(len(row["delete_key"]) for row in states),
        "fixed_arm_entries": len(states) * len(FIXED_POLICIES),
    }
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "split": "formal",
        "lengths": list(LENGTHS),
        "mechanisms": list(MECHANISMS),
        "fixed_policies": list(FIXED_POLICIES),
        "query_kinds": list(QUERY_KINDS),
        "counts": counts,
        "formal_sha256": sha(data_path),
        "formal_states_sha256": sha(state_path),
        "dependencies": dependencies,
        "split_design": {
            "episodes_per_subject": 4,
            "subjects_per_template_group": 4,
            "template_groups_per_fold": 4,
            "subject_and_template_group_each_nested_in_one_fold": True,
            "pilot_entity_value_and_template_overlap": False,
        },
        "opaque_value_namespace": "formal-only offset 4096; disjoint from pilot values",
        "opaque_value_format": "Z plus four spaced base-16 letters; exactly five Qwen3 tokens",
        "current_session_deletion_allowed": False,
        "model_scored": False,
        "formal_generated": True,
        "external_sets_loaded": False,
        "terminal_status": "frozen_v18_formal_before_model_scoring",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--pilot-data-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
