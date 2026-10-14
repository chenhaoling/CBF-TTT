"""Build the preregistered V19 locality-constrained confirmation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tasks.build_controlled_interference_v18 import (
    FIXED_POLICIES, LENGTHS, MECHANISMS, QUERY_KINDS, build_states, opaque_value,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v19_confirmation"
RELEASE_PROTOCOL = "v18_locality_constrained_oracle_posthoc_v1"
SEED = 1901
ROWS = 320
SUBJECTS = 80
TEMPLATE_GROUPS = 20
FOLDS = 5
VALUE_OFFSET = 8192
PRIMARY_LOCALITY_BUDGET = .03
MAXIMUM_LOCALITY_FAILURES = 9

SECTIONS = ("docket", "index", "register", "table", "directory")
LABELS = ("marker", "attribute", "coordinate", "channel")


def read_json(path):
    return json.loads(Path(path).read_text())


def opaque(namespace, index, width=10):
    digest = hashlib.sha256(f"v19:{namespace}:{index}".encode()).hexdigest().upper()
    return f"Y{digest[:width]}"


def verify_release(release_root, v18_data_root, v18_pilot_data_root):
    release_root = Path(release_root)
    v18_data_root, v18_pilot_data_root = map(Path, (v18_data_root, v18_pilot_data_root))
    summary_path, audit_path = release_root / "summary.json", release_root / "audit.json"
    summary, audit = read_json(summary_path), read_json(audit_path)
    v18_manifest = read_json(v18_data_root / "manifest.json")
    pilot_manifest = read_json(v18_pilot_data_root / "manifest.json")
    checks = (
        summary.get("protocol") == RELEASE_PROTOCOL,
        summary.get("posthoc") is True,
        summary.get("diagnostic_gate", {}).get("passed") is True,
        summary.get("primary_budget") == PRIMARY_LOCALITY_BUDGET,
        summary.get("stage_b_released") is False,
        summary.get("original_v18_status_unchanged") == "formal_no_sufficient_headroom",
        audit.get("protocol") == f"{RELEASE_PROTOCOL}_audit",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("stage_b_released") is False,
        audit.get("posthoc") is True,
        (release_root / "runner_exit_code.txt").read_text().strip() == "0",
        v18_manifest.get("protocol") == "controlled_memory_interference_v18_formal",
        pilot_manifest.get("protocol") == "controlled_memory_interference_v18_pilot",
        v18_manifest.get("formal_sha256") == sha(v18_data_root / "formal.jsonl"),
        pilot_manifest.get("pilot_sha256") == sha(v18_pilot_data_root / "pilot.jsonl"),
    )
    if not all(checks):
        raise ValueError("V19 construction requires the audited passing V18-LC diagnostic")
    return {
        "release_summary_sha256": sha(summary_path),
        "release_audit_sha256": sha(audit_path),
        "v18_manifest_sha256": sha(v18_data_root / "manifest.json"),
        "v18_formal_sha256": sha(v18_data_root / "formal.jsonl"),
        "v18_pilot_manifest_sha256": sha(v18_pilot_data_root / "manifest.json"),
        "v18_pilot_sha256": sha(v18_pilot_data_root / "pilot.jsonl"),
    }


def fact_statement(entity, key, value, template_group, variant):
    section = SECTIONS[template_group % len(SECTIONS)]
    label = LABELS[template_group // len(SECTIONS)]
    if variant == 0:
        return f"Within record {entity}, the {section} records {label} {key} as {value}."
    if variant == 1:
        return f"For item {entity}, {label} {key} appears as {value} in the {section}."
    raise ValueError(variant)


def query_specs(entity, keys, initial, current, locality_entity, locality_value,
                template_group):
    section = SECTIONS[template_group % len(SECTIONS)]
    label = LABELS[template_group // len(SECTIONS)]
    target_candidates = [initial[0], current[0]] + [current[index] for index in (3, 4, 5)]
    all_candidates = list(dict.fromkeys(current))
    locality_candidates = [locality_value] + current[:5]
    return [
        {"kind": "update_old_surface",
         "statement": f"Retrieve the effective token bound to {entity}, {label} {keys[0]}.",
         "answer": current[0], "candidates": target_candidates},
        {"kind": "update_current_surface",
         "statement": (f"After all revisions, which token does the {section} give for "
                       f"{entity} at {label} {keys[0]}?"),
         "answer": current[0], "candidates": target_candidates},
        {"kind": "retain_1",
         "statement": f"Look up {label} {keys[1]} for {entity}; return its token.",
         "answer": current[1], "candidates": all_candidates},
        {"kind": "retain_2",
         "statement": f"Which token remains under {entity}'s {label} {keys[2]}?",
         "answer": current[2], "candidates": all_candidates},
        {"kind": "locality",
         "statement": (f"An independent directory maps {locality_entity} to {locality_value}. "
                       f"Return the token mapped to {locality_entity}."),
         "answer": locality_value, "candidates": locality_candidates},
    ]


def build_episode(index, seed=SEED):
    if not 0 <= index < ROWS:
        raise ValueError(index)
    subject_index, within_subject = divmod(index, len(MECHANISMS))
    mechanism = MECHANISMS[within_subject]
    length = LENGTHS[(subject_index + within_subject) % len(LENGTHS)]
    template_group_index = subject_index % TEMPLATE_GROUPS
    fold = template_group_index % FOLDS
    template_group = f"v19_templates_{template_group_index:02d}"
    rng = random.Random(seed * 100_000 + index)
    entity = f"ENT-C-{opaque('entity', subject_index, 8)}"
    locality_entity = f"REF-C-{opaque('locality', index, 8)}"
    keys = [f"Q{opaque(f'key-{index}', key_index, 7)}" for key_index in range(6)]
    if mechanism == "irrelevant_near_duplicate":
        keys[3] = keys[0] + "B"
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
        raise RuntimeError("V19 session count mismatch")

    sessions = []
    for session_index, (key_index, value, role) in enumerate(facts):
        variant = rng.randrange(2)
        sessions.append({
            "session_id": f"v19-{index:03d}-s{session_index:03d}",
            "session_index": session_index,
            "key_id": keys[key_index],
            "key_index": key_index,
            "value": value,
            "role": role,
            "statement": fact_statement(
                entity, keys[key_index], value, template_group_index, variant),
            "template_id": f"v19_fact_{template_group_index:02d}_{variant}",
        })
    episode_id = hashlib.sha256(
        f"v19:{seed}:{index}:{subject_index}:{length}:{mechanism}".encode()
    ).hexdigest()[:20]
    return {
        "episode_id": episode_id,
        "split": "confirmation",
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
        "queries": query_specs(entity, keys, initial, current, locality_entity,
                                 locality_value, template_group_index),
    }


def build(output, release_root, v18_data_root, v18_pilot_data_root, seed=SEED):
    output = Path(output)
    dependencies = verify_release(release_root, v18_data_root, v18_pilot_data_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = [build_episode(index, seed) for index in range(ROWS)]
    states = [build_states(episode) for episode in episodes]
    data_path, state_path = output / "confirmation.jsonl", output / "states.jsonl"
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
        "split": "confirmation",
        "lengths": list(LENGTHS),
        "mechanisms": list(MECHANISMS),
        "fixed_policies": list(FIXED_POLICIES),
        "query_kinds": list(QUERY_KINDS),
        "counts": counts,
        "confirmation_sha256": sha(data_path),
        "states_sha256": sha(state_path),
        "dependencies": dependencies,
        "primary_locality_budget": PRIMARY_LOCALITY_BUDGET,
        "maximum_locality_failures": MAXIMUM_LOCALITY_FAILURES,
        "registered_success_gate": {
            "subject_and_template_macro_joint_gain_at_least": .10,
            "cluster_bootstrap_lower_95_above": 0,
            "update_accuracy_at_least": .90,
            "retained_key_accuracy_at_least": .90,
            "locality_accuracy_at_least": .97,
            "no_outer_fold_negative_joint_gain": True,
        },
        "bootstrap_draws": 20_000,
        "split_design": {
            "episodes_per_subject": 4,
            "subjects_per_template_group": 4,
            "template_groups_per_fold": 4,
            "subject_and_template_group_each_nested_in_one_fold": True,
            "v18_entity_value_and_template_overlap": False,
        },
        "opaque_value_namespace": "V19-only offset 8192; disjoint from all V18 values",
        "opaque_value_format": "Z plus four spaced base-16 letters; exactly five Qwen3 tokens",
        "current_session_deletion_allowed": False,
        "model_scored": False,
        "external_sets_loaded": False,
        "terminal_status": "frozen_v19_confirmation_before_model_scoring",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--v18-data-root", required=True)
    parser.add_argument("--v18-pilot-data-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
