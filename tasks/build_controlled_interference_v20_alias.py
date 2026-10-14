"""Build the gated V20 three-surface semantic-alias confirmation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tasks.build_controlled_interference_v18 import (
    FIXED_POLICIES, LENGTHS, MECHANISMS, QUERY_KINDS, opaque_value,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v20_semantic_alias"
RELEASE_PROTOCOL = "v19_causal_rule_explanation_v1"
ROWS = 320
SUBJECTS = 80
TEMPLATE_GROUPS = 20
FOLDS = 5
SEED = 2003
VALUE_OFFSET = 16_000

ALIAS_BANK = (
    ("primary contact", "main correspondent", "designated liaison"),
    ("delivery site", "shipping destination", "parcel endpoint"),
    ("invoice currency", "billing denomination", "payment money unit"),
    ("sign-in secret", "access credential", "authentication token"),
    ("telephone contact", "calling number", "phone endpoint"),
    ("support package", "assistance tier", "service plan"),
    ("display language", "interface locale", "preferred tongue"),
    ("clock region", "local time zone", "regional time setting"),
    ("fiscal category", "taxation class", "tax group"),
    ("alert route", "notification channel", "message pathway"),
    ("storage section", "warehouse zone", "inventory area"),
    ("merchandise category", "product class", "item family"),
    ("authorization status", "approval state", "clearance condition"),
    ("subscription renewal", "extension date", "rollover day"),
    ("client representative", "account owner", "relationship manager"),
    ("exposure rating", "risk grade", "hazard score"),
    ("release revision", "document version", "edition number"),
    ("traffic region", "routing zone", "network area"),
    ("preservation duration", "retention period", "storage lifetime"),
    ("security cipher", "encryption method", "encoding scheme"),
)
DOMAINS = ("profile", "ledger", "directory", "register", "account")


def read_json(path):
    return json.loads(Path(path).read_text())


def opaque(namespace, index, width=10):
    digest = hashlib.sha256(f"v20:{namespace}:{index}".encode()).hexdigest().upper()
    return f"X{digest[:width]}"


def verify_release(rule_root):
    rule_root = Path(rule_root)
    summary_path, audit_path = rule_root / "summary.json", rule_root / "audit.json"
    summary, audit = read_json(summary_path), read_json(audit_path)
    checks = (
        summary.get("protocol") == RELEASE_PROTOCOL,
        summary.get("explicit_key_explanation_gate", {}).get("passed") is True,
        summary.get("next_stage") == "build_v20_semantic_alias_confirmation",
        summary.get("query_or_answer_features_used") is False,
        summary.get("model_rescored") is False,
        summary.get("terminal_status") == "completed_v19_causal_rule_diagnostic",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("recomputed_explanation_gate")
            == summary.get("explicit_key_explanation_gate"),
        (rule_root / "runner_exit_code.txt").read_text().strip() == "0",
        (rule_root / "terminal_status.txt").read_text().strip()
            == "completed_v19_causal_rule_diagnostic",
    )
    if not all(checks):
        raise ValueError("V20 construction requires the audited passing V19 rule explanation")
    return {"rule_summary_sha256": sha(summary_path),
            "rule_audit_sha256": sha(audit_path),
            "v19_raw_rows_sha256": summary["raw_rows_sha256"],
            "v19_outcome_summary_sha256":
                summary["dependencies"]["outcome_summary_sha256"]}


def fact_statement(entity, key, value, template_group, variant):
    domain = DOMAINS[template_group % len(DOMAINS)]
    if variant == 0:
        return f"In {entity}'s {domain}, {key} is stored as {value}."
    if variant == 1:
        return f"The {domain} entry for {entity} assigns {value} to {key}."
    raise ValueError(variant)


def query_specs(entity, source_keys, query_key, initial, current,
                locality_entity, locality_value, template_group):
    domain = DOMAINS[template_group % len(DOMAINS)]
    target_candidates = [initial[0], current[0]] + [current[index] for index in (3, 4, 5)]
    all_candidates = list(dict.fromkeys(current))
    locality_candidates = [locality_value] + current[:5]
    return [
        {"kind": "update_old_surface",
         "statement": f"After all revisions, what is {entity}'s {source_keys[0]}?",
         "query_key_id": source_keys[0], "answer": current[0],
         "candidates": target_candidates},
        {"kind": "update_current_surface",
         "statement": f"Read the current {query_key} in the {domain} for {entity}.",
         "query_key_id": query_key, "answer": current[0],
         "candidates": target_candidates},
        {"kind": "retain_1", "statement": f"Return {entity}'s {source_keys[1]}.",
         "query_key_id": source_keys[1], "answer": current[1],
         "candidates": all_candidates},
        {"kind": "retain_2", "statement": f"Which value is {entity}'s {source_keys[2]}?",
         "query_key_id": source_keys[2], "answer": current[2],
         "candidates": all_candidates},
        {"kind": "locality",
         "statement": (f"An independent reference maps {locality_entity} to {locality_value}. "
                       f"What maps to {locality_entity}?"),
         "query_key_id": None, "answer": locality_value,
         "candidates": locality_candidates},
    ]


def build_episode(index, seed=SEED):
    if not 0 <= index < ROWS:
        raise ValueError(index)
    subject_index, within = divmod(index, 4)
    mechanism = MECHANISMS[within]
    length = LENGTHS[(subject_index + within) % 4]
    template_index = subject_index % TEMPLATE_GROUPS
    fold = template_index % FOLDS
    rng = random.Random(seed * 100_000 + index)
    entity = f"ENT-S-{opaque('entity', subject_index, 8)}"
    locality_entity = f"REF-S-{opaque('locality', index, 8)}"
    concepts = [(subject_index * 11 + index + offset * 7) % len(ALIAS_BANK)
                for offset in range(6)]
    used = set()
    for offset, value in enumerate(concepts):
        while value in used:
            value = (value + 1) % len(ALIAS_BANK)
        concepts[offset] = value; used.add(value)
    source_keys = [ALIAS_BANK[value][0] for value in concepts]
    update_key, query_key = ALIAS_BANK[concepts[0]][1:]
    if mechanism == "irrelevant_near_duplicate":
        source_keys[3] = source_keys[0] + " checksum"
    initial = [opaque_value(VALUE_OFFSET + index * 8 + key) for key in range(6)]
    current = list(initial); current[0] = opaque_value(VALUE_OFFSET + index * 8 + 6)
    locality_value = opaque_value(VALUE_OFFSET + index * 8 + 7)
    facts = [(key, initial[key], "initial") for key in range(6)]
    for filler in range(length - 7):
        if mechanism == "stale_repetition" and filler % 2 == 0:
            key, role = 0, "stale_repeat"
        elif mechanism == "irrelevant_near_duplicate" and filler % 2 == 0:
            key, role = 3, "near_duplicate"
        elif mechanism == "useful_repetition" and filler % 2 == 0:
            key, role = 1, "useful_repeat"
        else:
            key, role = 3 + filler % 3, "stable_filler"
        facts.append((key, initial[key], role))
    sessions = []
    for session_index, (key_index, value, role) in enumerate(facts):
        key = source_keys[key_index]; variant = rng.randrange(2)
        sessions.append({"session_id": f"v20-{index:03d}-s{session_index:03d}",
                         "session_index": session_index, "key_id": key,
                         "semantic_key_index": key_index, "value": value, "role": role,
                         "statement": fact_statement(
                             entity, key, value, template_index, variant),
                         "template_id": f"v20_fact_{template_index:02d}_{variant}"})
    current_id = f"v20-{index:03d}-s{length - 1:03d}"
    sessions.append({"session_id": current_id, "session_index": length - 1,
                     "key_id": update_key, "semantic_key_index": 0,
                     "value": current[0], "role": "current_update",
                     "statement": fact_statement(
                         entity, update_key, current[0], template_index, rng.randrange(2)),
                     "template_id": f"v20_fact_{template_index:02d}_update"})
    if len(sessions) != length:
        raise RuntimeError("V20 semantic-alias session count mismatch")
    episode_id = hashlib.sha256(
        f"v20:{seed}:{index}:{subject_index}:{length}:{mechanism}".encode()
    ).hexdigest()[:20]
    return {"episode_id": episode_id, "split": "semantic_alias", "row_index": index,
            "subject": hashlib.sha256(entity.encode()).hexdigest(),
            "subject_index": subject_index, "entity": entity,
            "template_group": f"v20_templates_{template_index:02d}",
            "template_group_index": template_index, "fold": fold, "length": length,
            "mechanism": mechanism, "query_step": length - 1,
            "semantic_concept_indices": concepts, "source_keys": source_keys,
            "update_key_id": update_key, "query_key_id": query_key,
            "initial_values": initial, "current_values": current,
            "primary_old_value": initial[0], "primary_current_value": current[0],
            "current_session_id": current_id, "locality_entity": locality_entity,
            "locality_value": locality_value, "sessions": sessions,
            "queries": query_specs(entity, source_keys, query_key, initial, current,
                                   locality_entity, locality_value, template_index)}


def build_states(episode):
    identifiers = [row["session_id"] for row in episode["sessions"]]
    current = episode["current_session_id"]
    exact_removed = [row["session_id"] for row in episode["sessions"][:-1]
                     if row["key_id"] == episode["update_key_id"]
                     and row["value"] != episode["primary_current_value"]]
    semantic_removed = [row["session_id"] for row in episode["sessions"][:-1]
                        if row["semantic_key_index"] == 0
                        and row["value"] != episode["primary_current_value"]]
    fixed = {"no_memory": [], "retain_all": identifiers,
             "global_reset": [current], "last_1": [current], "last_2": identifiers[-2:]}
    return {"episode_id": episode["episode_id"], "row_index": episode["row_index"],
            "fixed": fixed,
            "special": {
                "exact_conflict": [value for value in identifiers
                                   if value not in set(exact_removed)],
                "semantic_oracle": [value for value in identifiers
                                    if value not in set(semantic_removed)]},
            "exact_removed": exact_removed, "semantic_removed": semantic_removed}


def build(output, rule_root, seed=SEED):
    output = Path(output); dependencies = verify_release(rule_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = [build_episode(index, seed) for index in range(ROWS)]
    states = [build_states(row) for row in episodes]
    data_path, state_path = output / "alias.jsonl", output / "states.jsonl"
    data_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes))
    state_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states))
    counts = {"rows": ROWS, "subjects": SUBJECTS, "template_groups": TEMPLATE_GROUPS,
              "by_length": {str(value): sum(row["length"] == value for row in episodes)
                            for value in LENGTHS},
              "by_mechanism": {value: sum(row["mechanism"] == value for row in episodes)
                               for value in MECHANISMS},
              "by_fold": {str(value): sum(row["fold"] == value for row in episodes)
                          for value in range(FOLDS)},
              "semantic_oracle_deletions": sum(len(row["semantic_removed"]) for row in states),
              "exact_rule_deletions": sum(len(row["exact_removed"]) for row in states),
              "fixed_arm_entries": ROWS * len(FIXED_POLICIES),
              "special_arm_entries": ROWS * 2}
    manifest = {"protocol": PROTOCOL, "seed": seed, "split": "semantic_alias",
                "counts": counts, "lengths": list(LENGTHS),
                "mechanisms": list(MECHANISMS),
                "fixed_policies": list(FIXED_POLICIES),
                "query_kinds": list(QUERY_KINDS),
                "alias_bank": [list(row) for row in ALIAS_BANK],
                "alias_sha256": sha(data_path), "states_sha256": sha(state_path),
                "dependencies": dependencies, "value_offset": VALUE_OFFSET,
                "registered_gate": {"joint_gain_at_least": .10,
                                    "update_accuracy_at_least": .90,
                                    "retained_key_accuracy_at_least": .90,
                                    "locality_accuracy_at_least": .97},
                "model_scored": False, "controller_or_threshold_selected": False,
                "external_sets_loaded": False,
                "terminal_status": "frozen_v20_alias_before_model_scoring"}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
