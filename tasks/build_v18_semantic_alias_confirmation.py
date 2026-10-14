"""Build the gated, frozen V18 semantic-alias confirmation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tasks.build_controlled_interference_v18 import LENGTHS, MECHANISMS, opaque, opaque_value
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_semantic_alias_confirmation_v1"
ROWS = 320
SUBJECTS = 80
TEMPLATE_GROUPS = 20
FOLDS = 5
SEED = 1993
VALUE_OFFSET = 12_000
QUERY_KINDS = ("update_old_surface", "update_current_surface", "retain_1",
               "retain_2", "locality")

ALIAS_BANK = (
    ("customer identifier", "client ID", "account identity"),
    ("shipping destination", "delivery address", "parcel location"),
    ("billing currency", "payment denomination", "invoice money unit"),
    ("access credential", "login token", "authentication code"),
    ("contact telephone", "phone number", "calling number"),
    ("support tier", "service level", "assistance plan"),
    ("language preference", "preferred language", "interface locale"),
    ("time zone", "regional clock", "local time region"),
    ("tax category", "fiscal class", "taxation group"),
    ("notification channel", "alert method", "message route"),
    ("warehouse section", "storage zone", "inventory area"),
    ("product category", "item class", "merchandise group"),
    ("approval status", "authorization state", "clearance condition"),
    ("renewal date", "extension day", "subscription rollover date"),
    ("account manager", "client representative", "relationship owner"),
    ("risk rating", "exposure grade", "hazard score"),
    ("document version", "revision number", "release edition"),
    ("routing region", "network zone", "traffic area"),
    ("retention period", "storage duration", "preservation term"),
    ("encryption scheme", "cipher method", "security encoding"),
)
DOMAINS = ("profile", "record", "directory", "register", "account")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_controlled_release(root):
    root = Path(root); summary_path = root / "summary.json"
    summary, audit = read_json(summary_path), read_json(root / "audit.json")
    checks = (
        summary.get("protocol") == "v18_causal_deletion_policy_qwen_eval_v1",
        summary.get("controlled_policy_supported") is True,
        summary.get("controlled_policy_gate", {}).get("passed") is True,
        summary.get("terminal_status") == "controlled_policy_supported",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("recomputed_controlled_policy_gate", {}).get("passed") is True,
        (root / "runner_exit_code.txt").read_text().strip() == "0",
        (root / "terminal_status.txt").read_text().strip()
            == "completed_v18_causal_deletion_policy_eval",
    )
    if not all(checks):
        raise ValueError("semantic-alias construction requires positive controlled policy audit")
    return {"controlled_summary_sha256": sha(summary_path),
            "controlled_audit_sha256": sha(root / "audit.json"),
            "controlled_raw_sha256": summary["raw_rows_sha256"],
            "policy_checkpoint_sha256": summary["dependencies"]["policy_checkpoint_sha256"]}


def fact_statement(entity, key, value, template_group, variant):
    domain = DOMAINS[template_group % len(DOMAINS)]
    if variant == 0:
        return f"In the {domain} for {entity}, {key} is recorded as {value}."
    if variant == 1:
        return f"The {entity} {domain} assigns {value} to {key}."
    raise ValueError(variant)


def query_specs(entity, source_keys, update_key, query_key, initial, current,
                locality_entity, locality_value, template_group):
    domain = DOMAINS[template_group % len(DOMAINS)]
    target_candidates = [initial[0], current[0]] + [current[index] for index in (3, 4, 5)]
    all_candidates = list(dict.fromkeys(current)); locality_candidates = [locality_value] + current[:5]
    return [
        {"kind": "update_old_surface",
         "statement": f"What is {entity}'s {source_keys[0]}?", "query_key_id": source_keys[0],
         "answer": current[0], "candidates": target_candidates},
        {"kind": "update_current_surface",
         "statement": f"Read the current {query_key} in the {domain} for {entity}.",
         "query_key_id": query_key, "answer": current[0], "candidates": target_candidates},
        {"kind": "retain_1", "statement": f"Return {entity}'s {source_keys[1]}.",
         "query_key_id": source_keys[1], "answer": current[1], "candidates": all_candidates},
        {"kind": "retain_2", "statement": f"Which value is {entity}'s {source_keys[2]}?",
         "query_key_id": source_keys[2], "answer": current[2], "candidates": all_candidates},
        {"kind": "locality",
         "statement": (f"A public reference maps {locality_entity} to {locality_value}. "
                       f"What maps to {locality_entity}?"),
         "query_key_id": None, "answer": locality_value, "candidates": locality_candidates},
    ]


def build_episode(index, seed=SEED):
    if not 0 <= index < ROWS:
        raise ValueError(index)
    subject_index, within = divmod(index, 4); mechanism = MECHANISMS[within]
    length = LENGTHS[(subject_index + within) % 4]
    template_index = subject_index % TEMPLATE_GROUPS; fold = template_index % FOLDS
    rng = random.Random(seed * 100_000 + index)
    entity = f"ENT-A-{opaque('alias-entity', subject_index, 8)}"
    locality_entity = f"REF-A-{opaque('alias-locality', index, 8)}"
    concepts = [(subject_index * 7 + index + offset * 3) % len(ALIAS_BANK)
                for offset in range(6)]
    # Resolve rare modular collisions deterministically without changing the bank.
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
        sessions.append({"session_id": f"v18a-{index:03d}-s{session_index:03d}",
                         "session_index": session_index, "key_id": key,
                         "semantic_key_index": key_index, "value": value, "role": role,
                         "statement": fact_statement(
                             entity, key, value, template_index, variant),
                         "template_id": f"alias_fact_{template_index:02d}_{variant}"})
    current_id = f"v18a-{index:03d}-s{length - 1:03d}"
    sessions.append({"session_id": current_id, "session_index": length - 1,
                     "key_id": update_key, "semantic_key_index": 0,
                     "value": current[0], "role": "current_update",
                     "statement": fact_statement(
                         entity, update_key, current[0], template_index, rng.randrange(2)),
                     "template_id": f"alias_fact_{template_index:02d}_update"})
    if len(sessions) != length:
        raise RuntimeError("semantic-alias session count mismatch")
    episode_id = hashlib.sha256(
        f"v18-alias:{seed}:{index}:{subject_index}:{length}:{mechanism}".encode()
    ).hexdigest()[:20]
    return {"episode_id": episode_id, "split": "semantic_alias", "row_index": index,
            "subject": hashlib.sha256(entity.encode()).hexdigest(),
            "subject_index": subject_index, "entity": entity,
            "template_group": f"alias_templates_{template_index:02d}",
            "template_group_index": template_index, "fold": fold, "length": length,
            "mechanism": mechanism, "query_step": length - 1,
            "semantic_concept_indices": concepts, "source_keys": source_keys,
            "update_key_id": update_key, "query_key_id": query_key,
            "initial_values": initial, "current_values": current,
            "primary_old_value": initial[0], "primary_current_value": current[0],
            "current_session_id": current_id, "locality_entity": locality_entity,
            "locality_value": locality_value, "sessions": sessions,
            "queries": query_specs(entity, source_keys, update_key, query_key, initial,
                                   current, locality_entity, locality_value, template_index)}


def build_states(episode):
    identifiers = [row["session_id"] for row in episode["sessions"]]
    current = episode["current_session_id"]
    exact_removed = [row["session_id"] for row in episode["sessions"][:-1]
                     if row["key_id"] == episode["update_key_id"]
                     and row["value"] != episode["primary_current_value"]]
    semantic_removed = [row["session_id"] for row in episode["sessions"][:-1]
                        if row["semantic_key_index"] == 0
                        and row["value"] != episode["primary_current_value"]]
    return {"episode_id": episode["episode_id"], "row_index": episode["row_index"],
            "retain_all": identifiers, "global_reset": [current],
            "exact_conflict": [value for value in identifiers if value not in set(exact_removed)],
            "semantic_oracle": [value for value in identifiers
                                if value not in set(semantic_removed)],
            "exact_removed": exact_removed, "semantic_removed": semantic_removed}


def build(output, controlled_root, seed=SEED):
    output = Path(output); dependencies = verify_controlled_release(controlled_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = [build_episode(index, seed) for index in range(ROWS)]
    states = [build_states(row) for row in episodes]
    data_path, state_path = output / "alias.jsonl", output / "alias_states.jsonl"
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
              "exact_rule_deletions": sum(len(row["exact_removed"]) for row in states)}
    manifest = {"protocol": PROTOCOL, "seed": seed, "split": "semantic_alias",
                "counts": counts, "lengths": list(LENGTHS), "mechanisms": list(MECHANISMS),
                "query_kinds": list(QUERY_KINDS), "alias_bank": [list(row) for row in ALIAS_BANK],
                "alias_sha256": sha(data_path), "alias_states_sha256": sha(state_path),
                "dependencies": dependencies,
                "value_offset": VALUE_OFFSET, "model_scored": False,
                "controller_or_threshold_selected": False, "external_sets_loaded": False,
                "terminal_status": "frozen_semantic_alias_before_scoring"}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--controlled-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
