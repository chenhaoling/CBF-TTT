"""Build the frozen 40-row V18 controlled memory-interference pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_pilot"
SEED = 1801
LENGTHS = (8, 16, 32, 64)
MECHANISMS = ("clean_update", "stale_repetition", "irrelevant_near_duplicate",
              "useful_repetition")
FIXED_POLICIES = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
QUERY_KINDS = ("update_old_surface", "update_current_surface", "retain_1",
               "retain_2", "locality")


def opaque(namespace, index, width=10):
    digest = hashlib.sha256(f"v18:{namespace}:{index}".encode()).hexdigest().upper()
    return f"Z{digest[:width]}"


def opaque_value(index):
    """A unique code whose spaced base-16 digits are five Qwen3 tokens."""
    alphabet = "ABCDEFGHIJKLMNOP"
    if not 0 <= index < 16 ** 4:
        raise ValueError("opaque value index exceeds the registered four-digit space")
    digits = [alphabet[(index // (16 ** power)) % 16] for power in range(4)]
    return "Z " + " ".join(digits)


def fact_statement(entity, key, value, template):
    if template == 0:
        return f"Archive subject {entity} has code {value} in slot {key}."
    if template == 1:
        return f"For {entity}, the code stored under slot {key} is {value}."
    raise ValueError(template)


def query_specs(entity, keys, initial, current, locality_entity, locality_value):
    target_candidates = [initial[0], current[0]] + [current[index] for index in (3, 4, 5)]
    all_candidates = list(dict.fromkeys(current))
    locality_candidates = [locality_value] + current[:5]
    return [
        {"kind": "update_old_surface",
         "statement": f"What code does archive subject {entity} have in slot {keys[0]}?",
         "answer": current[0], "candidates": target_candidates},
        {"kind": "update_current_surface",
         "statement": f"Report the active code for {entity} under slot {keys[0]}.",
         "answer": current[0], "candidates": target_candidates},
        {"kind": "retain_1",
         "statement": f"What code is stored for {entity} in slot {keys[1]}?",
         "answer": current[1], "candidates": all_candidates},
        {"kind": "retain_2",
         "statement": f"Give the code assigned to slot {keys[2]} for {entity}.",
         "answer": current[2], "candidates": all_candidates},
        {"kind": "locality",
         "statement": (f"The public checksum for {locality_entity} is {locality_value}. "
                       f"What is the public checksum for {locality_entity}?"),
         "answer": locality_value, "candidates": locality_candidates},
    ]


def build_episode(index, length, mechanism, seed=SEED):
    rng = random.Random(seed * 10_000 + index)
    entity = f"ENT-P-{opaque('entity', index, 8)}"
    locality_entity = f"REF-P-{opaque('locality-entity', index, 8)}"
    keys = [f"K{opaque(f'key-{index}', key_index, 7)}" for key_index in range(6)]
    if mechanism == "irrelevant_near_duplicate":
        keys[3] = keys[0] + "A"
    initial = [opaque_value(index * 8 + key_index) for key_index in range(6)]
    current = list(initial)
    current[0] = opaque_value(index * 8 + 6)
    locality_value = opaque_value(index * 8 + 7)

    facts = [(key_index, initial[key_index], "initial") for key_index in range(6)]
    fillers = length - 7
    for filler in range(fillers):
        if mechanism == "stale_repetition" and filler % 2 == 0:
            key_index, role = 0, "stale_repeat"
        elif mechanism == "irrelevant_near_duplicate" and filler % 2 == 0:
            key_index, role = 3, "near_duplicate"
        elif mechanism == "useful_repetition" and filler % 2 == 0:
            key_index, role = 1, "useful_repeat"
        else:
            key_index, role = 3 + filler % 3, "stable_filler"
        facts.append((key_index, initial[key_index], role))
    if len(facts) != length - 1:
        raise RuntimeError("V18 historical-session count mismatch")
    facts.append((0, current[0], "current_update"))

    sessions = []
    for session_index, (key_index, value, role) in enumerate(facts):
        template = rng.randrange(2)
        sessions.append({
            "session_id": f"v18p-{index:03d}-s{session_index:03d}",
            "session_index": session_index, "key_id": keys[key_index],
            "key_index": key_index, "value": value, "role": role,
            "statement": fact_statement(entity, keys[key_index], value, template),
            "template_id": f"pilot_fact_{template}",
        })
    episode_id = hashlib.sha256(
        f"v18-pilot:{seed}:{index}:{length}:{mechanism}".encode()).hexdigest()[:20]
    queries = query_specs(entity, keys, initial, current, locality_entity, locality_value)
    return {
        "episode_id": episode_id, "split": "pilot", "row_index": index,
        "subject": hashlib.sha256(entity.encode()).hexdigest(),
        "entity": entity, "template_group": "pilot_templates_v1",
        "length": length, "mechanism": mechanism, "query_step": length - 1,
        "keys": keys, "initial_values": initial, "current_values": current,
        "target_key_id": keys[0], "primary_old_value": initial[0],
        "primary_current_value": current[0], "current_session_id": sessions[-1]["session_id"],
        "locality_entity": locality_entity, "locality_value": locality_value,
        "sessions": sessions, "queries": queries,
    }


def build_states(episode):
    identifiers = [session["session_id"] for session in episode["sessions"]]
    current = episode["current_session_id"]
    fixed = {
        "no_memory": [], "retain_all": identifiers,
        "global_reset": identifiers[-1:], "last_1": identifiers[-1:],
        "last_2": identifiers[-2:],
    }
    delete_one = [{"deleted_session_id": identifier,
                   "memory_session_ids": [value for value in identifiers if value != identifier]}
                  for identifier in identifiers[:-1]]
    delete_key = []
    for key in episode["keys"]:
        removed = [session["session_id"] for session in episode["sessions"][:-1]
                   if session["key_id"] == key]
        if not removed:
            continue
        delete_key.append({
            "deleted_key_id": key, "deleted_session_ids": removed,
            "memory_session_ids": [value for value in identifiers if value not in set(removed)],
        })
    if (tuple(fixed) != FIXED_POLICIES
            or any(current not in row["memory_session_ids"] for row in delete_one + delete_key)):
        raise RuntimeError("invalid V18 causal states")
    return {"episode_id": episode["episode_id"], "row_index": episode["row_index"],
            "fixed": fixed, "delete_one": delete_one, "delete_key": delete_key}


def build(output, seed=SEED):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for length_index, length in enumerate(LENGTHS):
        for within in range(10):
            index = length_index * 10 + within
            mechanism = MECHANISMS[(within + length_index) % len(MECHANISMS)]
            episodes.append(build_episode(index, length, mechanism, seed))
    states = [build_states(episode) for episode in episodes]
    data_path, state_path = output / "pilot.jsonl", output / "pilot_states.jsonl"
    with data_path.open("x") as sink:
        sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes))
    with state_path.open("x") as sink:
        sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in states))
    mechanism_counts = {name: sum(row["mechanism"] == name for row in episodes)
                        for name in MECHANISMS}
    counts = {
        "rows": len(episodes),
        "by_length": {str(length): sum(row["length"] == length for row in episodes)
                      for length in LENGTHS},
        "by_mechanism": mechanism_counts,
        "single_deletion_states": sum(len(row["delete_one"]) for row in states),
        "key_deletion_states": sum(len(row["delete_key"]) for row in states),
        "fixed_arm_entries": len(states) * len(FIXED_POLICIES),
    }
    manifest = {
        "protocol": PROTOCOL, "seed": seed, "split": "pilot",
        "lengths": list(LENGTHS), "mechanisms": list(MECHANISMS),
        "fixed_policies": list(FIXED_POLICIES), "query_kinds": list(QUERY_KINDS),
        "counts": counts, "pilot_sha256": sha(data_path),
        "pilot_states_sha256": sha(state_path),
        "opaque_value_namespace": "pilot-only; formal prefixes and pools remain ungenerated",
        "opaque_value_format": "Z plus four spaced base-16 letters; exactly five Qwen3 tokens",
        "slow_parameter_answerable_by_design": False,
        "current_session_deletion_allowed": False,
        "model_scored": False, "formal_generated": False, "external_sets_loaded": False,
        "terminal_status": "frozen_v18_pilot_before_model_scoring",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
