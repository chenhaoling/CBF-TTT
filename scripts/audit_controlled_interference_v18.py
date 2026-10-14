"""Independently audit the frozen V18 controlled-interference pilot corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_pilot"
LENGTHS = (8, 16, 32, 64)
MECHANISMS = ("clean_update", "stale_repetition", "irrelevant_near_duplicate",
              "useful_repetition")
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")
OPAQUE = re.compile(r"^Z(?: [A-P]){4}$")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, output):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    episodes, states = read_jsonl(root / "pilot.jsonl"), read_jsonl(root / "pilot_states.jsonl")
    identities_ok = content_ok = query_ok = mechanism_ok = states_ok = True
    all_values, all_entities = [], []
    for expected_index, (episode, state) in enumerate(zip(episodes, states)):
        identities_ok &= (
            episode.get("row_index") == state.get("row_index") == expected_index
            and episode.get("episode_id") == state.get("episode_id")
            and episode.get("split") == "pilot"
            and episode.get("template_group") == "pilot_templates_v1"
            and episode.get("length") in LENGTHS
            and episode.get("mechanism") in MECHANISMS)
        sessions = episode.get("sessions", [])
        keys = episode.get("keys", [])
        initial, current = episode.get("initial_values", []), episode.get("current_values", [])
        all_values.extend(initial + [current[0], episode.get("locality_value")]
                          if len(initial) == len(current) == 6 else [])
        all_entities.extend((episode.get("entity"), episode.get("locality_entity")))
        content_ok &= (
            len(sessions) == episode.get("length")
            and episode.get("query_step") == len(sessions) - 1
            and len(keys) == len(set(keys)) == 6
            and len(initial) == len(current) == 6
            and all(OPAQUE.fullmatch(value or "") for value in initial + [current[0]])
            and OPAQUE.fullmatch(episode.get("locality_value", "")) is not None
            and current[0] != initial[0] and current[1:] == initial[1:]
            and episode.get("target_key_id") == keys[0]
            and episode.get("primary_old_value") == initial[0]
            and episode.get("primary_current_value") == current[0]
            and sessions[-1].get("session_id") == episode.get("current_session_id")
            and sessions[-1].get("key_id") == keys[0]
            and sessions[-1].get("value") == current[0]
            and sessions[-1].get("role") == "current_update"
            and [session.get("session_index") for session in sessions] == list(range(len(sessions)))
            and len({session.get("session_id") for session in sessions}) == len(sessions)
            and all(session.get("key_id") in keys
                    and session.get("key_index") == keys.index(session.get("key_id"))
                    and session.get("key_id") in session.get("statement", "")
                    and session.get("value") in session.get("statement", "")
                    and episode.get("entity") in session.get("statement", "")
                    for session in sessions))
        queries = episode.get("queries", [])
        query_ok &= (
            len(queries) == 5 and tuple(query.get("kind") for query in queries) == KINDS
            and all(len(query.get("candidates", [])) == len(set(query.get("candidates", [])))
                    and query.get("answer") in query.get("candidates", [])
                    for query in queries)
            and queries[0].get("answer") == queries[1].get("answer") == current[0]
            and queries[2].get("answer") == current[1]
            and queries[3].get("answer") == current[2]
            and queries[4].get("answer") == episode.get("locality_value")
            and episode.get("locality_value") in queries[4].get("statement", ""))
        role_counts = {role: sum(session.get("role") == role for session in sessions)
                       for role in ("stale_repeat", "near_duplicate", "useful_repeat")}
        mechanism = episode.get("mechanism")
        mechanism_ok &= (
            (mechanism != "clean_update" or not any(role_counts.values()))
            and (mechanism != "stale_repetition" or role_counts["stale_repeat"] >= 1)
            and (mechanism != "irrelevant_near_duplicate"
                 or role_counts["near_duplicate"] >= 1 and keys[3] == keys[0] + "A")
            and (mechanism != "useful_repetition" or role_counts["useful_repeat"] >= 1))

        identifiers = [session["session_id"] for session in sessions]
        expected_fixed = {"no_memory": [], "retain_all": identifiers,
                          "global_reset": identifiers[-1:], "last_1": identifiers[-1:],
                          "last_2": identifiers[-2:]}
        delete_one = state.get("delete_one", [])
        delete_key = state.get("delete_key", [])
        states_ok &= (
            state.get("fixed") == expected_fixed
            and len(delete_one) == len(identifiers) - 1
            and [row.get("deleted_session_id") for row in delete_one] == identifiers[:-1]
            and all(row.get("memory_session_ids")
                    == [value for value in identifiers if value != row.get("deleted_session_id")]
                    and identifiers[-1] in row.get("memory_session_ids", [])
                    for row in delete_one)
            and 1 <= len(delete_key) <= 6
            and all(row.get("deleted_session_ids")
                    == [session["session_id"] for session in sessions[:-1]
                        if session["key_id"] == row.get("deleted_key_id")]
                    and row.get("memory_session_ids")
                    == [value for value in identifiers
                        if value not in set(row.get("deleted_session_ids", []))]
                    and identifiers[-1] in row.get("memory_session_ids", [])
                    for row in delete_key))

    counts = {
        "rows": len(episodes),
        "by_length": {str(length): sum(row.get("length") == length for row in episodes)
                      for length in LENGTHS},
        "by_mechanism": {name: sum(row.get("mechanism") == name for row in episodes)
                         for name in MECHANISMS},
        "single_deletion_states": sum(len(row.get("delete_one", [])) for row in states),
        "key_deletion_states": sum(len(row.get("delete_key", [])) for row in states),
        "fixed_arm_entries": len(states) * len(FIXED),
    }
    values_ok = (len(all_values) == len(set(all_values))
                 and all(value and value.startswith("Z") for value in all_values)
                 and len(all_entities) == len(set(all_entities)))
    checks = {
        "protocol_hashes_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("pilot_sha256") == sha(root / "pilot.jsonl")
            and manifest.get("pilot_states_sha256") == sha(root / "pilot_states.jsonl")
            and manifest.get("model_scored") is False
            and manifest.get("formal_generated") is False
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status") == "frozen_v18_pilot_before_model_scoring"
            and not (root / "formal.jsonl").exists()),
        "complete_unique_ordered_episode_state_identities": (
            len(episodes) == len(states) == 40
            and len({row.get("episode_id") for row in episodes}) == 40 and identities_ok),
        "balanced_registered_lengths_and_mechanisms": (
            counts["by_length"] == {str(length): 10 for length in LENGTHS}
            and counts["by_mechanism"] == {name: 10 for name in MECHANISMS}
            and manifest.get("counts") == counts),
        "opaque_disjoint_values_entities_and_session_content": content_ok and values_ok,
        "five_query_answers_and_candidate_membership": query_ok,
        "mechanism_specific_interference_invariants": mechanism_ok,
        "fixed_single_and_key_deletion_causal_states": states_ok,
        "registered_metadata_and_no_slow_answer_assumption": (
            manifest.get("lengths") == list(LENGTHS)
            and manifest.get("mechanisms") == list(MECHANISMS)
            and manifest.get("fixed_policies") == list(FIXED)
            and manifest.get("query_kinds") == list(KINDS)
            and manifest.get("slow_parameter_answerable_by_design") is False
            and manifest.get("opaque_value_format")
                == "Z plus four spaced base-16 letters; exactly five Qwen3 tokens"
            and manifest.get("current_session_deletion_allowed") is False),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "manifest_sha256": sha(root / "manifest.json"),
              "pilot_sha256": sha(root / "pilot.jsonl"),
              "pilot_states_sha256": sha(root / "pilot_states.jsonl"),
              "counts": counts, "model_scored": False, "formal_generated": False,
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 controlled-interference pilot audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
