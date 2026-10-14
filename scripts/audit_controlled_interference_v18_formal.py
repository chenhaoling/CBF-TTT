"""Independently audit the frozen V18 formal controlled-interference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_formal"
PILOT_PROTOCOL = "controlled_memory_interference_v18_pilot"
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


def audit(root, pilot_root, pilot_data_root, output):
    root, pilot_root, pilot_data_root = map(Path, (root, pilot_root, pilot_data_root))
    manifest = read_json(root / "manifest.json")
    episodes = read_jsonl(root / "formal.jsonl")
    states = read_jsonl(root / "formal_states.jsonl")
    pilot_summary, pilot_audit = (read_json(pilot_root / "summary.json"),
                                  read_json(pilot_root / "audit.json"))
    pilot_manifest = read_json(pilot_data_root / "manifest.json")
    pilot_data_audit_path = pilot_data_root.parent / "audit.json"
    if not pilot_data_audit_path.exists():
        pilot_data_audit_path = pilot_data_root / "audit.json"
    pilot_data_audit = read_json(pilot_data_audit_path)
    pilot_episodes = read_jsonl(pilot_data_root / "pilot.jsonl")

    identities_ok = content_ok = query_ok = mechanism_ok = states_ok = True
    all_values, all_entities, subjects_by_fold, templates_by_fold = [], [], {}, {}
    for expected_index, (episode, state) in enumerate(zip(episodes, states)):
        subject_index, within_subject = divmod(expected_index, 4)
        template_index = subject_index % 20
        expected_fold = template_index % 5
        identities_ok &= (
            episode.get("row_index") == state.get("row_index") == expected_index
            and episode.get("episode_id") == state.get("episode_id")
            and episode.get("split") == "formal"
            and episode.get("subject_index") == subject_index
            and episode.get("mechanism") == MECHANISMS[within_subject]
            and episode.get("length") == LENGTHS[(subject_index + within_subject) % 4]
            and episode.get("template_group_index") == template_index
            and episode.get("template_group") == f"formal_templates_{template_index:02d}"
            and episode.get("fold") == expected_fold)
        subjects_by_fold.setdefault(episode.get("subject"), set()).add(episode.get("fold"))
        templates_by_fold.setdefault(episode.get("template_group"), set()).add(
            episode.get("fold"))
        sessions, keys = episode.get("sessions", []), episode.get("keys", [])
        initial, current = episode.get("initial_values", []), episode.get("current_values", [])
        if len(initial) == len(current) == 6:
            all_values.extend(initial + [current[0], episode.get("locality_value")])
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
                    and session.get("template_id", "").startswith(
                        f"formal_fact_{template_index:02d}_")
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
        delete_one, delete_key = state.get("delete_one", []), state.get("delete_key", [])
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
        "subjects": len({row.get("subject") for row in episodes}),
        "template_groups": len({row.get("template_group") for row in episodes}),
        "folds": len({row.get("fold") for row in episodes}),
        "by_length": {str(length): sum(row.get("length") == length for row in episodes)
                      for length in LENGTHS},
        "by_mechanism": {name: sum(row.get("mechanism") == name for row in episodes)
                         for name in MECHANISMS},
        "by_fold": {str(fold): sum(row.get("fold") == fold for row in episodes)
                    for fold in range(5)},
        "single_deletion_states": sum(len(row.get("delete_one", [])) for row in states),
        "key_deletion_states": sum(len(row.get("delete_key", [])) for row in states),
        "fixed_arm_entries": len(states) * len(FIXED),
    }
    pilot_values = {value for row in pilot_episodes for value in
                    row["initial_values"] + [row["primary_current_value"],
                                              row["locality_value"]]}
    pilot_entities = {value for row in pilot_episodes
                      for value in (row["entity"], row["locality_entity"])}
    pilot_templates = {row["template_group"] for row in pilot_episodes}
    values_ok = (len(all_values) == len(set(all_values))
                 and not set(all_values) & pilot_values
                 and all(OPAQUE.fullmatch(value or "") for value in all_values))
    entities = set(all_entities)
    entities_ok = (len({row["entity"] for row in episodes}) == 80
                   and len({row["locality_entity"] for row in episodes}) == 320
                   and not entities & pilot_entities)
    folds_ok = (all(len(value) == 1 for value in subjects_by_fold.values())
                and all(len(value) == 1 for value in templates_by_fold.values())
                and all(sum(row["subject"] == subject for row in episodes) == 4
                        for subject in subjects_by_fold)
                and all(sum(row["template_group"] == template for row in episodes) == 16
                        for template in templates_by_fold))
    expected_dependencies = {
        "pilot_summary_sha256": sha(pilot_root / "summary.json"),
        "pilot_audit_sha256": sha(pilot_root / "audit.json"),
        "pilot_raw_rows_sha256": pilot_summary["raw_rows_sha256"],
        "pilot_data_manifest_sha256": sha(pilot_data_root / "manifest.json"),
        "pilot_data_audit_sha256": sha(pilot_data_audit_path),
        "pilot_data_sha256": sha(pilot_data_root / "pilot.jsonl"),
        "pilot_states_sha256": sha(pilot_data_root / "pilot_states.jsonl"),
    }
    pilot_ok = (
        pilot_summary.get("protocol") == "controlled_memory_interference_v18_qwen_pilot"
        and pilot_summary.get("corpus_admission_gate", {}).get("passed") is True
        and pilot_summary.get("terminal_status") == "pilot_corpus_admitted"
        and pilot_audit.get("audit_passed") is True
        and pilot_audit.get("summary_sha256") == sha(pilot_root / "summary.json")
        and pilot_manifest.get("protocol") == PILOT_PROTOCOL
        and pilot_data_audit.get("audit_passed") is True
        and pilot_data_audit.get("manifest_sha256") == sha(pilot_data_root / "manifest.json")
        and (pilot_root / "runner_exit_code.txt").read_text().strip() == "0")
    checks = {
        "admitted_independently_audited_pilot_release": pilot_ok,
        "protocol_hashes_dependencies_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("formal_sha256") == sha(root / "formal.jsonl")
            and manifest.get("formal_states_sha256") == sha(root / "formal_states.jsonl")
            and manifest.get("dependencies") == expected_dependencies
            and manifest.get("model_scored") is False
            and manifest.get("formal_generated") is True
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status") == "frozen_v18_formal_before_model_scoring"),
        "complete_unique_ordered_episode_state_identities": (
            len(episodes) == len(states) == 320
            and len({row.get("episode_id") for row in episodes}) == 320 and identities_ok),
        "balanced_lengths_mechanisms_subjects_templates_and_folds": (
            counts["by_length"] == {str(length): 80 for length in LENGTHS}
            and counts["by_mechanism"] == {name: 80 for name in MECHANISMS}
            and counts["by_fold"] == {str(fold): 64 for fold in range(5)}
            and counts["subjects"] == 80 and counts["template_groups"] == 20
            and manifest.get("counts") == counts),
        "subject_and_template_disjoint_outer_folds": folds_ok,
        "formal_pilot_disjoint_values_entities_and_templates": (
            values_ok and entities_ok
            and not {row["template_group"] for row in episodes} & pilot_templates),
        "opaque_session_content_and_current_update": content_ok,
        "five_query_answers_and_candidate_membership": query_ok,
        "mechanism_specific_interference_invariants": mechanism_ok,
        "fixed_single_and_key_deletion_causal_states": states_ok,
        "registered_metadata": (
            manifest.get("lengths") == list(LENGTHS)
            and manifest.get("mechanisms") == list(MECHANISMS)
            and manifest.get("fixed_policies") == list(FIXED)
            and manifest.get("query_kinds") == list(KINDS)
            and manifest.get("opaque_value_format")
                == "Z plus four spaced base-16 letters; exactly five Qwen3 tokens"
            and manifest.get("current_session_deletion_allowed") is False),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": sha(root / "manifest.json"),
        "formal_sha256": sha(root / "formal.jsonl"),
        "formal_states_sha256": sha(root / "formal_states.jsonl"),
        "counts": counts,
        "model_scored": False,
        "formal_generated": True,
        "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 controlled-interference formal audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--pilot-data-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
