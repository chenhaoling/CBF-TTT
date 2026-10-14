"""Independently audit the frozen V19 confirmation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v19_confirmation"
RELEASE_PROTOCOL = "v18_locality_constrained_oracle_posthoc_v1"
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


def _values(rows):
    return {value for row in rows for value in
            row["initial_values"] + [row["primary_current_value"], row["locality_value"]]}


def _entities(rows):
    return {value for row in rows for value in (row["entity"], row["locality_entity"])}


def audit(root, release_root, v18_data_root, v18_pilot_data_root, output):
    root, release_root, v18_data_root, v18_pilot_data_root = map(
        Path, (root, release_root, v18_data_root, v18_pilot_data_root))
    manifest = read_json(root / "manifest.json")
    episodes = read_jsonl(root / "confirmation.jsonl")
    states = read_jsonl(root / "states.jsonl")
    release_summary = read_json(release_root / "summary.json")
    release_audit = read_json(release_root / "audit.json")
    v18_manifest = read_json(v18_data_root / "manifest.json")
    pilot_manifest = read_json(v18_pilot_data_root / "manifest.json")
    v18_rows = read_jsonl(v18_data_root / "formal.jsonl")
    pilot_rows = read_jsonl(v18_pilot_data_root / "pilot.jsonl")

    release_ok = (
        release_summary.get("protocol") == RELEASE_PROTOCOL
        and release_summary.get("diagnostic_gate", {}).get("passed") is True
        and release_summary.get("primary_budget") == .03
        and release_summary.get("posthoc") is True
        and release_summary.get("stage_b_released") is False
        and release_summary.get("original_v18_status_unchanged")
            == "formal_no_sufficient_headroom"
        and release_audit.get("protocol") == f"{RELEASE_PROTOCOL}_audit"
        and release_audit.get("audit_passed") is True
        and release_audit.get("summary_sha256") == sha(release_root / "summary.json")
        and release_audit.get("stage_b_released") is False
        and (release_root / "runner_exit_code.txt").read_text().strip() == "0")

    identities_ok = content_ok = query_ok = mechanism_ok = states_ok = True
    subjects_by_fold, templates_by_fold = {}, {}
    for index, (episode, state) in enumerate(zip(episodes, states)):
        subject_index, within_subject = divmod(index, 4)
        template_index = subject_index % 20
        fold = template_index % 5
        identities_ok &= (
            episode.get("row_index") == state.get("row_index") == index
            and episode.get("episode_id") == state.get("episode_id")
            and episode.get("split") == "confirmation"
            and episode.get("subject_index") == subject_index
            and episode.get("mechanism") == MECHANISMS[within_subject]
            and episode.get("length") == LENGTHS[(subject_index + within_subject) % 4]
            and episode.get("template_group_index") == template_index
            and episode.get("template_group") == f"v19_templates_{template_index:02d}"
            and episode.get("fold") == fold)
        subjects_by_fold.setdefault(episode.get("subject"), set()).add(episode.get("fold"))
        templates_by_fold.setdefault(episode.get("template_group"), set()).add(
            episode.get("fold"))
        sessions, keys = episode.get("sessions", []), episode.get("keys", [])
        initial, current = episode.get("initial_values", []), episode.get("current_values", [])
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
            and [row.get("session_index") for row in sessions] == list(range(len(sessions)))
            and len({row.get("session_id") for row in sessions}) == len(sessions)
            and all(row.get("key_id") in keys
                    and row.get("key_index") == keys.index(row.get("key_id"))
                    and row.get("key_id") in row.get("statement", "")
                    and row.get("value") in row.get("statement", "")
                    and episode.get("entity") in row.get("statement", "")
                    and row.get("template_id", "").startswith(
                        f"v19_fact_{template_index:02d}_") for row in sessions))
        queries = episode.get("queries", [])
        query_ok &= (
            len(queries) == 5 and tuple(row.get("kind") for row in queries) == KINDS
            and all(len(row.get("candidates", [])) == len(set(row.get("candidates", [])))
                    and row.get("answer") in row.get("candidates", []) for row in queries)
            and queries[0].get("answer") == queries[1].get("answer") == current[0]
            and queries[2].get("answer") == current[1]
            and queries[3].get("answer") == current[2]
            and queries[4].get("answer") == episode.get("locality_value")
            and episode.get("locality_value") in queries[4].get("statement", ""))
        role_counts = {role: sum(row.get("role") == role for row in sessions)
                       for role in ("stale_repeat", "near_duplicate", "useful_repeat")}
        mechanism = episode.get("mechanism")
        mechanism_ok &= (
            (mechanism != "clean_update" or not any(role_counts.values()))
            and (mechanism != "stale_repetition" or role_counts["stale_repeat"] >= 1)
            and (mechanism != "irrelevant_near_duplicate"
                 or role_counts["near_duplicate"] >= 1 and keys[3] == keys[0] + "B")
            and (mechanism != "useful_repetition" or role_counts["useful_repeat"] >= 1))

        identifiers = [row["session_id"] for row in sessions]
        expected_fixed = {"no_memory": [], "retain_all": identifiers,
                          "global_reset": identifiers[-1:], "last_1": identifiers[-1:],
                          "last_2": identifiers[-2:]}
        delete_one, delete_key = state.get("delete_one", []), state.get("delete_key", [])
        states_ok &= (
            state.get("fixed") == expected_fixed
            and len(delete_one) == len(identifiers) - 1
            and [row.get("deleted_session_id") for row in delete_one] == identifiers[:-1]
            and all(row.get("memory_session_ids")
                    == [value for value in identifiers
                        if value != row.get("deleted_session_id")]
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
    all_values = _values(episodes)
    all_entities = _entities(episodes)
    disjoint_ok = (
        len(all_values) == 320 * 8
        and not all_values & (_values(v18_rows) | _values(pilot_rows))
        and not all_entities & (_entities(v18_rows) | _entities(pilot_rows))
        and not {row["template_group"] for row in episodes}
            & ({row["template_group"] for row in v18_rows}
               | {row["template_group"] for row in pilot_rows}))
    folds_ok = (all(len(value) == 1 for value in subjects_by_fold.values())
                and all(len(value) == 1 for value in templates_by_fold.values())
                and all(sum(row["subject"] == subject for row in episodes) == 4
                        for subject in subjects_by_fold)
                and all(sum(row["template_group"] == group for row in episodes) == 16
                        for group in templates_by_fold))
    dependencies = {
        "release_summary_sha256": sha(release_root / "summary.json"),
        "release_audit_sha256": sha(release_root / "audit.json"),
        "v18_manifest_sha256": sha(v18_data_root / "manifest.json"),
        "v18_formal_sha256": sha(v18_data_root / "formal.jsonl"),
        "v18_pilot_manifest_sha256": sha(v18_pilot_data_root / "manifest.json"),
        "v18_pilot_sha256": sha(v18_pilot_data_root / "pilot.jsonl"),
    }
    checks = {
        "audited_passing_posthoc_v18_lc_release": release_ok,
        "protocol_hashes_dependencies_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("confirmation_sha256") == sha(root / "confirmation.jsonl")
            and manifest.get("states_sha256") == sha(root / "states.jsonl")
            and manifest.get("dependencies") == dependencies
            and manifest.get("model_scored") is False
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status")
                == "frozen_v19_confirmation_before_model_scoring"),
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
        "v19_disjoint_from_all_v18_values_entities_and_templates": disjoint_ok,
        "opaque_session_content_and_current_update": content_ok,
        "five_query_answers_and_candidate_membership": query_ok,
        "mechanism_specific_interference_invariants": mechanism_ok,
        "fixed_single_and_key_deletion_causal_states": states_ok,
        "registered_primary_budget_and_success_gate": (
            manifest.get("primary_locality_budget") == .03
            and manifest.get("maximum_locality_failures") == 9
            and manifest.get("registered_success_gate") == {
                "subject_and_template_macro_joint_gain_at_least": .10,
                "cluster_bootstrap_lower_95_above": 0,
                "update_accuracy_at_least": .90,
                "retained_key_accuracy_at_least": .90,
                "locality_accuracy_at_least": .97,
                "no_outer_fold_negative_joint_gain": True})
    }
    result = {
        "protocol": f"{PROTOCOL}_audit",
        "audit_passed": all(checks.values()),
        "checks": checks,
        "manifest_sha256": sha(root / "manifest.json"),
        "confirmation_sha256": sha(root / "confirmation.jsonl"),
        "states_sha256": sha(root / "states.jsonl"),
        "counts": counts,
        "model_scored": False,
        "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V19 confirmation corpus audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--v18-data-root", required=True)
    parser.add_argument("--v18-pilot-data-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
