"""Independently audit the frozen V20 three-surface alias corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_controlled_interference_v18 import (
    FIXED_POLICIES, LENGTHS, MECHANISMS, QUERY_KINDS,
)
from tasks.build_controlled_interference_v20_alias import (
    ALIAS_BANK, PROTOCOL, RELEASE_PROTOCOL, ROWS, SEED, VALUE_OFFSET,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def values(rows):
    return {value for row in rows for value in
            row["initial_values"] + [row["primary_current_value"],
                                      row["locality_value"]]}


def audit(root, rule_root, v19_data_root, output):
    root, rule_root, v19_data_root = map(Path, (root, rule_root, v19_data_root))
    manifest = read_json(root / "manifest.json")
    episodes = read_jsonl(root / "alias.jsonl")
    states = read_jsonl(root / "states.jsonl")
    rule_summary = read_json(rule_root / "summary.json")
    rule_audit = read_json(rule_root / "audit.json")
    v19_manifest = read_json(v19_data_root / "manifest.json")
    v19_rows = read_jsonl(v19_data_root / "confirmation.jsonl")

    release_ok = (
        rule_summary.get("protocol") == RELEASE_PROTOCOL
        and rule_summary.get("explicit_key_explanation_gate", {}).get("passed") is True
        and rule_summary.get("next_stage") == "build_v20_semantic_alias_confirmation"
        and rule_summary.get("query_or_answer_features_used") is False
        and rule_summary.get("model_rescored") is False
        and rule_audit.get("audit_passed") is True
        and rule_audit.get("summary_sha256") == sha(rule_root / "summary.json")
        and rule_audit.get("recomputed_explanation_gate")
            == rule_summary.get("explicit_key_explanation_gate")
        and (rule_root / "runner_exit_code.txt").read_text().strip() == "0"
        and (rule_root / "terminal_status.txt").read_text().strip()
            == "completed_v19_causal_rule_diagnostic")

    identities_ok = content_ok = queries_ok = states_ok = True
    subjects_by_fold, templates_by_fold = {}, {}
    for index, pair in enumerate(zip(episodes, states)):
        episode, state = pair
        subject_index, within = divmod(index, 4)
        template_index = subject_index % 20
        expected_length = LENGTHS[(subject_index + within) % 4]
        identities_ok &= (
            episode.get("row_index") == state.get("row_index") == index
            and episode.get("episode_id") == state.get("episode_id")
            and episode.get("split") == "semantic_alias"
            and episode.get("subject_index") == subject_index
            and episode.get("mechanism") == MECHANISMS[within]
            and episode.get("length") == expected_length
            and episode.get("template_group_index") == template_index
            and episode.get("fold") == template_index % 5)
        subjects_by_fold.setdefault(episode.get("subject"), set()).add(episode.get("fold"))
        templates_by_fold.setdefault(episode.get("template_group"), set()).add(
            episode.get("fold"))

        sessions = episode.get("sessions", [])
        ids = [row.get("session_id") for row in sessions]
        current = sessions[-1] if sessions else {}
        concepts = episode.get("semantic_concept_indices", [])
        source = episode.get("source_keys", [])
        content_ok &= (
            len(sessions) == expected_length
            and len(ids) == len(set(ids)) == expected_length
            and [row.get("session_index") for row in sessions]
                == list(range(expected_length))
            and len(concepts) == len(set(concepts)) == len(source) == 6
            and all(0 <= value < len(ALIAS_BANK) for value in concepts)
            and all(source[offset] == ALIAS_BANK[value][0]
                    for offset, value in enumerate(concepts) if not (
                        episode.get("mechanism") == "irrelevant_near_duplicate"
                        and offset == 3))
            and (episode.get("mechanism") != "irrelevant_near_duplicate"
                 or source[3] == source[0] + " checksum")
            and episode.get("update_key_id") == ALIAS_BANK[concepts[0]][1]
            and episode.get("query_key_id") == ALIAS_BANK[concepts[0]][2]
            and len({source[0], episode.get("update_key_id"),
                     episode.get("query_key_id")}) == 3
            and current.get("session_id") == episode.get("current_session_id")
            and current.get("semantic_key_index") == 0
            and current.get("key_id") == episode.get("update_key_id")
            and current.get("value") == episode.get("primary_current_value")
            and all(row.get("key_id") != episode.get("update_key_id")
                    for row in sessions[:-1]))
        specs = episode.get("queries", [])
        queries_ok &= (
            [row.get("kind") for row in specs] == list(QUERY_KINDS)
            and all(row.get("answer") in row.get("candidates", []) for row in specs)
            and specs[0].get("query_key_id") == source[0]
            and specs[1].get("query_key_id") == episode.get("query_key_id")
            and specs[0].get("answer") == specs[1].get("answer")
                == episode.get("primary_current_value"))

        fixed, special = state.get("fixed", {}), state.get("special", {})
        semantic_expected = [row["session_id"] for row in sessions[:-1]
                             if row["semantic_key_index"] == 0
                             and row["value"] != episode["primary_current_value"]]
        semantic_state = [value for value in ids if value not in set(semantic_expected)]
        states_ok &= (
            set(fixed) == set(FIXED_POLICIES)
            and fixed.get("no_memory") == []
            and fixed.get("retain_all") == ids
            and fixed.get("global_reset") == [ids[-1]]
            and fixed.get("last_1") == [ids[-1]]
            and fixed.get("last_2") == ids[-2:]
            and state.get("exact_removed") == []
            and special.get("exact_conflict") == ids
            and state.get("semantic_removed") == semantic_expected
            and len(semantic_expected) >= 1
            and special.get("semantic_oracle") == semantic_state
            and ids[-1] in semantic_state)

    counts = {
        "rows": len(episodes),
        "subjects": len({row.get("subject") for row in episodes}),
        "template_groups": len({row.get("template_group") for row in episodes}),
        "by_length": {str(value): sum(row.get("length") == value for row in episodes)
                      for value in LENGTHS},
        "by_mechanism": {value: sum(row.get("mechanism") == value for row in episodes)
                         for value in MECHANISMS},
        "by_fold": {str(value): sum(row.get("fold") == value for row in episodes)
                    for value in range(5)},
        "semantic_oracle_deletions": sum(len(row.get("semantic_removed", []))
                                         for row in states),
        "exact_rule_deletions": sum(len(row.get("exact_removed", []))
                                    for row in states),
        "fixed_arm_entries": len(states) * len(FIXED_POLICIES),
        "special_arm_entries": len(states) * 2,
    }
    folds_ok = (
        all(len(value) == 1 for value in subjects_by_fold.values())
        and all(len(value) == 1 for value in templates_by_fold.values())
        and all(sum(row["subject"] == subject for row in episodes) == 4
                for subject in subjects_by_fold)
        and all(sum(row["template_group"] == template for row in episodes) == 16
                for template in templates_by_fold))
    dependencies = {
        "rule_summary_sha256": sha(rule_root / "summary.json"),
        "rule_audit_sha256": sha(rule_root / "audit.json"),
        "v19_raw_rows_sha256": rule_summary["raw_rows_sha256"],
        "v19_outcome_summary_sha256":
            rule_summary["dependencies"]["outcome_summary_sha256"],
    }
    checks = {
        "audited_passing_v19_explicit_key_explanation": release_ok,
        "protocol_hashes_dependencies_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("seed") == SEED
            and manifest.get("value_offset") == VALUE_OFFSET
            and manifest.get("alias_sha256") == sha(root / "alias.jsonl")
            and manifest.get("states_sha256") == sha(root / "states.jsonl")
            and manifest.get("dependencies") == dependencies
            and manifest.get("model_scored") is False
            and manifest.get("controller_or_threshold_selected") is False
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status") == "frozen_v20_alias_before_model_scoring"),
        "complete_unique_ordered_episode_state_identities": (
            len(episodes) == len(states) == ROWS
            and len({row.get("episode_id") for row in episodes}) == ROWS
            and identities_ok),
        "balanced_lengths_mechanisms_subjects_templates_and_folds": (
            counts["by_length"] == {str(value): 80 for value in LENGTHS}
            and counts["by_mechanism"] == {value: 80 for value in MECHANISMS}
            and counts["by_fold"] == {str(value): 64 for value in range(5)}
            and counts["subjects"] == 80 and counts["template_groups"] == 20
            and manifest.get("counts") == counts),
        "subject_and_template_disjoint_outer_folds": folds_ok,
        "v20_values_entities_and_templates_disjoint_from_v19": (
            not values(episodes) & values(v19_rows)
            and not {row["entity"] for row in episodes}
                & {row["entity"] for row in v19_rows}
            and not {row["template_group"] for row in episodes}
                & {row["template_group"] for row in v19_rows}
            and v19_manifest.get("protocol")
                == "controlled_memory_interference_v19_confirmation"),
        "three_distinct_surfaces_and_current_update_content": content_ok,
        "five_queries_and_candidate_membership": queries_ok,
        "exact_rule_noop_semantic_oracle_removes_stale_latent_slot": states_ok,
        "registered_primary_comparison_and_success_gate": (
            manifest.get("registered_gate") == {
                "joint_gain_at_least": .10,
                "update_accuracy_at_least": .90,
                "retained_key_accuracy_at_least": .90,
                "locality_accuracy_at_least": .97}),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "manifest_sha256": sha(root / "manifest.json"),
        "alias_sha256": sha(root / "alias.jsonl"),
        "states_sha256": sha(root / "states.jsonl"), "counts": counts,
        "model_scored": False, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V20 semantic-alias corpus audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--rule-root", required=True)
    parser.add_argument("--v19-data-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
