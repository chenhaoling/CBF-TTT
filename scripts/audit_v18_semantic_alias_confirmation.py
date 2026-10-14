"""Independent construction audit for the frozen V18 semantic-alias set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tasks.build_v18_semantic_alias_confirmation import (
    ALIAS_BANK, PROTOCOL, QUERY_KINDS, verify_controlled_release,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


OPAQUE = re.compile(r"^Z(?: [A-P]){4}$")


def audit(root, controlled_root, formal_data_root, pilot_data_root, tokenizer, output):
    from transformers import AutoTokenizer
    root, formal_data_root, pilot_data_root = map(
        Path, (root, formal_data_root, pilot_data_root))
    dependencies = verify_controlled_release(controlled_root)
    manifest = json.loads((root / "manifest.json").read_text())
    episodes = load_jsonl(root / "alias.jsonl"); states = load_jsonl(root / "alias_states.jsonl")
    formal = load_jsonl(formal_data_root / "formal.jsonl")
    pilot = load_jsonl(pilot_data_root / "pilot.jsonl")
    identities_ok = content_ok = queries_ok = states_ok = mechanisms_ok = True
    subjects_by_fold, templates_by_fold, values = {}, {}, []
    for index, (episode, state) in enumerate(zip(episodes, states)):
        subject_index, within = divmod(index, 4); template = subject_index % 20
        identities_ok &= (
            episode.get("row_index") == state.get("row_index") == index
            and episode.get("episode_id") == state.get("episode_id")
            and episode.get("subject_index") == subject_index
            and episode.get("template_group_index") == template
            and episode.get("fold") == template % 5
            and episode.get("split") == "semantic_alias")
        subjects_by_fold.setdefault(episode.get("subject"), set()).add(episode.get("fold"))
        templates_by_fold.setdefault(episode.get("template_group"), set()).add(episode.get("fold"))
        sessions = episode.get("sessions", []); source = episode.get("source_keys", [])
        primary_surfaces = {source[0], episode.get("update_key_id"), episode.get("query_key_id")} \
            if len(source) == 6 else set()
        values.extend(episode.get("initial_values", []) + [episode.get("primary_current_value"),
                                                            episode.get("locality_value")])
        content_ok &= (
            len(sessions) == episode.get("length") and len(source) == 6
            and len(primary_surfaces) == 3
            and sessions[-1].get("session_id") == episode.get("current_session_id")
            and sessions[-1].get("key_id") == episode.get("update_key_id")
            and sessions[-1].get("semantic_key_index") == 0
            and sessions[-1].get("value") == episode.get("primary_current_value")
            and sessions[-1].get("role") == "current_update"
            and [row.get("session_index") for row in sessions] == list(range(len(sessions)))
            and all(row.get("value") in row.get("statement", "")
                    and row.get("key_id") in row.get("statement", "")
                    and episode.get("entity") in row.get("statement", "") for row in sessions))
        queries = episode.get("queries", [])
        queries_ok &= (
            tuple(row.get("kind") for row in queries) == QUERY_KINDS
            and queries[0].get("query_key_id") == source[0]
            and queries[1].get("query_key_id") == episode.get("query_key_id")
            and queries[2].get("query_key_id") == source[1]
            and queries[3].get("query_key_id") == source[2]
            and queries[4].get("query_key_id") is None
            and queries[0].get("answer") == queries[1].get("answer")
                == episode.get("primary_current_value")
            and all(row.get("answer") in row.get("candidates", []) for row in queries))
        identifiers = [row["session_id"] for row in sessions]
        semantic_removed = [row["session_id"] for row in sessions[:-1]
                            if row["semantic_key_index"] == 0
                            and row["value"] != episode["primary_current_value"]]
        exact_removed = [row["session_id"] for row in sessions[:-1]
                         if row["key_id"] == episode["update_key_id"]
                         and row["value"] != episode["primary_current_value"]]
        states_ok &= (
            state.get("retain_all") == identifiers and state.get("global_reset") == identifiers[-1:]
            and state.get("exact_removed") == exact_removed == []
            and state.get("semantic_removed") == semantic_removed and bool(semantic_removed)
            and state.get("exact_conflict") == identifiers
            and state.get("semantic_oracle")
                == [value for value in identifiers if value not in set(semantic_removed)]
            and episode["current_session_id"] in state.get("semantic_oracle", []))
        roles = [row.get("role") for row in sessions]
        expected = {"stale_repetition": "stale_repeat",
                    "irrelevant_near_duplicate": "near_duplicate",
                    "useful_repetition": "useful_repeat"}.get(episode.get("mechanism"))
        mechanisms_ok &= (expected is None or expected in roles)
    formal_values = {value for row in formal for value in
                     row["initial_values"] + [row["primary_current_value"], row["locality_value"]]}
    pilot_values = {value for row in pilot for value in
                    row["initial_values"] + [row["primary_current_value"], row["locality_value"]]}
    tokenizer_object = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    tokenizer_ok = {len(tokenizer_object.encode(value, add_special_tokens=False))
                    for value in values} == {5}
    counts = {"rows": len(episodes), "subjects": len({row["subject"] for row in episodes}),
              "template_groups": len({row["template_group"] for row in episodes}),
              "by_length": {str(value): sum(row["length"] == value for row in episodes)
                            for value in (8, 16, 32, 64)},
              "by_mechanism": {value: sum(row["mechanism"] == value for row in episodes)
                               for value in ("clean_update", "stale_repetition",
                                             "irrelevant_near_duplicate", "useful_repetition")},
              "by_fold": {str(value): sum(row["fold"] == value for row in episodes)
                          for value in range(5)},
              "semantic_oracle_deletions": sum(len(row["semantic_removed"]) for row in states),
              "exact_rule_deletions": sum(len(row["exact_removed"]) for row in states)}
    checks = {
        "positive_controlled_policy_release": manifest.get("dependencies") == dependencies,
        "complete_unique_ordered_alias_state_identities": (
            len(episodes) == len(states) == 320
            and len({row["episode_id"] for row in episodes}) == 320 and identities_ok),
        "balanced_subject_template_fold_length_mechanism_design": (
            counts["subjects"] == 80 and counts["template_groups"] == 20
            and counts["by_length"] == {str(value): 80 for value in (8, 16, 32, 64)}
            and all(value == 80 for value in counts["by_mechanism"].values())
            and counts["by_fold"] == {str(value): 64 for value in range(5)}
            and all(len(value) == 1 for value in subjects_by_fold.values())
            and all(len(value) == 1 for value in templates_by_fold.values())
            and manifest.get("counts") == counts),
        "three_surface_latent_slot_and_session_content": content_ok,
        "query_surface_answers_and_candidates": queries_ok,
        "mechanism_specific_roles": mechanisms_ok,
        "exact_rule_miss_and_semantic_oracle_state": states_ok,
        "source_disjoint_unique_five_token_values": (
            len(values) == len(set(values)) and not set(values) & formal_values
            and not set(values) & pilot_values and all(OPAQUE.fullmatch(value) for value in values)
            and tokenizer_ok),
        "frozen_alias_bank_hashes_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("alias_bank") == [list(row) for row in ALIAS_BANK]
            and manifest.get("alias_sha256") == sha(root / "alias.jsonl")
            and manifest.get("alias_states_sha256") == sha(root / "alias_states.jsonl")
            and manifest.get("model_scored") is False
            and manifest.get("controller_or_threshold_selected") is False
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status") == "frozen_semantic_alias_before_scoring")}
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "manifest_sha256": sha(root / "manifest.json"),
              "alias_sha256": sha(root / "alias.jsonl"),
              "alias_states_sha256": sha(root / "alias_states.jsonl"),
              "counts": counts, "model_scored": False, "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("semantic-alias construction audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--controlled-root", required=True)
    parser.add_argument("--formal-data-root", required=True); parser.add_argument("--pilot-data-root", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
