"""Audit the frozen post-V19 natural semantic-conflict corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tasks.build_post_v19_natural_semantic_headroom import (
    FIXED_POLICIES,
    FOLDS,
    FORMAL_PER_FOLD,
    LENGTHS,
    MECHANISMS,
    PILOT_PER_FOLD,
    PROTOCOL,
    QUERY_KINDS,
    allocate,
    build_episode,
    build_states,
    load_jsonl,
    normalized_surface,
    verify_inventory,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def exact_structural_checks(episode, state):
    sessions = episode["sessions"]
    identifiers = [row["session_id"] for row in sessions]
    current = sessions[-1]
    semantic_expected = [row["session_id"] for row in sessions[:-1]
                         if row["subject_normalized"] == episode["subject_normalized"]
                         and row["relation_id"] == episode["relation_id"]
                         and row["value"] != episode["current_value"]]
    exact_expected = [row["session_id"] for row in sessions[:-1]
                      if row["subject_normalized"] == episode["subject_normalized"]
                      and row["key_id"] == current["key_id"]
                      and row["value"] != episode["current_value"]]
    fixed = state["fixed"]
    checks = (
        len(sessions) == episode["length"],
        identifiers == [f"p19n-{episode['split']}-{episode['row_index']:03d}-s{i:03d}"
                        for i in range(len(sessions))],
        current["role"] == "current_update",
        current["session_id"] == episode["current_session_id"],
        current["value"] == episode["current_value"],
        normalized_surface(episode["old_surface"])
            != normalized_surface(episode["current_surface"]),
        exact_expected == state["exact_removed"] == [],
        semantic_expected == state["semantic_removed"],
        bool(semantic_expected),
        current["session_id"] not in semantic_expected,
        fixed["no_memory"] == [],
        fixed["retain_all"] == identifiers,
        fixed["global_reset"] == [current["session_id"]],
        fixed["last_1"] == [current["session_id"]],
        fixed["last_2"] == identifiers[-2:],
        state["special"]["exact_conflict"] == identifiers,
        state["special"]["semantic_oracle"]
            == [value for value in identifiers if value not in set(semantic_expected)],
        [row["kind"] for row in episode["queries"]] == list(QUERY_KINDS),
        all(row["answer"] in row["candidates"] for row in episode["queries"]),
        len({row["source_row_sha256"] for row in sessions
             if row["role"] in {"retain_1", "retain_2"}}) == 2,
        any(row["role"] == "same_relation_other_subject" for row in sessions),
    )
    return all(checks)


def audit(data_root, inventory_root, output):
    data_root, output = Path(data_root), Path(output)
    rows_path, dependencies = verify_inventory(inventory_root)
    source_rows = load_jsonl(rows_path)
    manifest = json.loads((data_root / "manifest.json").read_text())
    selected = allocate(source_rows, manifest["seed"])
    exact_files = True
    structures = True
    split_rows, split_subjects, observed_hashes = {}, {}, {}
    regenerated = {}
    for split, expected_per_fold in (("pilot", PILOT_PER_FOLD),
                                     ("formal", FORMAL_PER_FOLD)):
        episodes = load_jsonl(data_root / f"{split}.jsonl")
        states = load_jsonl(data_root / f"{split}_states.jsonl")
        expected_episodes = [build_episode(row, selected[split], split, index,
                                           manifest["seed"])
                             for index, row in enumerate(selected[split])]
        expected_states = [build_states(row) for row in expected_episodes]
        regenerated[split] = (episodes == expected_episodes and states == expected_states)
        exact_files &= (
            manifest["files_sha256"][f"{split}.jsonl"]
                == sha(data_root / f"{split}.jsonl")
            and manifest["files_sha256"][f"{split}_states.jsonl"]
                == sha(data_root / f"{split}_states.jsonl")
        )
        structures &= (
            len(episodes) == expected_per_fold * FOLDS
            and len(states) == len(episodes)
            and [row["episode_id"] for row in episodes]
                == [row["episode_id"] for row in states]
            and all(exact_structural_checks(episode, state)
                    for episode, state in zip(episodes, states))
            and Counter(row["fold"] for row in episodes)
                == Counter({fold: expected_per_fold for fold in range(FOLDS)})
            and set(row["length"] for row in episodes) == set(LENGTHS)
            and set(row["mechanism"] for row in episodes) == set(MECHANISMS)
        )
        split_rows[split] = episodes
        split_subjects[split] = {row["subject_normalized"] for row in episodes}
        observed_hashes[split] = {row["source_row_sha256"] for row in episodes}
    relation_fold = {}
    relation_disjoint = True
    for episode in split_rows["pilot"] + split_rows["formal"]:
        relation = episode["relation_id"]
        if relation in relation_fold and relation_fold[relation] != episode["fold"]:
            relation_disjoint = False
        relation_fold[relation] = episode["fold"]
    checks = {
        "protocol_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("fixed_policies") == list(FIXED_POLICIES)
            and manifest.get("query_kinds") == list(QUERY_KINDS)
            and manifest.get("model_scored") is False
            and manifest.get("controller_trained") is False
            and manifest.get("sealed_benchmarks_loaded") is False
            and manifest.get("terminal_status")
                == "frozen_post_v19_natural_data_before_model_scoring"
        ),
        "audited_inventory_and_hash_dependencies": (
            manifest.get("dependencies") == dependencies
        ),
        "deterministic_selection_and_exact_reconstruction": all(regenerated.values()),
        "file_hashes_exact": exact_files,
        "complete_balanced_split_coverage": structures,
        "pilot_formal_source_and_subject_disjoint": (
            not (observed_hashes["pilot"] & observed_hashes["formal"])
            and not (split_subjects["pilot"] & split_subjects["formal"])
        ),
        "relation_and_subject_fold_isolation": (
            relation_disjoint
            and all(len({row["fold"] for row in split_rows[split]
                         if row["subject_normalized"] == subject}) == 1
                    for split in ("pilot", "formal")
                    for subject in split_subjects[split])
        ),
        "exact_surface_and_semantic_oracle_states": structures,
        "manifest_separation_checks": all(manifest.get("checks", {}).values()),
    }
    result = {
        "protocol": "audit_post_v19_natural_semantic_headroom_v1",
        "checks": checks,
        "audit_passed": all(checks.values()),
        "manifest_sha256": sha(data_root / "manifest.json"),
        "files_sha256": {name: sha(data_root / name) for name in (
            "pilot.jsonl", "pilot_states.jsonl", "formal.jsonl", "formal_states.jsonl")},
        "pilot_rows": len(split_rows["pilot"]),
        "formal_rows": len(split_rows["formal"]),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--output", required=True)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
