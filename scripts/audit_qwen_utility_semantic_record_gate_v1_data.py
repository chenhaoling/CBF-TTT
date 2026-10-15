"""Independent audit for the LAMA/ParaRel utility pilot corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import (
    BOUNDARIES,
    GROUPS_PER_MECHANISM,
    MECHANISMS,
    PROTOCOL,
    TWINS,
    build,
    validate,
    write_jsonl,
)


AUDIT_PROTOCOL = "qwen_utility_semantic_record_gate_v1_pilot_data_audit"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def input_fields_are_clean(episodes: list[dict]) -> bool:
    forbidden = {"relation_id", "semantic_key", "source_uuid", "source_fact_hash", "split"}
    for episode in episodes:
        for event in episode["events"]:
            if forbidden & set(event["record"]["input"]):
                return False
        for boundary in episode["boundaries"]:
            for query in boundary["queries"]:
                if forbidden & set(query["input"]):
                    return False
                if "semantic_key" in query:
                    return False
    return True


def twin_structure_is_valid(episodes: list[dict]) -> bool:
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["source_group"]].append(episode)
    for rows in groups.values():
        if len(rows) != TWINS or {row["twin"] for row in rows} != {0, 1}:
            return False
        left, right = sorted(rows, key=lambda row: row["twin"])
        if (left["mechanism"], left["length"]) != (right["mechanism"], right["length"]):
            return False
        if len(left["events"]) != len(right["events"]):
            return False
        for index, (a, b) in enumerate(zip(left["events"], right["events"])):
            if (a["reset"], a["revoke_keys"]) != (b["reset"], b["revoke_keys"]):
                return False
            if index == 1:
                if a["record"]["input"]["value"] == b["record"]["input"]["value"]:
                    return False
                if a["record"]["input"]["subject"] != b["record"]["input"]["subject"]:
                    return False
            elif a["record"]["input"] != b["record"]["input"]:
                return False
    return True


def query_structure_is_valid(episodes: list[dict]) -> bool:
    roles = {"current_exact", "current_rephrase", "retained_1", "retained_2", "anchor"}
    for episode in episodes:
        if len(episode["boundaries"]) != BOUNDARIES:
            return False
        for boundary in episode["boundaries"]:
            queries = boundary["queries"]
            if {row["role"] for row in queries} != roles:
                return False
            if any(len(row["candidates"]) != 6 or row["answer"] not in row["candidates"]
                   for row in queries):
                return False
            if boundary["reset"]:
                absent = [row for row in queries if row["role"] in {
                    "retained_1", "retained_2", "anchor"
                }]
                if any(row["answer"] != "unknown" for row in absent):
                    return False
    return True


def audit(source_root: Path, inspection_path: Path, data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    episode_path = data_root / "pilot_episodes.jsonl"
    manifest = json.loads(manifest_path.read_text())
    episodes = load_jsonl(episode_path)
    expected_episodes, expected_manifest = build(source_root, inspection_path)
    validate(expected_episodes, expected_manifest)
    with tempfile.TemporaryDirectory() as directory:
        rebuilt_path = Path(directory) / "pilot_episodes.jsonl"
        write_jsonl(rebuilt_path, expected_episodes)
        rebuilt_hash = sha256(rebuilt_path)
    expected_manifest["files_sha256"] = {"pilot_episodes.jsonl": rebuilt_hash}

    relations = manifest["splits"]
    relation_sets = {key: set(value) for key, value in relations.items()}
    split_disjoint = not (
        relation_sets["train"] & relation_sets["development"]
        or relation_sets["train"] & relation_sets["confirmation"]
        or relation_sets["development"] & relation_sets["confirmation"]
    )
    counts = {
        "source_groups": len({row["source_group"] for row in episodes}),
        "episodes": len(episodes),
        "boundaries": sum(len(row["boundaries"]) for row in episodes),
        "queries": sum(len(boundary["queries"]) for row in episodes
                       for boundary in row["boundaries"]),
        "mechanisms": dict(Counter(row["mechanism"] for row in episodes)),
        "lengths": dict(Counter(str(row["length"]) for row in episodes)),
    }
    expected_groups = len(MECHANISMS) * GROUPS_PER_MECHANISM
    checks = {
        "protocol_and_source_hashes": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("source_inspection_sha256") == sha256(inspection_path)
        ),
        "exact_row_reconstruction": episodes == expected_episodes,
        "episode_file_hash": (
            sha256(episode_path) == manifest.get("files_sha256", {}).get(episode_path.name)
            == rebuilt_hash
        ),
        "manifest_reconstruction": manifest == expected_manifest,
        "registered_counts": (
            counts["source_groups"] == expected_groups
            and counts["episodes"] == expected_groups * TWINS
            and counts["boundaries"] == expected_groups * TWINS * BOUNDARIES
            and all(value == GROUPS_PER_MECHANISM * TWINS
                    for value in counts["mechanisms"].values())
        ),
        "relation_split_isolation": split_disjoint,
        "input_audit_separation": input_fields_are_clean(episodes),
        "twin_structure": twin_structure_is_valid(episodes),
        "causal_queries": query_structure_is_valid(episodes),
        "sealed_and_model_free": (
            not manifest.get("model_outputs_loaded")
            and not manifest.get("development_episodes_generated")
            and not manifest.get("confirmation_episodes_generated")
            and not manifest.get("forbidden_sources_loaded")
        ),
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": all(checks.values()),
        "counts": counts,
        "manifest_sha256": sha256(manifest_path),
        "episodes_sha256": sha256(episode_path),
        "source_inspection_sha256": sha256(inspection_path),
        "development_episodes_generated": False,
        "confirmation_episodes_generated": False,
        "model_outputs_loaded": False,
        "terminal_status": (
            "pilot_data_audit_passed" if all(checks.values()) else "pilot_data_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_root, args.inspection, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
