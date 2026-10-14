"""Independently audit frozen MQuAKE rows before any V16 model score."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_outcome_v16"
FOLD_PROTOCOL = "mquake_robust_calibration_v11"
FOLDS = 5
LENGTHS = (16, 32, 48, 64)


def order_key(identifier):
    return hashlib.sha256(f"{PROTOCOL}|{identifier}".encode()).hexdigest()


def assign_folds(episodes):
    counts = Counter(row["primary_subject_hash"] for row in episodes)
    loads, assignment = [0] * FOLDS, {}
    ordered = sorted(counts.items(), key=lambda item: (
        -item[1], hashlib.sha256(f"{FOLD_PROTOCOL}|{item[0]}".encode()).hexdigest()))
    for subject, count in ordered:
        fold = min(range(FOLDS), key=lambda value: (loads[value], value))
        assignment[subject] = fold; loads[fold] += count
    return assignment, loads


def matched_hard(episode):
    return any(row["event_kind"] == "hard_retain" and row["hard_relation_matched"]
               for row in episode["sessions"])


def parse_text(text):
    prefix, marker = "Memory statement: ", "\nValue:"
    if not text.startswith(prefix) or marker not in text:
        raise ValueError("unexpected unit serialization")
    statement, value = text[len(prefix):].split(marker, 1)
    if not statement.strip() or not value.strip():
        raise ValueError("empty unit statement/value")
    return statement.strip(), value.strip()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def expected_ids(episodes, folds):
    pilot_cells, formal_cells = defaultdict(list), defaultdict(list)
    for row in episodes:
        if matched_hard(row):
            pilot_cells[folds[row["primary_subject_hash"]], row["length"]].append(row)
        formal_cells[row["primary_subject_hash"], row["length"]].append(row)
    pilot = []
    for cell in sorted((fold, length) for fold in range(FOLDS) for length in LENGTHS):
        pilot.extend(sorted(pilot_cells[cell], key=lambda row: order_key(row["episode_id"]))[:2])
    formal = []
    for cell in sorted(formal_cells):
        values = formal_cells[cell]; preferred = [row for row in values if matched_hard(row)]
        formal.append(min(preferred or values, key=lambda row: order_key(row["episode_id"])))
    return [row["episode_id"] for row in pilot], [row["episode_id"] for row in formal]


def validate_row(row, episode, texts, fold):
    sessions = episode["sessions"]
    old = next(value for value in sessions if value["event_kind"] == "primary_old")
    current = next(value for value in sessions if value["event_kind"] == "primary_update")
    query_step = sessions.index(current)
    if (row["episode_id"] != episode["episode_id"]
            or row["subject"] != episode["primary_subject_hash"]
            or row["length"] != episode["length"] or row["source_split"] != episode["split"]
            or row["fold"] != fold or row["query_step"] != query_step
            or row["primary_old_session_id"] != old["session_id"]
            or row["primary_current_session_id"] != current["session_id"]
            or current["target_session_id"] != old["session_id"]):
        raise ValueError("episode identity/query-time mismatch")
    visible = {value["session_id"]: value for value in sessions[:query_step]}
    if row["hard_retain_session_id"] not in visible \
            or row["unrelated_session_id"] not in visible:
        raise ValueError("future safety session leakage")
    hard = visible[row["hard_retain_session_id"]]
    unrelated = visible[row["unrelated_session_id"]]
    if hard["event_kind"] != "hard_retain" or unrelated["event_kind"] != "distractor" \
            or row["hard_relation_matched"] != bool(hard["hard_relation_matched"]):
        raise ValueError("safety type mismatch")
    units = {"old": old["units"][0], "current": current["units"][0],
             "hard_retain": hard["units"][0], "unrelated": unrelated["units"][0]}
    for name, identifier in units.items():
        statement, value = parse_text(texts[identifier])
        if row[name] != {"unit_id": identifier, "statement": statement, "value": value}:
            raise ValueError("anchor text mismatch")


def audit(data_root, sample_root, output):
    data_root, sample_root = Path(data_root), Path(sample_root)
    source_manifest = json.loads((data_root / "manifest.json").read_text())
    manifest = json.loads((sample_root / "manifest.json").read_text())
    episodes = load_jsonl(data_root / "train.jsonl") + load_jsonl(data_root / "tune.jsonl")
    texts = json.loads((data_root / "texts.json").read_text())
    folds, loads = assign_folds(episodes)
    expected_pilot, expected_formal = expected_ids(episodes, folds)
    pilot, formal = load_jsonl(sample_root / "pilot.jsonl"), load_jsonl(sample_root / "formal.jsonl")
    index = {row["episode_id"]: row for row in episodes}
    for row in pilot + formal:
        validate_row(row, index[row["episode_id"]], texts, folds[row["subject"]])
    checks = {
        "protocol": manifest["protocol"] == PROTOCOL,
        "source_protocol": source_manifest["protocol"] == manifest["source_protocol"]
                           == "mquake_temporal_sessions_v1",
        "source_manifest_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "source_file_hashes": all(source_manifest["files_sha256"][name]
                                  == manifest["source_files_sha256"][name]
                                  == sha(data_root / name)
                                  for name in ("train.jsonl", "tune.jsonl", "texts.json")),
        "fold_loads": manifest["fold_loads"] == loads,
        "pilot_selection": [row["episode_id"] for row in pilot] == expected_pilot,
        "formal_selection": [row["episode_id"] for row in formal] == expected_formal,
        "sizes": len(pilot) == manifest["pilot_rows"] == 40
                 and len(formal) == manifest["formal_rows"] == 272,
        "relation_counts": sum(row["hard_relation_matched"] for row in pilot)
                           == manifest["pilot_relation_matched_hard"] == 40
                           and sum(row["hard_relation_matched"] for row in formal)
                           == manifest["formal_relation_matched_hard"] == 252,
        "artifact_hashes": sha(sample_root / "pilot.jsonl") == manifest["pilot_sha256"]
                           and sha(sample_root / "formal.jsonl") == manifest["formal_sha256"],
        "pre_model_state": manifest["model_scored"] is False
                           and manifest["external_sets_loaded"] is False,
        "external_sealed": not any(manifest[key] for key in
                                   ("mquake_cf_used", "longmemeval_used", "locomo_used",
                                    "mmlu_used", "ruler_used")),
    }
    result = {"protocol": PROTOCOL, "audit_passed": all(checks.values()),
              "checks": checks, "pilot_rows": len(pilot), "formal_rows": len(formal),
              "subjects": len(folds), "fold_loads": loads,
              "sample_hashes": {name: sha(sample_root / name)
                                  for name in ("manifest.json", "pilot.jsonl", "formal.jsonl")},
              "model_scored": False, "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V16 data audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
