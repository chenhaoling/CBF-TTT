"""Freeze subject-balanced MQuAKE rows for post-V15 frozen-Qwen scoring."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_factorized_causal_forgetting_v4 import payload
from tasks.train_factorized_scope_candidate_v3 import scope_key
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_qwen_outcome_v16"
LENGTHS = (16, 32, 48, 64)
PILOT_PER_CELL = 2


def order_key(episode_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}|{episode_id}".encode()).hexdigest()


def event(episode: dict, kind: str) -> dict:
    rows = [row for row in episode["sessions"] if row["event_kind"] == kind]
    if len(rows) != 1:
        raise ValueError(f"episode must contain one {kind} event")
    return rows[0]


def query_unit(identifier: str, texts: dict[str, str]) -> dict:
    text = texts[identifier]
    return {"unit_id": identifier, "statement": scope_key(text), "value": payload(text)}


def identity(episode: dict, fold: int, texts: dict[str, str]) -> dict:
    old, current = event(episode, "primary_old"), event(episode, "primary_update")
    if current["target_session_id"] != old["session_id"]:
        raise ValueError("primary update does not target the primary old session")
    hard = [row for row in episode["sessions"]
            if row["event_kind"] == "hard_retain" and row["hard_relation_matched"]]
    unrelated = [row for row in episode["sessions"] if row["event_kind"] == "distractor"]
    if not hard or not unrelated:
        raise ValueError("episode lacks required safety sessions")
    # Corpus construction places the event anchor at units[0].  Sorting only
    # breaks ties between the two predefined hard-retain sessions/distractors.
    hard_row = min(hard, key=lambda row: order_key(row["session_id"]))
    unrelated_row = min(unrelated, key=lambda row: order_key(row["session_id"]))
    result = {
        "episode_id": episode["episode_id"], "subject": episode["primary_subject_hash"],
        "fold": fold, "length": episode["length"], "source_split": episode["split"],
        "primary_old_session_id": old["session_id"],
        "primary_current_session_id": current["session_id"],
        "old": query_unit(old["units"][0], texts),
        "current": query_unit(current["units"][0], texts),
        "hard_retain_session_id": hard_row["session_id"],
        "hard_retain": query_unit(hard_row["units"][0], texts),
        "unrelated_session_id": unrelated_row["session_id"],
        "unrelated": query_unit(unrelated_row["units"][0], texts),
    }
    if not all(result[name]["value"] for name in ("old", "current", "hard_retain", "unrelated")):
        raise ValueError("selected query unit has no value")
    return result


def select_pilot(episodes: list[dict], folds: dict[str, int]) -> list[dict]:
    cells = defaultdict(list)
    for row in episodes:
        cells[folds[row["primary_subject_hash"]], row["length"]].append(row)
    expected = {(fold, length) for fold in range(FOLDS) for length in LENGTHS}
    if set(cells) != expected or any(len(rows) < PILOT_PER_CELL for rows in cells.values()):
        raise ValueError("pilot fold/length cells are incomplete")
    selected = []
    for cell in sorted(expected):
        selected.extend(sorted(cells[cell], key=lambda row: order_key(row["episode_id"]))
                        [:PILOT_PER_CELL])
    return selected


def select_formal(episodes: list[dict]) -> list[dict]:
    cells = defaultdict(list)
    for row in episodes:
        cells[row["primary_subject_hash"], row["length"]].append(row)
    subjects = {row["primary_subject_hash"] for row in episodes}
    expected = {(subject, length) for subject in subjects for length in LENGTHS}
    if set(cells) != expected:
        raise ValueError("formal subject/length cells are incomplete")
    return [min(cells[cell], key=lambda row: order_key(row["episode_id"]))
            for cell in sorted(expected)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(data_root, output):
    data_root, output = Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, manifest = load_split(data_root, "train")
    tune, _ = load_split(data_root, "tune")
    episodes = train + tune
    if manifest["protocol"] != "mquake_temporal_sessions_v1" \
            or manifest["mquake_cf_used"] or manifest["longmemeval_used"] \
            or manifest["locomo_used"]:
        raise ValueError("unexpected MQuAKE source or external benchmark access")
    texts = load_texts(data_root, manifest)
    folds, fold_loads = subject_folds(episodes)
    pilot_source = select_pilot(episodes, folds)
    formal_source = select_formal(episodes)
    pilot = [identity(row, folds[row["primary_subject_hash"]], texts)
             for row in pilot_source]
    formal = [identity(row, folds[row["primary_subject_hash"]], texts)
              for row in formal_source]
    pilot_path, formal_path = output / "pilot.jsonl", output / "formal.jsonl"
    write_jsonl(pilot_path, pilot); write_jsonl(formal_path, formal)
    subjects = sorted(folds)
    result = {
        "protocol": PROTOCOL,
        "source_protocol": manifest["protocol"],
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "source_files_sha256": {
            name: manifest["files_sha256"][name]
            for name in ("train.jsonl", "tune.jsonl", "texts.json")
        },
        "selection_hash": "sha256(protocol|episode_id)",
        "pilot_rule": "two minimum-hash episodes per fold x length cell",
        "formal_rule": "one minimum-hash episode per subject x length cell",
        "folds": FOLDS, "lengths": list(LENGTHS), "fold_loads": fold_loads,
        "subjects": len(subjects), "pilot_rows": len(pilot), "formal_rows": len(formal),
        "pilot_sha256": sha(pilot_path), "formal_sha256": sha(formal_path),
        "external_sets_loaded": False, "model_scored": False,
        "mquake_cf_used": False, "longmemeval_used": False,
        "locomo_used": False, "mmlu_used": False, "ruler_used": False,
        "terminal_status": "frozen_pre_v15_outer_result",
    }
    if len(pilot) != FOLDS * len(LENGTHS) * PILOT_PER_CELL:
        raise RuntimeError("unexpected pilot size")
    if len(formal) != len(subjects) * len(LENGTHS):
        raise RuntimeError("unexpected formal size")
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
