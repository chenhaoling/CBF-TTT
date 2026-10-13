"""Build subject-disjoint sparse long-session forgetting episodes from ZsRE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "long_session_sparse_corpus_v1"
LENGTHS = (16, 32, 48, 64)
UNITS_PER_SESSION = 4
CALIBRATION_FIT_SESSIONS = 27


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def text_id(text: str, table: dict) -> str:
    identifier = stable(text)[:20]
    if identifier in table and table[identifier] != text:
        raise RuntimeError("text hash collision")
    table[identifier] = text
    return identifier


def load_source(root, split):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != "zsre_contextual_readout_v1":
        raise ValueError("unexpected ZsRE contextual source")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("ZsRE source hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], path


def flatten(rows):
    result = []
    for session in rows:
        for index, fact in enumerate(session["facts"]):
            if len(fact["locality_prompts"]) < 2:
                raise ValueError("every source fact needs two locality prompts")
            result.append({**fact, "source_session_id": session["id"],
                           "fact_index": index,
                           "chain_id": fact["source_row_sha256"]})
    if len({row["chain_id"] for row in result}) != len(result):
        raise ValueError("duplicate source fact")
    return result


def fact_text(prompt, value=None):
    return f"Memory statement: {prompt}" + ("" if value is None else f"\nValue: {value}")


def ranked_pool(facts, primary, key):
    return sorted((row for row in facts if row["chain_id"] != primary["chain_id"]),
                  key=lambda row: stable(f"{PROTOCOL}|{key}|{row['chain_id']}"))


def build_episode(primary, facts, length, split, table):
    old_position, update_position = length // 4, 3 * length // 4
    hard_positions = {old_position + 1: 0, update_position - 1: 1}
    key = f"{split}|{length}|{primary['chain_id']}"; pool = ranked_pool(facts, primary, key)
    if not pool:
        raise ValueError("empty distractor pool")
    cursor = 0; sessions = []

    def next_fact():
        nonlocal cursor
        row = pool[cursor % len(pool)]; cursor += 1; return row

    for position in range(length):
        kind, desired, target = "distractor", "retain", None
        if position == old_position:
            anchor = fact_text(primary["prompt"], primary["original"]); kind = "primary_old"
        elif position == update_position:
            anchor = fact_text(primary["rephrase_prompt"], primary["updated"])
            kind, desired = "primary_update", "local"
            target = f"{stable(key)[:12]}-s{old_position:02d}"
        elif position in hard_positions:
            anchor = fact_text(primary["locality_prompts"][hard_positions[position]])
            kind = "hard_retain"
        else:
            row = next_fact(); anchor = fact_text(row["prompt"], row["original"])
        units = [text_id(anchor, table)]
        while len(units) < UNITS_PER_SESSION:
            row = next_fact(); units.append(text_id(fact_text(row["prompt"], row["original"]), table))
        session_id = f"{stable(key)[:12]}-s{position:02d}"
        sessions.append({"session_id": session_id, "units": units, "desired": desired,
                         "target_session_id": target, "event_kind": kind})
    if sum(row["desired"] == "local" for row in sessions) != 1:
        raise RuntimeError("episode must contain one update")
    return {"episode_id": stable(key)[:20], "split": split, "length": length,
            "primary_chain_id": primary["chain_id"],
            "primary_subject_hash": stable(primary["subject"]), "sessions": sessions}


def write_jsonl(path, rows):
    with Path(path).open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    calibration, calibration_path = load_source(args.source, "calibration")
    development, development_path = load_source(args.source, "dev")
    ordered = sorted(calibration, key=lambda row: stable(f"{PROTOCOL}|split|{row['id']}"))
    fit_ids = {row["id"] for row in ordered[:CALIBRATION_FIT_SESSIONS]}
    sources = {
        "train": flatten([row for row in calibration if row["id"] in fit_ids]),
        "tune": flatten([row for row in calibration if row["id"] not in fit_ids]),
        "dev": flatten(development),
    }
    table, files, counts = {}, {}, {}
    for split, facts in sources.items():
        episodes = [build_episode(primary, facts, length, split, table)
                    for primary in facts for length in LENGTHS]
        path = output / f"{split}.jsonl"; write_jsonl(path, episodes)
        files[path.name] = sha(path); counts[split] = {
            "source_facts": len(facts), "episodes": len(episodes),
            "events": sum(row["length"] for row in episodes),
            "updates": len(episodes), "source_sessions": len({row["source_session_id"] for row in facts})}
    text_path = output / "texts.json"; write_json(text_path, table); files[text_path.name] = sha(text_path)
    train_subjects = {row["subject"] for row in sources["train"]}
    tune_subjects = {row["subject"] for row in sources["tune"]}
    dev_subjects = {row["subject"] for row in sources["dev"]}
    if train_subjects & tune_subjects or (train_subjects | tune_subjects) & dev_subjects:
        raise ValueError("subject leakage across corpus splits")
    manifest = {"protocol": PROTOCOL, "lengths": list(LENGTHS),
                "units_per_session": UNITS_PER_SESSION,
                "calibration_fit_sessions": CALIBRATION_FIT_SESSIONS,
                "counts": counts, "unique_texts": len(table), "files_sha256": files,
                "source_calibration_sha256": sha(calibration_path),
                "source_dev_sha256": sha(development_path),
                "subject_disjoint": True, "longmemeval_used": False,
                "locomo_used": False, "chain_identity_model_feature": False}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
