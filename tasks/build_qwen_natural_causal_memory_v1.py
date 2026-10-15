"""Build calibration/development natural causal-memory composition episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_causal_memory_v1_corpus"
SOURCE_PROTOCOL = "zsre_counterfactual_forgetting_v1"
EXPECTED_SOURCE = {
    "manifest.json": "c5056824efa867532a33ff460976aaaf6d709ff1900acc562fc26c4b5ae1026f",
    "calibration.jsonl": "04bcc19bfe0524b16294eb529fd1c168148743cf58987039abfab99bf6b3243d",
    "dev.jsonl": "79f6c29b49981dabd1f90aee5fe0dce748d87fc6351efd29f8fface5dbb4301d",
}
EXPECTED = {
    "calibration": {"sessions": 65, "subjects": 258, "events": 774},
    "dev": {"sessions": 64, "subjects": 256, "events": 768},
}
SPLITS = ("calibration", "dev")
CANDIDATES = 12


def normalize(value):
    return " ".join(str(value).strip().casefold().split())


def load_source(root, split):
    if split not in SPLITS:
        raise ValueError("natural causal memory cannot load sealed split")
    root = Path(root)
    manifest_path, path = root / "manifest.json", root / f"{split}.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("protocol") != SOURCE_PROTOCOL
        or manifest.get("test_scored")
        or sha(manifest_path) != EXPECTED_SOURCE["manifest.json"]
        or sha(path) != EXPECTED_SOURCE[path.name]
        or manifest["files_sha256"].get(path.name) != EXPECTED_SOURCE[path.name]
    ):
        raise ValueError("unexpected ZsRE counterfactual source")
    return manifest, [json.loads(line) for line in path.read_text().splitlines() if line]


def all_targets(sessions):
    return sorted({
        str(version["target"]).strip()
        for session in sessions for chain in session["chains"]
        for version in chain["versions"] if str(version["target"]).strip()
    })


def session_candidates(session, pool):
    values = []
    for chain in session["chains"]:
        for version in chain["versions"]:
            value = str(version["target"]).strip()
            if normalize(value) not in {normalize(item) for item in values}:
                values.append(value)
    if len(values) > CANDIDATES:
        raise ValueError("session has more than twelve distinct natural values")
    existing = {normalize(value) for value in values}
    eligible = [value for value in pool if normalize(value) not in existing]
    eligible.sort(key=lambda value: hashlib.sha256(
        f"{session['id']}|fill|{value}".encode()
    ).hexdigest())
    values.extend(eligible[:CANDIDATES - len(values)])
    if len(values) != CANDIDATES or len({normalize(value) for value in values}) != CANDIDATES:
        raise ValueError("could not construct twelve unique natural candidates")
    values.sort(key=lambda value: hashlib.sha256(
        f"{session['id']}|order|{value}".encode()
    ).hexdigest())
    return values


def materialize_split(source_root, split):
    _, sessions = load_source(source_root, split)
    pool = all_targets(sessions)
    output = []
    for session in sessions:
        output.append({
            "protocol": PROTOCOL,
            "id": session["id"],
            "split": split,
            "chains": session["chains"],
            "events": session["events"],
            "candidates": session_candidates(session, pool),
            "candidate_construction_used_model_scores": False,
        })
    return output


def validate_split(rows, split):
    expected = EXPECTED[split]
    if len(rows) != expected["sessions"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unexpected natural causal-memory session count")
    subjects, groups, events = set(), set(), 0
    for row in rows:
        if row["protocol"] != PROTOCOL or row["split"] != split:
            raise ValueError("natural causal-memory protocol mismatch")
        if len(row["chains"]) not in (4, 8):
            raise ValueError("unexpected partial/full counterfactual session width")
        primary = [chain for chain in row["chains"] if chain["kind"] == "primary"]
        auxiliary = [chain for chain in row["chains"] if chain["kind"] == "auxiliary"]
        if len(primary) != len(auxiliary) or len(primary) not in (2, 4):
            raise ValueError("primary/auxiliary pairing changed")
        if len(row["events"]) != 3 * len(primary):
            raise ValueError("counterfactual event sequence changed")
        expected_kinds = ["primary_initial", "hard_retain"] * len(primary) \
            + ["primary_update"] * len(primary)
        if [event["kind"] for event in row["events"]] != expected_kinds:
            raise ValueError("counterfactual event ordering changed")
        if any(len(chain["versions"]) != (2 if chain["kind"] == "primary" else 1)
               for chain in row["chains"]):
            raise ValueError("counterfactual version cardinality changed")
        if len(row["candidates"]) != CANDIDATES \
                or len({normalize(value) for value in row["candidates"]}) != CANDIDATES:
            raise ValueError("natural candidate cardinality changed")
        targets = {normalize(version["target"]) for chain in row["chains"]
                   for version in chain["versions"]}
        if not targets <= {normalize(value) for value in row["candidates"]} \
                or row["candidate_construction_used_model_scores"]:
            raise ValueError("natural candidates omit a causal target")
        subjects.update(chain["subject"] for chain in primary)
        groups.update(chain["id"] for chain in primary)
        events += len(row["events"])
    if len(subjects) != expected["subjects"] or len(groups) != expected["subjects"] \
            or events != expected["events"]:
        raise ValueError("natural causal-memory subject/event counts changed")
    return {"sessions": len(rows), "subjects": len(subjects), "events": events,
            "session_ids": {row["id"] for row in rows}, "subjects_set": subjects,
            "groups": groups}


def materialize(source_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = {split: materialize_split(source_root, split) for split in SPLITS}
    audits = {split: validate_split(rows[split], split) for split in SPLITS}
    if audits["calibration"]["session_ids"] & audits["dev"]["session_ids"] \
            or audits["calibration"]["subjects_set"] & audits["dev"]["subjects_set"] \
            or audits["calibration"]["groups"] & audits["dev"]["groups"]:
        raise ValueError("calibration/development natural memory leakage")
    output.mkdir(parents=True)
    paths = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[split]))
        paths[split] = path
    manifest = {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source_root": str(Path(source_root).resolve()),
        "source_sha256": EXPECTED_SOURCE,
        "files_sha256": {path.name: sha(path) for path in paths.values()},
        "counts": {split: {key: value for key, value in audits[split].items()
                            if key in ("sessions", "subjects", "events")}
                   for split in SPLITS},
        "candidate_count": CANDIDATES,
        "candidate_construction_used_model_scores": False,
        "session_id_overlap": 0,
        "subject_overlap": 0,
        "group_overlap": 0,
        "calibration_scored": False,
        "development_scored": False,
        "test_loaded": False,
        "test_scored": False,
        "external_benchmarks_loaded": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load(root, split):
    if split not in SPLITS:
        raise ValueError("natural causal memory exposes calibration/dev only")
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{split}.jsonl"
    if manifest.get("protocol") != PROTOCOL \
            or sha(path) != manifest["files_sha256"].get(path.name):
        raise ValueError("natural causal-memory corpus hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    validate_split(rows, split)
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root, args.output), indent=2))


if __name__ == "__main__":
    main()
