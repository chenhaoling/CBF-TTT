"""Build calibration/development-only natural-value bridge episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_value_bridge_v1_corpus"
SOURCE_PROTOCOL = "zsre_contextual_readout_v1"
SOURCE_SHA256 = "0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c"
EXPECTED_FILE_SHA256 = {
    "calibration": "d19bb92ca6f487ce046e75a0a65e384b831a7bbc988e81436df68b3cd7beec6f",
    "dev": "352f4e5162eec2133707bcd45ff998ab44774536a426e27c311d2cd6c482a65f",
}
EXPECTED_COUNTS = {"calibration": 258, "dev": 256}
SPLITS = ("calibration", "dev")


def normalize(value):
    return " ".join(str(value).strip().casefold().split())


def identifier(split, fact):
    return hashlib.sha256(
        f"natural-value-v1|{split}|{fact['source_row_sha256']}".encode()
    ).hexdigest()[:20]


def load_source(root, split):
    if split not in SPLITS:
        raise ValueError("natural-value bridge cannot load sealed split")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{split}.jsonl"
    if (
        manifest.get("protocol") != SOURCE_PROTOCOL
        or manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("test_loaded")
        or manifest.get("test_scored")
        or sha(path) != EXPECTED_FILE_SHA256[split]
        or manifest["files_sha256"].get(path.name) != EXPECTED_FILE_SHA256[split]
    ):
        raise ValueError("unexpected or unsealed ZsRE contextual source")
    return manifest, [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]


def build_split(source_root, split):
    _, sessions = load_source(source_root, split)
    facts = [fact for session in sessions for fact in session["facts"]]
    facts = [fact for fact in facts if normalize(fact["original"]) != normalize(fact["updated"])]
    if len(facts) != EXPECTED_COUNTS[split]:
        raise ValueError("natural-value non-noop count changed")
    return episodes_from_facts(facts, split)


def episodes_from_facts(facts, split):
    if split not in SPLITS:
        raise ValueError("unknown natural-value split")
    pool = sorted({
        str(fact[field]).strip() for fact in facts for field in ("original", "updated")
        if str(fact[field]).strip()
    })
    rows = []
    for fact in facts:
        row_id = identifier(split, fact)
        correct, wrong = str(fact["updated"]).strip(), str(fact["original"]).strip()
        excluded = {normalize(correct), normalize(wrong)}
        eligible = [value for value in pool if normalize(value) not in excluded]
        eligible.sort(key=lambda value: hashlib.sha256(
            f"{row_id}|candidate|{value}".encode()
        ).hexdigest())
        candidates = [correct, wrong, *eligible[:4]]
        if len(candidates) != 6 or len({normalize(value) for value in candidates}) != 6:
            raise ValueError("natural-value candidate construction failed")
        candidates.sort(key=lambda value: hashlib.sha256(
            f"{row_id}|order|{value}".encode()
        ).hexdigest())
        rows.append({
            "protocol": PROTOCOL,
            "id": row_id,
            "split": split,
            "subject": fact["subject"],
            "source_row_sha256": fact["source_row_sha256"],
            "queries": {
                "exact": fact["prompt"],
                "rephrase": fact["rephrase_prompt"],
            },
            "correct": correct,
            "wrong": wrong,
            "candidates": candidates,
            "candidate_construction_used_model_scores": False,
        })
    rows.sort(key=lambda row: row["id"])
    return rows


def validate(rows_by_split):
    ids, subjects, sources = {}, {}, {}
    for split in SPLITS:
        rows = rows_by_split[split]
        if len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError("natural-value split size mismatch")
        ids[split] = {row["id"] for row in rows}
        subjects[split] = {normalize(row["subject"]) for row in rows}
        sources[split] = {row["source_row_sha256"] for row in rows}
        if len(ids[split]) != len(rows) or len(subjects[split]) != len(rows):
            raise ValueError("natural-value duplicate ID or subject")
        for row in rows:
            if (
                row["protocol"] != PROTOCOL or row["split"] != split
                or not row["queries"]["exact"] or not row["queries"]["rephrase"]
                or normalize(row["correct"]) == normalize(row["wrong"])
                or row["correct"] not in row["candidates"]
                or row["wrong"] not in row["candidates"]
                or len(row["candidates"]) != 6
                or len({normalize(value) for value in row["candidates"]}) != 6
                or row["candidate_construction_used_model_scores"]
            ):
                raise ValueError("invalid natural-value episode")
    if ids["calibration"] & ids["dev"] or subjects["calibration"] & subjects["dev"] \
            or sources["calibration"] & sources["dev"]:
        raise ValueError("natural-value calibration/development leakage")
    return {
        "counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "id_overlap": 0,
        "subject_overlap": 0,
        "source_row_overlap": 0,
        "causal_updated_values_verified": True,
        "model_free_candidates_verified": True,
        "test_file_loaded": False,
    }


def materialize(source_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = {split: build_split(source_root, split) for split in SPLITS}
    audit = validate(rows)
    output.mkdir(parents=True)
    paths = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[split]))
        paths[split] = path
    source_manifest = Path(source_root) / "manifest.json"
    manifest = {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source_root": str(Path(source_root).resolve()),
        "source_manifest_sha256": sha(source_manifest),
        "source_sha256": SOURCE_SHA256,
        "source_split_sha256": EXPECTED_FILE_SHA256,
        "files_sha256": {path.name: sha(path) for path in paths.values()},
        "counts": EXPECTED_COUNTS,
        "audit": audit,
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
        raise ValueError("natural-value bridge exposes calibration/dev only")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{split}.jsonl"
    if manifest.get("protocol") != PROTOCOL or sha(path) != manifest["files_sha256"].get(path.name):
        raise ValueError("natural-value corpus hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise ValueError("natural-value corpus size mismatch")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root, args.output), indent=2))


if __name__ == "__main__":
    main()
