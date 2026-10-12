"""Build a deterministic subject-disjoint ZsRE memory readout benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


SOURCE_SHA256 = "0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c"
SPLIT_SEED = "20261201"


def canonical(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def select_rows(rows):
    by_subject = {}
    for row in rows:
        subject = row["subject"].strip()
        if not subject or not row.get("prompt") or not row.get("rephrase_prompt"):
            raise ValueError("incomplete ZsRE identity fields")
        ground = row.get("ground_truth")
        if not isinstance(ground, list) or not ground or not ground[0].strip():
            raise ValueError("missing ZsRE ground truth")
        if not row.get("target_new", "").strip():
            raise ValueError("missing ZsRE target")
        by_subject.setdefault(subject, []).append(row)
    selected = []
    for subject, candidates in by_subject.items():
        row = min(candidates, key=lambda item: hashlib.sha256(canonical(item).encode()).hexdigest())
        locality = row.get("locality", {}).get("Relation_Specificity", [])
        if len(locality) != 2 or any(not item.get("prompt") for item in locality):
            raise ValueError("unexpected ZsRE locality structure")
        selected.append({
            "subject": subject, "prompt": row["prompt"].strip(),
            "rephrase_prompt": row["rephrase_prompt"].strip(),
            "original": row["ground_truth"][0].strip(),
            "updated": row["target_new"].strip(),
            "locality_prompts": [item["prompt"].strip() for item in locality],
            "source_row_sha256": hashlib.sha256(canonical(row).encode()).hexdigest(),
        })
    selected.sort(key=lambda row: hashlib.sha256(
        f"{SPLIT_SEED}|{row['subject']}".encode()).hexdigest())
    return selected


def split_rows(rows):
    if len(rows) != 1068:
        raise ValueError(f"expected 1068 unique subjects, got {len(rows)}")
    return {"calibration": rows[:267], "dev": rows[267:534], "test": rows[534:]}


def sessions(rows, split, width=8):
    output = []
    for offset in range(0, len(rows), width):
        facts = rows[offset:offset + width]
        identifier = hashlib.sha256(
            f"{SPLIT_SEED}|{split}|{offset}".encode()).hexdigest()[:16]
        output.append({"id": identifier, "split": split, "facts": facts})
    return output


def build(source, output):
    source, output = Path(source), Path(output)
    if sha(source) != SOURCE_SHA256:
        raise ValueError("unexpected ZsRE source hash")
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(source.read_text())
    split = split_rows(select_rows(rows))
    output.mkdir(parents=True)
    files, subjects = {}, {}
    for name, part in split.items():
        packed = sessions(part, name)
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packed))
        files[path.name] = sha(path)
        subjects[name] = sorted(row["subject"] for row in part)
    if any(set(subjects[a]) & set(subjects[b]) for a in subjects for b in subjects if a < b):
        raise ValueError("subject leakage")
    manifest = {
        "protocol": "zsre_contextual_readout_v1", "source": str(source),
        "source_sha256": SOURCE_SHA256, "split_seed": SPLIT_SEED,
        "selection": "one canonical row per subject", "session_width": 8,
        "subjects": {name: len(part) for name, part in subjects.items()},
        "sessions": {name: (len(part) + 7) // 8 for name, part in split.items()},
        "subject_hashes": {name: hashlib.sha256(json.dumps(part).encode()).hexdigest()
                           for name, part in subjects.items()},
        "files_sha256": files, "test_loaded": False, "test_scored": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
