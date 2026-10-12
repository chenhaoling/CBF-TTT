"""Build deterministic natural-text sequential-update chains from duplicate ZsRE rows."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from tasks.build_zsre_memory_benchmark import SOURCE_SHA256, canonical
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "zsre_natural_forgetting_v1"
CHAIN_SEED = "zsre-natural-forgetting-v1"


def normalize(text):
    return " ".join(text.strip().lower().split())


def source_hash(row):
    return hashlib.sha256(canonical(row).encode()).hexdigest()


def ownership(data_root):
    data_root = Path(data_root); output = {}
    for split in ("calibration", "dev", "test"):
        rows = [json.loads(line) for line in (data_root / f"{split}.jsonl").read_text().splitlines() if line]
        output[split] = {fact["subject"]: fact["source_row_sha256"]
                         for session in rows for fact in session["facts"]}
    if any(set(output[left]) & set(output[right]) for left in output for right in output if left < right):
        raise ValueError("canonical subject leakage")
    return output


def build_chains(rows, split_subjects, split):
    by_subject = collections.defaultdict(list)
    for row in rows:
        if row["subject"] in split_subjects: by_subject[row["subject"]].append(row)
    chains = []
    for subject, subject_rows in by_subject.items():
        groups = collections.defaultdict(list)
        for row in subject_rows:
            key = tuple(sorted(normalize(value) for value in row["ground_truth"]))
            groups[key].append(row)
        for original_key, group in groups.items():
            by_target = collections.defaultdict(list)
            for row in group: by_target[normalize(row["target_new"])].append(row)
            if len(by_target) < 2: continue
            selected = [min(values, key=source_hash) for values in by_target.values()]
            canonical_sha = split_subjects[subject]
            selected.sort(key=lambda row: (source_hash(row) != canonical_sha, source_hash(row)))
            versions = []
            for index, row in enumerate(selected):
                locality = row.get("locality", {}).get("Relation_Specificity", [])
                if len(locality) != 2 or any(not item.get("prompt") for item in locality):
                    raise ValueError("unexpected locality structure")
                versions.append({"version": index, "prompt": row["prompt"].strip(),
                                 "rephrase_prompt": row["rephrase_prompt"].strip(),
                                 "target": row["target_new"].strip(),
                                 "locality_prompts": [item["prompt"].strip() for item in locality],
                                 "source_row_sha256": source_hash(row)})
            identifier = hashlib.sha256((CHAIN_SEED + "|" + split + "|" + subject + "|" +
                                         "|".join(original_key)).encode()).hexdigest()[:16]
            chains.append({"id": identifier, "split": split, "subject": subject,
                           "original_key": list(original_key), "versions": versions})
    chains.sort(key=lambda row: hashlib.sha256((CHAIN_SEED + "|" + row["id"]).encode()).hexdigest())
    return chains


def pack(chains, split, width=8):
    sessions = []
    for offset in range(0, len(chains), width):
        part = chains[offset:offset + width]
        identifier = hashlib.sha256(f"{CHAIN_SEED}|{split}|{offset}".encode()).hexdigest()[:16]
        events = []
        for version in range(max(len(chain["versions"]) for chain in part)):
            for chain in part:
                if version < len(chain["versions"]):
                    events.append({"chain_id": chain["id"], "version": version,
                                   "action": "retain" if version == 0 else "local"})
        sessions.append({"id": identifier, "split": split, "chains": part, "events": events})
    return sessions


def build(source, canonical_root, output):
    source, output = Path(source), Path(output)
    if sha(source) != SOURCE_SHA256: raise ValueError("unexpected ZsRE source hash")
    if output.exists(): raise FileExistsError(output)
    rows = json.loads(source.read_text()); owners = ownership(canonical_root)
    output.mkdir(parents=True); files, counts, subjects = {}, {}, {}
    for split in ("calibration", "dev", "test"):
        chains = build_chains(rows, owners[split], split); sessions = pack(chains, split)
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sessions))
        files[path.name] = sha(path); subjects[split] = sorted(chain["subject"] for chain in chains)
        counts[split] = {"subjects": len(subjects[split]), "chains": len(chains),
                         "versions": sum(len(chain["versions"]) for chain in chains),
                         "local_transitions": sum(len(chain["versions"]) - 1 for chain in chains),
                         "sessions": len(sessions), "events": sum(len(row["events"]) for row in sessions)}
    if any(set(subjects[left]) & set(subjects[right]) for left in subjects for right in subjects if left < right):
        raise ValueError("sequential subject leakage")
    manifest = {"protocol": PROTOCOL, "source": str(source), "source_sha256": SOURCE_SHA256,
                "canonical_manifest_sha256": sha(Path(canonical_root) / "manifest.json"),
                "chain_seed": CHAIN_SEED, "session_width": 8,
                "selection": "same-subject same-original distinct-target chains",
                "counts": counts, "subject_hashes": {split: hashlib.sha256(
                    json.dumps(values).encode()).hexdigest() for split, values in subjects.items()},
                "files_sha256": files, "dev_loaded": False, "test_loaded": False, "test_scored": False}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    print(json.dumps(build(args.source, args.canonical_root, args.output), indent=2))


if __name__ == "__main__": main()
