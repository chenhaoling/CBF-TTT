"""Add same-subject independent locality facts to natural ZsRE update chains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.build_zsre_memory_benchmark import SOURCE_SHA256, canonical
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "zsre_hard_retain_forgetting_v1"
SEED = "zsre-hard-retain-v1"


def source_hash(row):
    return hashlib.sha256(canonical(row).encode()).hexdigest()


def normalize(values):
    return sorted(" ".join(value.strip().lower().split()) for value in values if value.strip())


def auxiliary(primary, row):
    locality = row.get("locality", {}).get("Relation_Specificity", [])
    if len(locality) != 2: raise ValueError("expected two locality aliases")
    left, right = normalize(locality[0].get("ground_truth", [])), normalize(locality[1].get("ground_truth", []))
    if not left or left != right: raise ValueError("locality aliases disagree")
    identifier = hashlib.sha256(f"{SEED}|{primary['id']}|aux".encode()).hexdigest()[:16]
    version = {"version": 0, "prompt": locality[0]["prompt"].strip(),
               "rephrase_prompt": locality[1]["prompt"].strip(),
               "target": locality[0]["ground_truth"][0].strip(),
               "source_row_sha256": source_hash(row)}
    return {"id": identifier, "split": primary["split"], "subject": primary["subject"],
            "kind": "auxiliary", "parent_chain_id": primary["id"], "versions": [version]}


def pack(primary_chains, split, width=4):
    output = []
    for offset in range(0, len(primary_chains), width):
        part = primary_chains[offset:offset + width]
        chains = [chain for pair in part for chain in pair]
        events = []
        for primary, aux in part:
            events.extend(({"chain_id": primary["id"], "version": 0, "action": "retain", "kind": "primary_initial"},
                           {"chain_id": aux["id"], "version": 0, "action": "retain", "kind": "hard_retain"}))
        maximum = max(len(primary["versions"]) for primary, _ in part)
        for version in range(1, maximum):
            for primary, _ in part:
                if version < len(primary["versions"]):
                    events.append({"chain_id": primary["id"], "version": version,
                                   "action": "local", "kind": "primary_update"})
        identifier = hashlib.sha256(f"{SEED}|{split}|{offset}".encode()).hexdigest()[:16]
        output.append({"id": identifier, "split": split, "chains": chains, "events": events})
    return output


def build(source, v1_root, output):
    source, v1_root, output = Path(source), Path(v1_root), Path(output)
    if sha(source) != SOURCE_SHA256: raise ValueError("unexpected source hash")
    if output.exists(): raise FileExistsError(output)
    rows = json.loads(source.read_text()); by_hash = {source_hash(row): row for row in rows}
    v1_manifest = json.loads((v1_root / "manifest.json").read_text())
    if v1_manifest["protocol"] != "zsre_natural_forgetting_v1": raise ValueError("v1 protocol mismatch")
    output.mkdir(parents=True); files, counts, subject_hashes = {}, {}, {}
    for split in ("calibration", "dev", "test"):
        path = v1_root / f"{split}.jsonl"
        if sha(path) != v1_manifest["files_sha256"][path.name]: raise ValueError("v1 split mismatch")
        sessions = [json.loads(line) for line in path.read_text().splitlines() if line]
        primaries = [chain for session in sessions for chain in session["chains"]]
        paired = []
        for chain in primaries:
            primary = json.loads(json.dumps(chain)); primary["kind"] = "primary"
            row = by_hash[primary["versions"][0]["source_row_sha256"]]
            paired.append((primary, auxiliary(primary, row)))
        packed = pack(paired, split); out_path = output / f"{split}.jsonl"
        out_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packed))
        files[out_path.name] = sha(out_path); subjects = sorted(chain[0]["subject"] for chain in paired)
        subject_hashes[split] = hashlib.sha256(json.dumps(subjects).encode()).hexdigest()
        counts[split] = {"subjects": len(subjects), "primary_chains": len(paired),
                         "auxiliary_chains": len(paired), "primary_versions": sum(
                             len(primary["versions"]) for primary, _ in paired),
                         "hard_retain_events": len(paired), "local_transitions": sum(
                             len(primary["versions"]) - 1 for primary, _ in paired),
                         "sessions": len(packed), "events": sum(len(row["events"]) for row in packed)}
    manifest = {"protocol": PROTOCOL, "source_sha256": SOURCE_SHA256,
                "v1_manifest_sha256": sha(v1_root / "manifest.json"), "seed": SEED,
                "primary_session_width": 4, "ideal_active_records": 8,
                "counts": counts, "subject_hashes": subject_hashes, "files_sha256": files,
                "construction_splits": ["calibration", "dev", "test"],
                "selection_used_heldout_labels": False, "test_scored": False}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--v1-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    print(json.dumps(build(args.source, args.v1_root, args.output), indent=2))


if __name__ == "__main__": main()
