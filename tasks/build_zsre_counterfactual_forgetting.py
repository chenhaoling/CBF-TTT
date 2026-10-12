"""Build full subject-disjoint ZsRE counterfactual forgetting sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.build_zsre_memory_benchmark import SOURCE_SHA256, canonical
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "zsre_counterfactual_forgetting_v1"
SEED = "zsre-counterfactual-forgetting-v1"


def normalize(value):
    return " ".join(value.strip().lower().split())


def source_hash(row):
    return hashlib.sha256(canonical(row).encode()).hexdigest()


def chains(fact, source_row, split):
    if source_hash(source_row) != fact["source_row_sha256"]:
        raise ValueError("source row hash mismatch")
    if normalize(fact["original"]) == normalize(fact["updated"]):
        raise ValueError("counterfactual target must differ from original")
    locality = source_row.get("locality", {}).get("Relation_Specificity", [])
    if len(locality) != 2:
        raise ValueError("expected two locality aliases")
    answers = [sorted(normalize(value) for value in row.get("ground_truth", []) if value.strip())
               for row in locality]
    if not answers[0] or answers[0] != answers[1]:
        raise ValueError("locality aliases disagree")
    primary_id = hashlib.sha256(f"{SEED}|{split}|{fact['source_row_sha256']}|primary".encode()).hexdigest()[:16]
    auxiliary_id = hashlib.sha256(f"{SEED}|{split}|{fact['source_row_sha256']}|aux".encode()).hexdigest()[:16]
    primary = {"id": primary_id, "split": split, "subject": fact["subject"], "kind": "primary",
               "versions": [
                   {"version": 0, "prompt": fact["prompt"],
                    "rephrase_prompt": fact["rephrase_prompt"], "target": fact["original"],
                    "source_row_sha256": fact["source_row_sha256"]},
                   {"version": 1, "prompt": fact["rephrase_prompt"],
                    "rephrase_prompt": fact["prompt"], "target": fact["updated"],
                    "source_row_sha256": fact["source_row_sha256"]},
               ]}
    auxiliary = {"id": auxiliary_id, "split": split, "subject": fact["subject"],
                 "kind": "auxiliary", "parent_chain_id": primary_id, "versions": [{
                     "version": 0, "prompt": locality[0]["prompt"].strip(),
                     "rephrase_prompt": locality[1]["prompt"].strip(),
                     "target": locality[0]["ground_truth"][0].strip(),
                     "source_row_sha256": fact["source_row_sha256"]}]}
    return primary, auxiliary


def pack(pairs, split, width=4):
    output = []
    for offset in range(0, len(pairs), width):
        part = pairs[offset:offset + width]
        events = []
        for primary, auxiliary in part:
            events.extend((
                {"chain_id": primary["id"], "version": 0, "action": "retain",
                 "kind": "primary_initial"},
                {"chain_id": auxiliary["id"], "version": 0, "action": "retain",
                 "kind": "hard_retain"},
            ))
        events.extend({"chain_id": primary["id"], "version": 1, "action": "local",
                       "kind": "primary_update"} for primary, _ in part)
        identifier = hashlib.sha256(f"{SEED}|{split}|session|{offset}".encode()).hexdigest()[:16]
        output.append({"id": identifier, "split": split,
                       "chains": [chain for pair in part for chain in pair], "events": events})
    return output


def build(source, context_root, output):
    source, context_root, output = Path(source), Path(context_root), Path(output)
    if sha(source) != SOURCE_SHA256: raise ValueError("unexpected source hash")
    if output.exists(): raise FileExistsError(output)
    source_rows = json.loads(source.read_text()); by_hash = {source_hash(row): row for row in source_rows}
    context_manifest = json.loads((context_root / "manifest.json").read_text())
    if context_manifest["protocol"] != "zsre_contextual_readout_v1":
        raise ValueError("context protocol mismatch")
    output.mkdir(parents=True); files, counts, subject_hashes = {}, {}, {}
    for split in ("calibration", "dev", "test"):
        context_path = context_root / f"{split}.jsonl"
        if sha(context_path) != context_manifest["files_sha256"][context_path.name]:
            raise ValueError("context split hash mismatch")
        context_sessions = [json.loads(line) for line in context_path.read_text().splitlines() if line]
        facts = [fact for session in context_sessions for fact in session["facts"]]
        subjects = [fact["subject"] for fact in facts]
        if len(subjects) != len(set(subjects)):
            raise ValueError("context split must contain one fact per subject")
        pairs = [chains(fact, by_hash[fact["source_row_sha256"]], split) for fact in facts]
        sessions = pack(pairs, split); path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sessions))
        files[path.name] = sha(path)
        counts[split] = {"subjects": len(subjects), "primary_chains": len(pairs),
                         "auxiliary_chains": len(pairs), "primary_versions": 2 * len(pairs),
                         "hard_retain_events": len(pairs), "local_transitions": len(pairs),
                         "sessions": len(sessions), "events": 3 * len(pairs)}
        subject_hashes[split] = hashlib.sha256(json.dumps(sorted(subjects)).encode()).hexdigest()
    manifest = {"protocol": PROTOCOL, "source_sha256": SOURCE_SHA256,
                "context_manifest_sha256": sha(context_root / "manifest.json"), "seed": SEED,
                "primary_session_width": 4, "ideal_active_records": 8,
                "sequence": ["original_retain", "same_subject_hard_retain", "counterfactual_local"],
                "counts": counts, "subject_hashes": subject_hashes, "files_sha256": files,
                "construction_splits": ["calibration", "dev", "test"],
                "selection_used_heldout_labels": False, "test_scored": False}
    write_json(output / "manifest.json", manifest); return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--context-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    print(json.dumps(build(args.source, args.context_root, args.output), indent=2))


if __name__ == "__main__": main()
