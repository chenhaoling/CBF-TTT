"""Build same-subject CounterFact sequences for zero-shot forgetting transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "counterfact_zero_shot_forgetting_v1"
SOURCE_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
SOURCE_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
SEED = "counterfact-zero-shot-forgetting-v1"


def normalize(value):
    return " ".join(value.strip().lower().split())


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def format_prompt(record):
    rewrite = record["requested_rewrite"]
    prompt = rewrite["prompt"].format(rewrite["subject"]).strip()
    paraphrases = [value.strip() for value in record.get("paraphrase_prompts", []) if value.strip()]
    if not prompt or not paraphrases:
        raise ValueError("CounterFact record lacks prompt or paraphrase")
    return prompt, paraphrases[0]


def valid_primary(record):
    rewrite = record.get("requested_rewrite", {})
    true = rewrite.get("target_true", {}).get("str", "")
    new = rewrite.get("target_new", {}).get("str", "")
    return bool(rewrite.get("subject") and rewrite.get("relation_id") and true.strip()
                and new.strip() and normalize(true) != normalize(new)
                and record.get("paraphrase_prompts"))


def valid_auxiliary(record):
    rewrite = record.get("requested_rewrite", {})
    true = rewrite.get("target_true", {}).get("str", "")
    return bool(rewrite.get("subject") and rewrite.get("relation_id")
                and true.strip() and record.get("paraphrase_prompts"))


def select_pairs(rows):
    by_subject = {}
    for row in rows:
        subject = row.get("requested_rewrite", {}).get("subject")
        if subject:
            by_subject.setdefault(subject, []).append(row)
    selected = []
    for subject, records in sorted(by_subject.items()):
        candidates = []
        for primary in records:
            if not valid_primary(primary):
                continue
            for auxiliary in records:
                if not valid_auxiliary(auxiliary):
                    continue
                p = primary["requested_rewrite"]; a = auxiliary["requested_rewrite"]
                if p["relation_id"] == a["relation_id"]:
                    continue
                key = hashlib.sha256(
                    f"{SEED}|{subject}|{primary['case_id']}|{auxiliary['case_id']}".encode()).hexdigest()
                candidates.append((key, primary, auxiliary))
        if candidates:
            _, primary, auxiliary = min(candidates, key=lambda item: item[0])
            selected.append((primary, auxiliary))
    return selected


def make_chains(primary_row, auxiliary_row):
    primary = primary_row["requested_rewrite"]; auxiliary = auxiliary_row["requested_rewrite"]
    if primary["subject"] != auxiliary["subject"] or primary["relation_id"] == auxiliary["relation_id"]:
        raise ValueError("pair must share subject and differ in relation")
    primary_prompt, primary_rephrase = format_prompt(primary_row)
    auxiliary_prompt, auxiliary_rephrase = format_prompt(auxiliary_row)
    subject = primary["subject"]
    identity = f"{subject}|{primary_row['case_id']}|{auxiliary_row['case_id']}"
    primary_id = hashlib.sha256(f"{SEED}|{identity}|primary".encode()).hexdigest()[:16]
    auxiliary_id = hashlib.sha256(f"{SEED}|{identity}|auxiliary".encode()).hexdigest()[:16]
    common = {"split": "transfer_test", "subject": subject}
    primary_chain = {"id": primary_id, **common, "kind": "primary", "versions": [
        {"version": 0, "prompt": primary_prompt, "rephrase_prompt": primary_rephrase,
         "target": primary["target_true"]["str"].strip(), "source_row_sha256": row_hash(primary_row)},
        {"version": 1, "prompt": primary_rephrase, "rephrase_prompt": primary_prompt,
         "target": primary["target_new"]["str"].strip(), "source_row_sha256": row_hash(primary_row)},
    ]}
    auxiliary_chain = {"id": auxiliary_id, **common, "kind": "auxiliary",
                       "parent_chain_id": primary_id, "versions": [{
                           "version": 0, "prompt": auxiliary_prompt,
                           "rephrase_prompt": auxiliary_rephrase,
                           "target": auxiliary["target_true"]["str"].strip(),
                           "source_row_sha256": row_hash(auxiliary_row)}]}
    return primary_chain, auxiliary_chain


def pack(pairs, width=4):
    sessions = []
    for offset in range(0, len(pairs), width):
        part = pairs[offset:offset + width]; events = []
        for primary, auxiliary in part:
            events.extend((
                {"chain_id": primary["id"], "version": 0, "action": "retain",
                 "kind": "primary_initial"},
                {"chain_id": auxiliary["id"], "version": 0, "action": "retain",
                 "kind": "hard_retain"},
            ))
        events.extend({"chain_id": primary["id"], "version": 1, "action": "local",
                       "kind": "primary_update"} for primary, _ in part)
        session_id = hashlib.sha256(f"{SEED}|session|{offset}".encode()).hexdigest()[:16]
        sessions.append({"id": session_id, "split": "transfer_test",
                         "chains": [chain for pair in part for chain in pair], "events": events})
    return sessions


def build(source, output):
    source, output = Path(source), Path(output)
    if sha(source) != SOURCE_SHA256:
        raise ValueError("unexpected CounterFact source hash")
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads(source.read_text()); pairs_raw = select_pairs(rows)
    valid_primary_rows = sum(valid_primary(row) for row in rows)
    valid_auxiliary_rows = sum(valid_auxiliary(row) for row in rows)
    by_subject = {}
    for row in rows:
        subject = row.get("requested_rewrite", {}).get("subject")
        if subject:
            by_subject.setdefault(subject, []).append(row)
    eligible_subjects = sum(any(
        valid_primary(primary) and valid_auxiliary(auxiliary)
        and primary["requested_rewrite"]["relation_id"]
        != auxiliary["requested_rewrite"]["relation_id"]
        for primary in records for auxiliary in records
    ) for records in by_subject.values())
    pairs = [make_chains(primary, auxiliary) for primary, auxiliary in pairs_raw]
    subjects = [pair[0]["subject"] for pair in pairs]
    if len(subjects) != len(set(subjects)):
        raise ValueError("selected CounterFact subjects are not unique")
    sessions = pack(pairs); output.mkdir(parents=True)
    path = output / "transfer_test.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sessions))
    selected_cases = sorted((primary["case_id"], auxiliary["case_id"])
                            for primary, auxiliary in pairs_raw)
    manifest = {
        "protocol": PROTOCOL,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_rows": len(rows),
        "valid_primary_rows": valid_primary_rows,
        "valid_auxiliary_rows": valid_auxiliary_rows,
        "subjects_with_valid_primary_auxiliary_pair": eligible_subjects,
        "eligible_subjects": len(pairs),
        "sessions": len(sessions),
        "events": 3 * len(pairs),
        "primary_chains": len(pairs),
        "auxiliary_chains": len(pairs),
        "primary_session_width": 4,
        "sequence": ["original_retain", "same_subject_hard_retain", "counterfactual_local"],
        "subject_hash": hashlib.sha256(json.dumps(sorted(subjects)).encode()).hexdigest(),
        "selected_case_pairs_hash": hashlib.sha256(json.dumps(selected_cases).encode()).hexdigest(),
        "files_sha256": {path.name: sha(path)},
        "controller_training_rows": 0,
        "counterfact_tuning_rows": 0,
        "selection_used_model_scores": False,
        "test_scored": False,
    }
    if manifest["eligible_subjects"] != manifest["subjects_with_valid_primary_auxiliary_pair"]:
        raise ValueError("eligible-subject accounting mismatch")
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
