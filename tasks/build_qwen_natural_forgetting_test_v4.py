"""Conditionally materialize the unopened natural forgetting V4 test corpus."""

from __future__ import annotations

import json
from pathlib import Path

from tasks.build_qwen_natural_causal_memory_v1 import (
    CANDIDATES,
    EXPECTED_SOURCE,
    PROTOCOL as NATURAL_PROTOCOL,
    SOURCE_PROTOCOL,
    all_targets,
    canonicalize_targets,
    load,
    normalize,
    session_candidates,
)
from tasks.evaluate_qwen_natural_forgetting_consensus_v4 import PROTOCOL as CONSENSUS_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_forgetting_test_v4_corpus"
EXPECTED_TEST_SHA256 = "8c31c7a8e8268dd7281fa15525f07643c3ead1767d38f945b5e26c5eda8db63f"
EXPECTED = {"sessions": 129, "subjects": 516, "events": 1548}


def release_allowed(summary, audit, terminal_status, exit_code):
    return bool(
        summary.get("protocol") == CONSENSUS_PROTOCOL
        and summary.get("gate", {}).get("test_authorized")
        and summary.get("selected") is not None
        and audit.get("passed")
        and terminal_status == "completed_qwen_natural_forgetting_consensus_v4_test_release"
        and exit_code == "0"
        and not summary.get("development_loaded")
        and not summary.get("test_loaded")
        and not summary.get("external_benchmarks_loaded")
    )


def validate(rows):
    if len(rows) != EXPECTED["sessions"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unexpected test session count")
    subjects=set(); groups=set(); events=0
    for row in rows:
        if row["protocol"] != PROTOCOL or row["split"] != "test":
            raise ValueError("test protocol mismatch")
        if len(row["chains"]) not in (4,8): raise ValueError("test width changed")
        primary=[chain for chain in row["chains"] if chain["kind"]=="primary"]
        auxiliary=[chain for chain in row["chains"] if chain["kind"]=="auxiliary"]
        if len(primary)!=len(auxiliary) or len(primary) not in (2,4):
            raise ValueError("test pairing changed")
        if len(row["events"])!=3*len(primary): raise ValueError("test event count changed")
        kinds=["primary_initial","hard_retain"]*len(primary)+["primary_update"]*len(primary)
        if [event["kind"] for event in row["events"]]!=kinds:
            raise ValueError("test schedule changed")
        if len(row["candidates"])!=CANDIDATES \
                or len({normalize(value) for value in row["candidates"]})!=CANDIDATES:
            raise ValueError("test candidate count changed")
        targets={version["target"] for chain in row["chains"] for version in chain["versions"]}
        if not targets<=set(row["candidates"]) or row["candidate_construction_used_model_scores"]:
            raise ValueError("test candidates invalid")
        subjects.update(chain["subject"] for chain in primary)
        groups.update(chain["id"] for chain in primary); events+=len(row["events"])
    if len(subjects)!=EXPECTED["subjects"] or len(groups)!=EXPECTED["subjects"] \
            or events!=EXPECTED["events"]:
        raise ValueError("test subject/event coverage changed")
    return {"sessions":len(rows),"subjects":len(subjects),"events":events,
            "session_ids":{row["id"] for row in rows},"subjects_set":subjects,"groups":groups}


def materialize(source_root, existing_data_root, consensus_root, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    source_root, existing_data_root, consensus_root = map(
        Path,(source_root,existing_data_root,consensus_root))
    summary_path=consensus_root/"summary.json"; audit_path=consensus_root/"execution_audit.json"
    summary=json.loads(summary_path.read_text()); audit=json.loads(audit_path.read_text())
    if not release_allowed(summary,audit,(consensus_root/"terminal_status.txt").read_text().strip(),
                           (consensus_root/"runner_exit_code.txt").read_text().strip()):
        raise ValueError("V4 test release is not authorized")
    source_manifest_path=source_root/"manifest.json"; source_test=source_root/"test.jsonl"
    source_manifest=json.loads(source_manifest_path.read_text())
    if source_manifest.get("protocol")!=SOURCE_PROTOCOL or source_manifest.get("test_scored") \
            or sha(source_manifest_path)!=EXPECTED_SOURCE["manifest.json"] \
            or sha(source_test)!=EXPECTED_TEST_SHA256 \
            or source_manifest["files_sha256"].get("test.jsonl")!=EXPECTED_TEST_SHA256:
        raise ValueError("unexpected unopened source test")
    source_rows=[json.loads(line) for line in source_test.read_text().splitlines() if line]
    pool=all_targets(source_rows); rows=[]
    for session in source_rows:
        candidates=session_candidates(session,pool)
        chains,changes=canonicalize_targets(session,candidates)
        rows.append({"protocol":PROTOCOL,"source_protocol":NATURAL_PROTOCOL,
            "id":session["id"],"split":"test","chains":chains,"events":session["events"],
            "candidates":candidates,"target_surface_canonicalizations":changes,
            "candidate_construction_used_model_scores":False})
    test_audit=validate(rows)
    existing=[]
    for split in ("calibration","dev"):
        _,part=load(existing_data_root,split); existing.extend(part)
    existing_sessions={row["id"] for row in existing}
    existing_subjects={chain["subject"] for row in existing for chain in row["chains"]
                       if chain["kind"]=="primary"}
    existing_groups={chain["id"] for row in existing for chain in row["chains"]
                     if chain["kind"]=="primary"}
    if test_audit["session_ids"]&existing_sessions \
            or test_audit["subjects_set"]&existing_subjects \
            or test_audit["groups"]&existing_groups:
        raise ValueError("test overlaps calibration/development")
    output.mkdir(parents=True); path=output/"test.jsonl"
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
    manifest={"protocol":PROTOCOL,"source_protocol":SOURCE_PROTOCOL,
        "source_manifest_sha256":sha(source_manifest_path),
        "source_test_sha256":sha(source_test),"test_sha256":sha(path),
        "counts":{key:value for key,value in test_audit.items()
                  if key in ("sessions","subjects","events")},
        "selected_candidate":summary["selected"]["candidate"],
        "consensus_summary_sha256":sha(summary_path),
        "consensus_audit_sha256":sha(audit_path),
        "candidate_count":CANDIDATES,
        "target_surface_canonicalizations":sum(
            row["target_surface_canonicalizations"] for row in rows),
        "candidate_construction_used_model_scores":False,
        "session_overlap":0,"subject_overlap":0,"group_overlap":0,
        "test_loaded":True,"test_scored":False,"external_benchmarks_loaded":False}
    write_json(output/"manifest.json",manifest); return manifest


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source_root","existing_data_root","consensus_root","output"):
        parser.add_argument(f"--{name.replace('_','-')}",required=True)
    args=parser.parse_args(); print(json.dumps(materialize(
        args.source_root,args.existing_data_root,args.consensus_root,args.output),indent=2))


if __name__=="__main__": main()
