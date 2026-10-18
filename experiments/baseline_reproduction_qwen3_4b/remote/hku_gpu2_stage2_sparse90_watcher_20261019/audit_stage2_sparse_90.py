#!/usr/bin/env python3
import datetime as dt
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time

PROVENANCE = "/home/ctj/in_place_ttt_public_corpus_full_20261018.partial/stage2_128k.provenance.jsonl"
OUTPUT = "/home/ctj/in_place_ttt_public_corpus_full_run_20261018/stage2_sparse_90pct_audit.json"
BUILDER_PID = 1397987
START = 102990
STOP = 102996
POLL_SECONDS = 30
TIMEOUT_SECONDS = 10800


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def completed_line_count(path):
    result = subprocess.run(
        ["tail", "-n", "4", path], check=True, text=True, capture_output=True
    )
    for line in reversed(result.stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        index = row.get("stage_record_index")
        if isinstance(index, int):
            return index + 1
    return 0


def pid_live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def write_atomic(payload):
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, OUTPUT)


started = time.monotonic()
while True:
    count = completed_line_count(PROVENANCE)
    if count >= STOP:
        break
    if not pid_live(BUILDER_PID):
        write_atomic({
            "protocol": "stage2_sparse_90pct_provenance_audit_v1",
            "passed": False,
            "errors": ["builder_exited_before_fixed_boundary"],
            "observed_line_count": count,
            "created_at_utc": utc_now(),
        })
        sys.exit(1)
    if time.monotonic() - started > TIMEOUT_SECONDS:
        write_atomic({
            "protocol": "stage2_sparse_90pct_provenance_audit_v1",
            "passed": False,
            "errors": ["timeout_before_fixed_boundary"],
            "observed_line_count": count,
            "created_at_utc": utc_now(),
        })
        sys.exit(2)
    time.sleep(POLL_SECONDS)

with open(PROVENANCE, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in itertools.islice(handle, START, STOP)]

errors = []
expected_indices = list(range(START, STOP))
observed_indices = [row.get("stage_record_index") for row in rows]
if len(rows) != STOP - START:
    errors.append("row_count_mismatch")
if observed_indices != expected_indices:
    errors.append("stage_record_index_mismatch")

expected_sources = ["fineweb" if i % 2 == 0 else "longcrawl" for i in expected_indices]
observed_sources = [row.get("source") for row in rows]
if observed_sources != expected_sources:
    errors.append("source_alternation_mismatch")

expected_source_indices = [i // 2 for i in expected_indices]
observed_source_indices = [row.get("source_record_index") for row in rows]
if observed_source_indices != expected_source_indices:
    errors.append("source_record_index_pair_mismatch")

checks = {
    "exact_six_rows": len(rows) == 6,
    "stage_exact": all(row.get("stage") == "stage2_128k" for row in rows),
    "stage_record_indices_exact": observed_indices == expected_indices,
    "sources_strictly_alternating": observed_sources == expected_sources,
    "source_record_indices_paired": observed_source_indices == expected_source_indices,
    "content_token_count_exact": all(row.get("content_token_count") == 131071 for row in rows),
    "training_token_count_exact": all(row.get("training_token_count") == 131072 for row in rows),
    "raw_source_tokens_consumed_exact": all(row.get("raw_source_tokens_consumed") == 131071 for row in rows),
    "synthetic_eos_tokens_zero": all(row.get("synthetic_eos_tokens") == 0 for row in rows),
    "training_token_sha256_valid": all(
        isinstance(row.get("training_token_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", row["training_token_sha256"])
        for row in rows
    ),
    "spans_present": all(isinstance(row.get("spans"), list) and row["spans"] for row in rows),
}
errors.extend(name for name, passed in checks.items() if not passed)

script_sha256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
payload = {
    "protocol": "stage2_sparse_90pct_provenance_audit_v1",
    "fixed_before_boundary_observation": True,
    "builder_pid": BUILDER_PID,
    "provenance_path": PROVENANCE,
    "fixed_start_inclusive": START,
    "fixed_stop_exclusive": STOP,
    "observed_line_count": count,
    "observed_stage_record_indices": observed_indices,
    "observed_sources": observed_sources,
    "observed_source_record_indices": observed_source_indices,
    "observed_content_token_counts": [row.get("content_token_count") for row in rows],
    "observed_training_token_counts": [row.get("training_token_count") for row in rows],
    "observed_raw_source_tokens_consumed": [row.get("raw_source_tokens_consumed") for row in rows],
    "observed_synthetic_eos_tokens": [row.get("synthetic_eos_tokens") for row in rows],
    "observed_span_counts": [len(row.get("spans", [])) for row in rows],
    "checks": checks,
    "errors": sorted(set(errors)),
    "passed": not errors,
    "script_sha256": script_sha256,
    "created_at_utc": utc_now(),
}
write_atomic(payload)
sys.exit(0 if payload["passed"] else 3)
