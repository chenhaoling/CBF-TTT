"""Independent no-forward audit for the fixed candidate-ranking diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_cbf_dynamic_memory import audit_data, load_checkpoint
from tasks.evaluate_cbf_candidate_ranking import (
    ARMS, EXPOSURES, POLICIES, SETS, aggregate, build_candidate_sets, ranking, summarize,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_dynamic_memory import select_view
from tasks.train_cbf_event_writer import write_json


def audit(root: Path, source: Path, data_root: Path, formal_summary: Path) -> dict:
    rows, data_manifest, source_manifest = audit_data(data_root)
    selected, groups = select_view(rows, 128, 32)
    train_probe = groups["train"][:8]
    expected = [row for row in selected if row["split"] == "dev" or
                (row["split"] == "train" and row["group_id"] in train_probe)]
    expected_ids = {row["id"] for row in expected}
    by_id = {row["id"]: row for row in expected}
    reports = {}
    initial = {}
    for arm in ARMS:
        output = root / arm
        manifest = json.loads((output / "manifest.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        source_manifest_path = source / arm / "training_manifest.json"
        if manifest["protocol"] != "candidate_ranking_v1" or manifest["arm"] != arm:
            raise ValueError("candidate-ranking manifest mismatch")
        if manifest["data_sha256"] != data_manifest["data_sha256"] or manifest["test_scored"]:
            raise ValueError("candidate-ranking data/test invariant failed")
        if manifest["source_training_manifest_sha256"] != sha(source_manifest_path):
            raise ValueError("candidate-ranking source manifest changed")
        if complete["rows_per_checkpoint"] != len(expected) or not complete["source_checkpoints_unchanged"]:
            raise ValueError("candidate-ranking completion mismatch")
        checkpoints = {}
        for exposure in EXPOSURES:
            checkpoint = source / arm / f"writer_{exposure}.pt"
            if manifest["source_checkpoints_sha256"][str(exposure)] != sha(checkpoint):
                raise ValueError("candidate-ranking source checkpoint changed")
            saved = load_checkpoint(checkpoint)
            if exposure == 0:
                initial[arm] = saved["writer"]
            raw_path = output / f"ranking_{exposure}.jsonl"
            records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
            if len(records) != len(expected) or {row["id"] for row in records} != expected_ids:
                raise ValueError("candidate-ranking row matrix mismatch")
            for record in records:
                original = by_id[record["id"]]
                expected_sets = build_candidate_sets(selected, original)
                if record["candidate_sets"] != expected_sets or record["truth"] != original["answer"]:
                    raise ValueError("candidate-ranking set/label mismatch")
                if set(record["policies"]) != set(POLICIES):
                    raise ValueError("candidate-ranking policy matrix mismatch")
                for policy in POLICIES:
                    result = record["policies"][policy]
                    scores = result["scores"]
                    if [item["answer"] for item in scores] != expected_sets["global16"]:
                        raise ValueError("candidate-ranking score order mismatch")
                    if any(not math.isfinite(item[key]) for item in scores
                           for key in ("digit_nll", "answer_nll")):
                        raise ValueError("candidate-ranking nonfinite score")
                    recomputed = {name: ranking(scores, members, original["answer"])
                                  for name, members in expected_sets.items()}
                    if result["rankings"] != recomputed:
                        raise ValueError("candidate-ranking rank mismatch")
                    if any(not math.isfinite(result[key]) for key in
                           ("seconds", "peak_allocated_gib", "peak_reserved_gib")):
                        raise ValueError("candidate-ranking resource mismatch")
            stored = json.loads((output / f"ranking_{exposure}.summary.json").read_text())
            if stored["rows_sha256"] != sha(raw_path) or aggregate(records) != {
                key: stored[key] for key in ("train_probe", "dev")
            }:
                raise ValueError("candidate-ranking aggregate mismatch")
            if max(value for policy in stored["bridges"].values() for value in policy.values()) > 2e-5:
                raise ValueError("candidate-ranking bridge tolerance failed")
            checkpoints[str(exposure)] = {
                "rows": len(records), "rows_sha256": stored["rows_sha256"],
                "max_bridge_error": max(value for policy in stored["bridges"].values()
                                        for value in policy.values()),
            }
        reports[arm] = {
            "checkpoints": checkpoints, "peak_allocated_gib": complete["peak_allocated_gib"],
            "peak_reserved_gib": complete["peak_reserved_gib"],
        }
    if set(initial[ARMS[0]]) != set(initial[ARMS[1]]) or any(
        not initial[ARMS[0]][name].equal(initial[ARMS[1]][name]) for name in initial[ARMS[0]]
    ):
        raise ValueError("candidate-ranking initial writer tensors differ")
    stored_summary = json.loads((root / "summary.json").read_text())
    if summarize(root, formal_summary, save=False) != stored_summary:
        raise ValueError("candidate-ranking top-level summary mismatch")
    formal = json.loads(formal_summary.read_text())
    expected_fullkv = formal["arms"]["joint_binding"]["checkpoints"]["4096"]["dev"]["means"]["full_kv"]
    if stored_summary["audited_full_kv_control"]["source_sha256"] != sha(formal_summary) or \
       stored_summary["audited_full_kv_control"]["dev_code_correct"] != expected_fullkv["code_correct"]:
        raise ValueError("candidate-ranking Full-KV control mismatch")
    result = {
        "protocol": "candidate_ranking_v1_audit", "passed": True,
        "data": {"rows": len(rows), "data_sha256": data_manifest["data_sha256"],
                 "semantic_seed": source_manifest["seed"], "test_tokenized": False},
        "evaluated_rows_per_checkpoint": len(expected), "arms": reports,
        "primary_interpretation": stored_summary["primary_interpretation"], "test_scored": False,
    }
    write_json(root / "execution_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-summary", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.root), Path(args.source), Path(args.data_root),
                           Path(args.formal_summary)), indent=2))


if __name__ == "__main__":
    main()
