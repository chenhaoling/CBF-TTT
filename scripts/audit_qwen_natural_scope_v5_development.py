"""Independent audit of the one-time natural scope V5 development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_qwen_natural_scope_v5_development import (
    ARMS,
    PROTOCOL,
    aggregate,
    build_trace,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_value_bridge_v1 import render_query
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import build_examples


def audit(root, repo, data_root, v5_root):
    root, repo, data_root, v5_root = map(Path, (root, repo, data_root, v5_root))
    result = root / "result"; summary = json.loads((result / "summary.json").read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(summary["tokenizer"], use_fast=True)
    rows = [json.loads(line) for line in
            (result / "development_rows.jsonl").read_text().splitlines() if line]
    negatives = [json.loads(line) for line in
                 (result / "development_no_match.jsonl").read_text().splitlines() if line]
    _, sessions = load(data_root, "dev")
    expected_rows, states = [], []
    for session in sessions:
        part, state = build_trace(session); expected_rows.extend(part); states.append(state)
    examples = build_examples(sessions, expected_rows, states)
    expected_negatives = [row for row in examples if row["kind"] == "no_match"]
    identity = ("session_id", "phase", "surface", "chain_id", "active_version",
                "query", "answer", "group_id", "chain_kind", "final", "candidates")
    row_reconstruction = len(rows) == len(expected_rows) == 5120 and all(
        all(observed[key] == expected[key] for key in identity)
        for observed, expected in zip(rows, expected_rows)
    )
    negative_reconstruction = len(negatives) == len(expected_negatives) == 640 and all(
        (observed["session_id"], observed["phase"], observed["surface"])
        == (expected["session_id"], expected["phase"], expected["surface"])
        for observed, expected in zip(negatives, expected_negatives)
    )
    score_checks = True
    for row in rows:
        for arm in ARMS:
            value = row["arms"][arm]
            prediction = min(row["candidates"], key=lambda candidate: (
                value["scores"][candidate], candidate))
            score_checks &= prediction == value["prediction"]
            score_checks &= value["candidate_correct"] == int(prediction == row["answer"])
            score_checks &= value["answer_nll"] == value["scores"][row["answer"]]
            if row["final"]:
                _, encoded = encode_candidates(
                    tokenizer, render_query(row["query"]), row["candidates"]
                )
                score_checks &= value["generation"]["token_exact"] == int(
                    value["generation"]["ids"] == encoded[row["answer"]]
                )
    decision_checks = all(
        row["v5_defer"] == int(row["v5_probability"] < summary["reader_threshold"])
        and row["v1_defer"] in (0, 1) for row in negatives
    )
    recomputed = aggregate(rows, negatives)
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_natural_scope_v5_development_pass"
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "protocol_and_seals": summary["protocol"] == PROTOCOL
            and summary["development_scored"] and not summary["calibration_loaded"]
            and not summary["test_loaded"] and not summary["external_benchmarks_loaded"],
        "row_reconstruction": row_reconstruction,
        "no_match_reconstruction": negative_reconstruction,
        "score_and_decision_consistency": score_checks and decision_checks,
        "artifact_hashes": summary["development_rows_sha256"]
            == sha(result / "development_rows.jsonl")
            and summary["development_no_match_sha256"]
            == sha(result / "development_no_match.jsonl")
            and summary["reader_checkpoint_sha256"]
            == sha(v5_root / "result" / "constrained_scope_gate.pt")
            and summary["plan_sha256"]
            == sha(repo / "QWEN_NATURAL_SCOPE_V5_DEVELOPMENT_PLAN.md"),
        "independent_aggregate_and_gate": all(
            recomputed[key] == summary[key] for key in recomputed),
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": recomputed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "data_root", "v5_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.repo, args.data_root, args.v5_root
    ); write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
