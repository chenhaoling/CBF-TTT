"""Independent integrity and aggregate audit for constrained scope V5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.select_qwen_natural_scope_constrained_v5 import PROTOCOL, select_record
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_residual_gate_v4 import end_to_end


def audit(root, repo, v2_root, v4_root):
    root, repo, v2_root, v4_root = map(Path, (root, repo, v2_root, v4_root))
    result = root / "result"
    summary = json.loads((result / "summary.json").read_text())
    rows = [json.loads(line) for line in (result / "oof_rows.jsonl").read_text().splitlines()
            if line]
    v2_rows = [json.loads(line) for line in
               (v2_root / "result" / "oof_rows.jsonl").read_text().splitlines() if line]
    v4_search = json.loads((v4_root / "result" / "search.json").read_text())
    selected = select_record(v4_search["records"])
    recomputed = end_to_end(rows, v2_rows)
    decision = {key: selected[key] for key in (
        "correct_top_recall", "natural_correct_top_examples",
        "natural_wrong_top_specificity", "natural_wrong_top_examples",
        "no_match_specificity", "no_match_examples", "balanced_accuracy")}
    gate = {
        "oof_correct_top_recall_at_least_point_92": decision["correct_top_recall"] >= .92,
        "oof_wrong_top_specificity_at_least_point_85":
            decision["natural_wrong_top_specificity"] >= .85,
        "oof_no_match_specificity_at_least_point_92": decision["no_match_specificity"] >= .92,
        "oof_balanced_accuracy_at_least_point_92": decision["balanced_accuracy"] >= .92,
        "candidate_accuracy_at_least_point_93": recomputed["candidate_accuracy"] >= .93,
        "oracle_gap_at_most_point_055": recomputed["oracle_gap"] <= .055,
        "final_candidate_accuracy_at_least_point_91": recomputed["final_candidate_accuracy"] >= .91,
        "final_generation_exact_at_least_point_74": recomputed["final_generation_exact"] >= .74,
        "gain_over_v2_ci_lower_above_zero": recomputed["gain_bootstrap_95ci"][0] > 0,
    }
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_natural_scope_constrained_v5_pass"
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "protocol_and_seals": summary["protocol"] == PROTOCOL
            and summary["calibration_scored"] and not summary["development_loaded"]
            and not summary["test_loaded"] and not summary["external_benchmarks_loaded"],
        "immutable_input_hashes": summary["v2_features_sha256"]
            == sha(v2_root / "result" / "calibration_features.pt")
            and summary["v2_rows_sha256"] == sha(v2_root / "result" / "oof_rows.jsonl")
            and summary["v4_search_sha256"] == sha(v4_root / "result" / "search.json")
            and summary["v4_summary_sha256"] == sha(v4_root / "result" / "summary.json"),
        "output_hashes": summary["oof_rows_sha256"] == sha(result / "oof_rows.jsonl")
            and summary["checkpoint_sha256"] == sha(result / "constrained_scope_gate.pt")
            and summary["plan_sha256"] == sha(repo / "QWEN_NATURAL_SCOPE_CONSTRAINED_V5_PLAN.md"),
        "constrained_selection": selected == summary["selected"]
            and decision == summary["decision_metrics"],
        "oof_coverage": len(rows) == 5144
            and len({(row["session_id"], row["phase"], row["surface"], row["group_id"],
                      row["chain_kind"]) for row in rows}) == len(rows),
        "independent_aggregate": recomputed == summary["end_to_end"],
        "independent_gate": gate == summary["gate"]["checks"]
            and summary["gate"]["passed"] == all(gate.values()),
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": recomputed,
            "recomputed_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "v2_root", "v4_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.repo, args.v2_root, args.v4_root
    ); write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
