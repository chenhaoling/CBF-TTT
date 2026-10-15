"""Independent integrity and aggregate audit for natural scope transfer V2."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_qwen_natural_scope_transfer_v2 import (
    PROTOCOL,
    STRATA,
    bootstrap_group,
)
from tasks.train_cbf_event_writer import write_json


def audit(root, repo):
    import torch

    root, repo = Path(root), Path(repo)
    result = root / "result"
    summary = json.loads((result / "summary.json").read_text())
    search = json.loads((result / "search.json").read_text())
    feature = torch.load(result / "calibration_features.pt", map_location="cpu",
                         weights_only=False)
    rows = [json.loads(line) for line in (result / "oof_rows.jsonl").read_text().splitlines()
            if line]
    metadata = feature["metadata"]
    selected = max(
        [row for row in search["records"] if row["no_match_specificity"] >= .92
         and row["natural_wrong_top_specificity"] >= .85],
        key=lambda row: (row["correct_top_recall"], row["no_match_specificity"],
                         row["natural_wrong_top_specificity"], row["balanced_accuracy"],
                         row["anchor"], -row["learning_rate"], -row["epochs"],
                         -abs(row["threshold"] - .5)),
    )
    strata = Counter(row["stratum"] for row in metadata)
    final = [row for row in rows if row["final"]]
    gains = {}
    for group in sorted({row["group_id"] for row in rows}):
        part = [row for row in rows if row["group_id"] == group]
        gains[group] = statistics.fmean(
            row["candidate_correct"] - row["v1_candidate_correct"] for row in part
        )
    recomputed = {
        "candidate_accuracy": statistics.fmean(row["candidate_correct"] for row in rows),
        "answer_nll": statistics.fmean(row["answer_nll"] for row in rows),
        "oracle_oracle_candidate_accuracy": statistics.fmean(
            row["oracle_candidate_correct"] for row in rows),
        "final_candidate_accuracy": statistics.fmean(
            row["candidate_correct"] for row in final),
        "final_generation_exact": statistics.fmean(
            row["generation_exact"] for row in final),
        "gain_over_v1_oracle_auto": statistics.fmean(gains.values()),
        "gain_bootstrap_95ci": bootstrap_group(gains),
        "subject_groups": len(gains),
        "score_sources": dict(sorted(Counter(row["score_source"] for row in rows).items())),
        "by_surface": {surface: statistics.fmean(
            row["candidate_correct"] for row in rows if row["surface"] == surface
        ) for surface in ("exact", "rephrase")},
    }
    recomputed["oracle_gap"] = (recomputed["oracle_oracle_candidate_accuracy"]
                                - recomputed["candidate_accuracy"])
    checks_gate = {
        "oof_correct_top_recall_at_least_point_90":
            summary["decision_metrics"]["correct_top_recall"] >= .90,
        "oof_no_match_specificity_at_least_point_92":
            summary["decision_metrics"]["no_match_specificity"] >= .92,
        "oof_wrong_top_specificity_at_least_point_85":
            summary["decision_metrics"]["natural_wrong_top_specificity"] >= .85,
        "oof_balanced_accuracy_at_least_point_91":
            summary["decision_metrics"]["balanced_accuracy"] >= .91,
        "candidate_accuracy_at_least_point_90": recomputed["candidate_accuracy"] >= .90,
        "oracle_gap_at_most_point_06": recomputed["oracle_gap"] <= .06,
        "final_candidate_accuracy_at_least_point_88":
            recomputed["final_candidate_accuracy"] >= .88,
        "final_generation_exact_at_least_point_70":
            recomputed["final_generation_exact"] >= .70,
        "gain_ci_lower_above_zero": recomputed["gain_bootstrap_95ci"][0] > 0,
    }
    checks = {
        "terminal_status_and_exit":
            (root / "terminal_status.txt").read_text().strip()
            == "completed_qwen_natural_scope_transfer_v2_calibration_stop"
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "protocol_and_seals": summary["protocol"] == PROTOCOL
            and summary["calibration_scored"]
            and not summary["development_loaded"]
            and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
        "artifact_hashes": summary["calibration_features_sha256"]
            == sha(result / "calibration_features.pt")
            and summary["oof_rows_sha256"] == sha(result / "oof_rows.jsonl")
            and summary["checkpoint_sha256"] == sha(result / "scope_transfer_gate.pt")
            and summary["plan_sha256"] == sha(repo / "QWEN_NATURAL_SCOPE_TRANSFER_V2_PLAN.md"),
        "feature_shapes_and_strata": feature["features"].shape[0] == len(metadata)
            == feature["labels"].shape[0]
            and set(strata) == set(STRATA)
            and dict(sorted(strata.items())) == summary["examples"],
        "search_selection": search["selected"] == selected == summary["selected"]
            and len(search["records"]) == 408,
        "oof_coverage": len(rows) == 5144
            and len({(row["session_id"], row["phase"], row["surface"], row["group_id"],
                      row["chain_kind"])
                     for row in rows}) == len(rows),
        "independent_aggregate": all(
            recomputed[key] == summary["end_to_end"][key] for key in recomputed),
        "independent_gate": checks_gate == summary["gate"]["checks"]
            and summary["gate"]["passed"] == all(checks_gate.values()),
    }
    output = {"protocol": PROTOCOL + "_audit", "checks": checks,
              "passed": all(checks.values()), "recomputed": recomputed,
              "recomputed_gate": checks_gate}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = audit(args.root, args.repo); write_json(Path(args.output), result)
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
