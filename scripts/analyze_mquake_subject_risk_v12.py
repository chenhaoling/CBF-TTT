"""Read-only failure attribution for completed MQuAKE subject-risk V12."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from scripts.analyze_mquake_robust_calibration_v11 import quantiles, read_jsonl, summarize_outcomes
from tasks.train_mquake_robust_calibration_v11 import subject_folds


def auc(labels, scores):
    """Mann-Whitney AUROC with average ranks for ties."""
    ordered = sorted(zip(scores, labels), key=lambda row: row[0])
    rank_sum = 0.; position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[position:end])
        position = end
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def diagnose(root, data_root):
    root, data_root = Path(root), Path(data_root)
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    folds, loads = subject_folds(episodes)
    controller = read_jsonl(root / "oof_controller.jsonl")
    candidates = read_jsonl(root / "oof_candidate_rows.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())

    primary, decision_rows = [], []
    for record in controller:
        episode = episode_by_id[record["episode_id"]]
        fold = folds[record["subject"]]
        for event, decision in zip(episode["sessions"], record["decisions"]):
            if decision["step"] == 0:
                continue
            label = int(event["event_kind"] == "primary_update")
            decision_rows.append({"fold": fold, "subject": record["subject"],
                                  "kind": event["event_kind"], "label": label,
                                  "probability": decision["probability"]})
            if not label:
                continue
            target = event["target_session_id"]
            candidate_matches = decision["top"] == target
            accepts = decision["deleted"] is not None
            target_deleted = decision["deleted"] == target
            if target_deleted:
                outcome = "correct_deletion"
            elif candidate_matches and not accepts:
                outcome = "correct_candidate_threshold_reject"
            elif accepts:
                outcome = "wrong_candidate_deleted"
            else:
                outcome = "wrong_candidate_threshold_reject"
            primary.append({"episode_id": record["episode_id"], "subject": record["subject"],
                            "fold": fold, "length": record["length"],
                            "candidate_matches_target": candidate_matches,
                            "threshold_accepts": accepts, "target_deleted": target_deleted,
                            "probability": decision["probability"], "outcome": outcome})

    positives = [row for row in candidates if row["label"]]
    candidate_by_fold = {}
    for fold in range(5):
        rows = [row for row in positives if folds[row["subject"]] == fold]
        candidate_by_fold[str(fold)] = {
            "episodes": len(rows),
            "shortlist_recall": statistics.fmean(row["shortlist_correct"] for row in rows),
            "top1": statistics.fmean(row["candidate_correct"] for row in rows),
        }
    probability_by_kind = {
        kind: quantiles(row["probability"] for row in decision_rows if row["kind"] == kind)
        for kind in sorted({row["kind"] for row in decision_rows})
    }
    ranking = {
        "pooled_auc": auc([row["label"] for row in decision_rows],
                          [row["probability"] for row in decision_rows]),
        "by_fold": {},
    }
    for fold in range(5):
        rows = [row for row in decision_rows if row["fold"] == fold]
        ranking["by_fold"][str(fold)] = auc(
            [row["label"] for row in rows], [row["probability"] for row in rows])

    thresholds = {
        fold: {seed: {"threshold": row["threshold"],
                      "feasible": row["calibration_feasible"]}
               for seed, row in details["models"].items()}
        for fold, details in training["folds"].items()
    }
    return {
        "protocol": "mquake_subject_risk_v12_diagnostic",
        "analysis_type": "read_only_posthoc_failure_attribution",
        "no_model_calls": True,
        "no_threshold_selection": True,
        "source_terminal_status": summary["terminal_status"],
        "source_gate_passed": summary["oof_gate"]["passed"],
        "fold_loads_recomputed": loads,
        "selected_thresholds": thresholds,
        "primary_update": summarize_outcomes(primary),
        "primary_update_by_fold": {
            str(fold): summarize_outcomes([row for row in primary if row["fold"] == fold])
            for fold in range(5)
        },
        "candidate_by_fold": candidate_by_fold,
        "gate_ranking": ranking,
        "probability_by_event_kind": probability_by_kind,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
