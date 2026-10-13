"""Diagnose the frozen V11 OOF trajectories without rescoring or retuning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


PROTOCOL = "mquake_robust_calibration_v11"
FOLDS = 5


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def subject_folds(episodes):
    counts = Counter(row["primary_subject_hash"] for row in episodes)
    loads = [0] * FOLDS
    assignment = {}
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            hashlib.sha256(f"{PROTOCOL}|{item[0]}".encode()).hexdigest(),
        ),
    )
    for subject, count in ordered:
        fold = min(range(FOLDS), key=lambda value: (loads[value], value))
        assignment[subject] = fold
        loads[fold] += count
    return assignment, loads


def quantiles(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None

    def at(fraction):
        position = fraction * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "min": values[0],
        "p10": at(.10),
        "p25": at(.25),
        "median": at(.50),
        "p75": at(.75),
        "p90": at(.90),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_outcomes(rows):
    counts = Counter(row["outcome"] for row in rows)
    total = len(rows)
    return {
        "episodes": total,
        "counts": dict(sorted(counts.items())),
        "rates": {key: ratio(value, total) for key, value in sorted(counts.items())},
        "candidate_matches_target": ratio(
            sum(row["candidate_matches_target"] for row in rows), total
        ),
        "threshold_accepts": ratio(sum(row["threshold_accepts"] for row in rows), total),
        "target_deleted": ratio(sum(row["target_deleted"] for row in rows), total),
        "primary_probability": quantiles(row["probability"] for row in rows),
        "probability_when_candidate_matches": quantiles(
            row["probability"] for row in rows if row["candidate_matches_target"]
        ),
        "probability_when_candidate_misses": quantiles(
            row["probability"] for row in rows if not row["candidate_matches_target"]
        ),
    }


def diagnose(root, data_root):
    root = Path(root)
    episodes = read_jsonl(Path(data_root) / "train.jsonl") + read_jsonl(
        Path(data_root) / "tune.jsonl"
    )
    episode_by_id = {row["episode_id"]: row for row in episodes}
    folds, loads = subject_folds(episodes)
    controller = read_jsonl(root / "oof_controller.jsonl")
    candidates = read_jsonl(root / "oof_candidate_rows.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())

    primary = []
    retain_probabilities = defaultdict(list)
    for record in controller:
        episode = episode_by_id[record["episode_id"]]
        fold = folds[record["subject"]]
        for event, decision in zip(episode["sessions"], record["decisions"]):
            if event["event_kind"] != "primary_update":
                retain_probabilities[event["event_kind"]].append(decision["probability"])
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
            primary.append(
                {
                    "episode_id": record["episode_id"],
                    "subject": record["subject"],
                    "fold": fold,
                    "length": record["length"],
                    "candidate_matches_target": candidate_matches,
                    "threshold_accepts": accepts,
                    "target_deleted": target_deleted,
                    "probability": decision["probability"],
                    "outcome": outcome,
                }
            )

    positive_candidates = [row for row in candidates if row["label"]]
    candidate_by_fold = {}
    for fold in range(FOLDS):
        rows = [row for row in positive_candidates if folds[row["subject"]] == fold]
        candidate_by_fold[str(fold)] = {
            "positives": len(rows),
            "shortlist_recall": statistics.fmean(row["shortlist_correct"] for row in rows),
            "top1": statistics.fmean(row["candidate_correct"] for row in rows),
        }
    candidate_subject_rows = defaultdict(list)
    for row in positive_candidates:
        candidate_subject_rows[row["subject"]].append(row)
    candidate_subject_summary = []
    for subject, rows in candidate_subject_rows.items():
        candidate_subject_summary.append(
            {
                "subject": subject,
                "fold": folds[subject],
                "episodes": len(rows),
                "shortlist_recall": statistics.fmean(
                    row["shortlist_correct"] for row in rows
                ),
                "top1": statistics.fmean(row["candidate_correct"] for row in rows),
            }
        )
    candidate_subject_summary.sort(
        key=lambda row: (row["top1"], row["shortlist_recall"], row["subject"])
    )

    by_fold = {
        str(fold): summarize_outcomes([row for row in primary if row["fold"] == fold])
        for fold in range(FOLDS)
    }
    by_length = {
        str(length): summarize_outcomes(
            [row for row in primary if row["length"] == length]
        )
        for length in sorted({row["length"] for row in primary})
    }
    subject_rows = defaultdict(list)
    for row in primary:
        subject_rows[row["subject"]].append(row)
    subject_summary = []
    for subject, rows in subject_rows.items():
        item = summarize_outcomes(rows)
        subject_summary.append(
            {
                "subject": subject,
                "fold": rows[0]["fold"],
                "episodes": item["episodes"],
                "target_deleted": item["target_deleted"],
                "candidate_matches_target": item["candidate_matches_target"],
                "threshold_accepts": item["threshold_accepts"],
            }
        )
    subject_summary.sort(key=lambda row: (row["target_deleted"], row["subject"]))

    thresholds = {
        fold: {
            seed: details["threshold"]
            for seed, details in fold_row["models"].items()
        }
        for fold, fold_row in training["folds"].items()
    }
    result = {
        "protocol": "mquake_robust_calibration_v11_diagnostic",
        "analysis_type": "read_only_posthoc_failure_attribution",
        "no_model_calls": True,
        "no_threshold_selection": True,
        "source_terminal_status": summary["terminal_status"],
        "source_gate_passed": summary["oof_gate"]["passed"],
        "fold_loads_recomputed": loads,
        "selected_member_thresholds": thresholds,
        "primary_update": summarize_outcomes(primary),
        "primary_update_by_fold": by_fold,
        "primary_update_by_length": by_length,
        "oracle_memory_candidate_by_fold": candidate_by_fold,
        "oracle_memory_candidate_by_subject": {
            "count": len(candidate_subject_summary),
            "imperfect_subjects": sum(
                row["top1"] < 1 for row in candidate_subject_summary
            ),
            "worst_10": candidate_subject_summary[:10],
        },
        "retain_probability_by_event_kind": {
            key: quantiles(values) for key, values in sorted(retain_probabilities.items())
        },
        "subjects": {
            "count": len(subject_summary),
            "zero_target_deletion": sum(
                row["target_deleted"] == 0 for row in subject_summary
            ),
            "perfect_target_deletion": sum(
                row["target_deleted"] == 1 for row in subject_summary
            ),
            "worst_10": subject_summary[:10],
            "best_10": subject_summary[-10:],
        },
    }
    return result


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
