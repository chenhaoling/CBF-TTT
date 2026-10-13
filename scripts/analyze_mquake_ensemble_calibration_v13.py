"""Read-only paired attribution of V12 and completed V13 OOF trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from scripts.analyze_mquake_robust_calibration_v11 import (
    quantiles,
    read_jsonl,
    summarize_outcomes,
)
from scripts.analyze_mquake_subject_risk_v12 import auc
from tasks.train_factorized_causal_forgetting_v4 import paired_bootstrap
from tasks.train_mquake_robust_calibration_v11 import subject_folds


def decision_outcome(event, decision):
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
    return {
        "candidate_matches_target": candidate_matches,
        "threshold_accepts": accepts,
        "target_deleted": target_deleted,
        "probability": decision["probability"],
        "outcome": outcome,
    }


def binary_transitions(left, right):
    counts = Counter(f"{int(a)}->{int(b)}" for a, b in zip(left, right))
    return {
        "count": len(left),
        "counts": {key: counts.get(key, 0) for key in ("0->0", "0->1", "1->0", "1->1")},
        "left_rate": statistics.fmean(left),
        "right_rate": statistics.fmean(right),
        "delta": statistics.fmean(right) - statistics.fmean(left),
    }


def diagnose(v12_root, v13_root, data_root):
    v12_root, v13_root, data_root = map(Path, (v12_root, v13_root, data_root))
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_by_id = {row["episode_id"]: row for row in episodes}
    folds, loads = subject_folds(episodes)
    v12 = read_jsonl(v12_root / "oof_controller.jsonl")
    v13 = read_jsonl(v13_root / "oof_controller.jsonl")
    v12_by_id = {row["episode_id"]: row for row in v12}
    v13_by_id = {row["episode_id"]: row for row in v13}
    if set(v12_by_id) != set(v13_by_id) or set(v13_by_id) != set(episode_by_id):
        raise ValueError("V12/V13/data episode identities differ")
    summary12 = json.loads((v12_root / "summary.json").read_text())
    summary13 = json.loads((v13_root / "summary.json").read_text())
    training13 = json.loads((v13_root / "training.json").read_text())

    primary12, primary13, paired_primary, decision_rows13 = [], [], [], []
    safety_pairs = defaultdict(lambda: ([], []))
    ordered_ids = [row["episode_id"] for row in v13]
    for identifier in ordered_ids:
        episode = episode_by_id[identifier]
        left, right = v12_by_id[identifier], v13_by_id[identifier]
        if left["subject"] != right["subject"] or left["length"] != right["length"]:
            raise ValueError(f"V12/V13 record mismatch: {identifier}")
        fold = folds[right["subject"]]
        for event, old, new in zip(episode["sessions"], left["decisions"], right["decisions"]):
            if event["event_kind"] != "primary_update":
                old_values, new_values = safety_pairs[event["event_kind"]]
                old_values.append(old["correct"])
                new_values.append(new["correct"])
            if new["step"]:
                decision_rows13.append({
                    "fold": fold,
                    "kind": event["event_kind"],
                    "label": int(event["event_kind"] == "primary_update"),
                    "probability": new["probability"],
                })
            if event["event_kind"] != "primary_update":
                continue
            old_outcome = decision_outcome(event, old)
            new_outcome = decision_outcome(event, new)
            common = {"episode_id": identifier, "subject": right["subject"],
                      "fold": fold, "length": right["length"]}
            primary12.append({**common, **old_outcome})
            primary13.append({**common, **new_outcome})
            paired_primary.append({
                **common,
                "v12_outcome": old_outcome["outcome"],
                "v13_outcome": new_outcome["outcome"],
            })

    pair_transition = binary_transitions(
        [v12_by_id[key]["pair_state_correct"] for key in ordered_ids],
        [v13_by_id[key]["pair_state_correct"] for key in ordered_ids],
    )
    stale_transition = binary_transitions(
        [v12_by_id[key]["stale_deleted"] for key in ordered_ids],
        [v13_by_id[key]["stale_deleted"] for key in ordered_ids],
    )
    current_transition = binary_transitions(
        [v12_by_id[key]["current_retained"] for key in ordered_ids],
        [v13_by_id[key]["current_retained"] for key in ordered_ids],
    )
    outcome_transitions = Counter(
        f"{row['v12_outcome']}->{row['v13_outcome']}" for row in paired_primary
    )
    ranking = {
        "pooled_auc": auc(
            [row["label"] for row in decision_rows13],
            [row["probability"] for row in decision_rows13],
        ),
        "by_fold": {},
    }
    for fold in range(5):
        rows = [row for row in decision_rows13 if row["fold"] == fold]
        ranking["by_fold"][str(fold)] = auc(
            [row["label"] for row in rows], [row["probability"] for row in rows]
        )
    return {
        "protocol": "mquake_ensemble_calibration_v13_diagnostic",
        "analysis_type": "read_only_paired_v12_v13_attribution",
        "no_model_calls": True,
        "no_threshold_selection": True,
        "v12_terminal_status": summary12["terminal_status"],
        "v13_terminal_status": summary13["terminal_status"],
        "fold_loads_recomputed": loads,
        "selected_final_thresholds": {
            fold: details["threshold"] for fold, details in training13["folds"].items()
        },
        "pair_state_transition": pair_transition,
        "stale_deletion_transition": stale_transition,
        "current_retention_transition": current_transition,
        "v13_minus_v12_pair_bootstrap": paired_bootstrap(v13, v12, 38201),
        "primary_update_v12": summarize_outcomes(primary12),
        "primary_update_v13": summarize_outcomes(primary13),
        "primary_outcome_transitions": dict(sorted(outcome_transitions.items())),
        "primary_update_v13_by_fold": {
            str(fold): summarize_outcomes([row for row in primary13 if row["fold"] == fold])
            for fold in range(5)
        },
        "safety_action_transitions": {
            kind: binary_transitions(left, right)
            for kind, (left, right) in sorted(safety_pairs.items())
        },
        "v13_gate_ranking": ranking,
        "v13_probability_by_event_kind": {
            kind: quantiles(row["probability"] for row in decision_rows13
                            if row["kind"] == kind)
            for kind in sorted({row["kind"] for row in decision_rows13})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-root", required=True)
    parser.add_argument("--v13-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args.v12_root, args.v13_root, args.data_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
