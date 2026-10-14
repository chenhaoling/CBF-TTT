#!/usr/bin/env python3
"""Attribute V14 cross-fold failures to relation support and margin shift.

Relation IDs are used only after OOF evaluation for diagnosis.  They are never
fed to the V14 controller or used to change its registered decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = probability * (len(values) - 1)
    left = math.floor(position); right = math.ceil(position)
    if left == right:
        return values[left]
    return values[left] * (right - position) + values[right] * (position - left)


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "q10": None, "q25": None,
                "median": None, "q75": None, "q90": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "q10": quantile(values, .10),
        "q25": quantile(values, .25),
        "median": quantile(values, .50),
        "q75": quantile(values, .75),
        "q90": quantile(values, .90),
    }


def rate(values):
    return statistics.fmean(values) if values else None


def relation_id(row):
    rewrites = row.get("requested_rewrite", [])
    if len(rewrites) != 1:
        raise ValueError("each MQuAKE-T case must have one rewrite")
    return str(rewrites[0]["relation_id"])


def case_id(episode):
    prefix = "mquake-t-"
    value = episode["primary_chain_id"]
    if not value.startswith(prefix):
        raise ValueError(f"unexpected chain ID: {value}")
    return value[len(prefix):]


def support_bin(value):
    if value == 0:
        return "0"
    if value < 20:
        return "1-19"
    if value < 100:
        return "20-99"
    if value < 500:
        return "100-499"
    return "500+"


def run(args):
    data_root = Path(args.data_root)
    result_root = Path(args.v14_root) / "result"
    manifest = json.loads((data_root / "manifest.json").read_text())
    source = Path(args.source)
    if sha(source) != manifest["source_sha256"]:
        raise ValueError("MQuAKE-T source hash differs from the dataset manifest")

    cases = json.loads(source.read_text())
    case_relations = {str(row["case_id"]): relation_id(row) for row in cases}
    if len(case_relations) != len(cases):
        raise ValueError("duplicate MQuAKE-T case ID")
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    folds, fold_loads = subject_folds(episodes)
    by_episode = {row["episode_id"]: row for row in episodes}
    if len(by_episode) != len(episodes):
        raise ValueError("duplicate episode ID")

    episode_relation = {}
    relation_counts = Counter()
    relation_fold_counts = defaultdict(Counter)
    for episode in episodes:
        relation = case_relations[case_id(episode)]
        episode_relation[episode["episode_id"]] = relation
        fold = folds[episode["primary_subject_hash"]]
        relation_counts[relation] += 1
        relation_fold_counts[relation][fold] += 1

    records = read_jsonl(result_root / "oof_controller.jsonl")
    if {row["episode_id"] for row in records} != set(by_episode):
        raise ValueError("V14 OOF episodes differ from train+tune episodes")

    rows = []
    relation_outcomes = defaultdict(list)
    for record in records:
        episode = by_episode[record["episode_id"]]
        fold = folds[record["subject"]]
        relation = episode_relation[record["episode_id"]]
        updates = [row for row in record["decisions"]
                   if row["event_kind"] == "primary_update"]
        if len(updates) != 1:
            raise ValueError("episode must contain one primary update decision")
        update = updates[0]
        event = episode["sessions"][update["step"]]
        retain = [row for row in record["decisions"]
                  if row["event_kind"] in {"distractor", "hard_retain"}
                  and row["history_size"] > 0]
        row = {
            "episode_id": record["episode_id"],
            "subject": record["subject"],
            "fold": fold,
            "relation": relation,
            "length": record["length"],
            "pair": record["pair_state_correct"],
            "stale_deleted": record["stale_deleted"],
            "current_retained": record["current_retained"],
            "update_margin": update["margin"],
            "update_score": update["score"],
            "update_baseline": update["baseline"],
            "threshold": update["threshold"],
            "candidate_correct": int(update["top"] == event["target_session_id"]),
            "target_deleted": int(update["deleted"] == event["target_session_id"]),
            "maximum_retain_margin": max((value["margin"] for value in retain), default=None),
            "false_retain_deletions": sum(value["deleted"] is not None for value in retain),
        }
        rows.append(row); relation_outcomes[relation].append(row)

    fold_rows = {}
    for fold in range(FOLDS):
        values = [row for row in rows if row["fold"] == fold]
        calibration_fold = (fold + 1) % FOLDS
        fit_folds = set(range(FOLDS)) - {fold, calibration_fold}
        fit_counts = Counter(
            episode_relation[row["episode_id"]] for row in episodes
            if folds[row["primary_subject_hash"]] in fit_folds
        )
        calibration_counts = Counter(
            episode_relation[row["episode_id"]] for row in episodes
            if folds[row["primary_subject_hash"]] == calibration_fold
        )
        bins = defaultdict(list)
        for row in values:
            bins[support_bin(fit_counts[row["relation"]])].append(row)
        fold_rows[str(fold)] = {
            "episodes": len(values),
            "subjects": len({row["subject"] for row in values}),
            "relations": len({row["relation"] for row in values}),
            "relations_unseen_in_fit": len({row["relation"] for row in values
                                             if fit_counts[row["relation"]] == 0}),
            "relations_unseen_in_calibration": len({row["relation"] for row in values
                                                     if calibration_counts[row["relation"]] == 0}),
            "episodes_unseen_relation_in_fit": sum(fit_counts[row["relation"]] == 0
                                                    for row in values),
            "episodes_unseen_relation_in_calibration": sum(
                calibration_counts[row["relation"]] == 0 for row in values),
            "pair": rate([row["pair"] for row in values]),
            "candidate_correct": rate([row["candidate_correct"] for row in values]),
            "target_deleted": rate([row["target_deleted"] for row in values]),
            "update_margin": distribution([row["update_margin"] for row in values]),
            "maximum_retain_margin": distribution([
                row["maximum_retain_margin"] for row in values
                if row["maximum_retain_margin"] is not None
            ]),
            "by_fit_relation_support": {
                key: {
                    "episodes": len(group),
                    "relations": len({row["relation"] for row in group}),
                    "pair": rate([row["pair"] for row in group]),
                    "candidate_correct": rate([row["candidate_correct"] for row in group]),
                    "update_margin": distribution([row["update_margin"] for row in group]),
                }
                for key, group in sorted(bins.items())
            },
        }

    relation_rows = {}
    for relation, values in sorted(relation_outcomes.items()):
        by_fold = defaultdict(list)
        for row in values:
            by_fold[row["fold"]].append(row)
        relation_rows[relation] = {
            "episodes": len(values),
            "subjects": len({row["subject"] for row in values}),
            "folds": len(by_fold),
            "pair": rate([row["pair"] for row in values]),
            "candidate_correct": rate([row["candidate_correct"] for row in values]),
            "update_margin": distribution([row["update_margin"] for row in values]),
            "by_fold": {
                str(fold): {"episodes": len(group),
                            "pair": rate([row["pair"] for row in group]),
                            "update_margin": distribution([
                                row["update_margin"] for row in group])}
                for fold, group in sorted(by_fold.items())
            },
        }

    # Diagnostic only: predict each episode from the same relation in other folds.
    cross_fold_predictions = []
    for row in rows:
        references = [other["pair"] for other in relation_outcomes[row["relation"]]
                      if other["fold"] != row["fold"]]
        if references:
            cross_fold_predictions.append((statistics.fmean(references), row["pair"]))
    if cross_fold_predictions:
        brier = statistics.fmean((prediction - target) ** 2
                                 for prediction, target in cross_fold_predictions)
        binary_accuracy = statistics.fmean(
            (prediction >= .5) == bool(target)
            for prediction, target in cross_fold_predictions
        )
    else:
        brier = binary_accuracy = None

    output = {
        "protocol": "mquake_online_margin_v14_relation_diagnostic",
        "diagnostic_only": True,
        "relation_ids_used_by_controller": False,
        "future_or_validation_labels_used_by_controller": False,
        "v14_decisions_changed": False,
        "data_source_sha256": sha(source),
        "v14_summary_sha256": sha(result_root / "summary.json"),
        "v14_training_sha256": sha(result_root / "training.json"),
        "fold_loads": fold_loads,
        "episodes": len(rows),
        "relations": len(relation_rows),
        "folds": fold_rows,
        "cross_fold_relation_prediction": {
            "covered_episodes": len(cross_fold_predictions),
            "coverage": len(cross_fold_predictions) / len(rows),
            "brier": brier,
            "binary_accuracy_at_half": binary_accuracy,
        },
        "relations_detail": relation_rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--v14-root", required=True)
    parser.add_argument("--output", required=True)
    result = run(parser.parse_args())
    compact = {key: result[key] for key in (
        "protocol", "episodes", "relations", "folds",
        "cross_fold_relation_prediction",
    )}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
