#!/usr/bin/env python3
"""Screen causal score normalizers on frozen V14 OOF traces.

This is an architecture diagnostic, not an exact policy evaluation: changing a
threshold can change later memory contents and resolver outputs.  Every outer
fold threshold is selected from the other folds only.  A surviving normalizer
must still be rerun with exact causal replay in a new registered experiment.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds


FPR_TARGETS = (.0005, .001, .0025, .005)
SCALE_FLOOR = .1


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


def transforms(score, history):
    if not history:
        return {}
    median = statistics.median(history)
    margin = score - median
    mad = statistics.median(abs(value - median) for value in history)
    q25 = quantile(history, .25); q75 = quantile(history, .75)
    mean = statistics.fmean(history)
    std = statistics.pstdev(history) if len(history) > 1 else 0.
    below = sum(value < score for value in history)
    equal = sum(value == score for value in history)
    percentile = (below + .5 * equal) / len(history)
    return {
        "median_margin": margin,
        "max_gap": score - max(history),
        "mad_z": margin / max(SCALE_FLOOR, 1.4826 * mad),
        "iqr_z": margin / max(SCALE_FLOOR, q75 - q25),
        "mean_z": (score - mean) / max(SCALE_FLOOR, std),
        "upper_iqr": (score - q75) / max(SCALE_FLOOR, q75 - q25),
        "percentile": percentile,
    }


def threshold_at_fpr(negatives, target):
    values = sorted(negatives)
    if not values:
        raise ValueError("calibration negatives are empty")
    allowed = math.floor(target * len(values))
    if allowed == 0:
        return math.nextafter(values[-1], math.inf)
    index = len(values) - allowed
    return values[index]


def fraction_above(values, threshold):
    values = sorted(values)
    return (len(values) - bisect_left(values, threshold)) / len(values) if values else None


def run(args):
    data_root = Path(args.data_root)
    v14_root = Path(args.v14_root)
    train = read_jsonl(data_root / "train.jsonl")
    tune = read_jsonl(data_root / "tune.jsonl")
    episodes = train + tune
    folds, fold_loads = subject_folds(episodes)
    episode_map = {row["episode_id"]: row for row in episodes}
    records = read_jsonl(v14_root / "result" / "oof_controller.jsonl")
    if {row["episode_id"] for row in records} != set(episode_map):
        raise ValueError("V14 trace and MQuAKE episodes differ")

    feature_rows = []
    episode_counts = defaultdict(int)
    candidate_counts = defaultdict(int)
    for record in records:
        episode = episode_map[record["episode_id"]]
        fold = folds[record["subject"]]
        events = {index: event for index, event in enumerate(episode["sessions"])}
        history = []
        episode_counts[fold] += 1
        for decision in record["decisions"]:
            if decision["history_size"] != len(history):
                raise ValueError("V14 causal history size mismatch")
            # Step zero has no prior memory.  V14 emits a placeholder score but
            # deliberately does not add it to the causal score history.
            if decision["top"] is None:
                continue
            values = transforms(decision["score"], history)
            if not values:
                history.append(decision["score"])
                continue
            event = events[decision["step"]]
            executable = (event["desired"] == "local"
                          and decision["top"] == event["target_session_id"])
            if executable:
                candidate_counts[fold] += 1
            feature_rows.append({
                "fold": fold,
                "episode_id": record["episode_id"],
                "desired": event["desired"],
                "executable": executable,
                "features": values,
            })
            history.append(decision["score"])

    names = sorted(next(iter(feature_rows))["features"])
    output = {}
    for name in names:
        per_target = {}
        for target in FPR_TARGETS:
            folds_out = {}
            for fold in range(FOLDS):
                calibration = [row for row in feature_rows if row["fold"] != fold]
                validation = [row for row in feature_rows if row["fold"] == fold]
                calibration_negatives = [row["features"][name] for row in calibration
                                         if not row["executable"]]
                threshold = threshold_at_fpr(calibration_negatives, target)
                negatives = [row["features"][name] for row in validation
                             if not row["executable"]]
                positives = [row["features"][name] for row in validation
                             if row["executable"]]
                executable_recall = fraction_above(positives, threshold)
                pair_proxy = (sum(value >= threshold for value in positives)
                              / episode_counts[fold])
                folds_out[str(fold)] = {
                    "threshold": threshold,
                    "validation_negative_fpr": fraction_above(negatives, threshold),
                    "validation_executable_recall": executable_recall,
                    "validation_pair_proxy": pair_proxy,
                    "candidate_ceiling": candidate_counts[fold] / episode_counts[fold],
                    "episodes": episode_counts[fold],
                }
            pair_values = [row["validation_pair_proxy"] for row in folds_out.values()]
            fpr_values = [row["validation_negative_fpr"] for row in folds_out.values()]
            per_target[str(target)] = {
                "folds": folds_out,
                "macro_pair_proxy": statistics.fmean(pair_values),
                "minimum_fold_pair_proxy": min(pair_values),
                "macro_negative_fpr": statistics.fmean(fpr_values),
                "maximum_fold_negative_fpr": max(fpr_values),
            }
        output[name] = per_target

    result = {
        "protocol": "mquake_v14_causal_normalizer_fixed_trace_screen",
        "diagnostic_only": True,
        "exact_causal_replay": False,
        "threshold_uses_validation_fold": False,
        "v14_decisions_changed": False,
        "relation_ids_used": False,
        "position_or_length_features_used": False,
        "scale_floor": SCALE_FLOOR,
        "fpr_targets": list(FPR_TARGETS),
        "fold_loads": fold_loads,
        "v14_summary_sha256": sha(v14_root / "result" / "summary.json"),
        "features": output,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--v14-root", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
