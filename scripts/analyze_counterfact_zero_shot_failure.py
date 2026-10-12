"""Diagnose an observed CounterFact zero-shot run without fitting a new method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS


PROTOCOL = "counterfact_zero_shot_failure_diagnostic_v1"


def quantile(values, fraction):
    values = sorted(values)
    return values[round(fraction * (len(values) - 1))]


def event_summary(rows):
    probabilities = [row["controller_probability"] for row in rows]
    predicted = [row for row in rows if row["predicted"] == "local"]
    return {
        "events": len(rows),
        "mean_probability": statistics.fmean(probabilities),
        "probability_quantiles": {
            "p10": quantile(probabilities, .10),
            "p25": quantile(probabilities, .25),
            "p50": quantile(probabilities, .50),
            "p75": quantile(probabilities, .75),
            "p90": quantile(probabilities, .90),
        },
        "predicted_local_rate": len(predicted) / len(rows),
        "action_accuracy": statistics.fmean(row["correct"] for row in rows),
        "triggered_candidate_accuracy": (
            statistics.fmean(row["correct"] for row in predicted) if predicted else None),
    }


def auc(positive, negative):
    wins = sum((left > right) + .5 * (left == right)
               for left in positive for right in negative)
    return wins / (len(positive) * len(negative))


def analyze(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    records = {}
    hashes_verified = {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"transfer_test_controller_seed{seed}.jsonl"
        hashes_verified[str(seed)] = training["raw_rows_sha256"][path.name] == sha(path)
        records[str(seed)] = [json.loads(line) for line in path.read_text().splitlines() if line]
    reference = records[str(CONTROLLER_SEEDS[0])]
    decisions_identical = all(
        [row["decisions"] for row in records[str(seed)]]
        == [row["decisions"] for row in reference]
        for seed in CONTROLLER_SEEDS[1:])
    decisions = [row for record in reference for row in record["decisions"]]
    kinds = ("primary_initial", "hard_retain", "primary_update")
    by_kind = {kind: event_summary(
        [row for row in decisions if row["event_kind"] == kind]) for kind in kinds}

    positive = [row["controller_probability"] for row in decisions
                if row["desired"] == "local"]
    # Step zero has no candidate and is structurally forced to retain, so it is
    # excluded from the score-separability diagnostic.
    negative = [row["controller_probability"] for row in decisions
                if row["desired"] == "retain" and row["step"] > 0]
    grid = {}
    scored = [row for row in decisions if row["step"] > 0]
    for index in range(1, 20):
        threshold = index / 20
        predicted_local = [row["controller_probability"] >= threshold for row in scored]
        desired_local = [row["desired"] == "local" for row in scored]
        grid[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "binary_action_accuracy": statistics.fmean(
                prediction == target for prediction, target in zip(predicted_local, desired_local)),
            "local_recall": statistics.fmean(
                row["controller_probability"] >= threshold
                for row in scored if row["desired"] == "local"),
            "retain_recall": statistics.fmean(
                row["controller_probability"] < threshold
                for row in scored if row["desired"] == "retain"),
        }
    best = max(grid.values(), key=lambda row: (row["binary_action_accuracy"],
                                                -abs(row["threshold"] - .5)))
    return {
        "protocol": PROTOCOL,
        "source_summary_sha256": sha(root / "summary.json"),
        "source_training_sha256": sha(root / "training.json"),
        "raw_hashes_verified": hashes_verified,
        "controller_decisions_identical_across_readout_seeds": decisions_identical,
        "units": summary["units"],
        "selected_threshold": summary["selected_threshold"],
        "by_event_kind": by_kind,
        "local_vs_retain_probability_auc": auc(positive, negative),
        "noncausal_fixed_trajectory_threshold_grid": grid,
        "best_noncausal_fixed_trajectory_threshold": best,
        "interpretation_limits": [
            "The threshold grid reclassifies probabilities on the observed 0.675 trajectory.",
            "It does not replay the memory state and cannot be reported as a tuned policy result.",
            "Triggered candidate accuracy covers only events where the frozen policy attempted deletion.",
        ],
        "diagnostic_checks_passed": all(hashes_verified.values()) and decisions_identical,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["diagnostic_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
