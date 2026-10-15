"""Independent audit of grouped OOF Barrier-Q calibration results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    ACTIONS,
    ACTION_INDEX,
    CLASS_CONFIDENCES,
    OUTER_FOLDS,
    PROTOCOL,
    SEEDS,
    fold_for,
    load_labels,
    paired_bootstrap,
    policy_metrics,
    source_actions,
    target_tables,
)


def audit(root, repo, utility_root, data_root):
    import torch

    root, repo, utility_root, data_root = map(Path, (root, repo, utility_root, data_root))
    result = root / "result"; summary = json.loads((result / "summary.json").read_text())
    _, labels = load_labels(utility_root); _, sessions = load(data_root, "calibration")
    artifact = torch.load(result / "features.pt", map_location="cpu", weights_only=False)
    gain, nll, risk, mass = target_tables(labels)
    target_checks = all((
        torch.equal(artifact["gain"], gain), torch.equal(artifact["nll"], nll),
        torch.equal(artifact["risk"], risk), torch.equal(artifact["mass"], mass),
        artifact["features"].shape[0] == 774,
        torch.isfinite(artifact["features"]).all().item(),
        len(artifact["feature_names"]) == artifact["features"].shape[1]
            == len(set(artifact["feature_names"])),
    ))
    rows = [json.loads(line) for line in
            (result / "oof_predictions.jsonl").read_text().splitlines() if line]
    identity = [(row["session_id"], row["event_step"]) for row in rows]
    expected_identity = [(row["session_id"], row["event_step"]) for row in labels]
    oof = {name: {seed: [] for seed in SEEDS}
           for name in ("barrier_q", "gain_q", "direct_classification")}
    prediction_checks = len(rows) == 774 and identity == expected_identity
    for source, label in zip(rows, labels):
        prediction_checks &= source["outer_fold"] == fold_for(source["session_id"])
        prediction_checks &= source["safe_oracle_action"] == label["safe_oracle_action"]
        for name in oof:
            for seed in SEEDS:
                action = source["predictions"][name][str(seed)]
                prediction_checks &= action in ACTION_INDEX
                oof[name][seed].append(ACTION_INDEX[action])

    all_sessions = {row["id"] for row in sessions}; fold_checks = True
    search = json.loads((result / "search.json").read_text())
    selected_recomputed = {}
    for outer in range(OUTER_FOLDS):
        fold = summary["folds"][str(outer)]
        train, validation = set(fold["train_sessions"]), set(fold["validation_sessions"])
        fold_checks &= not train & validation and train | validation == all_sessions
        fold_checks &= validation == {value for value in all_sessions if fold_for(value) == outer}
        rows_search = search[str(outer)]
        feasible = [row for row in rows_search if row["feasible"]]
        if feasible:
            selected = max(feasible, key=lambda row: (
                row["mean_candidate_accuracy"], -row["mean_unrelated_errors"],
                -row["config"]["hidden"], -abs(row["barrier"] - .10),
                -abs(row["min_gain"] - .01), -abs(row["tie"] - .005)))
        else:
            selected = min(rows_search, key=lambda row: (
                row["mean_unrelated_errors"], -row["mean_candidate_accuracy"],
                row["config"]["hidden"], abs(row["barrier"] - .10),
                abs(row["min_gain"] - .01), abs(row["tie"] - .005)))
        selected_recomputed[str(outer)] = selected
        fold_checks &= selected == fold["selected"]
        fold_checks &= fold["feasible"] == bool(feasible)
        fold_checks &= fold["classifier_confidence"]["confidence"] in CLASS_CONFIDENCES

    metrics = {name: {str(seed): policy_metrics(labels, actions)
                      for seed, actions in per_seed.items()}
               for name, per_seed in oof.items()}
    metric_checks = all(metrics[name][str(seed)] == summary["metrics"][name][str(seed)]
                        for name in metrics for seed in SEEDS)
    fixed_actions = {action: [ACTION_INDEX[action]] * len(labels) for action in ACTIONS}
    fixed_metrics = {action: policy_metrics(labels, values)
                     for action, values in fixed_actions.items()}
    strongest = max(ACTIONS, key=lambda action: fixed_metrics[action]["candidate_accuracy"])
    retain = fixed_actions["retain_all"]
    source = source_actions(labels, range(len(labels)))
    safe = [ACTION_INDEX[row["safe_oracle_action"]] for row in labels]
    unconstrained = [ACTION_INDEX[row["unconstrained_oracle_action"]] for row in labels]
    names = artifact["feature_names"]; features = artifact["features"]
    frozen = [ACTION_INDEX["scope_top1_a0"]
              if features[index, names.index("v5_probability")] >= .30 else 0
              for index in range(len(labels))]
    baseline_recomputed = {**fixed_metrics,
        "frozen_scope_delete_if_retrieve": policy_metrics(labels, frozen),
        "source_action_rule": policy_metrics(labels, source),
        "safe_oracle": policy_metrics(labels, safe),
        "unconstrained_oracle": policy_metrics(labels, unconstrained)}
    baseline_checks = strongest == summary["strongest_fixed_deployable"] and all(
        baseline_recomputed[name] == summary["baselines"][name]
        for name in baseline_recomputed)
    comparison_retain = paired_bootstrap(labels, oof["barrier_q"], retain, 202610231)
    comparison_fixed = paired_bootstrap(
        labels, oof["barrier_q"], fixed_actions[strongest], 202610232)
    comparison_checks = (comparison_retain == summary["comparisons"]["barrier_q_vs_retain"]
                         and comparison_fixed
                         == summary["comparisons"]["barrier_q_vs_strongest_fixed"])
    mean = lambda name, key: statistics.fmean(
        metrics[name][str(seed)][key] for seed in SEEDS)
    primary_accuracy = mean("barrier_q", "candidate_accuracy")
    gainq_accuracy = mean("gain_q", "candidate_accuracy")
    primary_errors = mean("barrier_q", "unrelated_errors_added")
    gainq_errors = mean("gain_q", "unrelated_errors_added")
    retain_accuracy = baseline_recomputed["retain_all"]["candidate_accuracy"]
    oracle_accuracy = baseline_recomputed["safe_oracle"]["candidate_accuracy"]
    headroom = (primary_accuracy - retain_accuracy) / (oracle_accuracy - retain_accuracy)
    gate = {
        "gain_over_retain_at_least_point_0075": primary_accuracy - retain_accuracy >= .0075,
        "retain_gain_ci_lower_above_zero": comparison_retain["bootstrap_95ci"][0] > 0,
        "gain_over_strongest_fixed_at_least_point_0025":
            primary_accuracy - baseline_recomputed[strongest]["candidate_accuracy"] >= .0025,
        "fixed_gain_ci_lower_above_zero": comparison_fixed["bootstrap_95ci"][0] > 0,
        "captures_at_least_point_30_oracle_headroom": headroom >= .30,
        "unrelated_errors_no_more_than_source_rule":
            primary_errors <= baseline_recomputed["source_action_rule"]["unrelated_errors_added"],
        "barrier_reduces_gain_q_errors": primary_errors < gainq_errors,
        "barrier_within_point_005_of_gain_q": primary_accuracy >= gainq_accuracy - .005,
        "at_least_four_feasible_outer_folds":
            sum(row["feasible"] for row in summary["folds"].values()) >= 4,
        "all_seeds_improve_retain": all(
            metrics["barrier_q"][str(seed)]["candidate_accuracy"] > retain_accuracy
            for seed in SEEDS),
        "all_seeds_beat_strongest_fixed": all(
            metrics["barrier_q"][str(seed)]["candidate_accuracy"]
            > baseline_recomputed[strongest]["candidate_accuracy"] for seed in SEEDS),
    }
    mean_checks = summary["mean"] == {
        "barrier_q_candidate_accuracy": primary_accuracy,
        "gain_q_candidate_accuracy": gainq_accuracy,
        "barrier_q_unrelated_errors": primary_errors,
        "gain_q_unrelated_errors": gainq_errors,
        "oracle_headroom_captured": headroom,
    }
    checkpoint = torch.load(result / "barrier_q.pt", map_location="cpu", weights_only=False)
    checkpoint_checks = all((checkpoint["protocol"] == PROTOCOL,
                             tuple(checkpoint["actions"]) == ACTIONS,
                             tuple(checkpoint["seeds"]) == SEEDS,
                             checkpoint["config"] == summary["final_config"],
                             set(checkpoint["checkpoints"]) == {str(seed) for seed in SEEDS}))
    hash_checks = all((summary["hashes"]["features"] == sha(result / "features.pt"),
                       summary["hashes"]["oof_predictions"]
                           == sha(result / "oof_predictions.jsonl"),
                       summary["hashes"]["search"] == sha(result / "search.json"),
                       summary["hashes"]["checkpoint"] == sha(result / "barrier_q.pt"),
                       summary["hashes"]["utility_summary"] == sha(utility_root / "summary.json"),
                       summary["hashes"]["plan"]
                           == sha(repo / "QWEN_NATURAL_FORGETTING_BARRIER_Q_V1_PLAN.md")))
    terminal = ("completed_qwen_natural_forgetting_barrier_q_v1_development"
                if all(gate.values()) else
                "completed_qwen_natural_forgetting_barrier_q_v1_calibration_stop")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip() == terminal
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "feature_targets_and_shapes": target_checks,
        "complete_grouped_oof_predictions": prediction_checks,
        "fold_exclusivity_and_search_selection": fold_checks,
        "policy_metrics": metric_checks,
        "baselines_and_strongest_fixed": baseline_checks,
        "paired_bootstrap_comparisons": comparison_checks,
        "means_and_gate": mean_checks and gate == summary["gate"]["checks"]
            and summary["gate"]["development_authorized"] == all(gate.values()),
        "checkpoint_and_hashes": checkpoint_checks and hash_checks,
        "heldout_seals": summary["calibration_scored"]
            and not summary["development_loaded"] and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": {
                "metrics": metrics, "strongest_fixed": strongest,
                "comparisons": {"retain": comparison_retain, "fixed": comparison_fixed},
                "mean": {"barrier_q_candidate_accuracy": primary_accuracy,
                         "gain_q_candidate_accuracy": gainq_accuracy,
                         "barrier_q_unrelated_errors": primary_errors,
                         "gain_q_unrelated_errors": gainq_errors,
                         "oracle_headroom_captured": headroom},
                "gate": gate}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "utility_root", "data_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.repo, args.utility_root, args.data_root)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
