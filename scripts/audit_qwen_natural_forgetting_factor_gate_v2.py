"""Independent audit of nested grouped factorized forgetting gate V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    ACTION_INDEX,
    SEEDS,
    fold_for,
    load_labels,
    paired_bootstrap,
    policy_metrics,
    source_actions,
)
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    ENGINEERED,
    FEATURE_SETS,
    NO_SCHEDULE,
    OUTER_FOLDS,
    PROTOCOL,
    factor_targets,
)


PLAN = "QWEN_NATURAL_FORGETTING_FACTOR_GATE_V2_PLAN.md"


def audit(root, repo, utility_root, v1_root):
    import torch

    root, repo, utility_root, v1_root = map(Path, (root, repo, utility_root, v1_root))
    result = root / "result"; summary = json.loads((result / "summary.json").read_text())
    _, labels = load_labels(utility_root)
    factor_artifact = torch.load(result / "factor_features.pt", map_location="cpu",
                                 weights_only=False)
    v1_artifact = torch.load(v1_root / "result" / "features.pt", map_location="cpu",
                             weights_only=False)
    benefit, risk = factor_targets(labels); names = v1_artifact["feature_names"]
    expected_tables = {key: v1_artifact["features"][:, [names.index(name) for name in values]]
                       for key, values in FEATURE_SETS.items()}
    feature_checks = all((
        tuple(factor_artifact["feature_names"]["engineered"]) == ENGINEERED,
        tuple(factor_artifact["feature_names"]["no_schedule"]) == NO_SCHEDULE,
        not any(name.startswith("reader_") for name in ENGINEERED),
        torch.equal(factor_artifact["benefit"], benefit),
        torch.equal(factor_artifact["risk"], risk),
        all(torch.equal(factor_artifact["feature_sets"][key], value)
            for key, value in expected_tables.items()),
    ))
    rows = [json.loads(line) for line in
            (result / "oof_predictions.jsonl").read_text().splitlines() if line]
    expected_identity = [(row["session_id"], row["event_step"]) for row in labels]
    observed_identity = [(row["session_id"], row["event_step"]) for row in rows]
    predictions = {feature_set: {
        "factor_gate": {seed: [] for seed in SEEDS},
        "benefit_only": {seed: [] for seed in SEEDS},
    } for feature_set in FEATURE_SETS}
    prediction_checks = len(rows) == 774 and observed_identity == expected_identity
    for index, row in enumerate(rows):
        prediction_checks &= row["outer_fold"] == fold_for(row["session_id"])
        prediction_checks &= row["benefit_target"] == int(benefit[index])
        prediction_checks &= row["risk_target"] == int(risk[index])
        outer = str(row["outer_fold"])
        for feature_set in FEATURE_SETS:
            selected = summary["folds"][feature_set][outer]["selected"]
            for seed in SEEDS:
                value = row["feature_sets"][feature_set][str(seed)]
                expected_factor = ("scope_top1_a0"
                    if value["benefit_probability"] >= selected["benefit_threshold"]
                    and value["risk_probability"] <= selected["risk_threshold"]
                    else "retain_all")
                expected_benefit = ("scope_top1_a0"
                    if value["benefit_probability"] >= selected["benefit_threshold"]
                    else "retain_all")
                prediction_checks &= value["factor_action"] == expected_factor
                prediction_checks &= value["benefit_only_action"] == expected_benefit
                predictions[feature_set]["factor_gate"][seed].append(
                    ACTION_INDEX[value["factor_action"]])
                predictions[feature_set]["benefit_only"][seed].append(
                    ACTION_INDEX[value["benefit_only_action"]])

    searches = json.loads((result / "search.json").read_text()); fold_checks = True
    all_sessions = {row["session_id"] for row in labels}
    for feature_set in FEATURE_SETS:
        for outer in range(OUTER_FOLDS):
            fold = summary["folds"][feature_set][str(outer)]
            train, validation = set(fold["train_sessions"]), set(fold["validation_sessions"])
            fold_checks &= not train & validation and train | validation == all_sessions
            fold_checks &= validation == {value for value in all_sessions if fold_for(value) == outer}
            grid = searches[feature_set][str(outer)]; feasible = [row for row in grid if row["feasible"]]
            if feasible:
                selected = max(feasible, key=lambda row: (
                    row["mean_candidate_accuracy"], -row["mean_unrelated_errors"],
                    -row["config"]["hidden"], row["benefit_threshold"],
                    -row["risk_threshold"]))
            else:
                selected = min(grid, key=lambda row: (
                    row["mean_unrelated_errors"], -row["mean_candidate_accuracy"],
                    row["config"]["hidden"], -row["benefit_threshold"],
                    row["risk_threshold"]))
            fold_checks &= selected == fold["selected"]
            fold_checks &= fold["feasible"] == bool(feasible)

    metrics = {}
    for feature_set in FEATURE_SETS:
        metrics[feature_set] = {name: {str(seed): policy_metrics(labels, actions)
                                      for seed, actions in table.items()}
                                for name, table in predictions[feature_set].items()}
    metric_checks = all(metrics[feature_set][name][str(seed)]
                        == summary["metrics"][feature_set][name][str(seed)]
                        for feature_set in FEATURE_SETS
                        for name in ("factor_gate", "benefit_only") for seed in SEEDS)
    retain = [0] * len(labels); fixed = [ACTION_INDEX["scope_top1_a0"]] * len(labels)
    source = source_actions(labels, range(len(labels))); binary_oracle = []
    for row in labels:
        retained = row["actions"]["retain_all"]; local = row["actions"]["scope_top1_a0"]
        beneficial = (local["correct"] > retained["correct"]
                      or (local["correct"] == retained["correct"]
                          and local["total_answer_nll"] < retained["total_answer_nll"]))
        safe = local["unrelated_correct"] >= retained["unrelated_correct"]
        binary_oracle.append(ACTION_INDEX["scope_top1_a0"] if beneficial and safe else 0)
    full_oracle = [ACTION_INDEX[row["safe_oracle_action"]] for row in labels]
    baselines = {
        "retain_all": policy_metrics(labels, retain),
        "always_scope_top1_a0": policy_metrics(labels, fixed),
        "source_action_rule": policy_metrics(labels, source),
        "safe_binary_oracle": policy_metrics(labels, binary_oracle),
        "full_safe_oracle": policy_metrics(labels, full_oracle),
    }
    baseline_checks = baselines == summary["baselines"]
    primary = predictions["engineered"]["factor_gate"]
    comparison_retain = paired_bootstrap(labels, primary, retain, 202610241)
    comparison_fixed = paired_bootstrap(labels, primary, fixed, 202610242)
    comparison_checks = (comparison_retain == summary["comparisons"]["factor_vs_retain"]
                         and comparison_fixed == summary["comparisons"]["factor_vs_fixed"])
    mean = lambda table, key: statistics.fmean(table[str(seed)][key] for seed in SEEDS)
    primary_accuracy = mean(metrics["engineered"]["factor_gate"], "candidate_accuracy")
    benefit_accuracy = mean(metrics["engineered"]["benefit_only"], "candidate_accuracy")
    primary_errors = mean(metrics["engineered"]["factor_gate"], "unrelated_errors_added")
    benefit_errors = mean(metrics["engineered"]["benefit_only"], "unrelated_errors_added")
    retain_accuracy = baselines["retain_all"]["candidate_accuracy"]
    fixed_accuracy = baselines["always_scope_top1_a0"]["candidate_accuracy"]
    full_oracle_accuracy = baselines["full_safe_oracle"]["candidate_accuracy"]
    headroom = (primary_accuracy - retain_accuracy) / (full_oracle_accuracy - retain_accuracy)
    v1_summary = json.loads((v1_root / "result" / "summary.json").read_text())
    v1_barrier = statistics.fmean(
        v1_summary["metrics"]["barrier_q"][str(seed)]["candidate_accuracy"] for seed in SEEDS)
    v1_direct = statistics.fmean(
        v1_summary["metrics"]["direct_classification"][str(seed)]["candidate_accuracy"]
        for seed in SEEDS)
    mean_values = {"factor_candidate_accuracy": primary_accuracy,
                   "benefit_only_candidate_accuracy": benefit_accuracy,
                   "factor_unrelated_errors": primary_errors,
                   "benefit_only_unrelated_errors": benefit_errors,
                   "full_oracle_headroom_captured": headroom,
                   "v1_barrier_accuracy": v1_barrier, "v1_direct_accuracy": v1_direct}
    gate = {
        "gain_over_retain_at_least_point_0075": primary_accuracy - retain_accuracy >= .0075,
        "retain_gain_ci_lower_above_zero": comparison_retain["bootstrap_95ci"][0] > 0,
        "gain_over_fixed_at_least_point_0025": primary_accuracy - fixed_accuracy >= .0025,
        "fixed_gain_ci_lower_above_zero": comparison_fixed["bootstrap_95ci"][0] > 0,
        "captures_at_least_point_30_full_oracle_headroom": headroom >= .30,
        "unrelated_errors_no_more_than_source_rule":
            primary_errors <= baselines["source_action_rule"]["unrelated_errors_added"],
        "risk_gate_reduces_benefit_only_errors": primary_errors < benefit_errors,
        "risk_gate_within_point_005_of_benefit_only": primary_accuracy >= benefit_accuracy - .005,
        "beats_v1_barrier_q": primary_accuracy > v1_barrier,
        "beats_v1_direct_classification": primary_accuracy > v1_direct,
        "at_least_four_feasible_outer_folds":
            sum(row["feasible"] for row in summary["folds"]["engineered"].values()) >= 4,
        "all_seeds_improve_retain": all(
            metrics["engineered"]["factor_gate"][str(seed)]["candidate_accuracy"]
            > retain_accuracy for seed in SEEDS),
        "all_seeds_beat_fixed": all(
            metrics["engineered"]["factor_gate"][str(seed)]["candidate_accuracy"]
            > fixed_accuracy for seed in SEEDS),
    }
    checkpoint = torch.load(result / "factor_gate.pt", map_location="cpu", weights_only=False)
    checkpoint_checks = all((checkpoint["protocol"] == PROTOCOL,
        tuple(checkpoint["actions"]) == ("retain_all", "scope_top1_a0"),
        tuple(checkpoint["feature_names"]["engineered"]) == ENGINEERED,
        tuple(checkpoint["feature_names"]["no_schedule"]) == NO_SCHEDULE,
        tuple(checkpoint["seeds"]) == SEEDS,
        checkpoint["configs"] == summary["final_configs"]))
    hashes = summary["hashes"]
    hash_checks = all((hashes["source_features"] == sha(v1_root / "result" / "features.pt"),
        hashes["factor_features"] == sha(result / "factor_features.pt"),
        hashes["oof_predictions"] == sha(result / "oof_predictions.jsonl"),
        hashes["search"] == sha(result / "search.json"),
        hashes["checkpoint"] == sha(result / "factor_gate.pt"),
        hashes["utility_summary"] == sha(utility_root / "summary.json"),
        hashes["v1_summary"] == sha(v1_root / "result" / "summary.json"),
        hashes["v1_audit"] == sha(v1_root / "execution_audit.json"),
        hashes["plan"] == sha(repo / PLAN)))
    terminal = ("completed_qwen_natural_forgetting_factor_gate_v2_development"
                if all(gate.values()) else
                "completed_qwen_natural_forgetting_factor_gate_v2_calibration_stop")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip() == terminal
            and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "low_dimensional_features_and_targets": feature_checks,
        "complete_probability_and_action_oof": prediction_checks,
        "nested_group_isolation_and_selection": fold_checks,
        "policy_metrics": metric_checks,
        "baselines_and_binary_oracles": baseline_checks,
        "paired_bootstrap": comparison_checks,
        "means_and_gate": summary["mean"] == mean_values
            and summary["gate"]["checks"] == gate
            and summary["gate"]["development_authorized"] == all(gate.values()),
        "checkpoint_and_hashes": checkpoint_checks and hash_checks,
        "heldout_seals": summary["calibration_scored"]
            and not summary["development_loaded"] and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": {
                "metrics": metrics, "baselines": baselines,
                "comparisons": {"retain": comparison_retain, "fixed": comparison_fixed},
                "mean": mean_values, "gate": gate}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "utility_root", "v1_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.repo, args.utility_root, args.v1_root)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
