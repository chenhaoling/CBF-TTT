"""Nested grouped low-dimensional benefit/risk forgetting gate V2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    ACTION_INDEX,
    BOOTSTRAP_DRAWS,
    OUTER_FOLDS,
    INNER_FOLDS,
    SEEDS,
    fold_for,
    inner_fold_for,
    load_labels,
    paired_bootstrap,
    policy_metrics,
    source_actions,
)


PROTOCOL = "qwen_natural_forgetting_factor_gate_v2"
PLAN = "QWEN_NATURAL_FORGETTING_FACTOR_GATE_V2_PLAN.md"
HIDDENS = (0, 16, 32)
LEARNING_RATES = (.001, .005)
WEIGHT_DECAYS = (0., .01)
EPOCHS = 300
BENEFIT_THRESHOLDS = (.3, .5, .7, .8, .9)
RISK_THRESHOLDS = (.001, .005, .01, .02, .05, .1, .2, .3, .5)
BOOTSTRAP_SEED = 202610241
ENGINEERED = (
    "has_memory", "memory_size", "event_position", "top_recency",
    "top_cosine", "cosine_margin", "cosine_mean", "cosine_std",
    "cosine_entropy", "v5_probability", "lexical_jaccard",
    "lexical_containment", "length_ratio",
)
NO_SCHEDULE = tuple(name for name in ENGINEERED
                    if name not in ("memory_size", "event_position", "top_recency"))
FEATURE_SETS = {"engineered": ENGINEERED, "no_schedule": NO_SCHEDULE}


def factor_targets(labels):
    import torch

    benefit, risk = [], []
    for row in labels:
        retain = row["actions"]["retain_all"]
        local = row["actions"]["scope_top1_a0"]
        benefit.append(float(
            local["correct"] > retain["correct"]
            or (local["correct"] == retain["correct"]
                and local["total_answer_nll"] < retain["total_answer_nll"])
        ))
        risk.append(float(local["unrelated_correct"] < retain["unrelated_correct"]))
    return torch.tensor(benefit), torch.tensor(risk)


class BinaryHead:
    @staticmethod
    def create(input_dim, hidden, seed):
        import torch
        torch.manual_seed(seed)
        if hidden == 0:
            return torch.nn.Linear(input_dim, 1)
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )


def normalize(features, indices):
    mean = features[indices].mean(0)
    std = features[indices].std(0, unbiased=False).clamp_min(1e-6)
    return mean, std


def fit_head(features, target, indices, config, seed, offset):
    import torch

    mean, std = normalize(features, indices); x = (features[indices] - mean) / std
    model = BinaryHead.create(features.shape[1], config["hidden"], seed + offset)
    positives = target[indices].sum(); negatives = len(indices) - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(1, 20)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True); logits = model(x).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target[indices], pos_weight=pos_weight)
        loss.backward(); optimizer.step()
    return model, mean, std


def predict(model, mean, std, features):
    import torch
    with torch.no_grad(): return model((features - mean) / std).squeeze(-1).sigmoid()


def factor_actions(benefit_probability, risk_probability, benefit_threshold,
                   risk_threshold, use_risk=True):
    local = ACTION_INDEX["scope_top1_a0"]
    return [local if float(benefit) >= benefit_threshold
            and (not use_risk or float(risk) <= risk_threshold) else 0
            for benefit, risk in zip(benefit_probability, risk_probability)]


def configs():
    return [{"hidden": hidden, "learning_rate": lr, "weight_decay": decay}
            for hidden in HIDDENS for lr in LEARNING_RATES for decay in WEIGHT_DECAYS]


def fit_pair(features, benefit, risk, indices, config, seed):
    benefit_model, benefit_mean, benefit_std = fit_head(
        features, benefit, indices, config, seed, 0)
    risk_model, risk_mean, risk_std = fit_head(
        features, risk, indices, config, seed, 100_000)
    return {"benefit_model": benefit_model, "benefit_mean": benefit_mean,
            "benefit_std": benefit_std, "risk_model": risk_model,
            "risk_mean": risk_mean, "risk_std": risk_std}


def predict_pair(pair, features):
    return (predict(pair["benefit_model"], pair["benefit_mean"],
                    pair["benefit_std"], features),
            predict(pair["risk_model"], pair["risk_mean"],
                    pair["risk_std"], features))


def nested_feature_set(features, benefit, risk, labels, feature_set):
    import torch

    primary = {seed: [None] * len(labels) for seed in SEEDS}
    benefit_only = {seed: [None] * len(labels) for seed in SEEDS}
    probabilities = {seed: [None] * len(labels) for seed in SEEDS}
    folds, searches = {}, {}
    for outer in range(OUTER_FOLDS):
        validation = [index for index, row in enumerate(labels)
                      if fold_for(row["session_id"]) == outer]
        validation_set = set(validation)
        training = [index for index in range(len(labels)) if index not in validation_set]
        grid = []
        cached = {}
        for config_index, config in enumerate(configs()):
            per_seed = {}
            for seed in SEEDS:
                held_benefit = torch.empty(len(training)); held_risk = torch.empty(len(training))
                position = {index: offset for offset, index in enumerate(training)}
                for inner in range(INNER_FOLDS):
                    inner_validation = [index for index in training
                        if inner_fold_for(labels[index]["session_id"], outer) == inner]
                    inner_set = set(inner_validation)
                    inner_training = [index for index in training if index not in inner_set]
                    pair = fit_pair(features, benefit, risk, inner_training, config, seed)
                    pb, pr = predict_pair(pair, features[inner_validation])
                    for local, index in enumerate(inner_validation):
                        held_benefit[position[index]] = pb[local]
                        held_risk[position[index]] = pr[local]
                per_seed[seed] = (held_benefit, held_risk)
            cached[config_index] = per_seed
            for benefit_threshold in BENEFIT_THRESHOLDS:
                for risk_threshold in RISK_THRESHOLDS:
                    metrics = []
                    for seed in SEEDS:
                        pb, pr = per_seed[seed]
                        actions = factor_actions(
                            pb, pr, benefit_threshold, risk_threshold, True)
                        metrics.append(policy_metrics(labels, actions, training))
                    grid.append({
                        "config_index": config_index, "config": config,
                        "benefit_threshold": benefit_threshold,
                        "risk_threshold": risk_threshold,
                        "mean_candidate_accuracy": statistics.fmean(
                            row["candidate_accuracy"] for row in metrics),
                        "mean_unrelated_errors": statistics.fmean(
                            row["unrelated_errors_added"] for row in metrics),
                    })
        retain = policy_metrics(labels, [0] * len(training), training)
        source = policy_metrics(labels, source_actions(labels, training), training)
        for row in grid:
            row["feasible"] = (row["mean_unrelated_errors"]
                               <= source["unrelated_errors_added"]
                               and row["mean_candidate_accuracy"]
                               >= retain["candidate_accuracy"] + .005)
        feasible = [row for row in grid if row["feasible"]]
        if feasible:
            selected = max(feasible, key=lambda row: (
                row["mean_candidate_accuracy"], -row["mean_unrelated_errors"],
                -row["config"]["hidden"], row["benefit_threshold"],
                -row["risk_threshold"])); fold_feasible = True
        else:
            selected = min(grid, key=lambda row: (
                row["mean_unrelated_errors"], -row["mean_candidate_accuracy"],
                row["config"]["hidden"], -row["benefit_threshold"],
                row["risk_threshold"])); fold_feasible = False
        config = selected["config"]
        for seed in SEEDS:
            pair = fit_pair(features, benefit, risk, training, config, seed)
            pb, pr = predict_pair(pair, features[validation])
            guarded = factor_actions(pb, pr, selected["benefit_threshold"],
                                     selected["risk_threshold"], True)
            unguarded = factor_actions(pb, pr, selected["benefit_threshold"],
                                       selected["risk_threshold"], False)
            for local, index in enumerate(validation):
                primary[seed][index] = guarded[local]
                benefit_only[seed][index] = unguarded[local]
                probabilities[seed][index] = [float(pb[local]), float(pr[local])]
        folds[str(outer)] = {
            "train_sessions": sorted({labels[index]["session_id"] for index in training}),
            "validation_sessions": sorted({labels[index]["session_id"] for index in validation}),
            "selected": selected, "feasible": fold_feasible,
            "inner_retain": retain, "inner_source_rule": source,
        }
        searches[str(outer)] = grid
    return {"primary": primary, "benefit_only": benefit_only,
            "probabilities": probabilities, "folds": folds, "search": searches,
            "feature_set": feature_set}


def median_config(folds):
    rows = [folds[str(index)]["selected"] for index in range(OUTER_FOLDS)]
    return {
        "hidden": int(statistics.median(row["config"]["hidden"] for row in rows)),
        "learning_rate": statistics.median(
            row["config"]["learning_rate"] for row in rows),
        "weight_decay": statistics.median(row["config"]["weight_decay"] for row in rows),
        "benefit_threshold": statistics.median(row["benefit_threshold"] for row in rows),
        "risk_threshold": statistics.median(row["risk_threshold"] for row in rows),
    }


def run(args):
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    utility_summary, labels = load_labels(args.utility_root)
    v1_root = Path(args.v1_root); v1_summary = json.loads(
        (v1_root / "result" / "summary.json").read_text())
    v1_audit = json.loads((v1_root / "execution_audit.json").read_text())
    if v1_summary["protocol"] != "qwen_natural_forgetting_barrier_q_v1" or not v1_audit["passed"]:
        raise ValueError("V1 source is not audited")
    artifact_path = v1_root / "result" / "features.pt"
    if v1_summary["hashes"]["features"] != sha(artifact_path):
        raise ValueError("V1 feature artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    benefit, risk = factor_targets(labels); names = artifact["feature_names"]
    if any(name not in names for values in FEATURE_SETS.values() for name in values):
        raise ValueError("registered low-dimensional feature is missing")
    feature_tables = {key: artifact["features"][:, [names.index(name) for name in values]]
                      for key, values in FEATURE_SETS.items()}
    torch.save({"feature_sets": feature_tables, "feature_names": FEATURE_SETS,
                "benefit": benefit, "risk": risk}, root / "factor_features.pt")
    started = time.perf_counter()
    results = {name: nested_feature_set(table, benefit, risk, labels, name)
               for name, table in feature_tables.items()}
    metrics = {}
    for feature_set, result in results.items():
        metrics[feature_set] = {
            "factor_gate": {str(seed): policy_metrics(labels, actions)
                            for seed, actions in result["primary"].items()},
            "benefit_only": {str(seed): policy_metrics(labels, actions)
                             for seed, actions in result["benefit_only"].items()},
        }
    retain = [0] * len(labels); fixed = [ACTION_INDEX["scope_top1_a0"]] * len(labels)
    source = source_actions(labels, range(len(labels)))
    binary_oracle = []
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
    primary = results["engineered"]["primary"]
    benefit_only = results["engineered"]["benefit_only"]
    nonretain_rate = statistics.fmean(action != 0 for actions in primary.values()
                                      for action in actions)
    random_metrics = {}
    for seed in SEEDS:
        rng = random.Random(seed + 800_000)
        actions = [ACTION_INDEX["scope_top1_a0"] if rng.random() < nonretain_rate else 0
                   for _ in labels]
        random_metrics[str(seed)] = policy_metrics(labels, actions)
    metrics["rate_matched_random"] = random_metrics
    comparison_retain = paired_bootstrap(labels, primary, retain, BOOTSTRAP_SEED)
    comparison_fixed = paired_bootstrap(labels, primary, fixed, BOOTSTRAP_SEED + 1)
    mean = lambda table, key: statistics.fmean(
        table[str(seed)][key] for seed in SEEDS)
    primary_accuracy = mean(metrics["engineered"]["factor_gate"], "candidate_accuracy")
    benefit_accuracy = mean(metrics["engineered"]["benefit_only"], "candidate_accuracy")
    primary_errors = mean(metrics["engineered"]["factor_gate"], "unrelated_errors_added")
    benefit_errors = mean(metrics["engineered"]["benefit_only"], "unrelated_errors_added")
    retain_accuracy = baselines["retain_all"]["candidate_accuracy"]
    fixed_accuracy = baselines["always_scope_top1_a0"]["candidate_accuracy"]
    full_oracle_accuracy = baselines["full_safe_oracle"]["candidate_accuracy"]
    headroom = (primary_accuracy - retain_accuracy) / (full_oracle_accuracy - retain_accuracy)
    v1_barrier = statistics.fmean(
        v1_summary["metrics"]["barrier_q"][str(seed)]["candidate_accuracy"] for seed in SEEDS)
    v1_direct = statistics.fmean(
        v1_summary["metrics"]["direct_classification"][str(seed)]["candidate_accuracy"]
        for seed in SEEDS)
    checks = {
        "gain_over_retain_at_least_point_0075": primary_accuracy - retain_accuracy >= .0075,
        "retain_gain_ci_lower_above_zero": comparison_retain["bootstrap_95ci"][0] > 0,
        "gain_over_fixed_at_least_point_0025": primary_accuracy - fixed_accuracy >= .0025,
        "fixed_gain_ci_lower_above_zero": comparison_fixed["bootstrap_95ci"][0] > 0,
        "captures_at_least_point_30_full_oracle_headroom": headroom >= .30,
        "unrelated_errors_no_more_than_source_rule":
            primary_errors <= baselines["source_action_rule"]["unrelated_errors_added"],
        "risk_gate_reduces_benefit_only_errors": primary_errors < benefit_errors,
        "risk_gate_within_point_005_of_benefit_only":
            primary_accuracy >= benefit_accuracy - .005,
        "beats_v1_barrier_q": primary_accuracy > v1_barrier,
        "beats_v1_direct_classification": primary_accuracy > v1_direct,
        "at_least_four_feasible_outer_folds":
            sum(row["feasible"] for row in results["engineered"]["folds"].values()) >= 4,
        "all_seeds_improve_retain": all(
            metrics["engineered"]["factor_gate"][str(seed)]["candidate_accuracy"]
            > retain_accuracy for seed in SEEDS),
        "all_seeds_beat_fixed": all(
            metrics["engineered"]["factor_gate"][str(seed)]["candidate_accuracy"]
            > fixed_accuracy for seed in SEEDS),
    }
    checkpoints = {}
    final_configs = {}
    for feature_set, table in feature_tables.items():
        config = median_config(results[feature_set]["folds"]); final_configs[feature_set] = config
        checkpoints[feature_set] = {}
        train_config = {key: config[key] for key in ("hidden", "learning_rate", "weight_decay")}
        for seed in SEEDS:
            pair = fit_pair(table, benefit, risk, list(range(len(labels))), train_config, seed)
            checkpoints[feature_set][str(seed)] = {
                "benefit_state": pair["benefit_model"].state_dict(),
                "benefit_mean": pair["benefit_mean"], "benefit_std": pair["benefit_std"],
                "risk_state": pair["risk_model"].state_dict(),
                "risk_mean": pair["risk_mean"], "risk_std": pair["risk_std"],
            }
    checkpoint_path = root / "factor_gate.pt"
    torch.save({"protocol": PROTOCOL, "actions": ("retain_all", "scope_top1_a0"),
                "feature_names": FEATURE_SETS, "configs": final_configs,
                "seeds": SEEDS, "checkpoints": checkpoints}, checkpoint_path)
    with (root / "oof_predictions.jsonl").open("x") as sink:
        for index, row in enumerate(labels):
            sink.write(json.dumps({
                "session_id": row["session_id"], "event_step": row["event_step"],
                "outer_fold": fold_for(row["session_id"]),
                "benefit_target": int(benefit[index]), "risk_target": int(risk[index]),
                "feature_sets": {feature_set: {str(seed): {
                    "benefit_probability": result["probabilities"][seed][index][0],
                    "risk_probability": result["probabilities"][seed][index][1],
                    "factor_action": ("scope_top1_a0" if result["primary"][seed][index] else
                                      "retain_all"),
                    "benefit_only_action": ("scope_top1_a0"
                        if result["benefit_only"][seed][index] else "retain_all"),
                } for seed in SEEDS} for feature_set, result in results.items()},
            }) + "\n")
    searches = {feature_set: result["search"] for feature_set, result in results.items()}
    write_json(root / "search.json", searches)
    summary = {
        "protocol": PROTOCOL, "metrics": metrics, "baselines": baselines,
        "comparisons": {"factor_vs_retain": comparison_retain,
                        "factor_vs_fixed": comparison_fixed},
        "mean": {"factor_candidate_accuracy": primary_accuracy,
                 "benefit_only_candidate_accuracy": benefit_accuracy,
                 "factor_unrelated_errors": primary_errors,
                 "benefit_only_unrelated_errors": benefit_errors,
                 "full_oracle_headroom_captured": headroom,
                 "v1_barrier_accuracy": v1_barrier, "v1_direct_accuracy": v1_direct},
        "folds": {feature_set: result["folds"] for feature_set, result in results.items()},
        "final_configs": final_configs,
        "gate": {"checks": checks, "development_authorized": all(checks.values())},
        "resources": {"training_seconds": time.perf_counter() - started},
        "hashes": {"source_features": sha(artifact_path),
                   "factor_features": sha(root / "factor_features.pt"),
                   "oof_predictions": sha(root / "oof_predictions.jsonl"),
                   "search": sha(root / "search.json"),
                   "checkpoint": sha(checkpoint_path),
                   "utility_summary": sha(Path(args.utility_root) / "summary.json"),
                   "v1_summary": sha(v1_root / "result" / "summary.json"),
                   "v1_audit": sha(v1_root / "execution_audit.json"),
                   "plan": sha(PLAN)},
        "feature_sets": FEATURE_SETS,
        "grid": {"hiddens": HIDDENS, "learning_rates": LEARNING_RATES,
                 "weight_decays": WEIGHT_DECAYS, "epochs": EPOCHS,
                 "benefit_thresholds": BENEFIT_THRESHOLDS,
                 "risk_thresholds": RISK_THRESHOLDS},
        "seeds": SEEDS, "calibration_scored": True,
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
        "utility_summary": utility_summary,
    }
    write_json(root / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("utility_root", "v1_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
