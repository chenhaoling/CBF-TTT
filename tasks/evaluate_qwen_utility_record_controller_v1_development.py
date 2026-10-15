"""Diagnostic controller/baseline evaluation on relation-disjoint development."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from scripts.audit_qwen_utility_record_controller_v1_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES,
    safe_binary_action,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl, percentile
from tasks.train_qwen_utility_record_controller_v1 import SEEDS, make_model


PROTOCOL = "qwen_utility_record_controller_v1_development_diagnostic"
UTILITY_THRESHOLDS = (0.4, 0.5, 0.6)
RISK_PENALTIES = (0.0, 0.5, 1.0, 2.0)
CONFORMAL_LEVELS = (0.01, 0.03, 0.05)
COSINE_THRESHOLDS = (0.98, 0.99, 0.995)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610168


def stable_rank(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def development_split(groups: list[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(set(groups), key=lambda group: (
        stable_rank("controller-v1-development-diagnostic", group), group
    ))
    midpoint = len(ordered) // 2
    return set(ordered[:midpoint]), set(ordered[midpoint:])


def grid_action(alphas: list[int]) -> str:
    return "grid_" + "_".join(map(str, alphas))


def metric(rows: list[dict], actions: dict[tuple[str, int], str]) -> dict:
    queries = 0; correct = 0; nll = 0.0; unrelated_queries = 0
    unrelated_correct = 0; added_errors = 0; deletions = 0; candidates = 0
    teacher_agreement = 0
    per_mechanism = defaultdict(lambda: [0, 0])
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        name = actions[identity]
        outcome = row["actions"][name]
        baseline = row["actions"][row["retain_action"]]
        queries += outcome["queries"]; correct += outcome["correct"]
        nll += outcome["total_answer_nll"]
        unrelated_queries += outcome["unrelated_queries"]
        unrelated_correct += outcome["unrelated_correct"]
        added_errors += max(0, baseline["unrelated_correct"] - outcome["unrelated_correct"])
        alphas = ([0] * row["candidate_count"] if name == "global_clear"
                  else row["actions"][name]["spec"]["alphas"])
        deletions += sum(value == 0 for value in alphas); candidates += len(alphas)
        teacher_agreement += int(name == safe_binary_action(row))
        values = per_mechanism[row["mechanism"]]
        values[0] += outcome["correct"]; values[1] += outcome["queries"]
    return {
        "accuracy": correct / queries,
        "correct": correct,
        "queries": queries,
        "mean_answer_nll": nll / queries,
        "unrelated_accuracy": unrelated_correct / unrelated_queries,
        "added_unrelated_errors": added_errors,
        "deletion_fraction": deletions / candidates,
        "binary_teacher_action_agreement": teacher_agreement / len(rows),
        "per_mechanism_accuracy": {
            key: value[0] / value[1] for key, value in sorted(per_mechanism.items())
        },
    }


def fixed_policy_actions(rows: list[dict], features: dict, policy: str,
                         cosine_threshold: float | None = None) -> dict:
    output = {}
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]
        size = row["candidate_count"]
        if policy == "retain_all":
            name = row["retain_action"]
        elif policy == "global_clear":
            name = "global_clear"
        else:
            alphas = [1] * size
            if policy in {"last_1", "last_2"}:
                ordered = sorted(range(size), key=lambda index: (
                    feature["features"][index][12], index
                ))
                keep = set(ordered[:int(policy[-1])])
                alphas = [int(index in keep) for index in range(size)]
            elif policy == "exact_string_delete":
                alphas = [int(values[3] < .5) for values in feature["features"]]
            elif policy == "exact_subject_conflict":
                alphas = [int(values[2] < .5) for values in feature["features"]]
            elif policy == "explicit_lifecycle_physics":
                if any(values[20] >= .5 for values in feature["features"]):
                    name = "global_clear"; output[identity] = name; continue
                alphas = [int(values[3] < .5 and values[21] < .5)
                          for values in feature["features"]]
            elif policy == "frozen_cosine_delete":
                alphas = [int(values[4] < float(cosine_threshold))
                          for values in feature["features"]]
            else:
                raise ValueError(f"unknown policy {policy}")
            name = grid_action(alphas)
        if name not in row["actions"]:
            raise ValueError(f"missing action {name}")
        output[identity] = name
    return output


def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def predict_checkpoint(checkpoint: dict, feature_rows: list[dict]) -> dict:
    import torch

    model = make_model(len(FEATURE_NAMES)); model.load_state_dict(checkpoint["state_dict"])
    model.eval(); output = {}
    with torch.inference_mode():
        for row in feature_rows:
            values = torch.tensor([
                [(value - mean) / std for value, mean, std
                 in zip(candidate, checkpoint["means"], checkpoint["stds"])]
                for candidate in row["features"]
            ], dtype=torch.float32)
            utility, risk = model(values)
            output[(row["episode_id"], row["boundary_index"])] = {
                "delete": utility.softmax(-1)[:, 1].tolist(),
                "risk": risk.sigmoid().tolist(),
            }
    return output


def conformal_threshold(predictions: dict, feature_rows: list[dict], groups: set[str],
                        level: float) -> float:
    positives = []
    for row in feature_rows:
        if row["source_group"] not in groups:
            continue
        prediction = predictions[(row["episode_id"], row["boundary_index"])]
        positives.extend(probability for probability, target
                         in zip(prediction["risk"], row["targets"]["risk_targets"])
                         if target == 1)
    if not positives:
        return 0.0
    ordered = sorted(positives)
    index = min(len(ordered) - 1, max(0, math.ceil(level * (len(ordered) + 1)) - 1))
    return ordered[index]


def controller_actions(rows: list[dict], features: dict, predictions: dict,
                       utility_threshold: float, risk_penalty: float,
                       risk_threshold: float) -> dict:
    output = {}
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]; prediction = predictions[identity]
        if any(values[20] >= .5 for values in feature["features"]):
            output[identity] = "global_clear"; continue
        alphas = []
        for values, delete_probability, risk_probability in zip(
            feature["features"], prediction["delete"], prediction["risk"]
        ):
            forced_delete = values[3] >= .5 or values[21] >= .5
            neural_delete = (
                delete_probability - risk_penalty * risk_probability >= utility_threshold
                and risk_probability < risk_threshold
            )
            alphas.append(int(not (forced_delete or neural_delete)))
        output[identity] = grid_action(alphas)
    return output


def select_safe(results: dict[str, dict]) -> str | None:
    safe = [name for name, value in results.items() if value["added_unrelated_errors"] == 0]
    if not safe:
        return None
    return max(safe, key=lambda name: (
        results[name]["accuracy"], -results[name]["mean_answer_nll"], name
    ))


def bootstrap_gain(rows: list[dict], left: dict, right: dict) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    gains = {}
    for group, values in grouped.items():
        numerator = sum(
            row["actions"][left[(row["episode_id"], row["boundary_index"])]]["correct"]
            - row["actions"][right[(row["episode_id"], row["boundary_index"])]]["correct"]
            for row in values
        )
        denominator = sum(row["actions"][row["retain_action"]]["queries"] for row in values)
        gains[group] = numerator / denominator
    groups = sorted(gains); rng = random.Random(BOOTSTRAP_SEED); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(gains[group] for group in sample))
    return [percentile(draws, .025), percentile(draws, .975)]


def evaluate(args) -> dict:
    feature_root, label_root = Path(args.feature_root), Path(args.label_root)
    training_root = Path(args.training_root)
    feature_audit = json.loads((feature_root / "development_feature_audit.json").read_text())
    label_summary = json.loads((label_root / "summary.json").read_text())
    if feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL or not feature_audit.get("passed"):
        raise ValueError("development feature audit did not pass")
    labels = []; feature_rows = []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    features = {(row["episode_id"], row["boundary_index"]): row for row in feature_rows}
    calibration_groups, evaluation_groups = development_split(
        [row["source_group"] for row in labels]
    )
    calibration_rows = [row for row in labels if row["source_group"] in calibration_groups]
    evaluation_rows = [row for row in labels if row["source_group"] in evaluation_groups]

    baseline_actions = {}
    baseline_calibration = {}; baseline_evaluation = {}
    fixed = ("retain_all", "global_clear", "last_1", "last_2",
             "exact_string_delete", "exact_subject_conflict", "explicit_lifecycle_physics")
    for name in fixed:
        actions = fixed_policy_actions(labels, features, name)
        baseline_actions[name] = actions
        baseline_calibration[name] = metric(calibration_rows, actions)
        baseline_evaluation[name] = metric(evaluation_rows, actions)
    for threshold in COSINE_THRESHOLDS:
        name = f"frozen_cosine_delete_{threshold}"
        actions = fixed_policy_actions(labels, features, "frozen_cosine_delete", threshold)
        baseline_actions[name] = actions
        baseline_calibration[name] = metric(calibration_rows, actions)
        baseline_evaluation[name] = metric(evaluation_rows, actions)
    strongest_baseline = select_safe(baseline_calibration)
    if strongest_baseline is None:
        raise ValueError("no safe deterministic development baseline")

    controller_results = {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        predictions = predict_checkpoint(checkpoint, feature_rows)
        candidates = {}
        candidate_actions = {}
        conformal_thresholds = {
            str(level): conformal_threshold(
                predictions, feature_rows, calibration_groups, level
            )
            for level in CONFORMAL_LEVELS
        }
        for utility in UTILITY_THRESHOLDS:
            for penalty in RISK_PENALTIES:
                for level in CONFORMAL_LEVELS:
                    risk_threshold = conformal_thresholds[str(level)]
                    name = f"u{utility}_p{penalty}_c{level}"
                    actions = controller_actions(
                        labels, features, predictions, utility, penalty, risk_threshold
                    )
                    candidates[name] = metric(calibration_rows, actions)
                    candidate_actions[name] = actions
        selected = select_safe(candidates)
        if selected is None:
            controller_results[str(seed)] = {
                "selected": None,
                "calibration_grid": candidates,
                "conformal_risk_thresholds": conformal_thresholds,
                "terminal_status": "no_safe_controller_configuration",
            }
            continue
        actions = candidate_actions[selected]
        evaluation = metric(evaluation_rows, actions)
        selected_parts = selected.split("_")
        controller_results[str(seed)] = {
            "selected": selected,
            "selected_config": {
                "utility_threshold": float(selected_parts[0][1:]),
                "risk_penalty": float(selected_parts[1][1:]),
                "conformal_level": float(selected_parts[2][1:]),
            },
            "conformal_risk_thresholds": conformal_thresholds,
            "calibration_grid": candidates,
            "selected_calibration": candidates[selected],
            "evaluation": evaluation,
            "gain_over_strongest_baseline": (
                evaluation["accuracy"] - baseline_evaluation[strongest_baseline]["accuracy"]
            ),
            "gain_over_strongest_baseline_group_bootstrap_95ci": bootstrap_gain(
                evaluation_rows, actions, baseline_actions[strongest_baseline]
            ),
            "terminal_status": "controller_configuration_evaluated",
        }

    binary_oracle_actions = {
        (row["episode_id"], row["boundary_index"]): safe_binary_action(row) for row in labels
    }
    full_oracle_actions = {
        (row["episode_id"], row["boundary_index"]): row["safe_oracle_action"] for row in labels
    }
    oracle = {
        "binary": metric(evaluation_rows, binary_oracle_actions),
        "full": metric(evaluation_rows, full_oracle_actions),
    }
    seed_passes = [value for value in controller_results.values() if value["selected"]]
    diagnostic_checks = {
        "all_seeds_have_safe_configuration": len(seed_passes) == len(SEEDS),
        "all_seeds_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0 for row in seed_passes
        ),
        "all_seeds_improve_strongest_baseline": all(
            row["gain_over_strongest_baseline"] > 0 for row in seed_passes
        ),
        "all_seed_cluster_ci_lower_positive": all(
            row["gain_over_strongest_baseline_group_bootstrap_95ci"][0] > 0
            for row in seed_passes
        ),
        "confirmation_and_external_sealed": True,
    }
    # The registered V1 release gate is already impossible because full-set
    # oracle headroom over retain-all is below ten points.
    release_checks = {
        "oracle_headroom_at_least_point_10": label_summary["safe_oracle_gain"] >= .10,
        **diagnostic_checks,
    }
    return {
        "protocol": PROTOCOL,
        "partition": "development",
        "calibration_groups": sorted(calibration_groups),
        "evaluation_groups": sorted(evaluation_groups),
        "strongest_safe_deterministic_baseline": strongest_baseline,
        "baselines_calibration": baseline_calibration,
        "baselines_evaluation": baseline_evaluation,
        "controllers": controller_results,
        "oracles_evaluation": oracle,
        "diagnostic_checks": diagnostic_checks,
        "release_gate": {"checks": release_checks, "passed": all(release_checks.values())},
        "formal_development_oracle_gain": label_summary["safe_oracle_gain"],
        "feature_audit_sha256": sha256(feature_root / "development_feature_audit.json"),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "training_audit_sha256": sha256(training_root / "training_audit.json"),
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "sealed_confirmation" if all(release_checks.values()) else "prospective_v2",
        "terminal_status": "controller_v1_development_diagnostic_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
