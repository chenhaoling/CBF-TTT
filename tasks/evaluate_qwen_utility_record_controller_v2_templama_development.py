"""Evaluate the frozen V1 forgetting controller on TempLAMA development."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v1_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    COSINE_THRESHOLDS,
    controller_actions,
    fixed_policy_actions,
    grid_action,
    load_checkpoint,
    predict_checkpoint,
    select_safe,
)
from tasks.extract_qwen_utility_record_controller_v1_features import (
    safe_binary_action,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl, percentile
from tasks.train_qwen_utility_record_controller_v1 import SEEDS


PROTOCOL = "qwen_utility_record_controller_v2_templama_development"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v2_templama_development_audit"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 202610173
RANDOM_DRAWS = 20
EXPECTED_TRAINING_AUDIT_SHA256 = (
    "4260e090d6ebae309d0f0c0e39637de8691b9bea3fdb044ba40d244fa09e8d07"
)
FROZEN_CONFIGS = {
    2610161: {
        "utility_threshold": 0.6,
        "risk_penalty": 2.0,
        "conformal_level": 0.05,
        "risk_threshold": 0.004868066404014826,
    },
    2610162: {
        "utility_threshold": 0.6,
        "risk_penalty": 2.0,
        "conformal_level": 0.05,
        "risk_threshold": 0.004372645635157824,
    },
    2610163: {
        "utility_threshold": 0.6,
        "risk_penalty": 2.0,
        "conformal_level": 0.01,
        "risk_threshold": 1.1618091377840756e-8,
    },
}


def action_alphas(row: dict, name: str) -> list[int]:
    if name == "global_clear":
        return [0] * row["candidate_count"]
    return [int(value) for value in row["actions"][name]["spec"]["alphas"]]


def metric(rows: list[dict], actions: dict[tuple[str, int], str],
           features: dict[tuple[str, int], dict]) -> dict:
    queries = 0
    correct = 0
    nll = 0.0
    unrelated_queries = 0
    unrelated_correct = 0
    added_errors = 0
    deletions = 0
    candidates = 0
    deletion_true_positive = 0
    teacher_action_agreement = 0
    teacher_record_agreement = 0
    by_mechanism = defaultdict(lambda: [0, 0])
    by_relation = defaultdict(lambda: [0, 0])
    by_role = defaultdict(lambda: [0, 0, 0.0])
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        name = actions[identity]
        outcome = row["actions"][name]
        baseline = row["actions"][row["retain_action"]]
        feature = features[identity]
        predicted_alphas = action_alphas(row, name)
        teacher_alphas = [
            int(not value) for value in feature["targets"]["delete_targets"]
        ]
        predicted_delete = [int(value == 0) for value in predicted_alphas]
        teacher_delete = feature["targets"]["delete_targets"]
        queries += outcome["queries"]
        correct += outcome["correct"]
        nll += outcome["total_answer_nll"]
        unrelated_queries += outcome["unrelated_queries"]
        unrelated_correct += outcome["unrelated_correct"]
        added_errors += max(
            0, baseline["unrelated_correct"] - outcome["unrelated_correct"]
        )
        deletions += sum(predicted_delete)
        candidates += len(predicted_delete)
        deletion_true_positive += sum(
            predicted and target
            for predicted, target in zip(predicted_delete, teacher_delete)
        )
        teacher_action_agreement += int(name == safe_binary_action(row))
        teacher_record_agreement += sum(
            predicted == target
            for predicted, target in zip(predicted_alphas, teacher_alphas)
        )
        mechanism = by_mechanism[row["mechanism"]]
        mechanism[0] += outcome["correct"]
        mechanism[1] += outcome["queries"]
        relation = by_relation[row["target_relation"]]
        relation[0] += outcome["correct"]
        relation[1] += outcome["queries"]
        for role, value in outcome["per_query_role"].items():
            aggregate = by_role[role]
            aggregate[0] += value["correct"]
            aggregate[1] += value["queries"]
            aggregate[2] += value["total_answer_nll"]
    return {
        "accuracy": correct / queries,
        "correct": correct,
        "queries": queries,
        "mean_answer_nll": nll / queries,
        "unrelated_accuracy": unrelated_correct / unrelated_queries,
        "added_unrelated_errors": added_errors,
        "deletion_fraction": deletions / candidates,
        "deletion_precision": (
            deletion_true_positive / deletions if deletions else 1.0
        ),
        "binary_teacher_action_agreement": teacher_action_agreement / len(rows),
        "binary_teacher_record_agreement": teacher_record_agreement / candidates,
        "per_mechanism_accuracy": {
            key: value[0] / value[1]
            for key, value in sorted(by_mechanism.items())
        },
        "per_relation_accuracy": {
            key: value[0] / value[1]
            for key, value in sorted(by_relation.items())
        },
        "per_query_role": {
            key: {
                "accuracy": value[0] / value[1],
                "mean_answer_nll": value[2] / value[1],
                "queries": value[1],
            }
            for key, value in sorted(by_role.items())
        },
    }


def group_gain(rows: list[dict], left: dict, right: dict) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    output = {}
    for group, values in grouped.items():
        numerator = sum(
            row["actions"][left[(row["episode_id"], row["boundary_index"])]]
            ["correct"]
            - row["actions"][right[(row["episode_id"], row["boundary_index"])]]
            ["correct"]
            for row in values
        )
        denominator = sum(
            row["actions"][row["retain_action"]]["queries"] for row in values
        )
        output[group] = numerator / denominator
    return output


def bootstrap(values: dict[str, float], rng: random.Random) -> list[float]:
    keys = sorted(values)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(statistics.fmean(values[key] for key in sample))
    return [percentile(draws, .025), percentile(draws, .975)]


def pooled_group_bootstrap(group_gains: dict[int, dict[str, float]],
                           groups: list[str]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    seeds = sorted(group_gains)
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(
            group_gains[seeds[rng.randrange(len(seeds))]][group]
            for group in sample
        ))
    return [percentile(draws, .025), percentile(draws, .975)]


def pooled_relation_bootstrap(rows: list[dict], group_gains: dict[int, dict[str, float]]) \
        -> list[float]:
    relation_by_group = {}
    for row in rows:
        relation_by_group[row["source_group"]] = row["target_relation"]
    by_seed_relation = {}
    for seed, gains in group_gains.items():
        grouped = defaultdict(list)
        for group, gain in gains.items():
            grouped[relation_by_group[group]].append(gain)
        by_seed_relation[seed] = {
            relation: statistics.fmean(values)
            for relation, values in grouped.items()
        }
    relations = sorted(next(iter(by_seed_relation.values())))
    seeds = sorted(by_seed_relation)
    rng = random.Random(BOOTSTRAP_SEED + 1)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [relations[rng.randrange(len(relations))] for _ in relations]
        draws.append(statistics.fmean(
            by_seed_relation[seeds[rng.randrange(len(seeds))]][relation]
            for relation in sample
        ))
    return [percentile(draws, .025), percentile(draws, .975)]


def background_multiway_bootstrap(
    group_gains: dict[int, dict[str, float]],
    backgrounds: dict[str, list[str]],
) -> list[float]:
    """Bayesian two-way bootstrap over target groups and background IDs."""
    groups = sorted(backgrounds)
    background_ids = sorted({value for group in groups for value in backgrounds[group]})
    seeds = sorted(group_gains)
    rng = random.Random(BOOTSTRAP_SEED + 2)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        group_weights = {group: rng.expovariate(1.0) for group in groups}
        background_weights = {
            identity: rng.expovariate(1.0) for identity in background_ids
        }
        weighted = []
        weights = []
        for group in groups:
            background_weight = statistics.fmean(
                background_weights[value] for value in backgrounds[group]
            )
            weight = group_weights[group] * background_weight
            seed = seeds[rng.randrange(len(seeds))]
            weighted.append(weight * group_gains[seed][group])
            weights.append(weight)
        draws.append(sum(weighted) / sum(weights))
    return [percentile(draws, .025), percentile(draws, .975)]


def random_matched_actions(rows: list[dict], reference: dict, draw: int) -> dict:
    rng = random.Random(BOOTSTRAP_SEED + 1000 + draw)
    output = {}
    for row in sorted(rows, key=lambda value: (
        value["episode_id"], value["boundary_index"]
    )):
        identity = (row["episode_id"], row["boundary_index"])
        deletions = action_alphas(row, reference[identity]).count(0)
        deleted = set(rng.sample(range(row["candidate_count"]), deletions))
        name = grid_action([
            int(index not in deleted) for index in range(row["candidate_count"])
        ])
        if name not in row["actions"]:
            raise ValueError("deletion-matched random action missing")
        output[identity] = name
    return output


def majority_actions(rows: list[dict], actions_by_seed: dict[int, dict]) -> dict:
    output = {}
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        votes = [
            action_alphas(row, actions_by_seed[seed][identity])
            for seed in sorted(actions_by_seed)
        ]
        alphas = [
            int(sum(values[slot] for values in votes) >= 2)
            for slot in range(row["candidate_count"])
        ]
        output[identity] = grid_action(alphas)
    return output


def evaluate(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    pilot_root = Path(args.pilot_root)
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    label_summary = json.loads((label_root / "summary.json").read_text())
    pilot_summary = json.loads((pilot_root / "summary.json").read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("TempLAMA development feature audit did not pass")
    if (training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL
            or not training_audit.get("passed")):
        raise ValueError("V1 controller training audit did not pass")
    if sha256(training_audit_path) != EXPECTED_TRAINING_AUDIT_SHA256:
        raise ValueError("V1 controller training audit changed")

    labels = []
    feature_rows = []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(
            feature_root / f"features_shard{shard}.jsonl"
        ))
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    twin_zero = [row for row in episodes if row["twin"] == 0]
    backgrounds = {
        row["source_group"]: row["audit"]["background_subject_ids"]
        for row in twin_zero
    }
    background_heldout = {
        row["source_group"] for row in twin_zero
        if row["audit"]["background_heldout"]
    }

    baseline_actions = {}
    baseline_results = {}
    fixed = (
        "retain_all", "global_clear", "last_1", "last_2",
        "exact_string_delete", "exact_subject_conflict",
        "explicit_lifecycle_physics",
    )
    for name in fixed:
        actions = fixed_policy_actions(labels, features, name)
        baseline_actions[name] = actions
        baseline_results[name] = metric(labels, actions, features)
    for threshold in COSINE_THRESHOLDS:
        name = f"frozen_cosine_delete_{threshold}"
        actions = fixed_policy_actions(
            labels, features, "frozen_cosine_delete", threshold
        )
        baseline_actions[name] = actions
        baseline_results[name] = metric(labels, actions, features)
    strongest_baseline = select_safe(baseline_results)
    if strongest_baseline is None:
        raise ValueError("no safe deterministic TempLAMA baseline")
    strongest_actions = baseline_actions[strongest_baseline]
    strongest_metric = baseline_results[strongest_baseline]

    controller_results = {}
    controller_actions_by_seed = {}
    group_gains = {}
    prediction_seconds = {}
    for seed in SEEDS:
        checkpoint = load_checkpoint(
            training_root / f"seed_{seed}" / "controller.pt"
        )
        started = time.perf_counter()
        predictions = predict_checkpoint(checkpoint, feature_rows)
        prediction_seconds[seed] = time.perf_counter() - started
        config = FROZEN_CONFIGS[seed]
        actions = controller_actions(
            labels, features, predictions,
            config["utility_threshold"], config["risk_penalty"],
            config["risk_threshold"],
        )
        controller_actions_by_seed[seed] = actions
        result = metric(labels, actions, features)
        gains = group_gain(labels, actions, strongest_actions)
        group_gains[seed] = gains
        relation_gain = {
            relation: result["per_relation_accuracy"][relation]
            - strongest_metric["per_relation_accuracy"][relation]
            for relation in result["per_relation_accuracy"]
        }
        mechanism_gain = {
            mechanism: result["per_mechanism_accuracy"][mechanism]
            - strongest_metric["per_mechanism_accuracy"][mechanism]
            for mechanism in result["per_mechanism_accuracy"]
        }
        random_metrics = []
        for draw in range(RANDOM_DRAWS):
            random_actions = random_matched_actions(labels, actions, draw)
            random_metrics.append(metric(labels, random_actions, features))
        gain = result["accuracy"] - strongest_metric["accuracy"]
        oracle_gap = None
        controller_results[str(seed)] = {
            "frozen_config": config,
            "checkpoint_sha256": sha256(
                training_root / f"seed_{seed}" / "controller.pt"
            ),
            "prediction_seconds": prediction_seconds[seed],
            "evaluation": result,
            "gain_over_strongest_baseline": gain,
            "gain_over_strongest_baseline_group_bootstrap_95ci": bootstrap(
                gains, random.Random(BOOTSTRAP_SEED + seed)
            ),
            "per_mechanism_gain": mechanism_gain,
            "per_relation_gain": relation_gain,
            "macro_relation_gain": statistics.fmean(relation_gain.values()),
            "deletion_matched_random": {
                "draws": RANDOM_DRAWS,
                "accuracy_mean": statistics.fmean(
                    value["accuracy"] for value in random_metrics
                ),
                "accuracy_min": min(value["accuracy"] for value in random_metrics),
                "accuracy_max": max(value["accuracy"] for value in random_metrics),
                "added_unrelated_errors_mean": statistics.fmean(
                    value["added_unrelated_errors"] for value in random_metrics
                ),
            },
        }

    oracle_actions = {
        (row["episode_id"], row["boundary_index"]): safe_binary_action(row)
        for row in labels
    }
    oracle = metric(labels, oracle_actions, features)
    oracle_headroom = oracle["accuracy"] - strongest_metric["accuracy"]
    for seed in SEEDS:
        gain = controller_results[str(seed)]["gain_over_strongest_baseline"]
        controller_results[str(seed)]["oracle_gap_closure"] = (
            gain / oracle_headroom if oracle_headroom > 0 else 0.0
        )

    majority = majority_actions(labels, controller_actions_by_seed)
    majority_metric = metric(labels, majority, features)
    groups = sorted(backgrounds)
    pooled_interval = pooled_group_bootstrap(group_gains, groups)
    relation_interval = pooled_relation_bootstrap(labels, group_gains)
    background_interval = background_multiway_bootstrap(group_gains, backgrounds)
    heldout_gains = {
        seed: {group: value for group, value in gains.items()
               if group in background_heldout}
        for seed, gains in group_gains.items()
    }
    heldout_interval = pooled_group_bootstrap(
        heldout_gains, sorted(background_heldout)
    )
    heldout_mean = statistics.fmean(
        value for gains in heldout_gains.values() for value in gains.values()
    )

    seed_rows = [controller_results[str(seed)] for seed in SEEDS]
    mean_gain = statistics.fmean(
        row["gain_over_strongest_baseline"] for row in seed_rows
    )
    mean_gap_closure = statistics.fmean(
        row["oracle_gap_closure"] for row in seed_rows
    )
    mechanisms = sorted(strongest_metric["per_mechanism_accuracy"])
    mean_mechanism_gain = {
        mechanism: statistics.fmean(
            row["per_mechanism_gain"][mechanism] for row in seed_rows
        ) for mechanism in mechanisms
    }
    relations = sorted(strongest_metric["per_relation_accuracy"])
    mean_relation_gain = {
        relation: statistics.fmean(
            row["per_relation_gain"][relation] for row in seed_rows
        ) for relation in relations
    }
    macro_relation_gain = statistics.fmean(mean_relation_gain.values())
    release_checks = {
        "safe_binary_oracle_headroom_at_least_point_03": oracle_headroom >= .03,
        "every_seed_gain_at_least_point_01": all(
            row["gain_over_strongest_baseline"] >= .01 for row in seed_rows
        ),
        "mean_seed_gain_at_least_point_02": mean_gain >= .02,
        "pooled_group_bootstrap_lower_above_zero": pooled_interval[0] > 0,
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0
            for row in seed_rows
        ),
        "mean_oracle_gap_closure_at_least_point_40": mean_gap_closure >= .40,
        "overwrite_and_stale_mean_gains_nonnegative": (
            mean_mechanism_gain["semantic_overwrite"] >= 0
            and mean_mechanism_gain["stale_repetition"] >= 0
        ),
        "relation_robustness": (
            macro_relation_gain > 0
            and sum(value >= 0 for value in mean_relation_gain.values()) >= 8
            and min(mean_relation_gain.values()) >= -.02
        ),
        "upstream_audits_and_training_lock_passed": (
            feature_audit["passed"] and training_audit["passed"]
            and sha256(training_audit_path) == EXPECTED_TRAINING_AUDIT_SHA256
        ),
    }
    candidate_passed = all(release_checks.values())
    return {
        "protocol": PROTOCOL,
        "partition": "templama_validation_development",
        "groups": len(groups),
        "strongest_safe_deterministic_baseline": strongest_baseline,
        "baselines": baseline_results,
        "controllers": controller_results,
        "majority_controller": majority_metric,
        "safe_binary_oracle": oracle,
        "safe_binary_oracle_headroom": oracle_headroom,
        "resource_pilot_full_oracle": {
            "safe_oracle_accuracy": pilot_summary["safe_oracle_accuracy"],
            "retain_all_accuracy": pilot_summary["retain_all_accuracy"],
            "safe_oracle_gain": pilot_summary["safe_oracle_gain"],
            "summary_sha256": sha256(pilot_root / "summary.json"),
        },
        "pooled_three_seed": {
            "mean_gain": mean_gain,
            "group_bootstrap_95ci": pooled_interval,
            "relation_cluster_bootstrap_95ci": relation_interval,
            "background_multiway_bootstrap_95ci": background_interval,
            "background_heldout_groups": len(background_heldout),
            "background_heldout_mean_gain": heldout_mean,
            "background_heldout_group_bootstrap_95ci": heldout_interval,
            "mean_oracle_gap_closure": mean_gap_closure,
            "mean_per_mechanism_gain": mean_mechanism_gain,
            "mean_per_relation_gain": mean_relation_gain,
            "macro_relation_gain": macro_relation_gain,
            "nonnegative_relations": sum(
                value >= 0 for value in mean_relation_gain.values()
            ),
            "worst_relation_gain": min(mean_relation_gain.values()),
        },
        "release_candidate_gate": {
            "checks": release_checks,
            "passed_pending_independent_audit": candidate_passed,
        },
        "feature_audit_sha256": sha256(feature_audit_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "training_audit_sha256": sha256(training_audit_path),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "independent_development_result_audit",
        "terminal_status": "templama_development_evaluation_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
