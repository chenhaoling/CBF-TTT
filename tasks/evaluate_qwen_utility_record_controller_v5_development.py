"""Evaluate V5 exact-subject residual forgetting on TempLAMA development."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    COSINE_THRESHOLDS,
    fixed_policy_actions,
    grid_action,
    select_safe,
)
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    BOOTSTRAP_SEED,
    RANDOM_DRAWS,
    action_alphas,
    background_multiway_bootstrap,
    bootstrap,
    group_gain,
    majority_actions,
    metric,
    pooled_group_bootstrap,
    pooled_relation_bootstrap,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.train_qwen_utility_record_controller_v5_residual import (
    LOCALITY_RULE,
    PROTOCOL as TRAINING_PROTOCOL,
    SEEDS,
    predict_state,
    residual_actions,
    residual_examples,
    residual_inference_candidates,
    residual_precision,
)


PROTOCOL = "qwen_utility_record_controller_v5_templama_development"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v5_templama_development_audit"


def safe_single_residual_oracle_actions(labels: list[dict],
                                        feature_rows: list[dict]) -> dict:
    """Best safe physics-or-one-extra-delete action at each boundary."""
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    candidates = residual_inference_candidates(labels, feature_rows)
    by_boundary = defaultdict(list)
    for candidate in candidates:
        by_boundary[(candidate["episode_id"], candidate["boundary_index"])].append(
            candidate
        )
    physics = fixed_policy_actions(labels, features, "explicit_lifecycle_physics")
    output = {}
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        baseline_name = physics[identity]
        retain = row["actions"][row["retain_action"]]
        best_name = baseline_name
        best = row["actions"][best_name]
        for candidate in by_boundary[identity]:
            name = candidate["candidate_action"]
            value = row["actions"][name]
            safe = value["unrelated_correct"] >= retain["unrelated_correct"]
            if safe and (
                value["correct"], -value["total_answer_nll"]
            ) > (
                best["correct"], -best["total_answer_nll"]
            ):
                best_name, best = name, value
        output[identity] = best_name
    return output


def random_residual_matched_actions(labels: list[dict],
                                    feature_rows: list[dict],
                                    reference: dict, draw: int) -> dict:
    """Randomize only V5's extra deletion while preserving physics actions."""
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    physics = fixed_policy_actions(labels, features, "explicit_lifecycle_physics")
    candidates = residual_inference_candidates(labels, feature_rows)
    by_boundary = defaultdict(list)
    for candidate in candidates:
        by_boundary[(candidate["episode_id"], candidate["boundary_index"])].append(
            candidate["slot"]
        )
    rng = random.Random(BOOTSTRAP_SEED + 5000 + draw)
    output = {}
    for row in sorted(labels, key=lambda value: (
        value["episode_id"], value["boundary_index"]
    )):
        identity = (row["episode_id"], row["boundary_index"])
        baseline_name = physics[identity]
        if baseline_name == "global_clear":
            output[identity] = baseline_name
            continue
        baseline = action_alphas(row, baseline_name)
        selected = action_alphas(row, reference[identity])
        residual_count = sum(
            base == 1 and value == 0
            for base, value in zip(baseline, selected)
        )
        if residual_count not in (0, 1):
            raise ValueError("V5 reference selected more than one residual deletion")
        if not residual_count:
            output[identity] = baseline_name
            continue
        eligible = by_boundary[identity]
        if not eligible:
            raise ValueError("residual-matched random action has no eligible record")
        alphas = list(baseline)
        alphas[eligible[rng.randrange(len(eligible))]] = 0
        name = grid_action(alphas)
        if name not in row["actions"]:
            raise ValueError(f"missing random residual action {name}")
        output[identity] = name
    return output


def previous_result(root: Path) -> dict:
    result_path = root / "result.json"
    audit_path = root / "result_audit.json"
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("result_sha256") != sha256(result_path):
        raise ValueError(f"prior development result did not pass audit: {root}")
    controllers = result.get("controllers", {})
    return {
        "protocol": result["protocol"],
        "partition": result["partition"],
        "groups": result["groups"],
        "strongest_safe_deterministic_baseline": result[
            "strongest_safe_deterministic_baseline"
        ],
        "controller_gains": {
            seed: row["gain_over_strongest_baseline"]
            for seed, row in controllers.items()
        },
        "pooled_three_seed": result["pooled_three_seed"],
        "release_candidate_gate": result["release_candidate_gate"],
        "audit_release_passed": audit.get("release_passed", False),
        "result_sha256": sha256(result_path),
        "audit_sha256": sha256(audit_path),
    }


def evaluate(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    pilot_root = Path(args.pilot_root)
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    training_summary_path = training_root / "summary.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    training_summary = json.loads(training_summary_path.read_text())
    pilot_summary = json.loads((pilot_root / "summary.json").read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("TempLAMA development feature audit did not pass")
    if (training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL
            or not training_audit.get("passed")):
        raise ValueError("V5 residual training audit did not pass")
    if (training_summary.get("protocol") != TRAINING_PROTOCOL
            or not training_summary.get("passed")
            or training_audit.get("summary_sha256") != sha256(training_summary_path)):
        raise ValueError("V5 residual training summary changed after audit")

    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    if len(features) != len(feature_rows) or len(labels) != len(feature_rows):
        raise ValueError("TempLAMA development label/feature coverage mismatch")
    groups = sorted({row["source_group"] for row in labels})
    if len(groups) != 256:
        raise ValueError("V5 requires the complete 256-group development set")

    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    twin_zero = [row for row in episodes if row["twin"] == 0]
    backgrounds = {
        row["source_group"]: row["audit"]["background_subject_ids"]
        for row in twin_zero
    }
    heldout = {
        row["source_group"] for row in twin_zero
        if row["audit"]["background_heldout"]
    }
    if set(groups) != set(backgrounds) or not heldout:
        raise ValueError("V5 background diagnostics are incomplete")

    baseline_actions, baseline_results = {}, {}
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
    strongest = select_safe(baseline_results)
    if strongest is None:
        raise ValueError("no safe TempLAMA development baseline")
    physics_name = "explicit_lifecycle_physics"
    physics = baseline_results[physics_name]
    physics_actions = baseline_actions[physics_name]

    inference_candidates = residual_inference_candidates(labels, feature_rows)
    if any("benefit_target" in row or "risk_target" in row
           for row in inference_candidates):
        raise ValueError("V5 inference candidates contain counterfactual targets")
    scored_examples = residual_examples(labels, feature_rows)
    controller_results, actions_by_seed, group_gains = {}, {}, {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("protocol") != TRAINING_PROTOCOL or checkpoint["seed"] != seed:
            raise ValueError(f"invalid V5 checkpoint for seed {seed}")
        predictions = predict_state(
            checkpoint["state_dict"], inference_candidates,
            checkpoint["means"], checkpoint["stds"],
        )
        config = checkpoint["selected_config"]
        actions = residual_actions(labels, feature_rows, predictions, config)
        actions_by_seed[seed] = actions
        value = metric(labels, actions, features)
        precision, fraction, selected = residual_precision(
            scored_examples, actions, labels
        )
        value.update({
            "residual_deletion_precision": precision,
            "residual_deletion_fraction": fraction,
            "residual_deletions": selected,
        })
        gains = group_gain(labels, actions, physics_actions)
        group_gains[seed] = gains
        relation_gain = {
            relation: value["per_relation_accuracy"][relation]
            - physics["per_relation_accuracy"][relation]
            for relation in value["per_relation_accuracy"]
        }
        mechanism_gain = {
            mechanism: value["per_mechanism_accuracy"][mechanism]
            - physics["per_mechanism_accuracy"][mechanism]
            for mechanism in value["per_mechanism_accuracy"]
        }
        random_metrics = [
            metric(
                labels,
                random_residual_matched_actions(
                    labels, feature_rows, actions, draw + 100 * (seed - SEEDS[0])
                ),
                features,
            )
            for draw in range(RANDOM_DRAWS)
        ]
        controller_results[str(seed)] = {
            "selected_train_config": config,
            "checkpoint_sha256": sha256(checkpoint_path),
            "evaluation": value,
            "gain_over_lifecycle_physics": value["accuracy"] - physics["accuracy"],
            "gain_over_lifecycle_physics_group_bootstrap_95ci": bootstrap(
                gains, random.Random(BOOTSTRAP_SEED + 200_000 + seed)
            ),
            "per_mechanism_gain": mechanism_gain,
            "per_relation_gain": relation_gain,
            "macro_relation_gain": statistics.fmean(relation_gain.values()),
            "residual_deletion_matched_random": {
                "draws": RANDOM_DRAWS,
                "accuracy_mean": statistics.fmean(
                    row["accuracy"] for row in random_metrics
                ),
                "accuracy_min": min(row["accuracy"] for row in random_metrics),
                "accuracy_max": max(row["accuracy"] for row in random_metrics),
                "added_unrelated_errors_mean": statistics.fmean(
                    row["added_unrelated_errors"] for row in random_metrics
                ),
            },
        }

    oracle_actions = safe_single_residual_oracle_actions(labels, feature_rows)
    oracle = metric(labels, oracle_actions, features)
    oracle_headroom = oracle["accuracy"] - physics["accuracy"]
    for seed in SEEDS:
        gain = controller_results[str(seed)]["gain_over_lifecycle_physics"]
        controller_results[str(seed)]["oracle_gap_closure"] = (
            gain / oracle_headroom if oracle_headroom > 0 else 0.0
        )
    seed_rows = [controller_results[str(seed)] for seed in SEEDS]
    majority = metric(labels, majority_actions(labels, actions_by_seed), features)
    pooled_interval = pooled_group_bootstrap(group_gains, groups)
    relation_interval = pooled_relation_bootstrap(labels, group_gains)
    background_interval = background_multiway_bootstrap(group_gains, backgrounds)
    heldout_gains = {
        seed: {group: gain for group, gain in values.items() if group in heldout}
        for seed, values in group_gains.items()
    }
    heldout_interval = pooled_group_bootstrap(heldout_gains, sorted(heldout))
    mean_gain = statistics.fmean(
        row["gain_over_lifecycle_physics"] for row in seed_rows
    )
    mean_closure = statistics.fmean(row["oracle_gap_closure"] for row in seed_rows)
    mechanisms = sorted(physics["per_mechanism_accuracy"])
    mean_mechanism = {
        name: statistics.fmean(row["per_mechanism_gain"][name] for row in seed_rows)
        for name in mechanisms
    }
    relations = sorted(physics["per_relation_accuracy"])
    mean_relation = {
        name: statistics.fmean(row["per_relation_gain"][name] for row in seed_rows)
        for name in relations
    }
    macro_relation = statistics.fmean(mean_relation.values())
    release_checks = {
        "safe_single_residual_oracle_headroom_at_least_point_03": (
            oracle_headroom >= .03
        ),
        "every_seed_gain_at_least_point_01": all(
            row["gain_over_lifecycle_physics"] >= .01 for row in seed_rows
        ),
        "mean_seed_gain_at_least_point_02": mean_gain >= .02,
        "pooled_group_bootstrap_lower_above_zero": pooled_interval[0] > 0,
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0 for row in seed_rows
        ),
        "mean_oracle_gap_closure_at_least_point_40": mean_closure >= .40,
        "overwrite_and_stale_mean_gains_nonnegative": (
            mean_mechanism["semantic_overwrite"] >= 0
            and mean_mechanism["stale_repetition"] >= 0
        ),
        "relation_robustness": (
            macro_relation > 0
            and sum(value >= 0 for value in mean_relation.values()) >= 8
            and min(mean_relation.values()) >= -.02
        ),
        "upstream_audits_passed": (
            feature_audit["passed"] and training_audit["passed"]
        ),
    }
    prior = {
        "v2": previous_result(Path(args.v2_root)),
        "v3": previous_result(Path(args.v3_root)),
        "v4": previous_result(Path(args.v4_root)),
    }
    return {
        "protocol": PROTOCOL,
        "partition": "templama_validation_full_development",
        "groups": len(groups),
        "strongest_safe_deterministic_baseline": strongest,
        "release_baseline": physics_name,
        "baselines": baseline_results,
        "prior_audited_stages": prior,
        "controllers": controller_results,
        "majority_controller": majority,
        "safe_single_residual_oracle": oracle,
        "safe_single_residual_oracle_headroom": oracle_headroom,
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
            "background_heldout_groups": len(heldout),
            "background_heldout_mean_gain": statistics.fmean(
                gain for values in heldout_gains.values() for gain in values.values()
            ),
            "background_heldout_group_bootstrap_95ci": heldout_interval,
            "mean_oracle_gap_closure": mean_closure,
            "mean_per_mechanism_gain": mean_mechanism,
            "mean_per_relation_gain": mean_relation,
            "macro_relation_gain": macro_relation,
            "nonnegative_relations": sum(value >= 0 for value in mean_relation.values()),
            "worst_relation_gain": min(mean_relation.values()),
        },
        "release_candidate_gate": {
            "checks": release_checks,
            "passed_pending_independent_audit": all(release_checks.values()),
        },
        "feature_audit_sha256": sha256(feature_audit_path),
        "training_audit_sha256": sha256(training_audit_path),
        "training_summary_sha256": sha256(training_summary_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "inference_candidates": len(inference_candidates),
        "counterfactual_targets_available_to_controller": False,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "independent_v5_development_result_audit",
        "terminal_status": "templama_v5_development_evaluation_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--v4-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = evaluate(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
