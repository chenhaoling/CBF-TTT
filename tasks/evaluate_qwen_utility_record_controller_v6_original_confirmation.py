"""One-time original-source LAMA/ParaRel confirmation of frozen V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
    load_checkpoint,
)
from scripts.audit_qwen_utility_record_controller_v6_templama_test import (
    AUDIT_PROTOCOL as RELEASE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v6_original_confirmation_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v6_original_confirmation_labels import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    COSINE_THRESHOLDS,
    fixed_policy_actions,
    select_safe,
)
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    BOOTSTRAP_SEED,
    RANDOM_DRAWS,
    bootstrap,
    group_gain,
    majority_actions,
    metric,
    pooled_group_bootstrap,
    pooled_relation_bootstrap,
)
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    collateral_details,
    fact_key_matches,
    fact_key_projected_actions,
    random_fact_key_matched_actions,
    safe_fact_key_oracle_actions,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.train_qwen_utility_record_controller_v5_residual import (
    PROTOCOL as TRAINING_PROTOCOL,
    SEEDS,
    predict_state,
    residual_actions,
    residual_examples,
    residual_inference_candidates,
    residual_precision,
)


PROTOCOL = "qwen_utility_record_controller_v6_original_source_confirmation"
AUDIT_PROTOCOL = (
    "qwen_utility_record_controller_v6_original_source_confirmation_audit"
)


def evaluate(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    release_root = Path(args.release_root)
    data_audit_path = data_root / "data_audit.json"
    label_audit_path = label_root / "labels_audit.json"
    feature_audit_path = feature_root / "features_audit.json"
    training_audit_path = training_root / "training_audit.json"
    release_result_path = release_root / "result.json"
    release_audit_path = release_root / "result_audit.json"
    data_audit = json.loads(data_audit_path.read_text())
    label_audit = json.loads(label_audit_path.read_text())
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    release_audit = json.loads(release_audit_path.read_text())
    if not data_audit.get("passed"):
        raise ValueError("original-source data audit did not pass")
    if label_audit.get("protocol") != LABEL_AUDIT_PROTOCOL or not label_audit.get("passed"):
        raise ValueError("original-source label audit did not pass")
    if feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL or not feature_audit.get("passed"):
        raise ValueError("original-source feature audit did not pass")
    if training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL or not training_audit.get("passed"):
        raise ValueError("V6 training audit did not pass")
    if (release_audit.get("protocol") != RELEASE_AUDIT_PROTOCOL
            or not release_audit.get("passed")
            or not release_audit.get("confirmation_passed")
            or release_audit.get("result_sha256")
            != sha256(release_result_path)):
        raise ValueError("V6 TempLAMA test did not release original-source confirmation")

    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    episodes = load_jsonl(data_root / "confirmation_episodes.jsonl")
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    groups = sorted({row["source_group"] for row in labels})
    if len(groups) != 64 or len(labels) != len(features):
        raise ValueError("V6 original-source coverage is incomplete")
    matches = fact_key_matches(labels, episodes)
    baseline_actions, baseline_results = {}, {}
    for name in (
        "retain_all", "global_clear", "last_1", "last_2",
        "exact_string_delete", "exact_subject_conflict",
        "explicit_lifecycle_physics",
    ):
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
        raise ValueError("no safe deterministic original-source baseline")
    physics_name = "explicit_lifecycle_physics"
    physics_actions = baseline_actions[physics_name]
    physics = baseline_results[physics_name]

    candidates = residual_inference_candidates(labels, feature_rows)
    scored_examples = residual_examples(labels, feature_rows)
    controller_results, actions_by_seed, group_gains = {}, {}, {}
    raw_collateral, projected_collateral = {}, {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("protocol") != TRAINING_PROTOCOL or checkpoint["seed"] != seed:
            raise ValueError(f"invalid frozen V6 checkpoint for seed {seed}")
        predictions = predict_state(
            checkpoint["state_dict"], candidates,
            checkpoint["means"], checkpoint["stds"],
        )
        raw_actions = residual_actions(
            labels, feature_rows, predictions, checkpoint["selected_config"]
        )
        raw_value = metric(labels, raw_actions, features)
        actions = fact_key_projected_actions(
            labels, feature_rows, raw_actions, matches
        )
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
                random_fact_key_matched_actions(
                    labels, feature_rows, actions, matches,
                    draw + 100 * (seed - SEEDS[0]),
                ),
                features,
            )
            for draw in range(RANDOM_DRAWS)
        ]
        raw_collateral[str(seed)] = collateral_details(
            labels, raw_actions, physics_actions, matches
        )
        projected_collateral[str(seed)] = collateral_details(
            labels, actions, physics_actions, matches
        )
        controller_results[str(seed)] = {
            "selected_train_config": checkpoint["selected_config"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "subject_only_ablation": raw_value,
            "evaluation": value,
            "filtered_cross_fact_deletions": sum(
                raw_actions[key] != actions[key] for key in actions
            ),
            "gain_over_lifecycle_physics": value["accuracy"] - physics["accuracy"],
            "gain_over_lifecycle_physics_group_bootstrap_95ci": bootstrap(
                gains, random.Random(BOOTSTRAP_SEED + 400_000 + seed)
            ),
            "per_mechanism_gain": mechanism_gain,
            "per_relation_gain": relation_gain,
            "macro_relation_gain": statistics.fmean(relation_gain.values()),
            "fact_key_deletion_matched_random": {
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

    oracle_actions = safe_fact_key_oracle_actions(labels, feature_rows, matches)
    oracle = metric(labels, oracle_actions, features)
    headroom = oracle["accuracy"] - physics["accuracy"]
    for seed in SEEDS:
        gain = controller_results[str(seed)]["gain_over_lifecycle_physics"]
        controller_results[str(seed)]["oracle_gap_closure"] = (
            gain / headroom if headroom > 0 else 0.0
        )
    seed_rows = [controller_results[str(seed)] for seed in SEEDS]
    majority = metric(labels, majority_actions(labels, actions_by_seed), features)
    pooled_interval = pooled_group_bootstrap(group_gains, groups)
    relation_interval = pooled_relation_bootstrap(labels, group_gains)
    mean_gain = statistics.fmean(
        row["gain_over_lifecycle_physics"] for row in seed_rows
    )
    mean_closure = statistics.fmean(row["oracle_gap_closure"] for row in seed_rows)
    mean_mechanism = {
        name: statistics.fmean(row["per_mechanism_gain"][name] for row in seed_rows)
        for name in sorted(physics["per_mechanism_accuracy"])
    }
    mean_relation = {
        name: statistics.fmean(row["per_relation_gain"][name] for row in seed_rows)
        for name in sorted(physics["per_relation_accuracy"])
    }
    confirmation_checks = {
        "every_seed_gain_above_zero": all(
            row["gain_over_lifecycle_physics"] > 0 for row in seed_rows
        ),
        "mean_seed_gain_at_least_point_02": mean_gain >= .02,
        "pooled_group_bootstrap_lower_above_zero": pooled_interval[0] > 0,
        "relation_cluster_bootstrap_lower_above_zero": relation_interval[0] > 0,
        "every_seed_zero_added_unrelated_errors": all(
            row["evaluation"]["added_unrelated_errors"] == 0 for row in seed_rows
        ),
        "mean_oracle_gap_closure_at_least_point_30": mean_closure >= .30,
        "every_mechanism_nonnegative": min(mean_mechanism.values()) >= 0,
        "upstream_audits_passed": all((
            data_audit["passed"], label_audit["passed"], feature_audit["passed"],
            training_audit["passed"], release_audit["confirmation_passed"],
        )),
    }
    label_summary = json.loads((label_root / "summary.json").read_text())
    feature_summary = json.loads((feature_root / "summary.json").read_text())
    return {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V6_ORIGINAL_SOURCE_CONFIRMATION_PLAN.md",
        "partition": "lama_pararel_original_source_confirmation",
        "groups": len(groups),
        "strongest_safe_deterministic_baseline": strongest,
        "release_baseline": physics_name,
        "baselines": baseline_results,
        "controllers": controller_results,
        "majority_controller": majority,
        "safe_single_fact_key_oracle": oracle,
        "safe_single_fact_key_oracle_headroom": headroom,
        "collateral_localization": {
            "subject_only": raw_collateral,
            "fact_key": projected_collateral,
        },
        "pooled_three_seed": {
            "mean_gain": mean_gain,
            "group_bootstrap_95ci": pooled_interval,
            "relation_cluster_bootstrap_95ci": relation_interval,
            "mean_oracle_gap_closure": mean_closure,
            "mean_per_mechanism_gain": mean_mechanism,
            "mean_per_relation_gain": mean_relation,
            "macro_relation_gain": statistics.fmean(mean_relation.values()),
            "nonnegative_relations": sum(value >= 0 for value in mean_relation.values()),
            "worst_relation_gain": min(mean_relation.values()),
        },
        "original_source_confirmation_gate": {
            "checks": confirmation_checks,
            "passed_pending_independent_audit": all(confirmation_checks.values()),
        },
        "resource": {
            "label_two_gpu_critical_seconds": label_summary["resource"][
                "two_gpu_critical_seconds"
            ],
            "label_peak_reserved_gib": label_summary["resource"][
                "peak_reserved_gib"
            ],
            "feature_two_gpu_critical_seconds": feature_summary["resource"][
                "two_gpu_critical_seconds"
            ],
            "feature_peak_reserved_gib": feature_summary["resource"][
                "peak_reserved_gib"
            ],
        },
        "data_audit_sha256": sha256(data_audit_path),
        "label_audit_sha256": sha256(label_audit_path),
        "feature_audit_sha256": sha256(feature_audit_path),
        "training_audit_sha256": sha256(training_audit_path),
        "templama_release_result_sha256": sha256(release_result_path),
        "templama_release_audit_sha256": sha256(release_audit_path),
        "counterfactual_targets_available_to_controller": False,
        "templama_test_loaded": True,
        "parameters_frozen_before_original_source_outcomes": True,
        "v1_confirmation_loaded": True,
        "external_benchmarks_loaded": False,
        "next_stage": "independent_v6_original_source_result_audit",
        "terminal_status": "v6_original_source_confirmation_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = evaluate(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
