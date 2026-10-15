"""Evaluate domain-trained V4 controllers on TempLAMA validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v4_templama_selection import (
    AUDIT_PROTOCOL as SELECTION_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    COSINE_THRESHOLDS,
    fixed_policy_actions,
    load_checkpoint,
    predict_checkpoint,
    select_safe,
)
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    BOOTSTRAP_SEED,
    RANDOM_DRAWS,
    background_multiway_bootstrap,
    bootstrap,
    group_gain,
    majority_actions,
    metric,
    pooled_group_bootstrap,
    pooled_relation_bootstrap,
    random_matched_actions,
)
from tasks.extract_qwen_utility_record_controller_v1_features import safe_binary_action
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.select_qwen_utility_record_controller_v4_templama import (
    locality_controller_actions,
)
from tasks.train_qwen_utility_record_controller_v4_templama import SEEDS


PROTOCOL = "qwen_utility_record_controller_v4_templama_development"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v4_templama_development_audit"


def evaluate(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    selection_root = Path(args.selection_root)
    pilot_root = Path(args.pilot_root)
    feature_audit_path = feature_root / "development_feature_audit.json"
    selection_audit_path = selection_root / "selection_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    selection_audit = json.loads(selection_audit_path.read_text())
    selection = json.loads((selection_root / "result.json").read_text())
    pilot_summary = json.loads((pilot_root / "summary.json").read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("TempLAMA validation feature audit did not pass")
    if (selection_audit.get("protocol") != SELECTION_AUDIT_PROTOCOL
            or not selection_audit.get("passed")):
        raise ValueError("V4 validation selection audit did not pass")
    if selection_audit.get("result_sha256") != sha256(selection_root / "result.json"):
        raise ValueError("V4 selection changed after audit")

    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(
            feature_root / f"features_shard{shard}.jsonl"
        ))
    evaluation_groups = set(selection["evaluation_groups"])
    calibration_groups = set(selection["calibration_groups"])
    if (len(evaluation_groups) != 128 or len(calibration_groups) != 128
            or evaluation_groups & calibration_groups):
        raise ValueError("invalid audited V4 validation split")
    labels = [row for row in labels
              if row["source_group"] in evaluation_groups]
    feature_rows = [row for row in feature_rows
                    if row["source_group"] in evaluation_groups]
    features = {(row["episode_id"], row["boundary_index"]): row
                for row in feature_rows}
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    twin_zero = [row for row in episodes if row["twin"] == 0
                 and row["source_group"] in evaluation_groups]
    backgrounds = {
        row["source_group"]: row["audit"]["background_subject_ids"]
        for row in twin_zero
    }
    heldout = {row["source_group"] for row in twin_zero
               if row["audit"]["background_heldout"]}
    if ({row["source_group"] for row in labels} != evaluation_groups
            or not heldout):
        raise ValueError("V4 evaluation-half coverage is incomplete")

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
        raise ValueError("no safe TempLAMA validation baseline")
    baseline = baseline_results[strongest]
    strongest_actions = baseline_actions[strongest]

    controller_results = {}
    actions_by_seed = {}
    group_gains = {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        predictions = predict_checkpoint(load_checkpoint(checkpoint_path), feature_rows)
        config = selection["controllers"][str(seed)]["selected_config"]
        actions = locality_controller_actions(
            labels, features, predictions,
            config["utility_threshold"], config["risk_penalty"],
            config["risk_threshold"], config["locality_rule"],
        )
        actions_by_seed[seed] = actions
        value = metric(labels, actions, features)
        gains = group_gain(labels, actions, strongest_actions)
        group_gains[seed] = gains
        relation_gain = {
            relation: value["per_relation_accuracy"][relation]
            - baseline["per_relation_accuracy"][relation]
            for relation in value["per_relation_accuracy"]
        }
        mechanism_gain = {
            mechanism: value["per_mechanism_accuracy"][mechanism]
            - baseline["per_mechanism_accuracy"][mechanism]
            for mechanism in value["per_mechanism_accuracy"]
        }
        random_metrics = [
            metric(
                labels, random_matched_actions(labels, actions, draw), features
            ) for draw in range(RANDOM_DRAWS)
        ]
        controller_results[str(seed)] = {
            "selected_calibration_config": config,
            "selection_result_sha256": sha256(selection_root / "result.json"),
            "checkpoint_sha256": sha256(checkpoint_path),
            "evaluation": value,
            "gain_over_strongest_baseline": value["accuracy"] - baseline["accuracy"],
            "gain_over_strongest_baseline_group_bootstrap_95ci": bootstrap(
                gains, random.Random(BOOTSTRAP_SEED + 100_000 + seed)
            ),
            "per_mechanism_gain": mechanism_gain,
            "per_relation_gain": relation_gain,
            "macro_relation_gain": statistics.fmean(relation_gain.values()),
            "deletion_matched_random": {
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
    oracle_actions = {(row["episode_id"], row["boundary_index"]): safe_binary_action(row)
                      for row in labels}
    oracle = metric(labels, oracle_actions, features)
    headroom = oracle["accuracy"] - baseline["accuracy"]
    for seed in SEEDS:
        gain = controller_results[str(seed)]["gain_over_strongest_baseline"]
        controller_results[str(seed)]["oracle_gap_closure"] = (
            gain / headroom if headroom > 0 else 0.0
        )
    seed_rows = [controller_results[str(seed)] for seed in SEEDS]
    majority = metric(labels, majority_actions(labels, actions_by_seed), features)
    groups = sorted(backgrounds)
    pooled_interval = pooled_group_bootstrap(group_gains, groups)
    relation_interval = pooled_relation_bootstrap(labels, group_gains)
    background_interval = background_multiway_bootstrap(group_gains, backgrounds)
    heldout_gains = {seed: {group: gain for group, gain in values.items()
                            if group in heldout}
                     for seed, values in group_gains.items()}
    heldout_interval = pooled_group_bootstrap(heldout_gains, sorted(heldout))
    mean_gain = statistics.fmean(
        row["gain_over_strongest_baseline"] for row in seed_rows
    )
    mean_closure = statistics.fmean(row["oracle_gap_closure"] for row in seed_rows)
    mechanisms = sorted(baseline["per_mechanism_accuracy"])
    mean_mechanism = {name: statistics.fmean(
        row["per_mechanism_gain"][name] for row in seed_rows
    ) for name in mechanisms}
    relations = sorted(baseline["per_relation_accuracy"])
    mean_relation = {name: statistics.fmean(
        row["per_relation_gain"][name] for row in seed_rows
    ) for name in relations}
    macro_relation = statistics.fmean(mean_relation.values())
    release_checks = {
        "safe_binary_oracle_headroom_at_least_point_03": headroom >= .03,
        "every_seed_gain_at_least_point_01": all(
            row["gain_over_strongest_baseline"] >= .01 for row in seed_rows
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
        "upstream_audits_passed": feature_audit["passed"] and selection_audit["passed"],
    }
    return {
        "protocol": PROTOCOL,
        "partition": "templama_validation_evaluation",
        "groups": len(groups),
        "evaluation_groups": sorted(evaluation_groups),
        "calibration_groups": sorted(calibration_groups),
        "strongest_safe_deterministic_baseline": strongest,
        "baselines": baseline_results,
        "controllers": controller_results,
        "majority_controller": majority,
        "safe_binary_oracle": oracle,
        "safe_binary_oracle_headroom": headroom,
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
        "selection_audit_sha256": sha256(selection_audit_path),
        "selection_result_sha256": sha256(selection_root / "result.json"),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "calibration_outcomes_scored": False,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "independent_v4_development_result_audit",
        "terminal_status": "templama_v4_development_evaluation_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--selection-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = evaluate(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
