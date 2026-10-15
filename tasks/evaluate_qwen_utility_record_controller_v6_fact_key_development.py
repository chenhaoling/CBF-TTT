"""Evaluate the frozen V6 semantic fact-key safety projection."""

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
    fixed_policy_actions,
    grid_action,
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
from tasks.evaluate_qwen_utility_record_controller_v5_development import (
    PROTOCOL as V5_EVALUATION_PROTOCOL,
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


PROTOCOL = "qwen_utility_record_controller_v6_fact_key_development"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v6_fact_key_development_audit"


def fact_key_matches(labels: list[dict], episodes: list[dict]) -> dict:
    """Reconstruct candidate/incoming semantic-key equality from records."""
    episode_by_id = {row["episode_id"]: row for row in episodes}
    if len(episode_by_id) != len(episodes):
        raise ValueError("duplicate V6 episode identity")
    output = {}
    for row in labels:
        episode = episode_by_id[row["episode_id"]]
        event_index = row["event_index"]
        incoming = episode["events"][event_index]["record"]
        prior = {
            event["record"]["record_id"]: event["record"]
            for event in episode["events"][:event_index]
        }
        if set(row["candidate_record_ids"]) - set(prior):
            raise ValueError("V6 candidate is not in prior memory")
        incoming_key = incoming["audit"]["semantic_key"]
        incoming_relation = incoming["audit"]["relation_id"]
        for slot, record_id in enumerate(row["candidate_record_ids"]):
            record = prior[record_id]
            candidate_key = record["audit"]["semantic_key"]
            same_key = candidate_key == incoming_key
            if same_key != (
                record["audit"]["subject_norm"]
                == incoming["audit"]["subject_norm"]
                and record["audit"]["relation_id"] == incoming_relation
            ):
                raise ValueError("semantic key is inconsistent with subject/relation")
            output[(row["episode_id"], row["boundary_index"], slot)] = {
                "same_fact_key": same_key,
                "candidate_semantic_key": candidate_key,
                "incoming_semantic_key": incoming_key,
                "candidate_relation": record["audit"]["relation_id"],
                "incoming_relation": incoming_relation,
                "candidate_record_id": record_id,
            }
    return output


def fact_key_projected_actions(labels: list[dict], feature_rows: list[dict],
                               raw_actions: dict, matches: dict) -> dict:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    physics = fixed_policy_actions(labels, features, "explicit_lifecycle_physics")
    output = {}
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        baseline_name = physics[identity]
        raw_name = raw_actions[identity]
        if baseline_name == "global_clear":
            if raw_name != baseline_name:
                raise ValueError("V5 changed a lifecycle global-clear action")
            output[identity] = baseline_name
            continue
        baseline = action_alphas(row, baseline_name)
        raw = action_alphas(row, raw_name)
        if any(base == 0 and value == 1 for base, value in zip(baseline, raw)):
            raise ValueError("V5 restored a physics-deleted record")
        residual = [
            slot for slot, (base, value) in enumerate(zip(baseline, raw))
            if base == 1 and value == 0
        ]
        if len(residual) > 1:
            raise ValueError("V5 selected more than one residual deletion")
        output[identity] = (
            raw_name
            if not residual or matches[(*identity, residual[0])]["same_fact_key"]
            else baseline_name
        )
    return output


def safe_fact_key_oracle_actions(labels: list[dict], feature_rows: list[dict],
                                 matches: dict) -> dict:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    physics = fixed_policy_actions(labels, features, "explicit_lifecycle_physics")
    candidates = residual_inference_candidates(labels, feature_rows)
    by_boundary = defaultdict(list)
    for candidate in candidates:
        identity = (candidate["episode_id"], candidate["boundary_index"])
        if matches[(*identity, candidate["slot"])]["same_fact_key"]:
            by_boundary[identity].append(candidate)
    output = {}
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        retain = row["actions"][row["retain_action"]]
        best_name = physics[identity]
        best = row["actions"][best_name]
        for candidate in by_boundary[identity]:
            name = candidate["candidate_action"]
            value = row["actions"][name]
            if (
                value["unrelated_correct"] >= retain["unrelated_correct"]
                and (value["correct"], -value["total_answer_nll"])
                > (best["correct"], -best["total_answer_nll"])
            ):
                best_name, best = name, value
        output[identity] = best_name
    return output


def random_fact_key_matched_actions(labels: list[dict],
                                    feature_rows: list[dict], reference: dict,
                                    matches: dict, draw: int) -> dict:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    physics = fixed_policy_actions(labels, features, "explicit_lifecycle_physics")
    candidates = residual_inference_candidates(labels, feature_rows)
    eligible = defaultdict(list)
    for candidate in candidates:
        identity = (candidate["episode_id"], candidate["boundary_index"])
        if matches[(*identity, candidate["slot"])]["same_fact_key"]:
            eligible[identity].append(candidate["slot"])
    rng = random.Random(BOOTSTRAP_SEED + 9000 + draw)
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
            raise ValueError("V6 reference selected multiple residual deletions")
        if not residual_count:
            output[identity] = baseline_name
            continue
        if not eligible[identity]:
            raise ValueError("V6 random control has no fact-key candidate")
        alphas = list(baseline)
        alphas[eligible[identity][rng.randrange(len(eligible[identity]))]] = 0
        output[identity] = grid_action(alphas)
    return output


def collateral_details(labels: list[dict], actions: dict,
                       physics_actions: dict, matches: dict) -> list[dict]:
    output = []
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        outcome = row["actions"][actions[identity]]
        retain = row["actions"][row["retain_action"]]
        added = max(0, retain["unrelated_correct"] - outcome["unrelated_correct"])
        if not added:
            continue
        baseline = action_alphas(row, physics_actions[identity])
        selected = action_alphas(row, actions[identity])
        residual = [
            slot for slot, (base, value) in enumerate(zip(baseline, selected))
            if base == 1 and value == 0
        ]
        output.append({
            "episode_id": row["episode_id"],
            "boundary_index": row["boundary_index"],
            "source_group": row["source_group"],
            "mechanism": row["mechanism"],
            "relation": row["target_relation"],
            "added_unrelated_errors": added,
            "residual_slots": residual,
            "residual_same_fact_key": [
                matches[(*identity, slot)]["same_fact_key"] for slot in residual
            ],
        })
    return output


def evaluate(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    v5_root = Path(args.v5_root)
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    training_summary_path = training_root / "summary.json"
    v5_result_path = v5_root / "result.json"
    v5_audit_path = v5_root / "result_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    v5_result = json.loads(v5_result_path.read_text())
    v5_audit = json.loads(v5_audit_path.read_text())
    if feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL or not feature_audit.get("passed"):
        raise ValueError("V6 feature audit did not pass")
    if training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL or not training_audit.get("passed"):
        raise ValueError("V6 training audit did not pass")
    if (v5_result.get("protocol") != V5_EVALUATION_PROTOCOL
            or not v5_audit.get("passed")
            or v5_audit.get("result_sha256") != sha256(v5_result_path)):
        raise ValueError("V5 development result is not independently audited")
    if v5_audit.get("release_passed") is not False:
        raise ValueError("V6 requires the audited V5 safety-gate failure")

    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    groups = sorted({row["source_group"] for row in labels})
    if len(groups) != 256 or len(features) != len(labels):
        raise ValueError("V6 requires complete development coverage")
    matches = fact_key_matches(labels, episodes)
    twin_zero = [row for row in episodes if row["twin"] == 0]
    backgrounds = {
        row["source_group"]: row["audit"]["background_subject_ids"]
        for row in twin_zero
    }
    heldout = {
        row["source_group"] for row in twin_zero
        if row["audit"]["background_heldout"]
    }

    physics_actions = fixed_policy_actions(
        labels, features, "explicit_lifecycle_physics"
    )
    physics = metric(labels, physics_actions, features)
    candidates = residual_inference_candidates(labels, feature_rows)
    scored_examples = residual_examples(labels, feature_rows)
    controller_results, actions_by_seed, group_gains = {}, {}, {}
    original_collateral, projected_collateral = {}, {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("protocol") != TRAINING_PROTOCOL or checkpoint["seed"] != seed:
            raise ValueError(f"invalid V6 checkpoint for seed {seed}")
        predictions = predict_state(
            checkpoint["state_dict"], candidates,
            checkpoint["means"], checkpoint["stds"],
        )
        raw_actions = residual_actions(
            labels, feature_rows, predictions, checkpoint["selected_config"]
        )
        raw_metric = metric(labels, raw_actions, features)
        expected_v5 = v5_result["controllers"][str(seed)]["evaluation"]
        if raw_metric["accuracy"] != expected_v5["accuracy"] \
                or raw_metric["added_unrelated_errors"] != expected_v5["added_unrelated_errors"]:
            raise ValueError("V6 could not reproduce frozen V5 actions")
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
        original_collateral[str(seed)] = collateral_details(
            labels, raw_actions, physics_actions, matches
        )
        projected_collateral[str(seed)] = collateral_details(
            labels, actions, physics_actions, matches
        )
        controller_results[str(seed)] = {
            "selected_train_config": checkpoint["selected_config"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "v5_subject_only_evaluation": raw_metric,
            "evaluation": value,
            "filtered_cross_fact_deletions": sum(
                raw_actions[key] != actions[key] for key in actions
            ),
            "gain_over_lifecycle_physics": value["accuracy"] - physics["accuracy"],
            "gain_over_lifecycle_physics_group_bootstrap_95ci": bootstrap(
                gains, random.Random(BOOTSTRAP_SEED + 300_000 + seed)
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
    mean_mechanism = {
        name: statistics.fmean(row["per_mechanism_gain"][name] for row in seed_rows)
        for name in sorted(physics["per_mechanism_accuracy"])
    }
    mean_relation = {
        name: statistics.fmean(row["per_relation_gain"][name] for row in seed_rows)
        for name in sorted(physics["per_relation_accuracy"])
    }
    macro_relation = statistics.fmean(mean_relation.values())
    all_v5_errors_cross_fact = all(
        detail["residual_slots"]
        and not any(detail["residual_same_fact_key"])
        for rows in original_collateral.values() for detail in rows
    )
    release_checks = {
        "safe_single_fact_key_oracle_headroom_at_least_point_03": headroom >= .03,
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
            and v5_audit["passed"]
        ),
    }
    return {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V6_FACT_KEY_SAFETY_PLAN.md",
        "partition": "templama_validation_full_development_adaptive",
        "groups": len(groups),
        "release_baseline": "explicit_lifecycle_physics",
        "lifecycle_physics": physics,
        "controllers": controller_results,
        "majority_controller": majority,
        "safe_single_fact_key_oracle": oracle,
        "safe_single_fact_key_oracle_headroom": headroom,
        "collateral_localization": {
            "v5_subject_only": original_collateral,
            "v6_fact_key": projected_collateral,
            "all_v5_error_boundaries_cross_fact_key": all_v5_errors_cross_fact,
            "v5_error_boundaries": {
                seed: len(rows) for seed, rows in original_collateral.items()
            },
            "v6_error_boundaries": {
                seed: len(rows) for seed, rows in projected_collateral.items()
            },
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
        "semantic_key_candidates": sum(
            value["same_fact_key"] for value in matches.values()
        ),
        "candidate_records": len(matches),
        "feature_audit_sha256": sha256(feature_audit_path),
        "training_audit_sha256": sha256(training_audit_path),
        "training_summary_sha256": sha256(training_summary_path),
        "v5_result_sha256": sha256(v5_result_path),
        "v5_audit_sha256": sha256(v5_audit_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "counterfactual_targets_available_to_controller": False,
        "development_adaptive": True,
        "templama_test_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "independent_v6_development_result_audit",
        "terminal_status": "templama_v6_fact_key_development_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--v5-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = evaluate(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
