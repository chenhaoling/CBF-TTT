"""Select V4 domain-controller settings on the validation calibration half."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_qwen_utility_record_controller_v4_templama_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v2_templama_development_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
)
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    CONFORMAL_LEVELS,
    RISK_PENALTIES,
    UTILITY_THRESHOLDS,
    conformal_threshold,
    grid_action,
    load_checkpoint,
    metric,
    predict_checkpoint,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.train_qwen_utility_record_controller_v4_templama import SEEDS, stable_rank


PROTOCOL = "qwen_utility_record_controller_v4_templama_validation_selection"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v4_templama_validation_selection_audit"
SPLIT_SEED = 202610176
CALIBRATION_GROUPS = 128
LOCALITY_RULES = (
    "unrestricted",
    "exact_subject",
    "subject_value_conflict",
    "conflict_pair_positive",
    "conflict_cosine_098",
    "conflict_cosine_099",
    "conflict_cosine_0995",
    "conflict_pair_cosine_098",
    "conflict_pair_cosine_099",
    "conflict_pair_cosine_0995",
)


def locality_allows(values: list[float], rule: str) -> bool:
    if rule == "unrestricted":
        return True
    if rule == "exact_subject":
        return values[0] >= .5
    if rule == "subject_value_conflict":
        return values[2] >= .5
    if rule == "conflict_pair_positive":
        return values[2] >= .5 and values[10] >= 0
    pair_required = rule.startswith("conflict_pair_cosine_")
    prefix = "conflict_pair_cosine_" if pair_required else "conflict_cosine_"
    if not rule.startswith(prefix):
        raise ValueError(f"unknown V4 locality rule {rule}")
    threshold = {
        "098": .98,
        "099": .99,
        "0995": .995,
    }[rule.removeprefix(prefix)]
    return (
        values[2] >= .5
        and (not pair_required or values[10] >= 0)
        and values[4] >= threshold
    )


def locality_controller_actions(rows: list[dict], features: dict, predictions: dict,
                                utility_threshold: float, risk_penalty: float,
                                risk_threshold: float, locality_rule: str) -> dict:
    output = {}
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]
        prediction = predictions[identity]
        if any(values[20] >= .5 for values in feature["features"]):
            output[identity] = "global_clear"
            continue
        alphas = []
        for values, delete_probability, risk_probability in zip(
            feature["features"], prediction["delete"], prediction["risk"]
        ):
            forced_delete = values[3] >= .5 or values[21] >= .5
            neural_delete = (
                locality_allows(values, locality_rule)
                and delete_probability - risk_penalty * risk_probability
                >= utility_threshold
                and risk_probability < risk_threshold
            )
            alphas.append(int(not (forced_delete or neural_delete)))
        name = grid_action(alphas)
        if name not in row["actions"]:
            raise ValueError(f"selected V4 action {name} missing")
        output[identity] = name
    return output


def deletion_precision(rows: list[dict], features: dict,
                       actions: dict[tuple[str, int], str]) -> float:
    selected = 0
    correct = 0
    for row in rows:
        identity = (row["episode_id"], row["boundary_index"])
        name = actions[identity]
        alphas = ([0] * row["candidate_count"] if name == "global_clear"
                  else row["actions"][name]["spec"]["alphas"])
        targets = features[identity]["targets"]["delete_targets"]
        selected += sum(value == 0 for value in alphas)
        correct += sum(value == 0 and target == 1
                       for value, target in zip(alphas, targets))
    return correct / selected if selected else 1.0


def config_name(utility: float, penalty: float, level: float, rule: str) -> str:
    return f"u{utility}_p{penalty}_c{level}_{rule}"


def select_config(candidates: dict[str, dict]) -> str | None:
    eligible = [name for name, value in candidates.items()
                if value["added_unrelated_errors"] == 0]
    if not eligible:
        return None
    rule_order = {name: index for index, name in enumerate(LOCALITY_RULES)}
    return max(eligible, key=lambda name: (
        candidates[name]["accuracy"],
        candidates[name]["deletion_precision"],
        -candidates[name]["deletion_fraction"],
        -rule_order[candidates[name]["config"]["locality_rule"]],
        candidates[name]["config"]["utility_threshold"],
        candidates[name]["config"]["risk_penalty"],
        -candidates[name]["config"]["conformal_level"],
        name,
    ))


def validation_group_split(episodes: list[dict]) -> tuple[set, set, dict]:
    """Create the frozen, exact 128/128 relation-stratified group split."""
    groups_by_relation = {}
    for episode in episodes:
        group = episode["source_group"]
        relation = episode["audit"]["target_relation"]
        if group in groups_by_relation and groups_by_relation[group] != relation:
            raise ValueError(f"validation group spans relations: {group}")
        groups_by_relation[group] = relation
    by_relation = {}
    for group, relation in groups_by_relation.items():
        by_relation.setdefault(relation, []).append(group)
    calibration_counts = {
        relation: len(groups) // 2 for relation, groups in by_relation.items()
    }
    extras = CALIBRATION_GROUPS - sum(calibration_counts.values())
    odd_relations = [relation for relation, groups in by_relation.items()
                     if len(groups) % 2]
    ranked_odd = sorted(odd_relations, key=lambda relation: (
        stable_rank("utility-controller-v4-validation-extra", SPLIT_SEED,
                    relation),
        relation,
    ))
    if extras < 0 or extras > len(ranked_odd):
        raise ValueError("cannot construct exact V4 calibration half")
    for relation in ranked_odd[:extras]:
        calibration_counts[relation] += 1
    calibration, evaluation, allocation = set(), set(), {}
    for relation in sorted(by_relation):
        groups = sorted(by_relation[relation], key=lambda group: (
            stable_rank("utility-controller-v4-validation", SPLIT_SEED,
                        relation, group),
            group,
        ))
        count = calibration_counts[relation]
        calibration.update(groups[:count])
        evaluation.update(groups[count:])
        allocation[relation] = {
            "total": len(groups),
            "calibration": count,
            "evaluation": len(groups) - count,
        }
    if (len(calibration) != CALIBRATION_GROUPS
            or len(evaluation) != CALIBRATION_GROUPS
            or calibration & evaluation):
        raise ValueError("invalid V4 validation split")
    return calibration, evaluation, allocation


def select(args) -> dict:
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    feature_audit_path = feature_root / "development_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    data_audit_path = data_root / "data_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    data_audit = json.loads(data_audit_path.read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("TempLAMA validation feature audit did not pass")
    if (training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL
            or not training_audit.get("passed")):
        raise ValueError("V4 controller training audit mismatch")
    if (data_audit.get("protocol") != DATA_AUDIT_PROTOCOL
            or not data_audit.get("passed")
            or data_audit.get("manifest_sha256")
            != sha256(data_root / "manifest.json")):
        raise ValueError("TempLAMA validation data audit mismatch")
    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(
            feature_root / f"features_shard{shard}.jsonl"
        ))
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    calibration_groups, evaluation_groups, allocation = validation_group_split(
        episodes
    )
    labels = [row for row in labels
              if row["source_group"] in calibration_groups]
    feature_rows = [row for row in feature_rows
                    if row["source_group"] in calibration_groups]
    features = {(row["episode_id"], row["boundary_index"]): row
                for row in feature_rows}
    groups = {row["source_group"] for row in labels}
    if groups != calibration_groups:
        raise ValueError("V4 calibration labels do not cover the frozen split")
    results = {}
    for seed in SEEDS:
        checkpoint_path = training_root / f"seed_{seed}" / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        predictions = predict_checkpoint(checkpoint, feature_rows)
        thresholds = {
            str(level): conformal_threshold(
                predictions, feature_rows, groups, level
            ) for level in CONFORMAL_LEVELS
        }
        candidates = {}
        for utility in UTILITY_THRESHOLDS:
            for penalty in RISK_PENALTIES:
                for level in CONFORMAL_LEVELS:
                    for rule in LOCALITY_RULES:
                        name = config_name(utility, penalty, level, rule)
                        actions = locality_controller_actions(
                            labels, features, predictions, utility, penalty,
                            thresholds[str(level)], rule,
                        )
                        value = metric(labels, actions)
                        value["deletion_precision"] = deletion_precision(
                            labels, features, actions
                        )
                        value["config"] = {
                            "utility_threshold": utility,
                            "risk_penalty": penalty,
                            "conformal_level": level,
                            "risk_threshold": thresholds[str(level)],
                            "locality_rule": rule,
                        }
                        candidates[name] = value
        selected = select_config(candidates)
        if selected is None:
            results[str(seed)] = {
                "selected": None,
                "conformal_risk_thresholds": thresholds,
                "candidates": candidates,
                "checkpoint_sha256": sha256(checkpoint_path),
            }
        else:
            results[str(seed)] = {
                "selected": selected,
                "selected_config": candidates[selected]["config"],
                "selected_metric": candidates[selected],
                "conformal_risk_thresholds": thresholds,
                "candidates": candidates,
                "checkpoint_sha256": sha256(checkpoint_path),
            }
    selected_rows = [value for value in results.values() if value["selected"]]
    checks = {
        "all_three_seeds_selected": len(selected_rows) == len(SEEDS),
        "all_selected_zero_collateral": all(
            row["selected_metric"]["added_unrelated_errors"] == 0
            for row in selected_rows
        ),
        "complete_registered_grid": all(
            len(row["candidates"])
            == (len(UTILITY_THRESHOLDS) * len(RISK_PENALTIES)
                * len(CONFORMAL_LEVELS) * len(LOCALITY_RULES))
            for row in results.values()
        ),
        "exact_calibration_half_only": (
            len(groups) == CALIBRATION_GROUPS
            and not groups & evaluation_groups
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "partition": "templama_validation_calibration",
        "source_groups": len(groups),
        "calibration_groups": sorted(calibration_groups),
        "evaluation_groups": sorted(evaluation_groups),
        "split_allocation": allocation,
        "controllers": results,
        "checks": checks,
        "passed": passed,
        "mean_selected_accuracy": (
            statistics.fmean(row["selected_metric"]["accuracy"]
                             for row in selected_rows)
            if selected_rows else None
        ),
        "feature_audit_sha256": sha256(feature_audit_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "training_audit_sha256": sha256(training_audit_path),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_audit_path),
        "evaluation_outcomes_scored": False,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": (
            "templama_v4_selection_audit" if passed else "stop_v4"
        ),
        "terminal_status": "templama_v4_validation_selection_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = select(args)
    Path(args.output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
