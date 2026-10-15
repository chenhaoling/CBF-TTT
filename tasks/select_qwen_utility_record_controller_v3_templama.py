"""Select V3 semantic-locality projections on the TempLAMA train pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_qwen_utility_record_controller_v1_training import (
    AUDIT_PROTOCOL as TRAINING_AUDIT_PROTOCOL,
)
from scripts.audit_qwen_utility_record_controller_v3_templama_pilot_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
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
from tasks.train_qwen_utility_record_controller_v1 import SEEDS


PROTOCOL = "qwen_utility_record_controller_v3_templama_train_selection"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v3_templama_train_selection_audit"
EXPECTED_TRAINING_AUDIT_SHA256 = (
    "4260e090d6ebae309d0f0c0e39637de8691b9bea3fdb044ba40d244fa09e8d07"
)
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
        raise ValueError(f"unknown V3 locality rule {rule}")
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
            raise ValueError(f"selected V3 action {name} missing")
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


def select(args) -> dict:
    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    training_root = Path(args.training_root)
    feature_audit_path = feature_root / "pilot_feature_audit.json"
    training_audit_path = training_root / "training_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    training_audit = json.loads(training_audit_path.read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("TempLAMA pilot feature audit did not pass")
    if (training_audit.get("protocol") != TRAINING_AUDIT_PROTOCOL
            or not training_audit.get("passed")
            or sha256(training_audit_path) != EXPECTED_TRAINING_AUDIT_SHA256):
        raise ValueError("frozen V1 controller training audit mismatch")
    labels, feature_rows = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        feature_rows.extend(load_jsonl(
            feature_root / f"features_shard{shard}.jsonl"
        ))
    features = {(row["episode_id"], row["boundary_index"]): row
                for row in feature_rows}
    groups = {row["source_group"] for row in labels}
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
        "train_pilot_only": True,
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "partition": "templama_train_resource_pilot",
        "source_groups": len(groups),
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
        "templama_validation_loaded": False,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": (
            "templama_v3_selection_audit" if passed else "stop_v3"
        ),
        "terminal_status": "templama_v3_train_selection_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
