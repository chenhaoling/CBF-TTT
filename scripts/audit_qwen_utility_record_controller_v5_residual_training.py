"""Independent artifact audit for V5 exact-subject residual training."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v4_templama_train_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.label_qwen_utility_record_controller_v4_templama_train import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from tasks.train_qwen_utility_record_controller_v4_templama import (
    stratified_group_split,
)
from tasks.train_qwen_utility_record_controller_v5_residual import (
    BATCH_SIZE,
    CHECKPOINT_EPOCHS,
    CONFORMAL_LEVELS,
    DROPOUT,
    EXPECTED_BENEFIT_POSITIVES,
    EXPECTED_EXAMPLES,
    EXPECTED_RISK_POSITIVES,
    GRADIENT_NORM,
    LEARNING_RATE,
    LOCALITY_RULE,
    PROTOCOL,
    RISK_PENALTIES,
    SEEDS,
    UTILITY_THRESHOLDS,
    WEIGHT_DECAY,
    class_weights,
    load_feature_rows,
    load_labels,
    make_model,
    normalization,
    residual_examples,
    select_config,
)


AUDIT_PROTOCOL = "qwen_utility_record_controller_v5_residual_training_audit"


def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def close_lists(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= 1e-12 for a, b in zip(left, right)
    )


def audit(training_root: Path, feature_root: Path, label_root: Path) -> dict:
    import torch

    summary_path = training_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    feature_audit_path = feature_root / "train_feature_audit.json"
    label_audit_path = label_root / "train_labels_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    label_audit = json.loads(label_audit_path.read_text())
    labels = load_labels(label_root)
    feature_rows = load_feature_rows(feature_root)
    examples = residual_examples(labels, feature_rows)
    group_relations = {}
    for row in labels:
        group_relations[row["source_group"]] = row["target_relation"]
    group_rows = [
        {"source_group": group, "relation": relation}
        for group, relation in group_relations.items()
    ]
    expected_configs = {
        (epoch, utility, penalty, level)
        for epoch, utility, penalty, level in itertools.product(
            CHECKPOINT_EPOCHS, UTILITY_THRESHOLDS,
            RISK_PENALTIES, CONFORMAL_LEVELS,
        )
    }
    seed_summaries = []
    split_valid = True
    selection_valid = True
    candidates_valid = True
    checkpoints_valid = True
    traces_valid = True
    for index, seed in enumerate(SEEDS):
        seed_root = training_root / f"seed_{seed}"
        seed_summary = json.loads((seed_root / "summary.json").read_text())
        seed_summaries.append(seed_summary)
        split_train, split_validation, allocation = stratified_group_split(
            group_rows, seed
        )
        split_valid &= (
            seed_summary["train_groups"] == sorted(split_train)
            and seed_summary["validation_groups"] == sorted(split_validation)
            and seed_summary["group_allocation"] == allocation
            and not split_train & split_validation
            and split_train | split_validation == set(group_relations)
        )
        candidates = seed_summary["candidates"]
        observed_configs = {
            (
                value["config"]["epoch"],
                value["config"]["utility_threshold"],
                value["config"]["risk_penalty"],
                value["config"]["conformal_level"],
            )
            for value in candidates.values()
        }
        candidates_valid &= (
            len(candidates) == len(expected_configs)
            and observed_configs == expected_configs
            and all(
                value["config"]["locality_rule"] == LOCALITY_RULE
                and value["config"]["maximum_extra_deletions"] == 1
                and math.isfinite(value["config"]["risk_threshold"])
                and all(math.isfinite(float(value[key])) for key in (
                    "accuracy", "mean_answer_nll", "deletion_fraction",
                    "residual_deletion_precision",
                    "residual_deletion_fraction",
                ))
                for value in candidates.values()
            )
        )
        selected = select_config(candidates)
        selection_valid &= (
            seed_summary["seed"] == seed
            and seed_summary["passed"] is True
            and seed_summary["selected"] == selected
            and seed_summary["selected_config"]
            == candidates[selected]["config"]
            and seed_summary["selected_metric"] == candidates[selected]
            and candidates[selected]["added_unrelated_errors"] == 0
            and summary["seeds"][index] == seed_summary
        )
        inner_train = [row for row in examples
                       if row["source_group"] in split_train]
        expected_means, expected_stds = normalization(inner_train)
        expected_weights = class_weights(inner_train)
        checkpoint_path = seed_root / "controller.pt"
        checkpoint = load_checkpoint(checkpoint_path)
        model = make_model(len(FEATURE_NAMES))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        checkpoints_valid &= (
            seed_summary["checkpoint_sha256"] == sha256(checkpoint_path)
            and checkpoint["protocol"] == PROTOCOL
            and checkpoint["seed"] == seed
            and checkpoint["feature_names"] == list(FEATURE_NAMES)
            and checkpoint["input_dim"] == len(FEATURE_NAMES)
            and checkpoint["selected_config"] == seed_summary["selected_config"]
            and checkpoint["class_weights"] == expected_weights
            and close_lists(checkpoint["means"], expected_means)
            and close_lists(checkpoint["stds"], expected_stds)
            and checkpoint["feature_audit_sha256"] == sha256(feature_audit_path)
            and checkpoint["label_audit_sha256"] == sha256(label_audit_path)
            and all(torch.isfinite(value).all().item()
                    for value in checkpoint["state_dict"].values())
        )
        traces_valid &= (
            len(seed_summary["trace"]) == max(CHECKPOINT_EPOCHS)
            and all(
                math.isfinite(epoch["loss"])
                and math.isfinite(epoch["gradient_norm"])
                for epoch in seed_summary["trace"]
            )
        )
    counts = {
        "examples": len(examples),
        "benefit_positive": sum(row["benefit_target"] for row in examples),
        "risk_positive": sum(row["risk_target"] for row in examples),
        "source_groups": len(group_relations),
        "residual_source_groups": len({row["source_group"] for row in examples}),
        "relations": len(set(group_relations.values())),
    }
    checks = {
        "upstream_audits_and_hashes_locked": (
            feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and label_audit.get("protocol") == LABEL_AUDIT_PROTOCOL
            and label_audit.get("passed") is True
            and summary["feature_summary_sha256"]
            == sha256(feature_root / "summary.json")
            and summary["feature_audit_sha256"] == sha256(feature_audit_path)
            and summary["label_summary_sha256"]
            == sha256(label_root / "summary.json")
            and summary["label_audit_sha256"] == sha256(label_audit_path)
        ),
        "protocol_seeds_and_registered_support": (
            summary.get("protocol") == PROTOCOL
            and [row["seed"] for row in seed_summaries] == list(SEEDS)
            and counts == summary["counts"]
            and counts["examples"] == EXPECTED_EXAMPLES
            and counts["benefit_positive"] == EXPECTED_BENEFIT_POSITIVES
            and counts["risk_positive"] == EXPECTED_RISK_POSITIVES
            and counts["source_groups"] == 512
            and counts["relations"] == 9
        ),
        "relation_stratified_group_splits": split_valid,
        "complete_finite_registered_grid": candidates_valid,
        "independent_zero_collateral_selection": selection_valid,
        "checkpoint_integrity": checkpoints_valid,
        "finite_training_traces": traces_valid,
        "fixed_hyperparameters": summary["hyperparameters"] == {
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "utility_thresholds": list(UTILITY_THRESHOLDS),
            "risk_penalties": list(RISK_PENALTIES),
            "conformal_levels": list(CONFORMAL_LEVELS),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_norm": GRADIENT_NORM,
            "dropout": DROPOUT,
            "locality_rule": LOCALITY_RULE,
            "maximum_extra_deletions": 1,
        },
        "validation_test_and_external_sealed": (
            summary["templama_validation_loaded"] is False
            and summary["templama_test_loaded"] is False
            and summary["v1_confirmation_loaded"] is False
            and summary["external_benchmarks_loaded"] is False
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "counts": counts,
        "selected": {
            str(row["seed"]): row["selected"] for row in seed_summaries
        },
        "summary_sha256": sha256(summary_path),
        "feature_audit_sha256": sha256(feature_audit_path),
        "label_audit_sha256": sha256(label_audit_path),
        "next_stage": (
            "templama_v5_development_evaluation" if passed else "stop_v5"
        ),
        "terminal_status": (
            "templama_v5_residual_training_audit_passed" if passed
            else "templama_v5_residual_training_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.training_root, args.feature_root, args.label_root
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
