"""Post-hoc support audit for residual decisions beyond lifecycle physics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    fixed_policy_actions,
    grid_action,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.select_qwen_utility_record_controller_v4_templama import locality_allows


PROTOCOL = "qwen_utility_record_controller_v4_residual_diagnostic"
RULE = "conflict_cosine_099"


def analyze(labels: list[dict], feature_rows: list[dict]) -> dict:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    if len(features) != len(feature_rows):
        raise ValueError("duplicate diagnostic feature identity")
    actions = fixed_policy_actions(
        labels, features, "explicit_lifecycle_physics"
    )
    counts = Counter()
    per_mechanism = defaultdict(Counter)
    per_relation = defaultdict(Counter)
    total_queries = 0
    physics_correct = 0
    best_safe_correct = 0
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]
        baseline_name = actions[identity]
        baseline = row["actions"][baseline_name]
        retain = row["actions"][row["retain_action"]]
        total_queries += baseline["queries"]
        physics_correct += baseline["correct"]
        category_rows = (
            counts,
            per_mechanism[row["mechanism"]],
            per_relation[row["target_relation"]],
        )
        for target in category_rows:
            target["boundaries"] += 1
        if baseline_name == "global_clear":
            best_safe_correct += baseline["correct"]
            for target in category_rows:
                target["global_clear_boundaries"] += 1
            continue
        base_alphas = list(baseline["spec"]["alphas"])
        optional = [
            slot for slot, (alpha, values) in enumerate(zip(
                base_alphas, feature["features"]
            ))
            if alpha == 1 and locality_allows(values, RULE)
        ]
        for target in category_rows:
            target["eligible_records"] += len(optional)
            target["boundaries_with_eligible"] += int(bool(optional))
        single_positive = False
        best = baseline
        best_extra = 0
        for size in range(1, len(optional) + 1):
            for subset in itertools.combinations(optional, size):
                alphas = list(base_alphas)
                for slot in subset:
                    alphas[slot] = 0
                candidate = row["actions"][grid_action(alphas)]
                safe = (
                    candidate["unrelated_correct"]
                    >= retain["unrelated_correct"]
                )
                correct_gain = candidate["correct"] - baseline["correct"]
                nll_gain = baseline["total_answer_nll"] \
                    - candidate["total_answer_nll"]
                for target in category_rows:
                    target["residual_subsets"] += 1
                    target["collateral_subsets"] += int(not safe)
                    target["safe_accuracy_positive_subsets"] += int(
                        safe and correct_gain > 0
                    )
                    target["safe_nll_only_positive_subsets"] += int(
                        safe and correct_gain == 0 and nll_gain > 0
                    )
                if size == 1:
                    for target in category_rows:
                        target["single_residual_actions"] += 1
                        target["single_collateral_actions"] += int(not safe)
                        target["single_safe_accuracy_positive"] += int(
                            safe and correct_gain > 0
                        )
                    single_positive |= safe and correct_gain > 0
                if safe and (
                    candidate["correct"], -candidate["total_answer_nll"]
                ) > (best["correct"], -best["total_answer_nll"]):
                    best = candidate
                    best_extra = size
        best_safe_correct += best["correct"]
        gain = best["correct"] - baseline["correct"]
        for target in category_rows:
            target["boundaries_with_safe_accuracy_gain"] += int(gain > 0)
            target["additional_correct_from_best_safe"] += max(0, gain)
            target["pair_or_higher_only_gain"] += int(
                gain > 0 and best_extra >= 2 and not single_positive
            )
    def finalize(value: Counter) -> dict:
        output = dict(value)
        boundaries = value["boundaries"]
        output["eligible_records_per_boundary"] = (
            value["eligible_records"] / boundaries if boundaries else 0.0
        )
        output["boundary_optional_rate"] = (
            value["boundaries_with_eligible"] / boundaries if boundaries else 0.0
        )
        output["single_collateral_rate"] = (
            value["single_collateral_actions"]
            / value["single_residual_actions"]
            if value["single_residual_actions"] else 0.0
        )
        output["single_safe_accuracy_positive_rate"] = (
            value["single_safe_accuracy_positive"]
            / value["single_residual_actions"]
            if value["single_residual_actions"] else 0.0
        )
        return output
    return {
        **finalize(counts),
        "queries": total_queries,
        "physics_accuracy": physics_correct / total_queries,
        "best_safe_residual_accuracy": best_safe_correct / total_queries,
        "safe_residual_headroom": (
            best_safe_correct - physics_correct
        ) / total_queries,
        "per_mechanism": {
            key: finalize(value) for key, value in sorted(per_mechanism.items())
        },
        "per_relation": {
            key: finalize(value) for key, value in sorted(per_relation.items())
        },
    }


def load_pair(feature_root: Path, label_root: Path) -> tuple[list, list]:
    labels, features = [], []
    for shard in (0, 1):
        labels.extend(load_jsonl(label_root / f"labels_shard{shard}.jsonl"))
        features.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    return labels, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feature-root", type=Path, required=True)
    parser.add_argument("--train-label-root", type=Path, required=True)
    parser.add_argument("--validation-feature-root", type=Path, required=True)
    parser.add_argument("--validation-label-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_labels, train_features = load_pair(
        args.train_feature_root, args.train_label_root
    )
    validation_labels, validation_features = load_pair(
        args.validation_feature_root, args.validation_label_root
    )
    selection = json.loads((args.selection_root / "result.json").read_text())
    parts = {}
    for name, group_key in (
        ("calibration", "calibration_groups"),
        ("evaluation", "evaluation_groups"),
    ):
        groups = set(selection[group_key])
        parts[name] = analyze(
            [row for row in validation_labels if row["source_group"] in groups],
            [row for row in validation_features if row["source_group"] in groups],
        )
    result = {
        "protocol": PROTOCOL,
        "post_hoc_after_v4_evaluation": True,
        "locality_rule": RULE,
        "train": analyze(train_labels, train_features),
        **parts,
        "input_sha256": {
            "train_feature_summary": sha256(args.train_feature_root / "summary.json"),
            "train_label_summary": sha256(args.train_label_root / "summary.json"),
            "validation_feature_summary": sha256(
                args.validation_feature_root / "summary.json"
            ),
            "validation_label_summary": sha256(
                args.validation_label_root / "summary.json"
            ),
            "selection_result": sha256(args.selection_root / "result.json"),
        },
        "templama_test_loaded": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
