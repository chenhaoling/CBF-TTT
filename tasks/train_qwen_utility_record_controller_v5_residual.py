"""Train the V5 exact-subject residual benefit/risk controller."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v4_templama_train_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    fixed_policy_actions,
    grid_action,
)
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    metric,
)
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.label_qwen_utility_record_controller_v4_templama_train import (
    AUDIT_PROTOCOL as LABEL_AUDIT_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    load_jsonl,
)
from tasks.select_qwen_utility_record_controller_v4_templama import locality_allows
from tasks.train_qwen_utility_record_controller_v4_templama import (
    stratified_group_split,
)


PROTOCOL = "qwen_utility_record_controller_v5_exact_subject_residual_training"
SEEDS = (26101651, 26101652, 26101653)
CHECKPOINT_EPOCHS = (25, 50, 100)
UTILITY_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
RISK_PENALTIES = (0.0, 0.5, 1.0, 2.0)
CONFORMAL_LEVELS = (0.01, 0.03, 0.05)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRADIENT_NORM = 1.0
DROPOUT = 0.1
LOCALITY_RULE = "exact_subject"
EXPECTED_EXAMPLES = 4392
EXPECTED_BENEFIT_POSITIVES = 685
EXPECTED_RISK_POSITIVES = 201


def load_feature_rows(root: Path) -> list[dict]:
    rows = []
    for shard in (0, 1):
        rows.extend(load_jsonl(root / f"features_shard{shard}.jsonl"))
    return rows


def load_labels(root: Path) -> list[dict]:
    rows = []
    for shard in (0, 1):
        rows.extend(load_jsonl(root / f"labels_shard{shard}.jsonl"))
    return rows


def residual_inference_candidates(labels: list[dict],
                                  feature_rows: list[dict]) -> list[dict]:
    """Enumerate V5 candidates without reading counterfactual outcomes."""
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    if len(features) != len(feature_rows):
        raise ValueError("duplicate V5 feature identity")
    physics = fixed_policy_actions(
        labels, features, "explicit_lifecycle_physics"
    )
    candidates = []
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]
        baseline_name = physics[identity]
        if baseline_name == "global_clear":
            continue
        base_alphas = [
            int(value)
            for value in row["actions"][baseline_name]["spec"]["alphas"]
        ]
        for slot, (alpha, values) in enumerate(zip(
            base_alphas, feature["features"]
        )):
            if alpha != 1 or not locality_allows(values, LOCALITY_RULE):
                continue
            candidate_alphas = list(base_alphas)
            candidate_alphas[slot] = 0
            candidate_name = grid_action(candidate_alphas)
            if candidate_name not in row["actions"]:
                raise ValueError(f"missing V5 residual action {candidate_name}")
            candidates.append({
                "features": values,
                "episode_id": row["episode_id"],
                "boundary_index": row["boundary_index"],
                "slot": slot,
                "source_group": row["source_group"],
                "relation": row["target_relation"],
                "mechanism": row["mechanism"],
                "physics_action": baseline_name,
                "candidate_action": candidate_name,
            })
    return candidates


def residual_examples(labels: list[dict], feature_rows: list[dict]) -> list[dict]:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    if len(features) != len(feature_rows):
        raise ValueError("duplicate V5 feature identity")
    physics = fixed_policy_actions(
        labels, features, "explicit_lifecycle_physics"
    )
    examples = []
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        feature = features[identity]
        baseline_name = physics[identity]
        if baseline_name == "global_clear":
            continue
        baseline = row["actions"][baseline_name]
        retain = row["actions"][row["retain_action"]]
        base_alphas = [int(value) for value in baseline["spec"]["alphas"]]
        for slot, (alpha, values) in enumerate(zip(
            base_alphas, feature["features"]
        )):
            if alpha != 1 or not locality_allows(values, LOCALITY_RULE):
                continue
            candidate_alphas = list(base_alphas)
            candidate_alphas[slot] = 0
            candidate_name = grid_action(candidate_alphas)
            candidate = row["actions"][candidate_name]
            risk = int(
                candidate["unrelated_correct"] < retain["unrelated_correct"]
            )
            correct_gain = candidate["correct"] - baseline["correct"]
            examples.append({
                "features": values,
                "benefit_target": int(not risk and correct_gain > 0),
                "risk_target": risk,
                "correct_gain": correct_gain,
                "nll_gain": (
                    baseline["total_answer_nll"]
                    - candidate["total_answer_nll"]
                ),
                "episode_id": row["episode_id"],
                "boundary_index": row["boundary_index"],
                "slot": slot,
                "source_group": row["source_group"],
                "relation": row["target_relation"],
                "mechanism": row["mechanism"],
                "physics_action": baseline_name,
                "candidate_action": candidate_name,
            })
    return examples


def normalization(examples: list[dict]) -> tuple[list[float], list[float]]:
    means = [
        statistics.fmean(row["features"][index] for row in examples)
        for index in range(len(FEATURE_NAMES))
    ]
    stds = []
    for index, mean in enumerate(means):
        variance = statistics.fmean(
            (row["features"][index] - mean) ** 2 for row in examples
        )
        stds.append(max(math.sqrt(variance), 1e-6))
    return means, stds


def class_weights(examples: list[dict]) -> dict:
    benefit = Counter(row["benefit_target"] for row in examples)
    risk = Counter(row["risk_target"] for row in examples)
    if set(benefit) != {0, 1} or set(risk) != {0, 1}:
        raise ValueError("V5 residual training requires both target classes")
    return {
        "benefit_positive": benefit[0] / benefit[1],
        "risk_positive": risk[0] / risk[1],
        "counts": {"benefit": dict(benefit), "risk": dict(risk)},
    }


def make_model(input_dim: int):
    import torch.nn as nn

    class ResidualController(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
            )
            self.benefit = nn.Linear(64, 1)
            self.risk = nn.Linear(64, 1)

        def forward(self, values):
            hidden = self.body(values)
            return (
                self.benefit(hidden).squeeze(-1),
                self.risk(hidden).squeeze(-1),
            )

    return ResidualController()


def tensors(examples: list[dict], means: list[float], stds: list[float], device):
    import torch

    x = torch.tensor([
        [(value - mean) / std
         for value, mean, std in zip(row["features"], means, stds)]
        for row in examples
    ], dtype=torch.float32, device=device)
    benefit = torch.tensor(
        [row["benefit_target"] for row in examples],
        dtype=torch.float32, device=device,
    )
    risk = torch.tensor(
        [row["risk_target"] for row in examples],
        dtype=torch.float32, device=device,
    )
    return x, benefit, risk


def fit(examples: list[dict], seed: int, device) -> dict:
    import torch
    import torch.nn.functional as F

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = normalization(examples)
    weights = class_weights(examples)
    model = make_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x, benefit, risk = tensors(examples, means, stds, device)
    benefit_weight = torch.tensor(
        weights["benefit_positive"], dtype=torch.float32, device=device
    )
    risk_weight = torch.tensor(
        weights["risk_positive"], dtype=torch.float32, device=device
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train()
        permutation = torch.randperm(len(examples), generator=generator)
        losses, norms = [], []
        for start in range(0, len(examples), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE].to(device)
            optimizer.zero_grad(set_to_none=True)
            benefit_logits, risk_logits = model(x[indices])
            benefit_loss = F.binary_cross_entropy_with_logits(
                benefit_logits, benefit[indices], pos_weight=benefit_weight
            )
            risk_loss = F.binary_cross_entropy_with_logits(
                risk_logits, risk[indices], pos_weight=risk_weight
            )
            loss = benefit_loss + risk_loss
            if not torch.isfinite(loss):
                raise ValueError("non-finite V5 residual loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_NORM
            )
            optimizer.step()
            losses.append(float(loss.item()))
            norms.append(float(norm))
        trace.append({
            "epoch": epoch,
            "loss": statistics.fmean(losses),
            "gradient_norm": statistics.fmean(norms),
        })
        if epoch in CHECKPOINT_EPOCHS:
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            })
    return {
        "means": means,
        "stds": stds,
        "class_weights": weights,
        "trace": trace,
        "states": states,
    }


def predict_state(state: dict, examples: list[dict], means: list[float],
                  stds: list[float]) -> dict:
    import torch

    model = make_model(len(FEATURE_NAMES))
    model.load_state_dict(state, strict=True)
    model.eval()
    # Evaluation candidates intentionally carry no counterfactual targets.
    # Build the inference tensor directly so prediction cannot depend on label
    # fields through a shared training helper.
    x = torch.tensor([
        [(value - mean) / std
         for value, mean, std in zip(row["features"], means, stds)]
        for row in examples
    ], dtype=torch.float32)
    with torch.inference_mode():
        benefit, risk = model(x)
        benefit = benefit.sigmoid().tolist()
        risk = risk.sigmoid().tolist()
    return {
        (row["episode_id"], row["boundary_index"], row["slot"]): {
            "benefit": benefit[index],
            "risk": risk[index],
        }
        for index, row in enumerate(examples)
    }


def conformal_risk_threshold(predictions: dict, examples: list[dict],
                             level: float) -> float:
    positives = [
        predictions[(row["episode_id"], row["boundary_index"], row["slot"])][
            "risk"
        ]
        for row in examples if row["risk_target"] == 1
    ]
    if not positives:
        return 0.0
    ordered = sorted(positives)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(level * (len(ordered) + 1)) - 1),
    )
    return ordered[index]


def residual_actions(labels: list[dict], feature_rows: list[dict],
                     predictions: dict, config: dict) -> dict:
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in feature_rows
    }
    physics = fixed_policy_actions(
        labels, features, "explicit_lifecycle_physics"
    )
    output = {}
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        baseline_name = physics[identity]
        if baseline_name == "global_clear":
            output[identity] = baseline_name
            continue
        alphas = [
            int(value) for value in row["actions"][baseline_name]["spec"]["alphas"]
        ]
        candidates = []
        for slot, (alpha, values) in enumerate(zip(
            alphas, features[identity]["features"]
        )):
            key = (row["episode_id"], row["boundary_index"], slot)
            if (alpha != 1 or not locality_allows(values, LOCALITY_RULE)
                    or key not in predictions):
                continue
            prediction = predictions[key]
            score = (
                prediction["benefit"]
                - config["risk_penalty"] * prediction["risk"]
            )
            if (score >= config["utility_threshold"]
                    and prediction["risk"] < config["risk_threshold"]):
                candidates.append((
                    score, prediction["benefit"], -prediction["risk"],
                    -slot, slot,
                ))
        if candidates:
            alphas[max(candidates)[-1]] = 0
        name = grid_action(alphas)
        if name not in row["actions"]:
            raise ValueError(f"missing V5 residual action {name}")
        output[identity] = name
    return output


def residual_precision(examples: list[dict], actions: dict,
                       labels: list[dict]) -> tuple[float, float, int]:
    example_map = {
        (row["episode_id"], row["boundary_index"], row["slot"]): row
        for row in examples
    }
    selected = correct = eligible = 0
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        name = actions[identity]
        selected_alphas = ([0] * row["candidate_count"]
                           if name == "global_clear" else [
                               int(value) for value
                               in row["actions"][name]["spec"]["alphas"]
                           ])
        for slot in range(row["candidate_count"]):
            key = (row["episode_id"], row["boundary_index"], slot)
            if key not in example_map:
                continue
            eligible += 1
            base_alpha = int(
                row["actions"][example_map[key]["physics_action"]]["spec"][
                    "alphas"
                ][slot]
            )
            is_selected = base_alpha == 1 and selected_alphas[slot] == 0
            selected += int(is_selected)
            correct += int(is_selected and example_map[key]["benefit_target"] == 1)
    return (
        correct / selected if selected else 1.0,
        selected / eligible if eligible else 0.0,
        selected,
    )


def config_name(epoch: int, utility: float, penalty: float, level: float) -> str:
    return f"e{epoch}_u{utility}_p{penalty}_c{level}"


def select_config(candidates: dict[str, dict]) -> str | None:
    eligible = [
        name for name, value in candidates.items()
        if value["added_unrelated_errors"] == 0
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda name: (
        candidates[name]["accuracy"],
        candidates[name]["residual_deletion_precision"],
        -candidates[name]["residual_deletion_fraction"],
        -candidates[name]["config"]["epoch"],
        candidates[name]["config"]["utility_threshold"],
        candidates[name]["config"]["risk_penalty"],
        -candidates[name]["config"]["conformal_level"],
        name,
    ))


def train(args) -> dict:
    import torch

    feature_root = Path(args.feature_root)
    label_root = Path(args.label_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    feature_audit_path = feature_root / "train_feature_audit.json"
    label_audit_path = label_root / "train_labels_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    label_audit = json.loads(label_audit_path.read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("V5 feature audit did not pass")
    if (label_audit.get("protocol") != LABEL_AUDIT_PROTOCOL
            or not label_audit.get("passed")):
        raise ValueError("V5 label audit did not pass")
    labels = load_labels(label_root)
    feature_rows = load_feature_rows(feature_root)
    examples = residual_examples(labels, feature_rows)
    group_relations = {}
    for row in labels:
        group = row["source_group"]
        relation = row["target_relation"]
        if group in group_relations and group_relations[group] != relation:
            raise ValueError(f"V5 group spans relations: {group}")
        group_relations[group] = relation
    groups = set(group_relations)
    relations = set(group_relations.values())
    group_rows = [
        {"source_group": group, "relation": relation}
        for group, relation in group_relations.items()
    ]
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    seed_results = []
    for seed in SEEDS:
        train_groups, validation_groups, allocation = stratified_group_split(
            group_rows, seed
        )
        inner_train = [row for row in examples
                       if row["source_group"] in train_groups]
        inner_validation = [row for row in examples
                            if row["source_group"] in validation_groups]
        validation_labels = [row for row in labels
                             if row["source_group"] in validation_groups]
        validation_features = [row for row in feature_rows
                               if row["source_group"] in validation_groups]
        fitted = fit(inner_train, seed, device)
        candidates = {}
        for epoch in CHECKPOINT_EPOCHS:
            predictions = predict_state(
                fitted["states"][epoch], inner_validation,
                fitted["means"], fitted["stds"],
            )
            thresholds = {
                level: conformal_risk_threshold(
                    predictions, inner_validation, level
                )
                for level in CONFORMAL_LEVELS
            }
            for utility in UTILITY_THRESHOLDS:
                for penalty in RISK_PENALTIES:
                    for level in CONFORMAL_LEVELS:
                        name = config_name(epoch, utility, penalty, level)
                        config = {
                            "epoch": epoch,
                            "utility_threshold": utility,
                            "risk_penalty": penalty,
                            "conformal_level": level,
                            "risk_threshold": thresholds[level],
                            "locality_rule": LOCALITY_RULE,
                            "maximum_extra_deletions": 1,
                        }
                        actions = residual_actions(
                            validation_labels, validation_features,
                            predictions, config,
                        )
                        value = metric(
                            validation_labels, actions,
                            {(row["episode_id"], row["boundary_index"]): row
                             for row in validation_features},
                        )
                        precision, fraction, selected = residual_precision(
                            inner_validation, actions, validation_labels
                        )
                        value.update({
                            "residual_deletion_precision": precision,
                            "residual_deletion_fraction": fraction,
                            "residual_deletions": selected,
                            "config": config,
                        })
                        candidates[name] = value
        selected = select_config(candidates)
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        if selected is None:
            result = {
                "seed": seed,
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(validation_groups),
                "group_allocation": allocation,
                "candidates": candidates,
                "selected": None,
                "passed": False,
                "terminal_status": "no_zero_collateral_residual_policy",
            }
        else:
            selected_config = candidates[selected]["config"]
            checkpoint = {
                "protocol": PROTOCOL,
                "seed": seed,
                "feature_names": list(FEATURE_NAMES),
                "input_dim": len(FEATURE_NAMES),
                "state_dict": fitted["states"][selected_config["epoch"]],
                "means": fitted["means"],
                "stds": fitted["stds"],
                "class_weights": fitted["class_weights"],
                "selected_config": selected_config,
                "feature_audit_sha256": sha256(feature_audit_path),
                "label_audit_sha256": sha256(label_audit_path),
            }
            checkpoint_path = seed_root / "controller.pt"
            torch.save(checkpoint, checkpoint_path)
            result = {
                "seed": seed,
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(validation_groups),
                "group_allocation": allocation,
                "inner_train_examples": len(inner_train),
                "inner_validation_examples": len(inner_validation),
                "class_weights": fitted["class_weights"],
                "trace": fitted["trace"],
                "candidates": candidates,
                "selected": selected,
                "selected_config": selected_config,
                "selected_metric": candidates[selected],
                "checkpoint_sha256": sha256(checkpoint_path),
                "passed": True,
                "terminal_status": "residual_controller_seed_trained",
            }
        (seed_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        seed_results.append(result)
    counts = {
        "examples": len(examples),
        "benefit_positive": sum(row["benefit_target"] for row in examples),
        "risk_positive": sum(row["risk_target"] for row in examples),
        "source_groups": len(groups),
        "residual_source_groups": len({row["source_group"] for row in examples}),
        "relations": len(relations),
    }
    checks = {
        "upstream_audits_passed": (
            feature_audit["passed"] and label_audit["passed"]
        ),
        "registered_residual_support": (
            counts["examples"] == EXPECTED_EXAMPLES
            and counts["benefit_positive"] == EXPECTED_BENEFIT_POSITIVES
            and counts["risk_positive"] == EXPECTED_RISK_POSITIVES
            and counts["source_groups"] == 512
            and counts["relations"] == 9
            and 0 < counts["residual_source_groups"] <= 512
        ),
        "all_seeds_selected_zero_collateral": all(
            row["passed"]
            and row["selected_metric"]["added_unrelated_errors"] == 0
            for row in seed_results
        ),
        "group_disjoint_relation_stratified": all(
            not set(row["train_groups"]) & set(row["validation_groups"])
            and set(row["train_groups"]) | set(row["validation_groups"])
            == groups
            and set(row["group_allocation"]) == relations
            for row in seed_results
        ),
        "finite_training": all(
            math.isfinite(epoch["loss"])
            and math.isfinite(epoch["gradient_norm"])
            for row in seed_results if row["passed"]
            for epoch in row["trace"]
        ),
        "validation_test_sealed": True,
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "plan": "QWEN_UTILITY_RECORD_CONTROLLER_V5_EXACT_SUBJECT_RESIDUAL_PLAN.md",
        "checks": checks,
        "passed": passed,
        "counts": counts,
        "seeds": seed_results,
        "hyperparameters": {
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
        "seconds": time.perf_counter() - started,
        "feature_summary_sha256": sha256(feature_root / "summary.json"),
        "feature_audit_sha256": sha256(feature_audit_path),
        "label_summary_sha256": sha256(label_root / "summary.json"),
        "label_audit_sha256": sha256(label_audit_path),
        "templama_validation_loaded": False,
        "templama_test_loaded": False,
        "v1_confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "next_stage": "templama_v5_training_audit" if passed else "stop_v5",
        "terminal_status": (
            "templama_v5_residual_training_passed" if passed
            else "templama_v5_residual_training_failed"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
