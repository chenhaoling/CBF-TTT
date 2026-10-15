"""Train the frozen-spec utility/risk record controller on formal features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v1_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES,
    PROTOCOL as FEATURE_PROTOCOL,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


PROTOCOL = "qwen_utility_record_controller_v1_training"
SEEDS = (2610161, 2610162, 2610163)
CHECKPOINT_EPOCHS = (25, 50, 100)
FOLDS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRADIENT_NORM = 1.0
DROPOUT = 0.1


def stable_rank(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def relation_folds(relations: list[str]) -> dict[str, int]:
    ordered = sorted(relations, key=lambda relation: (
        stable_rank("utility-controller-v1-fold", relation), relation
    ))
    return {relation: index % FOLDS for index, relation in enumerate(ordered)}


def select_checkpoint(metrics: dict[int, dict]) -> int | None:
    eligible = [epoch for epoch, row in metrics.items()
                if row["unsafe_deletions"] == 0 and row["finite"]]
    if not eligible:
        return None
    return max(eligible, key=lambda epoch: (
        metrics[epoch]["boundary_action_accuracy"],
        metrics[epoch]["record_action_accuracy"],
        metrics[epoch]["risk_recall"],
        -epoch,
    ))


def load_rows(feature_root: Path) -> list[dict]:
    rows = []
    for shard in (0, 1):
        rows.extend(load_jsonl(feature_root / f"features_shard{shard}.jsonl"))
    return rows


def flatten(rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        for slot, features in enumerate(row["features"]):
            examples.append({
                "features": features,
                "delete_target": row["targets"]["delete_targets"][slot],
                "risk_target": row["targets"]["risk_targets"][slot],
                "utility_margin": row["targets"]["utility_margins"][slot],
                "identity": f"{row['episode_id']}|{row['boundary_index']}|{slot}",
                "boundary_identity": f"{row['episode_id']}|{row['boundary_index']}",
                "relation": row["audit"]["target_relation"],
                "source_group": row["source_group"],
            })
    return examples


def normalization(examples: list[dict]) -> tuple[list[float], list[float]]:
    width = len(FEATURE_NAMES)
    means = [statistics.fmean(row["features"][index] for row in examples)
             for index in range(width)]
    stds = []
    for index, mean in enumerate(means):
        variance = statistics.fmean(
            (row["features"][index] - mean) ** 2 for row in examples
        )
        stds.append(max(math.sqrt(variance), 1e-6))
    return means, stds


def class_weights(examples: list[dict]) -> dict:
    utility = Counter(row["delete_target"] for row in examples)
    risk = Counter(row["risk_target"] for row in examples)
    if set(utility) != {0, 1} or set(risk) != {0, 1}:
        raise ValueError("controller training requires both utility and risk classes")
    total = len(examples)
    return {
        "utility": [total / (2 * utility[0]), total / (2 * utility[1])],
        "risk_positive": risk[0] / risk[1],
        "counts": {"utility": dict(utility), "risk": dict(risk)},
    }


def make_model(input_dim: int):
    import torch.nn as nn

    class Controller(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
            )
            self.utility = nn.Linear(64, 2)
            self.risk = nn.Linear(64, 1)

        def forward(self, values):
            hidden = self.body(values)
            return self.utility(hidden), self.risk(hidden).squeeze(-1)

    return Controller()


def tensors(examples: list[dict], means: list[float], stds: list[float], device):
    import torch

    x = torch.tensor([
        [(value - mean) / std for value, mean, std in zip(row["features"], means, stds)]
        for row in examples
    ], dtype=torch.float32, device=device)
    y = torch.tensor([row["delete_target"] for row in examples],
                     dtype=torch.long, device=device)
    risk = torch.tensor([row["risk_target"] for row in examples],
                        dtype=torch.float32, device=device)
    margin = torch.tensor([
        1.0 + min(2.0, max(0.0, row["utility_margin"])) for row in examples
    ], dtype=torch.float32, device=device)
    return x, y, risk, margin


def evaluate(model, examples, means, stds, device) -> dict:
    import torch

    model.eval()
    x, y, risk, _ = tensors(examples, means, stds, device)
    with torch.inference_mode():
        utility_logits, risk_logits = model(x)
        delete_probability = utility_logits.softmax(-1)[:, 1]
        risk_probability = risk_logits.sigmoid()
    predicted_delete = (delete_probability >= .5) & (risk_probability < .5)
    boundaries = defaultdict(list)
    for index, row in enumerate(examples):
        boundaries[row["boundary_identity"]].append(index)
    boundary_correct = sum(
        all(int(predicted_delete[index].item()) == examples[index]["delete_target"]
            for index in indices)
        for indices in boundaries.values()
    )
    risk_positive = int(risk.sum().item())
    risk_detected = int(((risk_probability >= .5) & (risk == 1)).sum().item())
    result = {
        "examples": len(examples),
        "boundaries": len(boundaries),
        "record_action_accuracy": float(
            (predicted_delete.long() == y).float().mean().item()
        ),
        "boundary_action_accuracy": boundary_correct / len(boundaries),
        "unsafe_deletions": int((predicted_delete & (risk == 1)).sum().item()),
        "risk_recall": risk_detected / risk_positive if risk_positive else 1.0,
        "mean_delete_probability": float(delete_probability.mean().item()),
        "mean_risk_probability": float(risk_probability.mean().item()),
    }
    result["finite"] = all(math.isfinite(value) for value in result.values())
    return result


def fit(
    train_examples: list[dict],
    validation_examples: list[dict] | None,
    seed: int,
    epochs: int,
    device,
) -> tuple[object, dict]:
    import torch
    import torch.nn.functional as F

    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = normalization(train_examples)
    weights = class_weights(train_examples)
    model = make_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x, y, risk, margin = tensors(train_examples, means, stds, device)
    utility_weight = torch.tensor(weights["utility"], dtype=torch.float32, device=device)
    risk_positive = torch.tensor(weights["risk_positive"], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    trace = []; checkpoints = {}; states = {}
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_examples), generator=generator)
        losses = []; gradient_norms = []
        for start in range(0, len(train_examples), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE].to(device)
            optimizer.zero_grad(set_to_none=True)
            utility_logits, risk_logits = model(x[indices])
            utility_loss = F.cross_entropy(
                utility_logits, y[indices], weight=utility_weight, reduction="none"
            )
            utility_loss = (utility_loss * margin[indices]).mean()
            risk_loss = F.binary_cross_entropy_with_logits(
                risk_logits, risk[indices], pos_weight=risk_positive
            )
            loss = utility_loss + risk_loss
            if not torch.isfinite(loss):
                raise ValueError("non-finite controller loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
            optimizer.step()
            losses.append(float(loss.item())); gradient_norms.append(float(norm))
        epoch_row = {
            "epoch": epoch,
            "loss": statistics.fmean(losses),
            "gradient_norm": statistics.fmean(gradient_norms),
        }
        trace.append(epoch_row)
        if validation_examples is not None and epoch in CHECKPOINT_EPOCHS:
            checkpoints[epoch] = evaluate(
                model, validation_examples, means, stds, device
            )
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
    return model, {
        "means": means,
        "stds": stds,
        "class_weights": weights,
        "trace": trace,
        "checkpoints": checkpoints,
        "states": states,
    }


def train(args) -> dict:
    import torch

    feature_root, output_root = Path(args.feature_root), Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    feature_summary = json.loads((feature_root / "summary.json").read_text())
    feature_audit = json.loads((feature_root / "feature_audit.json").read_text())
    if feature_summary.get("protocol") != FEATURE_PROTOCOL or not feature_summary.get("passed"):
        raise ValueError("feature summary did not pass")
    if feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL or not feature_audit.get("passed"):
        raise ValueError("feature audit did not pass")
    rows = load_rows(feature_root); examples = flatten(rows)
    relations = sorted({row["relation"] for row in examples})
    folds = relation_folds(relations)
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    output_root.mkdir(parents=True)
    started = time.perf_counter(); seed_results = []
    for seed_index, seed in enumerate(SEEDS):
        validation_fold = seed_index
        train_examples = [row for row in examples if folds[row["relation"]] != validation_fold]
        validation_examples = [row for row in examples
                               if folds[row["relation"]] == validation_fold]
        _, inner = fit(
            train_examples, validation_examples, seed, max(CHECKPOINT_EPOCHS), device
        )
        selected_epoch = select_checkpoint(inner["checkpoints"])
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        inner_serializable = {key: value for key, value in inner.items() if key != "states"}
        if selected_epoch is None:
            seed_result = {
                "seed": seed,
                "validation_fold": validation_fold,
                "train_relations": sorted({row["relation"] for row in train_examples}),
                "validation_relations": sorted({row["relation"] for row in validation_examples}),
                "selected_epoch": None,
                "inner": inner_serializable,
                "passed": False,
                "terminal_status": "no_zero_risk_checkpoint",
            }
            (seed_root / "summary.json").write_text(
                json.dumps(seed_result, indent=2, sort_keys=True) + "\n"
            )
            seed_results.append(seed_result)
            continue
        final_model, final = fit(examples, None, seed, selected_epoch, device)
        checkpoint = {
            "protocol": PROTOCOL,
            "seed": seed,
            "epochs": selected_epoch,
            "feature_names": list(FEATURE_NAMES),
            "input_dim": len(FEATURE_NAMES),
            "state_dict": {key: value.detach().cpu()
                           for key, value in final_model.state_dict().items()},
            "means": final["means"],
            "stds": final["stds"],
            "class_weights": final["class_weights"],
            "feature_summary_sha256": sha256(feature_root / "summary.json"),
            "feature_audit_sha256": sha256(feature_root / "feature_audit.json"),
        }
        checkpoint_path = seed_root / "controller.pt"
        torch.save(checkpoint, checkpoint_path)
        seed_result = {
            "seed": seed,
            "validation_fold": validation_fold,
            "train_relations": sorted({row["relation"] for row in train_examples}),
            "validation_relations": sorted({row["relation"] for row in validation_examples}),
            "selected_epoch": selected_epoch,
            "inner": inner_serializable,
            "final": {
                "epochs": selected_epoch,
                "class_weights": final["class_weights"],
                "trace": final["trace"],
            },
            "checkpoint_sha256": sha256(checkpoint_path),
            "passed": True,
            "terminal_status": "controller_seed_trained",
        }
        (seed_root / "summary.json").write_text(
            json.dumps(seed_result, indent=2, sort_keys=True) + "\n"
        )
        seed_results.append(seed_result)

    checks = {
        "feature_gate_passed": feature_summary["passed"] and feature_audit["passed"],
        "seventeen_relations": len(relations) == 17,
        "relation_disjoint_inner_folds": all(
            not (set(row["train_relations"]) & set(row["validation_relations"]))
            for row in seed_results
        ),
        "all_seeds_zero_risk_checkpoint": all(row["passed"] for row in seed_results),
        "finite_training": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for row in seed_results for epoch in row["inner"]["trace"]
        ),
        "heldouts_sealed": (
            not feature_summary["development_loaded"]
            and not feature_summary["confirmation_loaded"]
            and not feature_summary["external_benchmarks_loaded"]
        ),
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "checks": checks,
        "passed": passed,
        "seeds": seed_results,
        "relation_folds": folds,
        "examples": len(examples),
        "boundaries": len(rows),
        "feature_dimension": len(FEATURE_NAMES),
        "hyperparameters": {
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_norm": GRADIENT_NORM,
            "dropout": DROPOUT,
        },
        "seconds": time.perf_counter() - started,
        "feature_summary_sha256": sha256(feature_root / "summary.json"),
        "feature_audit_sha256": sha256(feature_root / "feature_audit.json"),
        "next_stage": "development_construction" if passed else "stop",
        "development_loaded": False,
        "confirmation_loaded": False,
        "external_benchmarks_loaded": False,
        "terminal_status": (
            "utility_record_controller_training_passed" if passed
            else "utility_record_controller_training_failed"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
