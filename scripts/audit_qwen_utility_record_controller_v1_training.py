"""Independent artifact audit for utility/risk controller training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v1_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v1 import (
    BATCH_SIZE,
    CHECKPOINT_EPOCHS,
    DROPOUT,
    GRADIENT_NORM,
    LEARNING_RATE,
    PROTOCOL,
    SEEDS,
    WEIGHT_DECAY,
    make_model,
    relation_folds,
    select_checkpoint,
)


AUDIT_PROTOCOL = "qwen_utility_record_controller_v1_training_audit"


def load_checkpoint(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def audit(training_root: Path, feature_root: Path) -> dict:
    import torch

    summary_path = training_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    feature_audit_path = feature_root / "feature_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    seed_summaries = []
    checkpoints = []
    for seed in SEEDS:
        seed_root = training_root / f"seed_{seed}"
        seed_summary = json.loads((seed_root / "summary.json").read_text())
        checkpoint_path = seed_root / "controller.pt"
        if seed_summary.get("passed"):
            if seed_summary["checkpoint_sha256"] != sha256(checkpoint_path):
                raise ValueError("controller checkpoint hash mismatch")
            checkpoints.append((checkpoint_path, load_checkpoint(checkpoint_path)))
        seed_summaries.append(seed_summary)

    relations = sorted(summary["relation_folds"])
    selected_valid = all(
        row["selected_epoch"] == select_checkpoint({
            int(epoch): value for epoch, value in row["inner"]["checkpoints"].items()
        })
        for row in seed_summaries
    )
    folds_valid = (
        summary["relation_folds"] == relation_folds(relations)
        and all(not (set(row["train_relations"]) & set(row["validation_relations"]))
                and set(row["train_relations"]) | set(row["validation_relations"]) == set(relations)
                for row in seed_summaries)
    )
    checkpoint_valid = True
    for path, checkpoint in checkpoints:
        model = make_model(len(FEATURE_NAMES))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        checkpoint_valid &= (
            checkpoint["protocol"] == PROTOCOL
            and checkpoint["feature_names"] == list(FEATURE_NAMES)
            and checkpoint["input_dim"] == len(FEATURE_NAMES)
            and checkpoint["feature_summary_sha256"] == sha256(feature_root / "summary.json")
            and checkpoint["feature_audit_sha256"] == sha256(feature_audit_path)
            and checkpoint["epochs"] in CHECKPOINT_EPOCHS
            and len(checkpoint["means"]) == len(FEATURE_NAMES)
            and len(checkpoint["stds"]) == len(FEATURE_NAMES)
            and all(math.isfinite(value) for value in checkpoint["means"])
            and all(math.isfinite(value) and value > 0 for value in checkpoint["stds"])
            and all(torch.isfinite(value).all().item()
                    for value in checkpoint["state_dict"].values())
        )

    checks = {
        "feature_audit_locked": (
            feature_audit.get("protocol") == FEATURE_AUDIT_PROTOCOL
            and feature_audit.get("passed") is True
            and summary["feature_audit_sha256"] == sha256(feature_audit_path)
        ),
        "protocol_and_seeds": (
            summary.get("protocol") == PROTOCOL
            and [row["seed"] for row in seed_summaries] == list(SEEDS)
            and all(row["passed"] for row in seed_summaries)
        ),
        "relation_folds": len(relations) == 17 and folds_valid,
        "checkpoint_selection": selected_valid,
        "checkpoint_integrity": len(checkpoints) == len(SEEDS) and checkpoint_valid,
        "fixed_hyperparameters": summary["hyperparameters"] == {
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_norm": GRADIENT_NORM,
            "dropout": DROPOUT,
        },
        "finite_traces": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for row in seed_summaries
            for trace in (row["inner"]["trace"], row["final"]["trace"])
            for epoch in trace
        ),
        "zero_risk_inner_selection": all(
            row["inner"]["checkpoints"][str(row["selected_epoch"])]["unsafe_deletions"] == 0
            for row in seed_summaries
        ),
        "heldouts_sealed": (
            not summary["development_loaded"]
            and not summary["confirmation_loaded"]
            and not summary["external_benchmarks_loaded"]
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "checks": checks,
        "passed": passed,
        "seeds": list(SEEDS),
        "selected_epochs": {str(row["seed"]): row["selected_epoch"]
                            for row in seed_summaries},
        "summary_sha256": sha256(summary_path),
        "feature_audit_sha256": sha256(feature_audit_path),
        "next_stage": "development_construction" if passed else "stop",
        "terminal_status": (
            "utility_record_controller_training_audit_passed" if passed
            else "utility_record_controller_training_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.training_root, args.feature_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
