"""Train the preregistered V8 relation-posterior utility controller."""

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

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import (
    probability_metrics,
)
from tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features import (
    AUDIT_PROTOCOL as FEATURE_AUDIT_PROTOCOL,
    FEATURE_NAMES,
)
from tasks.train_qwen_utility_record_controller_v5_residual import (
    BATCH_SIZE,
    CHECKPOINT_EPOCHS,
    CONFORMAL_LEVELS,
    DROPOUT,
    GRADIENT_NORM,
    LEARNING_RATE,
    LOCALITY_RULE,
    RISK_PENALTIES,
    SEEDS,
    UTILITY_THRESHOLDS,
    WEIGHT_DECAY,
    class_weights,
    config_name,
    conformal_risk_threshold,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    EXPECTED_DEVELOPMENT,
    EXPECTED_TRAIN,
    development_gate,
    domain_metric,
    load_development_domain,
    load_training_domain,
    select_config,
)


PROTOCOL = "qwen_utility_record_controller_v8_relation_posterior_training"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V8_RELATION_POSTERIOR_PLAN.md"


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


def make_model(input_dim: int):
    import torch.nn as nn

    class Controller(nn.Module):
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
            return self.benefit(hidden).squeeze(-1), self.risk(hidden).squeeze(-1)

    return Controller()


def fit(examples: list[dict], seed: int, device) -> dict:
    import torch
    import torch.nn.functional as F

    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = normalization(examples)
    weights = class_weights(examples)
    model = make_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    x = torch.tensor([
        [(value - mean) / std for value, mean, std
         in zip(row["features"], means, stds)]
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
    benefit_weight = torch.tensor(
        weights["benefit_positive"], dtype=torch.float32, device=device
    )
    risk_weight = torch.tensor(
        weights["risk_positive"], dtype=torch.float32, device=device
    )
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    trace, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train(); permutation = torch.randperm(len(examples), generator=generator)
        losses, norms = [], []
        for start in range(0, len(examples), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE].to(device)
            optimizer.zero_grad(set_to_none=True)
            benefit_logits, risk_logits = model(x[indices])
            loss = F.binary_cross_entropy_with_logits(
                benefit_logits, benefit[indices], pos_weight=benefit_weight
            ) + F.binary_cross_entropy_with_logits(
                risk_logits, risk[indices], pos_weight=risk_weight
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite V8 loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
            optimizer.step()
            losses.append(float(loss)); norms.append(float(norm))
        trace.append({"epoch": epoch, "loss": statistics.fmean(losses),
                      "gradient_norm": statistics.fmean(norms)})
        if epoch in CHECKPOINT_EPOCHS:
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
    return {"means": means, "stds": stds, "class_weights": weights,
            "trace": trace, "states": states}


def predict_state(state: dict, examples: list[dict], means: list[float],
                  stds: list[float]) -> dict:
    import torch

    model = make_model(len(FEATURE_NAMES))
    model.load_state_dict(state, strict=True); model.eval()
    x = torch.tensor([
        [(value - mean) / std for value, mean, std
         in zip(row["features"], means, stds)]
        for row in examples
    ], dtype=torch.float32)
    with torch.inference_mode():
        benefit, risk = model(x)
        benefit = benefit.sigmoid().tolist(); risk = risk.sigmoid().tolist()
    return {
        (row["episode_id"], row["boundary_index"], row["slot"]): {
            "benefit": benefit[index], "risk": risk[index],
        } for index, row in enumerate(examples)
    }


def evaluate_state(state: dict, means: list[float], stds: list[float],
                   development: dict, epoch: int) -> tuple[dict, dict]:
    predictions = {
        name: predict_state(state, domain["examples"], means, stds)
        for name, domain in development.items()
    }
    output = {}
    for level in CONFORMAL_LEVELS:
        threshold = min(
            conformal_risk_threshold(
                predictions[name], domain["examples"], level
            ) for name, domain in development.items()
        )
        for utility in UTILITY_THRESHOLDS:
            for penalty in RISK_PENALTIES:
                name = config_name(epoch, utility, penalty, level)
                config = {
                    "epoch": epoch,
                    "utility_threshold": utility,
                    "risk_penalty": penalty,
                    "conformal_level": level,
                    "risk_threshold": threshold,
                    "locality_rule": LOCALITY_RULE,
                    "maximum_extra_deletions": 1,
                    "risk_threshold_rule": "minimum_across_development_domains",
                }
                output[name] = {
                    "config": config,
                    "domains": {
                        domain_name: domain_metric(
                            domain, predictions[domain_name], config
                        ) for domain_name, domain in development.items()
                    },
                }
    return output, predictions


def head_metrics(examples: list[dict], predictions: dict) -> dict:
    output = {}
    for head, target in (("benefit", "benefit_target"),
                         ("risk", "risk_target")):
        labels = [int(row[target]) for row in examples]
        scores = [predictions[(
            row["episode_id"], row["boundary_index"], row["slot"]
        )][head] for row in examples]
        output[head] = probability_metrics(labels, scores)
    return output


def add_arguments(parser) -> None:
    parser.add_argument("--templama-train-label-root", required=True)
    parser.add_argument("--lama-train-label-root", required=True)
    parser.add_argument("--templama-dev-data-root", required=True)
    parser.add_argument("--templama-dev-label-root", required=True)
    parser.add_argument("--lama-dev-data-root", required=True)
    parser.add_argument("--lama-dev-label-root", required=True)
    parser.add_argument("--v8-feature-root", required=True)
    parser.add_argument("--v8-feature-audit", required=True)
    parser.add_argument("--v7-diagnostic", required=True)


def load_inputs(args) -> tuple[dict, dict, dict]:
    feature_root = Path(args.v8_feature_root)
    feature_audit_path = Path(args.v8_feature_audit)
    feature_audit = json.loads(feature_audit_path.read_text())
    if (feature_audit.get("protocol") != FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("V8 feature audit did not pass")
    train = {
        "templama": load_training_domain(
            Path(args.templama_train_label_root), feature_root / "templama_train"
        ),
        "lama": load_training_domain(
            Path(args.lama_train_label_root), feature_root / "lama_train"
        ),
    }
    development = {
        "templama": load_development_domain(
            Path(args.templama_dev_label_root), feature_root / "templama_dev",
            Path(args.templama_dev_data_root), "development_episodes.jsonl",
        ),
        "lama": load_development_domain(
            Path(args.lama_dev_label_root), feature_root / "lama_dev",
            Path(args.lama_dev_data_root), "development_episodes.jsonl",
        ),
    }
    hashes = {
        "v8_feature_audit": sha256(feature_audit_path),
        "v7_diagnostic": sha256(Path(args.v7_diagnostic)),
    }
    return train, development, hashes


def extended_gate(seed_results: list[dict], development: dict,
                  v7_diagnostic: dict) -> dict:
    base = development_gate(seed_results, development, True)
    lama_gain = base["mean_gain"]["lama"]
    v7_lama_gain = v7_diagnostic["diagnosis_evidence"][
        "joint_lama_domain_specific_ceiling"
    ]["mean_gain"]
    lama_ap = statistics.fmean(
        row["head_metrics"]["lama"]["benefit"]["average_precision"]
        for row in seed_results
    )
    v7_ap = statistics.fmean(
        v7_diagnostic["ranking_and_calibration"][str(seed)]["lama"][
            "benefit"
        ]["overall"]["average_precision"] for seed in SEEDS
    )
    extra = {
        "lama_gain_over_v7_at_least_point_005": lama_gain - v7_lama_gain >= .005,
        "lama_benefit_ap_over_v7_at_least_point_03": lama_ap - v7_ap >= .03,
    }
    return {
        **base,
        "checks": {**base["checks"], **extra},
        "passed_pending_independent_audit": (
            base["passed_pending_independent_audit"] and all(extra.values())
        ),
        "v8_lama_benefit_average_precision": lama_ap,
        "v7_lama_benefit_average_precision": v7_ap,
        "lama_benefit_average_precision_gain": lama_ap - v7_ap,
        "v7_lama_gain": v7_lama_gain,
        "lama_gain_over_v7": lama_gain - v7_lama_gain,
    }


def train(args) -> dict:
    import torch

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    train_domains, development, hashes = load_inputs(args)
    v7_diagnostic = json.loads(Path(args.v7_diagnostic).read_text())
    combined = [
        {**row, "training_domain": domain_name}
        for domain_name, domain in train_domains.items()
        for row in domain["examples"]
    ]
    counts = {
        name: {
            "examples": len(domain["examples"]),
            "benefit": sum(row["benefit_target"] for row in domain["examples"]),
            "risk": sum(row["risk_target"] for row in domain["examples"]),
            "groups": len({row["source_group"] for row in domain["labels"]}),
        } for name, domain in train_domains.items()
    }
    dev_counts = {
        name: {"examples": len(domain["examples"]),
               "groups": len({row["source_group"] for row in domain["labels"]})}
        for name, domain in development.items()
    }
    if counts != EXPECTED_TRAIN or dev_counts != EXPECTED_DEVELOPMENT:
        raise ValueError("V8 residual support differs from registration")
    output_root.mkdir(parents=True)
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    started = time.perf_counter(); seed_results = []
    for seed in SEEDS:
        fitted = fit(combined, seed, device)
        candidates = {}; predictions_by_epoch = {}
        for epoch in CHECKPOINT_EPOCHS:
            values, predictions = evaluate_state(
                fitted["states"][epoch], fitted["means"], fitted["stds"],
                development, epoch,
            )
            candidates.update(values); predictions_by_epoch[epoch] = predictions
        selected = select_config(candidates)
        if selected is None:
            raise ValueError(f"V8 seed {seed} has no dual-domain safe config")
        config = candidates[selected]["config"]
        metrics = {
            name: head_metrics(
                domain["examples"], predictions_by_epoch[config["epoch"]][name]
            ) for name, domain in development.items()
        }
        seed_root = output_root / f"seed_{seed}"; seed_root.mkdir()
        checkpoint = {
            "protocol": PROTOCOL, "seed": seed,
            "feature_names": list(FEATURE_NAMES),
            "input_dim": len(FEATURE_NAMES),
            "state_dict": fitted["states"][config["epoch"]],
            "means": fitted["means"], "stds": fitted["stds"],
            "class_weights": fitted["class_weights"],
            "selected_config": config, "upstream_hashes": hashes,
        }
        checkpoint_path = seed_root / "controller.pt"
        torch.save(checkpoint, checkpoint_path)
        result = {
            "seed": seed, "selected": selected, "selected_config": config,
            "selected_metric": candidates[selected], "head_metrics": metrics,
            "class_weights": fitted["class_weights"], "trace": fitted["trace"],
            "candidate_count": len(candidates),
            "checkpoint_sha256": sha256(checkpoint_path), "passed": True,
        }
        (seed_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        seed_results.append(result)
    gate = extended_gate(seed_results, development, v7_diagnostic)
    checks = {
        "registered_training_support": counts == EXPECTED_TRAIN,
        "registered_development_support": dev_counts == EXPECTED_DEVELOPMENT,
        "registered_feature_dimension": len(FEATURE_NAMES) == 74,
        "all_checkpoints_written": all(
            (output_root / f"seed_{seed}" / "controller.pt").is_file()
            for seed in SEEDS
        ),
        "finite_training": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for row in seed_results for epoch in row["trace"]
        ),
        "heldout_sources_sealed": True,
    }
    summary = {
        "protocol": PROTOCOL, "plan": PLAN,
        "passed": all(checks.values()), "checks": checks,
        "counts": counts, "development_counts": dev_counts,
        "development_baselines": {
            name: {"lifecycle_physics": domain["physics"],
                   "safe_single_fact_key_oracle": domain["oracle"],
                   "safe_single_fact_key_oracle_headroom": domain["oracle_headroom"]}
            for name, domain in development.items()
        },
        "seeds": seed_results, "development_gate": gate,
        "seconds": time.perf_counter() - started,
        "feature_names": list(FEATURE_NAMES), "upstream_hashes": hashes,
        "templama_test_loaded": False, "lama_confirmation_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "v8_independent_training_audit",
        "terminal_status": "v8_relation_posterior_training_completed",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); add_arguments(parser)
    parser.add_argument("--output-root", required=True); parser.add_argument("--device")
    args = parser.parse_args(); result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
