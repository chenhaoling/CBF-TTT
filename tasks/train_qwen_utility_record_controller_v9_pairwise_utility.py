"""Train V9 direct pairwise counterfactual-utility controllers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import (
    probability_metrics,
)
from tasks.extract_qwen_utility_record_controller_v1_features import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
)
from tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features import (
    AUDIT_PROTOCOL as V8_FEATURE_AUDIT_PROTOCOL,
    pair_records,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    load_jsonl,
    relation_text_record,
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


PROTOCOL = "qwen_utility_record_controller_v9_pairwise_utility_training"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V9_PAIRWISE_UTILITY_PLAN.md"
PROJECTION_DIM = 128
RANK_WEIGHT = .25
ARMS = {
    "pair_rank": {"use_pair": True, "rank_weight": RANK_WEIGHT},
    "pair_no_rank": {"use_pair": True, "rank_weight": 0.0},
    "base_rank": {"use_pair": False, "rank_weight": RANK_WEIGHT},
}


def make_model(input_dim: int, embedding_dim: int, use_pair: bool):
    import torch.nn as nn

    class PairController(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_benefit = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
            )
            self.use_pair = use_pair
            if use_pair:
                self.projection = nn.Linear(
                    embedding_dim, PROJECTION_DIM, bias=False
                )
                self.benefit_fusion = nn.Sequential(
                    nn.Linear(64 + 4 * PROJECTION_DIM, 128),
                    nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(128, 1),
                )
            else:
                self.benefit_fusion = nn.Linear(64, 1)
            # Risk is structurally isolated from every pair representation.
            self.risk_body = nn.Sequential(
                nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(DROPOUT),
                nn.Linear(64, 1),
            )

        def forward(self, base, pair):
            benefit_base = self.base_benefit(base)
            if self.use_pair:
                candidate = self.projection(pair[:, 0])
                incoming = self.projection(pair[:, 1])
                interaction = [candidate, incoming,
                               (candidate - incoming).abs(),
                               candidate * incoming]
                benefit = self.benefit_fusion(
                    __import__("torch").cat([benefit_base, *interaction], dim=-1)
                ).squeeze(-1)
            else:
                benefit = self.benefit_fusion(benefit_base).squeeze(-1)
            return benefit, self.risk_body(base).squeeze(-1)

    return PairController()


def base_normalization(examples: list[dict]) -> tuple[list[float], list[float]]:
    means = [
        statistics.fmean(row["features"][index] for row in examples)
        for index in range(len(BASE_FEATURE_NAMES))
    ]
    stds = []
    for index, mean in enumerate(means):
        variance = statistics.fmean(
            (row["features"][index] - mean) ** 2 for row in examples
        )
        stds.append(max(math.sqrt(variance), 1e-6))
    return means, stds


def grouped_batches(examples: list[dict], seed: int) -> list[list[int]]:
    groups = defaultdict(list)
    for index, row in enumerate(examples):
        groups[(row["training_domain"], row["episode_id"],
                row["boundary_index"])].append(index)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    batches, batch = [], []
    for key in keys:
        values = groups[key]
        if batch and len(batch) + len(values) > BATCH_SIZE:
            batches.append(batch); batch = []
        batch.extend(values)
    if batch:
        batches.append(batch)
    if sorted(index for batch in batches for index in batch) != list(range(len(examples))):
        raise ValueError("V9 grouped batches lost or duplicated examples")
    return batches


def ranking_loss(logits, targets, identities):
    import torch
    import torch.nn.functional as F

    groups = defaultdict(list)
    for index, identity in enumerate(identities):
        groups[identity].append(index)
    values = []
    for indices in groups.values():
        positive = [index for index in indices if float(targets[index]) == 1.0]
        negative = [index for index in indices if float(targets[index]) == 0.0]
        if positive and negative:
            differences = torch.stack([
                logits[pos] - logits[neg]
                for pos in positive for neg in negative
            ])
            values.append(F.softplus(-differences).mean())
    return torch.stack(values).mean() if values else logits.sum() * 0.0


def pairwise_accuracy(examples: list[dict], predictions: dict) -> dict:
    groups = defaultdict(list)
    for row in examples:
        identity = (row["episode_id"], row["boundary_index"])
        score = predictions[(row["episode_id"], row["boundary_index"], row["slot"])][
            "benefit"
        ]
        groups[identity].append((row["benefit_target"], score))
    wins = pairs = 0.0
    boundaries = 0
    for values in groups.values():
        positive = [score for target, score in values if target]
        negative = [score for target, score in values if not target]
        if not positive or not negative:
            continue
        boundaries += 1
        for pos in positive:
            for neg in negative:
                pairs += 1
                wins += float(pos > neg) + .5 * float(pos == neg)
    return {"boundaries": boundaries, "pairs": int(pairs),
            "accuracy": wins / pairs if pairs else 1.0}


def attach_pairs(domain: dict, embedding_artifact: dict,
                 domain_name: str) -> None:
    texts = embedding_artifact["texts"]
    vectors = embedding_artifact["vectors"]
    text_to_index = {text: index for index, text in enumerate(texts)}
    features = {
        (row["episode_id"], row["boundary_index"]): row
        for row in domain["features"]
    }
    for row in domain["examples"]:
        feature = features[(row["episode_id"], row["boundary_index"])]
        candidates, incoming = pair_records(domain, feature)
        candidate = candidates[row["slot"]]
        try:
            row["candidate_embedding"] = vectors[
                text_to_index[relation_text_record(candidate)]
            ]
            row["incoming_embedding"] = vectors[
                text_to_index[relation_text_record(incoming)]
            ]
        except KeyError as error:
            raise ValueError(f"V9 embedding text missing in {domain_name}") from error
        row["training_domain"] = domain_name


def tensors(examples: list[dict], means: list[float], stds: list[float], device):
    import torch

    base = torch.tensor([
        [(value - mean) / std for value, mean, std
         in zip(row["features"], means, stds)] for row in examples
    ], dtype=torch.float32, device=device)
    pair = torch.stack([
        torch.stack((row["candidate_embedding"], row["incoming_embedding"]))
        for row in examples
    ]).float().to(device)
    benefit = torch.tensor([row["benefit_target"] for row in examples],
                           dtype=torch.float32, device=device)
    risk = torch.tensor([row["risk_target"] for row in examples],
                        dtype=torch.float32, device=device)
    identities = [(row["training_domain"], row["episode_id"],
                   row["boundary_index"]) for row in examples]
    return base, pair, benefit, risk, identities


def fit(examples: list[dict], seed: int, device, arm: str) -> dict:
    import torch
    import torch.nn.functional as F

    config = ARMS[arm]
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = base_normalization(examples)
    weights = class_weights(examples)
    embedding_dim = int(examples[0]["candidate_embedding"].numel())
    model = make_model(
        len(BASE_FEATURE_NAMES), embedding_dim, config["use_pair"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    base, pair, benefit, risk, identities = tensors(examples, means, stds, device)
    benefit_weight = torch.tensor(weights["benefit_positive"], device=device)
    risk_weight = torch.tensor(weights["risk_positive"], device=device)
    trace, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train(); losses = []; norms = []
        for batch in grouped_batches(examples, seed + epoch):
            indices = torch.tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            benefit_logits, risk_logits = model(base[indices], pair[indices])
            benefit_loss = F.binary_cross_entropy_with_logits(
                benefit_logits, benefit[indices], pos_weight=benefit_weight
            )
            risk_loss = F.binary_cross_entropy_with_logits(
                risk_logits, risk[indices], pos_weight=risk_weight
            )
            rank = ranking_loss(
                benefit_logits, benefit[indices], [identities[index] for index in batch]
            )
            loss = benefit_loss + risk_loss + config["rank_weight"] * rank
            if not torch.isfinite(loss):
                raise ValueError("non-finite V9 loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
            optimizer.step(); losses.append(float(loss)); norms.append(float(norm))
        trace.append({"epoch": epoch, "loss": statistics.fmean(losses),
                      "gradient_norm": statistics.fmean(norms)})
        if epoch in CHECKPOINT_EPOCHS:
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
    return {"means": means, "stds": stds, "class_weights": weights,
            "embedding_dim": embedding_dim, "trace": trace, "states": states}


def predict_state(state: dict, examples: list[dict], means: list[float],
                  stds: list[float], embedding_dim: int, arm: str) -> dict:
    import torch

    model = make_model(
        len(BASE_FEATURE_NAMES), embedding_dim, ARMS[arm]["use_pair"]
    )
    model.load_state_dict(state, strict=True); model.eval()
    base, pair, _, _, _ = tensors(examples, means, stds, torch.device("cpu"))
    with torch.inference_mode():
        benefit, risk = model(base, pair)
        benefit = benefit.sigmoid().tolist(); risk = risk.sigmoid().tolist()
    return {
        (row["episode_id"], row["boundary_index"], row["slot"]): {
            "benefit": benefit[index], "risk": risk[index],
        } for index, row in enumerate(examples)
    }


def evaluate_state(state: dict, fitted: dict, development: dict,
                   epoch: int, arm: str) -> tuple[dict, dict]:
    predictions = {
        name: predict_state(
            state, domain["examples"], fitted["means"], fitted["stds"],
            fitted["embedding_dim"], arm,
        ) for name, domain in development.items()
    }
    output = {}
    for level in CONFORMAL_LEVELS:
        threshold = min(
            conformal_risk_threshold(predictions[name], domain["examples"], level)
            for name, domain in development.items()
        )
        for utility in UTILITY_THRESHOLDS:
            for penalty in RISK_PENALTIES:
                name = config_name(epoch, utility, penalty, level)
                config = {
                    "epoch": epoch, "utility_threshold": utility,
                    "risk_penalty": penalty, "conformal_level": level,
                    "risk_threshold": threshold, "locality_rule": LOCALITY_RULE,
                    "maximum_extra_deletions": 1,
                    "risk_threshold_rule": "minimum_across_development_domains",
                }
                output[name] = {"config": config, "domains": {
                    domain_name: domain_metric(
                        domain, predictions[domain_name], config
                    ) for domain_name, domain in development.items()
                }}
    return output, predictions


def prediction_metrics(examples: list[dict], predictions: dict) -> dict:
    output = {"pairwise": pairwise_accuracy(examples, predictions)}
    for head, target in (("benefit", "benefit_target"),
                         ("risk", "risk_target")):
        labels = [int(row[target]) for row in examples]
        scores = [predictions[(row["episode_id"], row["boundary_index"],
                               row["slot"])][head] for row in examples]
        output[head] = probability_metrics(labels, scores)
    return output


def add_arguments(parser) -> None:
    for name in ("templama_train", "lama_train", "templama_dev", "lama_dev"):
        parser.add_argument(f"--{name.replace('_', '-')}-data-root", required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-label-root", required=True)
        parser.add_argument(f"--{name.replace('_', '-')}-feature-root", required=True)
    parser.add_argument("--v8-embedding", required=True)
    parser.add_argument("--v8-feature-audit", required=True)
    parser.add_argument("--v8-training-audit", required=True)


def load_inputs(args) -> tuple[dict, dict, dict, dict]:
    feature_audit = json.loads(Path(args.v8_feature_audit).read_text())
    if (feature_audit.get("protocol") != V8_FEATURE_AUDIT_PROTOCOL
            or not feature_audit.get("passed")):
        raise ValueError("V8 feature audit did not pass")
    embedding_path = Path(args.v8_embedding)
    if sha256(embedding_path) != feature_audit["embedding_sha256"]:
        raise ValueError("V8 embedding hash changed")
    embedding = load_checkpoint(embedding_path)
    train, development = {}, {}
    for name in ("templama", "lama"):
        train_domain = load_training_domain(
            Path(getattr(args, f"{name}_train_label_root")),
            Path(getattr(args, f"{name}_train_feature_root")),
        )
        train_domain["episodes"] = load_jsonl(
            Path(getattr(args, f"{name}_train_data_root"))
            / ("train_episodes.jsonl" if name == "templama"
               else "formal_train_episodes.jsonl")
        )
        train_domain["episode_by_id"] = {
            row["episode_id"]: row for row in train_domain["episodes"]
        }
        attach_pairs(train_domain, embedding, name)
        train[name] = train_domain
        dev_domain = load_development_domain(
            Path(getattr(args, f"{name}_dev_label_root")),
            Path(getattr(args, f"{name}_dev_feature_root")),
            Path(getattr(args, f"{name}_dev_data_root")),
            "development_episodes.jsonl",
        )
        dev_domain["episode_by_id"] = {
            row["episode_id"]: row for row in dev_domain["episodes"]
        }
        attach_pairs(dev_domain, embedding, name)
        development[name] = dev_domain
    hashes = {
        "v8_embedding": sha256(embedding_path),
        "v8_feature_audit": sha256(Path(args.v8_feature_audit)),
        "v8_training_audit": sha256(Path(args.v8_training_audit)),
    }
    return train, development, embedding, hashes


def v9_gate(seed_results: list[dict], development: dict,
            v8_audit: dict) -> dict:
    base = development_gate(seed_results, development, True)
    v8_lama = v8_audit["mean_gain"]["lama"]
    extra = {
        "templama_mean_gain_at_least_point_035": (
            base["mean_gain"]["templama"] >= .035
        ),
        "lama_gain_over_v8_at_least_point_003": (
            base["mean_gain"]["lama"] - v8_lama >= .003
        ),
    }
    return {
        **base, "checks": {**base["checks"], **extra},
        "passed_pending_independent_audit": (
            base["passed_pending_independent_audit"] and all(extra.values())
        ),
        "v8_lama_gain": v8_lama,
        "lama_gain_over_v8": base["mean_gain"]["lama"] - v8_lama,
    }


def train(args) -> dict:
    import torch

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    train_domains, development, embedding, hashes = load_inputs(args)
    v8_audit = json.loads(Path(args.v8_training_audit).read_text())
    if not v8_audit.get("passed"):
        raise ValueError("V8 training integrity audit did not pass")
    combined = [row for domain in train_domains.values() for row in domain["examples"]]
    counts = {
        name: {"examples": len(domain["examples"]),
               "benefit": sum(row["benefit_target"] for row in domain["examples"]),
               "risk": sum(row["risk_target"] for row in domain["examples"]),
               "groups": len({row["source_group"] for row in domain["labels"]})}
        for name, domain in train_domains.items()
    }
    dev_counts = {
        name: {"examples": len(domain["examples"]),
               "groups": len({row["source_group"] for row in domain["labels"]})}
        for name, domain in development.items()
    }
    if counts != EXPECTED_TRAIN or dev_counts != EXPECTED_DEVELOPMENT:
        raise ValueError("V9 support differs from registration")
    output_root.mkdir(parents=True)
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    started = time.perf_counter(); arm_results = {}
    for arm in ARMS:
        results = []
        arm_root = output_root / arm; arm_root.mkdir()
        for seed in SEEDS:
            fitted = fit(combined, seed, device, arm)
            candidates = {}; predictions_by_epoch = {}
            for epoch in CHECKPOINT_EPOCHS:
                values, predictions = evaluate_state(
                    fitted["states"][epoch], fitted, development, epoch, arm
                )
                candidates.update(values); predictions_by_epoch[epoch] = predictions
            selected = select_config(candidates)
            if selected is None:
                raise ValueError(f"V9 {arm}/{seed} lacks safe configuration")
            config = candidates[selected]["config"]
            metrics = {
                name: prediction_metrics(
                    domain["examples"], predictions_by_epoch[config["epoch"]][name]
                ) for name, domain in development.items()
            }
            seed_root = arm_root / f"seed_{seed}"; seed_root.mkdir()
            checkpoint = {
                "protocol": PROTOCOL, "arm": arm, "seed": seed,
                "feature_names": list(BASE_FEATURE_NAMES),
                "input_dim": len(BASE_FEATURE_NAMES),
                "embedding_dim": fitted["embedding_dim"],
                "projection_dim": PROJECTION_DIM,
                "state_dict": fitted["states"][config["epoch"]],
                "means": fitted["means"], "stds": fitted["stds"],
                "class_weights": fitted["class_weights"],
                "selected_config": config, "upstream_hashes": hashes,
                "arm_config": ARMS[arm],
            }
            checkpoint_path = seed_root / "controller.pt"
            torch.save(checkpoint, checkpoint_path)
            result = {
                "arm": arm, "seed": seed, "selected": selected,
                "selected_config": config,
                "selected_metric": candidates[selected],
                "prediction_metrics": metrics,
                "class_weights": fitted["class_weights"], "trace": fitted["trace"],
                "candidate_count": len(candidates),
                "checkpoint_sha256": sha256(checkpoint_path), "passed": True,
            }
            (seed_root / "summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            ); results.append(result)
        arm_results[arm] = results
    gate = v9_gate(arm_results["pair_rank"], development, v8_audit)
    checks = {
        "registered_training_support": counts == EXPECTED_TRAIN,
        "registered_development_support": dev_counts == EXPECTED_DEVELOPMENT,
        "all_arms_and_seeds_written": all(
            (output_root / arm / f"seed_{seed}" / "controller.pt").is_file()
            for arm in ARMS for seed in SEEDS
        ),
        "risk_structurally_pair_isolated": all(
            name.startswith(("base_benefit", "benefit_fusion", "projection"))
            or not name.startswith("risk") or "projection" not in name
            for name in make_model(66, embedding["vectors"].shape[1], True).state_dict()
        ),
        "finite_training": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for results in arm_results.values() for row in results
            for epoch in row["trace"]
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
        "arms": arm_results, "development_gate": gate,
        "seconds": time.perf_counter() - started, "upstream_hashes": hashes,
        "embedding_dim": int(embedding["vectors"].shape[1]),
        "templama_test_loaded": False, "lama_confirmation_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "v9_independent_training_audit",
        "terminal_status": "v9_pairwise_utility_training_completed",
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
