"""Train the locked V7 controller with equal-weighted three-source losses."""

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
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    BATCH_SIZE, CHECKPOINT_EPOCHS, DROPOUT, GRADIENT_NORM, LEARNING_RATE,
    SEEDS, WEIGHT_DECAY, class_weights, make_model, tensors,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    add_arguments as add_v7_arguments, development_gate, evaluate_state,
    load_inputs as load_v7_inputs, load_training_domain, select_config,
)


PROTOCOL = "qwen_utility_record_controller_v10_three_source_training"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v10_three_source_training_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V10_FEWREL_CORPUS_PLAN.md"
ARMS = ("three_source_equal", "fewrel_removed_equal")
V8_PROTOCOL = "qwen_utility_record_controller_v8_relation_posterior_training"


def equal_domain_normalization(domains: dict[str, list[dict]]) \
        -> tuple[list[float], list[float]]:
    means = [
        statistics.fmean(
            statistics.fmean(row["features"][index] for row in rows)
            for rows in domains.values()
        ) for index in range(len(FEATURE_NAMES))
    ]
    stds = []
    for index, mean in enumerate(means):
        variance = statistics.fmean(
            statistics.fmean(
                (row["features"][index] - mean) ** 2 for row in rows
            ) for rows in domains.values()
        )
        stds.append(max(math.sqrt(variance), 1e-6))
    return means, stds


def _cyclic_indices(permutation, start: int, count: int):
    import torch
    size = len(permutation)
    return torch.tensor(
        [int(permutation[(start + offset) % size]) for offset in range(count)],
        dtype=torch.long,
    )


def fit_equal_domains(domains: dict[str, list[dict]], seed: int, device) -> dict:
    """Give each source one third (or one half) of every optimizer step."""
    import torch
    import torch.nn.functional as F

    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = equal_domain_normalization(domains)
    weights = {name: class_weights(rows) for name, rows in domains.items()}
    model = make_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    values = {
        name: tensors(rows, means, stds, device)
        for name, rows in domains.items()
    }
    pos_weights = {
        name: (
            torch.tensor(weight["benefit_positive"], dtype=torch.float32,
                         device=device),
            torch.tensor(weight["risk_positive"], dtype=torch.float32,
                         device=device),
        ) for name, weight in weights.items()
    }
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    steps = math.ceil(max(map(len, domains.values())) / BATCH_SIZE)
    trace, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train()
        permutations = {
            name: torch.randperm(len(rows), generator=generator)
            for name, rows in domains.items()
        }
        losses, norms = [], []
        domain_losses = {name: [] for name in domains}
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            terms = []
            for name, rows in domains.items():
                indices = _cyclic_indices(
                    permutations[name], step * BATCH_SIZE, BATCH_SIZE
                ).to(device)
                x, benefit, risk = values[name]
                benefit_logits, risk_logits = model(x[indices])
                benefit_loss = F.binary_cross_entropy_with_logits(
                    benefit_logits, benefit[indices],
                    pos_weight=pos_weights[name][0],
                )
                risk_loss = F.binary_cross_entropy_with_logits(
                    risk_logits, risk[indices],
                    pos_weight=pos_weights[name][1],
                )
                term = benefit_loss + risk_loss
                terms.append(term); domain_losses[name].append(float(term.item()))
            loss = sum(terms) / len(terms)
            if not torch.isfinite(loss):
                raise ValueError("non-finite V10 equal-domain loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_NORM
            )
            optimizer.step()
            losses.append(float(loss.item())); norms.append(float(norm))
        trace.append({
            "epoch": epoch, "loss": statistics.fmean(losses),
            "gradient_norm": statistics.fmean(norms),
            "per_domain_loss": {
                name: statistics.fmean(values)
                for name, values in domain_losses.items()
            },
        })
        if epoch in CHECKPOINT_EPOCHS:
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            })
    return {
        "means": means, "stds": stds, "class_weights": weights,
        "equal_domain_steps_per_epoch": steps,
        "trace": trace, "states": states,
    }


def extended_gate(primary: list[dict], ablation: list[dict], development: dict,
                  v8_summary: dict, upstream_passed: bool) -> dict:
    base = development_gate(primary, development, upstream_passed)
    primary_gain = base["mean_gain"]
    ablation_gain = {
        domain: statistics.fmean(
            seed["selected_metric"]["domains"][domain][
                "gain_over_lifecycle_physics"
            ] for seed in ablation
        ) for domain in development
    }
    v8_gain = v8_summary["development_gate"]["mean_gain"]
    lama_gain_over_v8 = primary_gain["lama"] - v8_gain["lama"]
    fewrel_removal_lama_drop = primary_gain["lama"] - ablation_gain["lama"]
    checks = {
        **base["checks"],
        "templama_mean_gain_at_least_point_035": (
            primary_gain["templama"] >= .035
        ),
        "lama_gain_over_v8_at_least_point_003": lama_gain_over_v8 >= .003,
        "fewrel_removal_lama_drop_at_least_point_002": (
            fewrel_removal_lama_drop >= .002
        ),
    }
    return {
        **base, "checks": checks,
        "passed_pending_independent_audit": all(checks.values()),
        "ablation_mean_gain": ablation_gain,
        "v8_mean_gain": v8_gain,
        "lama_gain_over_v8": lama_gain_over_v8,
        "fewrel_removal_lama_drop": fewrel_removal_lama_drop,
    }


def _load_inputs(args):
    base_train, development, hashes = load_v7_inputs(args)
    feature_root, label_root = Path(args.fewrel_feature_root), Path(args.fewrel_label_root)
    feature_audit_path = feature_root / "feature_audit.json"
    label_audit_path = label_root / "formal_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    label_audit = json.loads(label_audit_path.read_text())
    if not feature_audit.get("passed") or not label_audit.get("passed") \
            or not label_audit.get("release_passed"):
        raise ValueError("V10 FewRel labels/features are not audited releases")
    fewrel = load_training_domain(label_root, feature_root)
    train_domains = {**base_train, "fewrel": fewrel}
    hashes.update({
        "fewrel_labels": sha256(label_audit_path),
        "fewrel_features": sha256(feature_audit_path),
    })
    v8_summary_path = Path(args.v8_training_root) / "summary.json"
    v8_audit_path = Path(args.v8_audit)
    v8_summary = json.loads(v8_summary_path.read_text())
    v8_audit = json.loads(v8_audit_path.read_text())
    if v8_summary.get("protocol") != V8_PROTOCOL \
            or not v8_audit.get("passed") \
            or v8_audit.get("summary_sha256") != sha256(v8_summary_path):
        raise ValueError("locked V8 comparison did not pass audit")
    hashes.update({"v8_summary": sha256(v8_summary_path),
                   "v8_audit": sha256(v8_audit_path)})
    return train_domains, development, hashes, v8_summary


def _train_arm(arm: str, domains: dict, development: dict, hashes: dict,
               output_root: Path, device) -> list[dict]:
    results = []
    examples = {name: value["examples"] for name, value in domains.items()}
    for seed in SEEDS:
        fitted = fit_equal_domains(examples, seed, device)
        candidates = {}
        for epoch in CHECKPOINT_EPOCHS:
            candidates.update(evaluate_state(
                fitted["states"][epoch], fitted["means"], fitted["stds"],
                development, epoch,
            ))
        selected = select_config(candidates)
        if selected is None:
            raise ValueError(f"V10 {arm} seed {seed} has no safe config")
        config = candidates[selected]["config"]
        seed_root = output_root / arm / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        checkpoint = {
            "protocol": PROTOCOL, "arm": arm, "seed": seed,
            "feature_names": list(FEATURE_NAMES),
            "input_dim": len(FEATURE_NAMES),
            "state_dict": fitted["states"][config["epoch"]],
            "means": fitted["means"], "stds": fitted["stds"],
            "class_weights": fitted["class_weights"],
            "selected_config": config,
            "upstream_audit_hashes": hashes,
            "training_domains": sorted(domains),
            "loss_rule": "equal_mean_of_per_domain_weighted_losses",
        }
        import torch
        checkpoint_path = seed_root / "controller.pt"
        torch.save(checkpoint, checkpoint_path)
        result = {
            "arm": arm, "seed": seed, "selected": selected,
            "selected_config": config,
            "selected_metric": candidates[selected],
            "class_weights": fitted["class_weights"],
            "equal_domain_steps_per_epoch": fitted[
                "equal_domain_steps_per_epoch"
            ],
            "trace": fitted["trace"],
            "candidate_count": len(candidates),
            "checkpoint_sha256": sha256(checkpoint_path), "passed": True,
        }
        (seed_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        results.append(result)
    return results


def train(args) -> dict:
    import torch

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    train_domains, development, hashes, v8_summary = _load_inputs(args)
    counts = {
        name: {
            "examples": len(domain["examples"]),
            "benefit": sum(row["benefit_target"] for row in domain["examples"]),
            "risk": sum(row["risk_target"] for row in domain["examples"]),
            "groups": len({row["source_group"] for row in domain["labels"]}),
            "relations": len({row["target_relation"] for row in domain["labels"]}),
        } for name, domain in train_domains.items()
    }
    if any(value["examples"] <= 0 or value["benefit"] <= 0
           or value["risk"] <= 0 for value in counts.values()):
        raise ValueError("V10 domain lacks required training target support")
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    arms = {
        "three_source_equal": train_domains,
        "fewrel_removed_equal": {
            name: value for name, value in train_domains.items()
            if name != "fewrel"
        },
    }
    results = {
        arm: _train_arm(arm, domains, development, hashes, output_root, device)
        for arm, domains in arms.items()
    }
    gate = extended_gate(
        results["three_source_equal"], results["fewrel_removed_equal"],
        development, v8_summary, True,
    )
    checks = {
        "three_nonempty_training_domains": set(counts) == {
            "templama", "lama", "fewrel"
        },
        "fewrel_full_registered_group_relation_support": (
            counts["fewrel"]["groups"] == 672
            and counts["fewrel"]["relations"] == 48
        ),
        "all_six_checkpoints_written": all(
            (output_root / arm / f"seed_{seed}" / "controller.pt").is_file()
            for arm in ARMS for seed in SEEDS
        ),
        "equal_domain_loss_rule": all(
            all(set(epoch["per_domain_loss"]) == set(arms[arm])
                for epoch in result["trace"])
            for arm in ARMS for result in results[arm]
        ),
        "all_selected_configs_safe": all(
            all(metric["added_unrelated_errors"] == 0
                for metric in result["selected_metric"]["domains"].values())
            for arm in ARMS for result in results[arm]
        ),
        "finite_training": all(
            math.isfinite(epoch["loss"])
            and math.isfinite(epoch["gradient_norm"])
            for arm in ARMS for result in results[arm]
            for epoch in result["trace"]
        ),
        "heldout_sources_not_loaded": True,
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL, "plan": PLAN, "checks": checks,
        "passed": passed, "counts": counts,
        "development_baselines": {
            name: {"lifecycle_physics": domain["physics"],
                   "safe_fact_key_oracle": domain["oracle"],
                   "oracle_headroom": domain["oracle_headroom"]}
            for name, domain in development.items()
        },
        "arms": results, "development_gate": gate,
        "seconds": time.perf_counter() - started,
        "upstream_audit_hashes": hashes,
        "feature_names": list(FEATURE_NAMES),
        "templama_test_loaded": False, "lama_confirmation_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "v10_independent_training_audit" if passed else "stop",
        "terminal_status": (
            "v10_three_source_training_completed" if passed
            else "v10_three_source_training_failed"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def add_arguments(parser) -> None:
    add_v7_arguments(parser)
    parser.add_argument("--fewrel-label-root", required=True)
    parser.add_argument("--fewrel-feature-root", required=True)
    parser.add_argument("--v8-training-root", required=True)
    parser.add_argument("--v8-audit", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
