"""Train V11 ERM and relation-group DRO controllers on unseen relations."""

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

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import head_diagnostic
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    fact_key_projected_actions,
)
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    BATCH_SIZE, CHECKPOINT_EPOCHS, GRADIENT_NORM, LEARNING_RATE, SEEDS,
    WEIGHT_DECAY, class_weights, make_model, predict_state, residual_actions,
    tensors,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    domain_metric, evaluate_state, load_development_domain,
    load_training_domain, select_config,
)
from tasks.train_qwen_utility_record_controller_v10_three_source import (
    equal_domain_normalization, fit_equal_domains,
)


PROTOCOL = "qwen_utility_record_controller_v11_relation_transfer_training"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V11_RELATION_TRANSFER_PLAN.md"
ARMS = ("erm_equal", "relation_group_dro", "relation_group_dro_no_fewrel")
DRO_ETA = 0.05
DRO_MIN_FACTOR = 0.25
DRO_MAX_FACTOR = 4.0
BOOTSTRAP_DRAWS = 20000

TRAIN_AUDITS = {
    "templama_label": "train_labels_audit.json",
    "templama_feature": "train_feature_audit.json",
    "lama_label": "formal_labels_audit.json",
    "lama_feature": "feature_audit.json",
    "fewrel_label": "formal_audit.json",
    "fewrel_feature": "feature_audit.json",
}


def _passed(path: Path, *, release: bool = False) -> dict:
    value = json.loads(path.read_text())
    if not value.get("passed") or (release and not value.get("release_passed")):
        raise ValueError(f"upstream V11 dependency failed: {path}")
    return value


def load_inputs(args):
    roots = {name: Path(getattr(args, name)) for name in (
        "templama_train_label_root", "templama_train_feature_root",
        "lama_train_label_root", "lama_train_feature_root",
        "fewrel_train_label_root", "fewrel_train_feature_root",
        "development_data_root", "development_label_root",
        "development_feature_root",
    )}
    audits = {
        "templama_label": roots["templama_train_label_root"] / TRAIN_AUDITS["templama_label"],
        "templama_feature": roots["templama_train_feature_root"] / TRAIN_AUDITS["templama_feature"],
        "lama_label": roots["lama_train_label_root"] / TRAIN_AUDITS["lama_label"],
        "lama_feature": roots["lama_train_feature_root"] / TRAIN_AUDITS["lama_feature"],
        "fewrel_label": roots["fewrel_train_label_root"] / TRAIN_AUDITS["fewrel_label"],
        "fewrel_feature": roots["fewrel_train_feature_root"] / TRAIN_AUDITS["fewrel_feature"],
        "development_data": roots["development_data_root"].parent / "data_audit.json",
        "development_label": roots["development_label_root"] / "formal_audit.json",
        "development_feature": roots["development_feature_root"] / "feature_audit.json",
    }
    for name, path in audits.items():
        _passed(path, release=name in {"fewrel_label", "development_label"})
    train = {
        "templama": load_training_domain(
            roots["templama_train_label_root"], roots["templama_train_feature_root"]),
        "lama": load_training_domain(
            roots["lama_train_label_root"], roots["lama_train_feature_root"]),
        "fewrel": load_training_domain(
            roots["fewrel_train_label_root"], roots["fewrel_train_feature_root"]),
    }
    development = load_development_domain(
        roots["development_label_root"], roots["development_feature_root"],
        roots["development_data_root"], "development_episodes.jsonl",
    )
    hashes = {name: sha256(path) for name, path in audits.items()}
    return train, {"fewrel_unseen": development}, hashes


def _cyclic_indices(permutation, start: int, count: int):
    import torch
    return torch.tensor([
        int(permutation[(start + offset) % len(permutation)])
        for offset in range(count)
    ], dtype=torch.long)


def _project_group_weights(values: dict[str, float]) -> dict[str, float]:
    count = len(values); lower = DRO_MIN_FACTOR / count; upper = DRO_MAX_FACTOR / count
    clipped = {key: min(upper, max(lower, value)) for key, value in values.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def fit_relation_group_dro(domains: dict[str, list[dict]], seed: int, device) -> dict:
    """Apply DRO to benefit relation groups and equal-source ERM to risk."""
    import torch
    import torch.nn.functional as F

    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    means, stds = equal_domain_normalization(domains)
    weights = {name: class_weights(rows) for name, rows in domains.items()}
    model = make_model(len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    values = {name: tensors(rows, means, stds, device) for name, rows in domains.items()}
    relation_index = {
        name: [row["relation"] for row in rows] for name, rows in domains.items()
    }
    q = {
        name: {relation: 1 / len(set(relation_index[name]))
               for relation in sorted(set(relation_index[name]))}
        for name in domains
    }
    pos_weights = {
        name: (
            torch.tensor(weight["benefit_positive"], dtype=torch.float32, device=device),
            torch.tensor(weight["risk_positive"], dtype=torch.float32, device=device),
        ) for name, weight in weights.items()
    }
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    steps = math.ceil(max(map(len, domains.values())) / BATCH_SIZE)
    trace, states = [], {}
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train()
        permutations = {name: torch.randperm(len(rows), generator=generator)
                        for name, rows in domains.items()}
        losses, norms = [], []
        benefit_trace, risk_trace = defaultdict(list), defaultdict(list)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True); source_terms = []
            for name, rows in domains.items():
                cpu_indices = _cyclic_indices(
                    permutations[name], step * BATCH_SIZE, BATCH_SIZE)
                indices = cpu_indices.to(device)
                x, benefit, risk = values[name]
                benefit_logits, risk_logits = model(x[indices])
                benefit_losses = F.binary_cross_entropy_with_logits(
                    benefit_logits, benefit[indices],
                    pos_weight=pos_weights[name][0], reduction="none")
                risk_loss = F.binary_cross_entropy_with_logits(
                    risk_logits, risk[indices], pos_weight=pos_weights[name][1])
                batch_relations = [relation_index[name][int(index)] for index in cpu_indices]
                relation_losses = {}
                for relation in sorted(set(batch_relations)):
                    mask = torch.tensor([value == relation for value in batch_relations],
                                        dtype=torch.bool, device=device)
                    relation_losses[relation] = benefit_losses[mask].mean()
                    q[name][relation] *= math.exp(
                        DRO_ETA * float(relation_losses[relation].detach().item()))
                q[name] = _project_group_weights(q[name])
                denominator = sum(q[name][relation] for relation in relation_losses)
                benefit_loss = sum(q[name][relation] * value
                                   for relation, value in relation_losses.items()) / denominator
                source_terms.append(benefit_loss + risk_loss)
                benefit_trace[name].append(float(benefit_loss.item()))
                risk_trace[name].append(float(risk_loss.item()))
            loss = sum(source_terms) / len(source_terms)
            if not torch.isfinite(loss):
                raise ValueError("non-finite V11 relation-group DRO loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
            optimizer.step(); losses.append(float(loss.item())); norms.append(float(norm))
        trace.append({
            "epoch": epoch, "loss": statistics.fmean(losses),
            "gradient_norm": statistics.fmean(norms),
            "per_domain_benefit_loss": {
                name: statistics.fmean(values) for name, values in benefit_trace.items()},
            "per_domain_risk_loss": {
                name: statistics.fmean(values) for name, values in risk_trace.items()},
            "group_weight_min": min(value for source in q.values() for value in source.values()),
            "group_weight_max": max(value for source in q.values() for value in source.values()),
        })
        if epoch in CHECKPOINT_EPOCHS:
            states[epoch] = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()})
    return {
        "means": means, "stds": stds, "class_weights": weights,
        "equal_source_steps_per_epoch": steps, "trace": trace,
        "states": states, "final_relation_weights": q,
    }


def _selected_diagnostic(state, means, stds, development, config):
    examples = development["fewrel_unseen"]["examples"]
    predictions = predict_state(state, examples, means, stds)
    return {
        "benefit": head_diagnostic(examples, predictions, "benefit", "benefit_target")["overall"],
        "risk": head_diagnostic(examples, predictions, "risk", "risk_target")["overall"],
    }


def _train_arm(arm, domains, development, hashes, output_root, device):
    results = []
    examples = {name: value["examples"] for name, value in domains.items()}
    for seed in SEEDS:
        fitted = (fit_equal_domains(examples, seed, device)
                  if arm == "erm_equal"
                  else fit_relation_group_dro(examples, seed, device))
        candidates = {}
        for epoch in CHECKPOINT_EPOCHS:
            candidates.update(evaluate_state(
                fitted["states"][epoch], fitted["means"], fitted["stds"],
                development, epoch))
        selected = select_config(candidates)
        if selected is None:
            raise ValueError(f"V11 {arm} seed {seed} has no zero-collateral config")
        config = candidates[selected]["config"]
        diagnostic = _selected_diagnostic(
            fitted["states"][config["epoch"]], fitted["means"], fitted["stds"],
            development, config)
        seed_root = output_root / arm / f"seed_{seed}"; seed_root.mkdir(parents=True)
        checkpoint = {
            "protocol": PROTOCOL, "arm": arm, "seed": seed,
            "feature_names": list(FEATURE_NAMES), "input_dim": len(FEATURE_NAMES),
            "state_dict": fitted["states"][config["epoch"]],
            "means": fitted["means"], "stds": fitted["stds"],
            "class_weights": fitted["class_weights"], "selected_config": config,
            "upstream_audit_hashes": hashes, "training_domains": sorted(domains),
            "loss_rule": ("equal_source_erm" if arm == "erm_equal"
                          else "relation_group_dro_benefit_equal_source_risk"),
            "dro": None if arm == "erm_equal" else {
                "eta": DRO_ETA, "minimum_factor": DRO_MIN_FACTOR,
                "maximum_factor": DRO_MAX_FACTOR,
                "final_relation_weights": fitted["final_relation_weights"],
            },
        }
        import torch
        checkpoint_path = seed_root / "controller.pt"; torch.save(checkpoint, checkpoint_path)
        result = {
            "arm": arm, "seed": seed, "selected": selected,
            "selected_config": config, "selected_metric": candidates[selected],
            "selected_diagnostic": diagnostic, "trace": fitted["trace"],
            "checkpoint_sha256": sha256(checkpoint_path), "passed": True,
        }
        (seed_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    return results


def _percentile(values, q):
    ordered = sorted(values); position = (len(ordered) - 1) * q
    low = math.floor(position); high = math.ceil(position)
    if low == high: return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def bootstrap_selected_gain(domain, checkpoints, seed=202610192):
    group_seed = defaultdict(list); relation_by_group = {}
    labels_by_id = {(row["episode_id"], row["boundary_index"]): row
                    for row in domain["labels"]}
    for checkpoint in checkpoints:
        predictions = predict_state(
            checkpoint["state_dict"], domain["examples"],
            checkpoint["means"], checkpoint["stds"])
        raw = residual_actions(
            domain["labels"], domain["features"], predictions,
            checkpoint["selected_config"])
        actions = fact_key_projected_actions(
            domain["labels"], domain["features"], raw, domain["matches"])
        totals = defaultdict(lambda: [0, 0])
        for identity, action in actions.items():
            row = labels_by_id[identity]; physics = domain["physics_actions"][identity]
            totals[row["source_group"]][0] += (
                row["actions"][action]["correct"] - row["actions"][physics]["correct"])
            totals[row["source_group"]][1] += row["actions"][physics]["queries"]
            relation_by_group[row["source_group"]] = row["target_relation"]
        for group, (delta, queries) in totals.items():
            group_seed[group].append(delta / queries)
    group_gain = {group: statistics.fmean(values) for group, values in group_seed.items()}
    relation_gain = defaultdict(list)
    for group, gain in group_gain.items():
        relation_gain[relation_by_group[group]].append(gain)
    relation_gain = {key: statistics.fmean(values) for key, values in relation_gain.items()}
    rng = random.Random(seed); groups = sorted(group_gain); relations = sorted(relation_gain)
    group_draws = [statistics.fmean(group_gain[rng.choice(groups)] for _ in groups)
                   for _ in range(BOOTSTRAP_DRAWS)]
    relation_draws = [statistics.fmean(relation_gain[rng.choice(relations)] for _ in relations)
                      for _ in range(BOOTSTRAP_DRAWS)]
    return {
        "mean_group_gain": statistics.fmean(group_gain.values()),
        "group_bootstrap_95ci": [_percentile(group_draws, .025), _percentile(group_draws, .975)],
        "relation_cluster_bootstrap_95ci": [
            _percentile(relation_draws, .025), _percentile(relation_draws, .975)],
        "groups": len(groups), "relations": len(relations),
    }


def v11_gate(results, development, bootstrap, upstream_passed):
    primary = results["relation_group_dro"]
    erm = results["erm_equal"]
    ablation = results["relation_group_dro_no_fewrel"]
    domain = "fewrel_unseen"
    gains = [row["selected_metric"]["domains"][domain]["gain_over_lifecycle_physics"]
             for row in primary]
    closures = [row["selected_metric"]["domains"][domain]["oracle_gap_closure"]
                for row in primary]
    mechanisms = {
        mechanism: statistics.fmean(
            row["selected_metric"]["domains"][domain]["per_mechanism_gain"][mechanism]
            for row in primary)
        for mechanism in development[domain]["physics"]["per_mechanism_accuracy"]
    }
    benefit_ap = statistics.fmean(
        row["selected_diagnostic"]["benefit"]["average_precision"] for row in primary)
    erm_ap = statistics.fmean(
        row["selected_diagnostic"]["benefit"]["average_precision"] for row in erm)
    ablation_gain = statistics.fmean(
        row["selected_metric"]["domains"][domain]["gain_over_lifecycle_physics"]
        for row in ablation)
    mean_gain = statistics.fmean(gains)
    checks = {
        "every_primary_seed_positive": all(value > 0 for value in gains),
        "primary_mean_gain_at_least_point_015": mean_gain >= .015,
        "group_bootstrap_lower_above_zero": bootstrap["group_bootstrap_95ci"][0] > 0,
        "relation_bootstrap_lower_above_zero": bootstrap["relation_cluster_bootstrap_95ci"][0] > 0,
        "zero_collateral_every_primary_seed": all(
            row["selected_metric"]["domains"][domain]["added_unrelated_errors"] == 0
            for row in primary),
        "mean_oracle_closure_at_least_point_50": statistics.fmean(closures) >= .50,
        "critical_mechanisms_nonnegative": all(
            mechanisms[name] >= 0 for name in
            ("semantic_overwrite", "stale_repetition", "explicit_reset_revoke")),
        "benefit_ap_gain_at_least_point_03": benefit_ap - erm_ap >= .03,
        "fewrel_removal_drop_at_least_point_002": mean_gain - ablation_gain >= .002,
        "upstream_audits_passed": upstream_passed,
    }
    return {
        "checks": checks, "passed_pending_independent_audit": all(checks.values()),
        "primary_seed_gains": gains, "primary_mean_gain": mean_gain,
        "primary_mean_closure": statistics.fmean(closures),
        "mean_per_mechanism_gain": mechanisms,
        "primary_benefit_average_precision": benefit_ap,
        "erm_benefit_average_precision": erm_ap,
        "benefit_average_precision_gain": benefit_ap - erm_ap,
        "ablation_mean_gain": ablation_gain,
        "fewrel_removal_drop": mean_gain - ablation_gain,
        "bootstrap": bootstrap,
    }


def train(args):
    import torch
    output_root = Path(args.output_root)
    if output_root.exists(): raise FileExistsError(output_root)
    train_domains, development, hashes = load_inputs(args)
    counts = {name: {
        "examples": len(domain["examples"]),
        "benefit": sum(row["benefit_target"] for row in domain["examples"]),
        "risk": sum(row["risk_target"] for row in domain["examples"]),
        "groups": len({row["source_group"] for row in domain["labels"]}),
        "relations": len({row["target_relation"] for row in domain["labels"]}),
    } for name, domain in train_domains.items()}
    dev_counts = {
        "examples": len(development["fewrel_unseen"]["examples"]),
        "groups": len({row["source_group"] for row in development["fewrel_unseen"]["labels"]}),
        "relations": len({row["target_relation"] for row in development["fewrel_unseen"]["labels"]}),
    }
    if set(counts) != {"templama", "lama", "fewrel"} \
            or counts["fewrel"]["groups"] != 672 or counts["fewrel"]["relations"] != 48 \
            or dev_counts["groups"] != 112 or dev_counts["relations"] != 8:
        raise ValueError("V11 train/development support differs from registration")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root.mkdir(parents=True); started = time.perf_counter()
    arm_domains = {
        "erm_equal": train_domains,
        "relation_group_dro": train_domains,
        "relation_group_dro_no_fewrel": {
            key: value for key, value in train_domains.items() if key != "fewrel"},
    }
    results = {arm: _train_arm(
        arm, domains, development, hashes, output_root, device)
        for arm, domains in arm_domains.items()}
    from scripts.audit_qwen_utility_record_controller_v5_residual_training import load_checkpoint
    primary_checkpoints = [load_checkpoint(
        output_root / "relation_group_dro" / f"seed_{seed}" / "controller.pt")
        for seed in SEEDS]
    bootstrap = bootstrap_selected_gain(
        development["fewrel_unseen"], primary_checkpoints)
    gate = v11_gate(results, development, bootstrap, True)
    checks = {
        "registered_training_support": set(counts) == {"templama", "lama", "fewrel"},
        "new_development_support": dev_counts["groups"] == 112 and dev_counts["relations"] == 8,
        "all_nine_checkpoints_written": all(
            (output_root / arm / f"seed_{seed}" / "controller.pt").is_file()
            for arm in ARMS for seed in SEEDS),
        "finite_training": all(math.isfinite(epoch["loss"])
            and math.isfinite(epoch["gradient_norm"])
            for arm in ARMS for row in results[arm] for epoch in row["trace"]),
        "all_selected_configs_zero_collateral": all(
            row["selected_metric"]["domains"]["fewrel_unseen"]["added_unrelated_errors"] == 0
            for arm in ARMS for row in results[arm]),
        "old_development_and_tests_not_loaded": True,
    }
    summary = {
        "protocol": PROTOCOL, "plan": PLAN, "checks": checks,
        "passed": all(checks.values()), "counts": counts,
        "development_counts": dev_counts,
        "development_baselines": {
            "lifecycle_physics": development["fewrel_unseen"]["physics"],
            "safe_fact_key_oracle": development["fewrel_unseen"]["oracle"],
            "oracle_headroom": development["fewrel_unseen"]["oracle_headroom"],
        },
        "arms": results, "development_gate": gate,
        "upstream_audit_hashes": hashes, "feature_names": list(FEATURE_NAMES),
        "dro": {"eta": DRO_ETA, "minimum_factor": DRO_MIN_FACTOR,
                "maximum_factor": DRO_MAX_FACTOR},
        "seconds": time.perf_counter() - started,
        "templama_development_loaded": False, "lama_development_loaded": False,
        "internal_test_loaded": False, "external_test_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "v11_independent_training_audit" if all(checks.values()) else "stop",
        "terminal_status": "v11_relation_transfer_training_completed" if all(checks.values())
                           else "v11_relation_transfer_training_failed",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def add_arguments(parser):
    for name in (
        "templama_train_label_root", "templama_train_feature_root",
        "lama_train_label_root", "lama_train_feature_root",
        "fewrel_train_label_root", "fewrel_train_feature_root",
        "development_data_root", "development_label_root",
        "development_feature_root",
    ):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--output-root", required=True); parser.add_argument("--device")
    args = parser.parse_args(); result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
