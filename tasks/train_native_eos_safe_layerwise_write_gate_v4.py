"""Frozen three-seed utility/risk controller for EOS-safe V4 admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.native_layerwise_write_gate_v2 import SINGLETON_ACTIONS


PROTOCOL = "native_eos_safe_layerwise_fastweight_admission_v4_controller"
SEEDS = (2026101801, 2026101802, 2026101803)
VARIANTS = ("no_support", "support_only", "full")
EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def feature_indices(variant: str) -> list[int]:
    # [4 uncertainty][7*5 geometry][7*(128 centroid+128 variance+3 relation)][2 tail]
    uncertainty_geometry = list(range(39))
    support = list(range(39, 1852))
    tail = [1852, 1853]
    if variant == "no_support":
        return uncertainty_geometry + tail
    if variant == "support_only":
        return support + tail
    if variant == "full":
        return list(range(1854))
    raise ValueError(variant)


def load_stage(root: Path, stage: str) -> tuple[list[dict], list[dict], dict]:
    audit_path = root / f"{stage}_independent_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("test_opened"):
        raise ValueError(f"{stage} labels did not pass independent audit")
    labels = read_jsonl(sorted(root.glob(f"{stage}_shard*_labels.jsonl")))
    features = read_jsonl(sorted(root.glob(f"{stage}_shard*_features.jsonl")))
    feature_index = {(row["group_id"], row["boundary"]): row for row in features}
    if len(feature_index) != len(features) or len(labels) != len(features):
        raise ValueError(f"{stage} feature/label coverage")
    for row in labels:
        feature = feature_index[row["group_id"], row["boundary"]]
        if canonical(feature) != row["feature_sha256"]:
            raise ValueError(f"{stage} feature dependency changed")
    ordered_features = [feature_index[row["group_id"], row["boundary"]] for row in labels]
    return labels, ordered_features, {"audit_sha256": sha256(audit_path)}


def arrays(labels: list[dict], features: list[dict]):
    import torch
    x = torch.tensor([row["features"] for row in features], dtype=torch.float32)
    benefit = torch.tensor([
        [row["benefit"][action] for action in SINGLETON_ACTIONS] for row in labels
    ], dtype=torch.float32)
    protected_delta = torch.tensor([
        [row["protected_nll_change"][action] for action in SINGLETON_ACTIONS] for row in labels
    ], dtype=torch.float32)
    accuracy_harm = torch.tensor([
        [row["protected_accuracy_harm"][action] for action in SINGLETON_ACTIONS] for row in labels
    ], dtype=torch.float32)
    eos_harm = torch.tensor([
        [row["eos_increase_harm"][action] for action in SINGLETON_ACTIONS] for row in labels
    ], dtype=torch.float32)
    return x, benefit, protected_delta, accuracy_harm, eos_harm


def _stats(value):
    mean = value.mean(0); std = value.std(0, unbiased=False).clamp_min(1e-5)
    return mean, std


def build_model(width: int):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(width, 256), torch.nn.GELU(), torch.nn.Dropout(.1),
        torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(.1),
        torch.nn.Linear(64, 28),
    )


def predict(model, x, stats: dict):
    import torch
    model.eval()
    with torch.inference_mode():
        raw = model((x - stats["x_mean"]) / stats["x_std"])
    benefit = raw[:, :7] * stats["benefit_std"] + stats["benefit_mean"]
    protected_delta = raw[:, 7:14] * stats["delta_std"] + stats["delta_mean"]
    accuracy_harm_probability = raw[:, 14:21].sigmoid()
    eos_harm_probability = raw[:, 21:].sigmoid()
    return benefit, protected_delta, accuracy_harm_probability, eos_harm_probability


def choose_actions(benefit, protected_delta, accuracy_harm_probability, eos_harm_probability) -> list[str]:
    choices = []
    for row in range(benefit.shape[0]):
        eligible = [index for index in range(7)
                    if float(benefit[row, index]) > 0
                    and float(protected_delta[row, index]) <= 0
                    and float(accuracy_harm_probability[row, index]) < .5
                    and float(eos_harm_probability[row, index]) < .5]
        if not eligible:
            choices.append("reject_all")
        else:
            index = max(eligible, key=lambda item: float(benefit[row, item]))
            choices.append(SINGLETON_ACTIONS[index])
    return choices


def action_metrics(labels: list[dict], choices: list[str]) -> dict:
    def query_values(row, action, role, key):
        return [query[key] for query in row["actions"][action]["queries"] if query["role"] == role]
    result = {
        "J": statistics.fmean(row["actions"][action]["J"] for row, action in zip(labels, choices)),
        "protected_accuracy": statistics.fmean(
            row["actions"][action]["protected_accuracy"] for row, action in zip(labels, choices)
        ),
        "protected_nll": statistics.fmean(
            row["actions"][action]["protected_nll"] for row, action in zip(labels, choices)
        ),
        "eos_rate": statistics.fmean(
            row["actions"][action]["eos_rate"] for row, action in zip(labels, choices)
        ),
        "related_accuracy": statistics.fmean(
            value for row, action in zip(labels, choices)
            for value in query_values(row, action, "related", "correct")
        ),
        "related_nll": statistics.fmean(
            value for row, action in zip(labels, choices)
            for value in query_values(row, action, "related", "answer_nll")
        ),
        "write_rate": sum(action != "reject_all" for action in choices) / len(choices),
        "actions": {name: choices.count(name) for name in ("reject_all", *SINGLETON_ACTIONS)},
    }
    return result


def grouped_bootstrap(labels: list[dict], choices: list[str], baseline: list[str], seed: int) -> dict:
    groups = sorted({row["group_id"] for row in labels})
    values = [statistics.fmean(
        row["actions"][base]["J"] - row["actions"][action]["J"]
        for row, action, base in zip(labels, choices, baseline) if row["group_id"] == group
    ) for group in groups]
    rng = random.Random(seed); draws = 20_000; count = len(values)
    estimates = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count))
                       for _ in range(draws))
    return {
        "mean": statistics.fmean(values),
        "ci95": [estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]],
        "groups": count, "draws": draws, "seed": seed,
    }


def threshold_policy(labels: list[dict], scores: list[float], threshold: float, direction: str) -> list[str]:
    if direction == "high":
        return ["admit_all" if value >= threshold else "reject_all" for value in scores]
    return ["admit_all" if value <= threshold else "reject_all" for value in scores]


def fit_threshold(labels: list[dict], scores: list[float]) -> dict:
    unique = sorted(set(scores))
    span = max(1.0, unique[-1] - unique[0])
    candidates = [unique[0] - span, *unique, unique[-1] + span]
    best = None
    for direction in ("high", "low"):
        for threshold in candidates:
            choices = threshold_policy(labels, scores, threshold, direction)
            value = action_metrics(labels, choices)["J"]
            key = (value, direction, threshold)
            if best is None or key < best[0]:
                best = (key, {"threshold": threshold, "direction": direction, "train_J": value})
    return best[1]


def numeric_heuristics(features: list[dict]) -> dict[str, list[str]]:
    support_in, novelty, duplicate = [], [], []
    for row in features:
        values = row["features"]
        cosines = [values[39 + index * 259 + 256] for index in range(7)]
        likelihoods = [values[39 + index * 259 + 257] for index in range(7)]
        recent = [values[39 + index * 259 + 258] > .5 for index in range(7)]
        eligible = [index for index in range(7) if recent[index]] or list(range(7))
        support_index = max(eligible, key=lambda index: likelihoods[index])
        novelty_index = min(eligible, key=lambda index: cosines[index])
        support_in.append(SINGLETON_ACTIONS[support_index])
        novelty.append(SINGLETON_ACTIONS[novelty_index])
        duplicate.append(
            "reject_all" if max(cosines[index] for index in eligible) >= .95
            else SINGLETON_ACTIONS[support_index]
        )
    return {"support_in": support_in, "novelty_only": novelty, "duplicate_skip": duplicate}


def train_one(train_labels, train_features, dev_labels, dev_features, variant: str, seed: int, output: Path):
    import torch
    torch.manual_seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    indices = feature_indices(variant)
    train_x, train_b, train_d, train_accuracy_harm, train_eos_harm = arrays(train_labels, train_features)
    dev_x, _, _, _, _ = arrays(dev_labels, dev_features)
    train_x, dev_x = train_x[:, indices], dev_x[:, indices]
    x_mean, x_std = _stats(train_x); b_mean, b_std = _stats(train_b); d_mean, d_std = _stats(train_d)
    stats = {"x_mean": x_mean, "x_std": x_std, "benefit_mean": b_mean,
             "benefit_std": b_std, "delta_mean": d_mean, "delta_std": d_std}
    target_b = (train_b - b_mean) / b_std; target_d = (train_d - d_mean) / d_std
    model = build_model(len(indices))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(EPOCHS):
        permutation = torch.randperm(len(train_x), generator=generator)
        epoch_losses = []
        for start in range(0, len(permutation), BATCH_SIZE):
            batch = permutation[start:start + BATCH_SIZE]
            raw = model((train_x[batch] - x_mean) / x_std)
            loss = (
                torch.nn.functional.mse_loss(raw[:, :7], target_b[batch])
                + torch.nn.functional.mse_loss(raw[:, 7:14], target_d[batch])
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    raw[:, 14:21], train_accuracy_harm[batch]
                )
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    raw[:, 21:], train_eos_harm[batch]
                )
            )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); epoch_losses.append(float(loss.detach()))
        losses.append(statistics.fmean(epoch_losses))
    train_prediction = predict(model, train_x, stats)
    dev_prediction = predict(model, dev_x, stats)
    train_choices = choose_actions(*train_prediction); dev_choices = choose_actions(*dev_prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": PROTOCOL, "variant": variant, "seed": seed,
        "indices": indices, "state_dict": model.state_dict(), "stats": stats,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "losses": losses,
    }
    torch.save(checkpoint, output)
    prediction_path = output.with_suffix(".predictions.jsonl")
    with prediction_path.open("x") as sink:
        for row, feature, benefit, delta, accuracy_harm, eos_harm, action in zip(
            dev_labels, dev_features, *dev_prediction, dev_choices
        ):
            sink.write(json.dumps({
                "protocol": PROTOCOL, "variant": variant, "seed": seed,
                "group_id": row["group_id"], "boundary": row["boundary"],
                "feature_sha256": canonical(feature),
                "predicted_benefit": benefit.tolist(),
                "predicted_protected_nll_change": delta.tolist(),
                "predicted_accuracy_harm_probability": accuracy_harm.tolist(),
                "predicted_eos_harm_probability": eos_harm.tolist(), "action": action,
            }) + "\n")
    return {
        "variant": variant, "seed": seed, "checkpoint": str(output),
        "checkpoint_sha256": sha256(output), "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path), "final_loss": losses[-1],
        "train_metrics": action_metrics(train_labels, train_choices),
        "development_metrics": action_metrics(dev_labels, dev_choices),
        "development_vs_always_write": grouped_bootstrap(
            dev_labels, dev_choices, ["admit_all"] * len(dev_labels), seed + 100
        ),
        "development_vs_never_write": grouped_bootstrap(
            dev_labels, dev_choices, ["reject_all"] * len(dev_labels), seed + 200
        ),
    }


def safe_oracle(labels: list[dict]) -> list[str]:
    choices = []
    for row in labels:
        reject = row["actions"]["reject_all"]
        eligible = [name for name in SINGLETON_ACTIONS
                    if row["actions"][name]["J"] < reject["J"]
                    and row["actions"][name]["protected_accuracy"] >= reject["protected_accuracy"]
                    and row["actions"][name]["eos_rate"] <= reject["eos_rate"]]
        choices.append(min(eligible, key=lambda name: row["actions"][name]["J"]) if eligible else "reject_all")
    return choices


def train(args) -> dict:
    root, output = Path(args.label_root), Path(args.output_root)
    if output.exists():
        raise FileExistsError(output)
    train_labels, train_features, train_dependency = load_stage(root, "train")
    dev_labels, dev_features, dev_dependency = load_stage(root, "development")
    output.mkdir(parents=True)
    runs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            runs.append(train_one(
                train_labels, train_features, dev_labels, dev_features,
                variant, seed, output / f"{variant}_seed{seed}.pt",
            ))
    train_x, _, _, _, _ = arrays(train_labels, train_features)
    dev_x, _, _, _, _ = arrays(dev_labels, dev_features)
    loss_rule = fit_threshold(train_labels, train_x[:, 0].tolist())
    train_norm = train_x[:, [5 + 5 * index for index in range(7)]].sum(1).tolist()
    dev_norm = dev_x[:, [5 + 5 * index for index in range(7)]].sum(1).tolist()
    norm_rule = fit_threshold(train_labels, train_norm)
    dev_heuristics = numeric_heuristics(dev_features)
    loss_choices = threshold_policy(dev_labels, dev_x[:, 0].tolist(), loss_rule["threshold"], loss_rule["direction"])
    norm_choices = threshold_policy(dev_labels, dev_norm, norm_rule["threshold"], norm_rule["direction"])
    baselines = {
        "always_write": action_metrics(dev_labels, ["admit_all"] * len(dev_labels)),
        "never_write": action_metrics(dev_labels, ["reject_all"] * len(dev_labels)),
        "loss_threshold": {**action_metrics(dev_labels, loss_choices), "configuration": loss_rule},
        "norm_threshold": {**action_metrics(dev_labels, norm_choices), "configuration": norm_rule},
        **{name: action_metrics(dev_labels, choices) for name, choices in dev_heuristics.items()},
        "safe_oracle": action_metrics(dev_labels, safe_oracle(dev_labels)),
    }
    run_index = {(row["variant"], row["seed"]): row for row in runs}
    never = baselines["never_write"]; always = baselines["always_write"]
    stronger = min((never, always), key=lambda value: value["J"])
    oracle_gap = stronger["J"] - baselines["safe_oracle"]["J"]
    release = {}
    for seed in SEEDS:
        full = run_index["full", seed]
        no_support = run_index["no_support", seed]
        metrics = full["development_metrics"]
        gap_closure = ((stronger["J"] - metrics["J"]) / oracle_gap) if oracle_gap > 0 else 0.0
        checks = {
            "better_than_always_write": metrics["J"] < always["J"],
            "better_than_never_write": metrics["J"] < never["J"],
            "bootstrap_vs_always_lower_positive": full["development_vs_always_write"]["ci95"][0] > 0,
            "bootstrap_vs_never_lower_positive": full["development_vs_never_write"]["ci95"][0] > 0,
            "oracle_gap_closed_at_least_30pct": gap_closure >= .30,
            "protected_accuracy_safe": metrics["protected_accuracy"] >= never["protected_accuracy"] - .005,
            "protected_nll_safe": metrics["protected_nll"] <= never["protected_nll"] + .01,
            "eos_safe": metrics["eos_rate"] <= never["eos_rate"] + .01,
            "full_better_than_no_support": metrics["J"] < no_support["development_metrics"]["J"],
        }
        release[str(seed)] = {"oracle_gap_closure": gap_closure, "checks": checks, "passed": all(checks.values())}
    result = {
        "protocol": PROTOCOL, "train_groups": len({row["group_id"] for row in train_labels}),
        "development_groups": len({row["group_id"] for row in dev_labels}),
        "train_labels": len(train_labels), "development_labels": len(dev_labels),
        "seeds": list(SEEDS), "variants": list(VARIANTS), "runs": runs,
        "baselines": baselines, "train_dependency": train_dependency,
        "development_dependency": dev_dependency, "test_opened": False,
        "development_release": release,
        "development_released_test": all(value["passed"] for value in release.values()),
    }
    (output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--label-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
