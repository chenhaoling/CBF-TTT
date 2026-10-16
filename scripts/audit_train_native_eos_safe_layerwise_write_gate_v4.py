"""Independent checkpoint/prediction/metric audit for V4 controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PROTOCOL,
    SEEDS,
    VARIANTS,
    WEIGHT_DECAY,
    action_metrics,
    arrays,
    build_model,
    canonical,
    choose_actions,
    feature_indices,
    grouped_bootstrap,
    load_stage,
    numeric_heuristics,
    predict,
    safe_oracle,
    threshold_policy,
)


def close(left, right, tolerance=1e-7):
    if isinstance(left, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)), abs(float(right)))
    return left == right


def audit(args) -> dict:
    import torch
    label_root, root = Path(args.label_root), Path(args.controller_root)
    summary_path = root / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol") != PROTOCOL or summary.get("test_opened"):
        raise ValueError("controller summary identity/test seal")
    train_labels, train_features, train_dependency = load_stage(label_root, "train")
    dev_labels, dev_features, dev_dependency = load_stage(label_root, "development")
    dev_full, _, _, _, _ = arrays(dev_labels, dev_features)
    reconstructed_runs = []
    for recorded in summary["runs"]:
        variant, seed = recorded["variant"], recorded["seed"]
        if variant not in VARIANTS or seed not in SEEDS:
            raise ValueError("unregistered controller run")
        checkpoint_path = Path(recorded["checkpoint"])
        if sha256(checkpoint_path) != recorded["checkpoint_sha256"]:
            raise ValueError("controller checkpoint hash")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        indices = feature_indices(variant)
        if checkpoint["indices"] != indices or checkpoint["epochs"] != EPOCHS \
                or checkpoint["batch_size"] != BATCH_SIZE \
                or checkpoint["learning_rate"] != LEARNING_RATE \
                or checkpoint["weight_decay"] != WEIGHT_DECAY:
            raise ValueError("controller configuration changed")
        model = build_model(len(indices)); model.load_state_dict(checkpoint["state_dict"])
        prediction = predict(model, dev_full[:, indices], checkpoint["stats"])
        choices = choose_actions(*prediction)
        metrics = action_metrics(dev_labels, choices)
        vs_always = grouped_bootstrap(dev_labels, choices, ["admit_all"] * len(dev_labels), seed + 100)
        vs_never = grouped_bootstrap(dev_labels, choices, ["reject_all"] * len(dev_labels), seed + 200)
        if not close(metrics, recorded["development_metrics"]) \
                or not close(vs_always, recorded["development_vs_always_write"]) \
                or not close(vs_never, recorded["development_vs_never_write"]):
            raise ValueError("controller metric reconstruction")
        prediction_path = Path(recorded["predictions"])
        if sha256(prediction_path) != recorded["predictions_sha256"]:
            raise ValueError("controller prediction hash")
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(dev_labels):
            raise ValueError("controller prediction coverage")
        for index, (row, label, feature, action) in enumerate(zip(rows, dev_labels, dev_features, choices)):
            if (row["group_id"], row["boundary"], row["feature_sha256"], row["action"]) != (
                label["group_id"], label["boundary"], canonical(feature), action
            ):
                raise ValueError("controller prediction identity/action")
            expected = [tensor[index].tolist() for tensor in prediction]
            observed = [row["predicted_benefit"], row["predicted_protected_nll_change"],
                        row["predicted_accuracy_harm_probability"],
                        row["predicted_eos_harm_probability"]]
            if not close(expected, observed):
                raise ValueError("controller prediction values")
        reconstructed_runs.append({
            "variant": variant, "seed": seed, "development_metrics": metrics,
            "development_vs_always_write": vs_always,
            "development_vs_never_write": vs_never,
        })
    run_index = {(row["variant"], row["seed"]): row for row in reconstructed_runs}
    baseline = summary["baselines"]
    expected_baselines = {
        "always_write": action_metrics(dev_labels, ["admit_all"] * len(dev_labels)),
        "never_write": action_metrics(dev_labels, ["reject_all"] * len(dev_labels)),
        "safe_oracle": action_metrics(dev_labels, safe_oracle(dev_labels)),
    }
    heuristics = numeric_heuristics(dev_features)
    expected_baselines.update({name: action_metrics(dev_labels, choices) for name, choices in heuristics.items()})
    for name, score_index in (("loss_threshold", 0), ("norm_threshold", None)):
        config = baseline[name]["configuration"]
        scores = (dev_full[:, score_index].tolist() if score_index is not None
                  else dev_full[:, [5 + 5 * index for index in range(7)]].sum(1).tolist())
        choices = threshold_policy(dev_labels, scores, config["threshold"], config["direction"])
        expected_baselines[name] = {**action_metrics(dev_labels, choices), "configuration": config}
    if not close(expected_baselines, baseline):
        raise ValueError("baseline metric reconstruction")
    never, always = baseline["never_write"], baseline["always_write"]
    stronger = min((never, always), key=lambda value: value["J"])
    oracle_gap = stronger["J"] - baseline["safe_oracle"]["J"]
    release = {}
    for seed in SEEDS:
        full, no_support = run_index["full", seed], run_index["no_support", seed]
        metrics = full["development_metrics"]
        closure = (stronger["J"] - metrics["J"]) / oracle_gap if oracle_gap > 0 else 0.0
        checks = {
            "better_than_always_write": metrics["J"] < always["J"],
            "better_than_never_write": metrics["J"] < never["J"],
            "bootstrap_vs_always_lower_positive": full["development_vs_always_write"]["ci95"][0] > 0,
            "bootstrap_vs_never_lower_positive": full["development_vs_never_write"]["ci95"][0] > 0,
            "oracle_gap_closed_at_least_30pct": closure >= .30,
            "protected_accuracy_safe": metrics["protected_accuracy"] >= never["protected_accuracy"] - .005,
            "protected_nll_safe": metrics["protected_nll"] <= never["protected_nll"] + .01,
            "eos_safe": metrics["eos_rate"] <= never["eos_rate"] + .01,
            "full_better_than_no_support": metrics["J"] < no_support["development_metrics"]["J"],
        }
        release[str(seed)] = {"oracle_gap_closure": closure, "checks": checks, "passed": all(checks.values())}
    if not close(release, summary["development_release"]) \
            or summary["development_released_test"] != all(value["passed"] for value in release.values()):
        raise ValueError("development release reconstruction")
    result = {
        "protocol": PROTOCOL + "_independent_audit", "passed": True,
        "checkpoints": len(reconstructed_runs), "seeds": list(SEEDS), "variants": list(VARIANTS),
        "summary_sha256": sha256(summary_path), "train_dependency": train_dependency,
        "development_dependency": dev_dependency, "development_release": release,
        "test_released": summary["development_released_test"], "test_opened": False,
    }
    (root / "training_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--label-root", required=True)
    parser.add_argument("--controller-root", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
