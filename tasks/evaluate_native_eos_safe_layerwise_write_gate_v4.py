"""One-shot sealed-test evaluation for the frozen V4 controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    PROTOCOL,
    SEEDS,
    VARIANTS,
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


def verify_controller_release(controller_root: Path) -> tuple[dict, dict, dict]:
    summary_path = controller_root / "training_summary.json"
    audit_path = controller_root / "training_independent_audit.json"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != PROTOCOL or summary.get("test_opened") \
            or not summary.get("development_released_test") \
            or not audit.get("passed") or not audit.get("test_released") \
            or audit.get("test_opened") \
            or audit.get("summary_sha256") != sha256(summary_path):
        raise ValueError("development did not independently release sealed test")
    return summary, audit, {
        "controller_summary_sha256": sha256(summary_path),
        "controller_independent_audit_sha256": sha256(audit_path),
    }


def load_checkpoint_prediction(recorded: dict, x):
    import torch

    checkpoint_path = Path(recorded["checkpoint"])
    if sha256(checkpoint_path) != recorded["checkpoint_sha256"]:
        raise ValueError("frozen controller checkpoint hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    indices = feature_indices(recorded["variant"])
    if checkpoint["protocol"] != PROTOCOL or checkpoint["variant"] != recorded["variant"] \
            or checkpoint["seed"] != recorded["seed"] or checkpoint["indices"] != indices:
        raise ValueError("frozen controller identity changed")
    model = build_model(len(indices)); model.load_state_dict(checkpoint["state_dict"])
    return predict(model, x[:, indices], checkpoint["stats"])


def release_checks(metrics: dict, full: dict, no_support: dict, baselines: dict) -> dict:
    never, always = baselines["never_write"], baselines["always_write"]
    stronger = min((never, always), key=lambda value: value["J"])
    oracle_gap = stronger["J"] - baselines["safe_oracle"]["J"]
    gap_closure = ((stronger["J"] - metrics["J"]) / oracle_gap) if oracle_gap > 0 else 0.0
    checks = {
        "better_than_always_write": metrics["J"] < always["J"],
        "better_than_never_write": metrics["J"] < never["J"],
        "bootstrap_vs_always_lower_positive": full["test_vs_always_write"]["ci95"][0] > 0,
        "bootstrap_vs_never_lower_positive": full["test_vs_never_write"]["ci95"][0] > 0,
        "oracle_gap_closed_at_least_30pct": gap_closure >= .30,
        "protected_accuracy_safe": metrics["protected_accuracy"] >= never["protected_accuracy"] - .005,
        "protected_nll_safe": metrics["protected_nll"] <= never["protected_nll"] + .01,
        "eos_safe": metrics["eos_rate"] <= never["eos_rate"] + .01,
        "full_better_than_no_support": metrics["J"] < no_support["test_metrics"]["J"],
    }
    return {"oracle_gap_closure": gap_closure, "checks": checks, "passed": all(checks.values())}


def evaluate(args) -> dict:
    import torch

    label_root = Path(args.label_root)
    controller_root = Path(args.controller_root)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError(output)
    controller_summary, _, controller_dependency = verify_controller_release(controller_root)
    labels, features, test_dependency = load_stage(label_root, "sealed_test")
    x, _, _, _, _ = arrays(labels, features)
    output.mkdir(parents=True)
    runs = []
    for recorded in controller_summary["runs"]:
        prediction = load_checkpoint_prediction(recorded, x)
        choices = choose_actions(*prediction)
        prediction_path = output / f'{recorded["variant"]}_seed{recorded["seed"]}.predictions.jsonl'
        with prediction_path.open("x") as sink:
            for row, feature, benefit, delta, accuracy_harm, eos_harm, action in zip(
                labels, features, *prediction, choices
            ):
                sink.write(json.dumps({
                    "protocol": PROTOCOL, "stage": "sealed_test",
                    "variant": recorded["variant"], "seed": recorded["seed"],
                    "group_id": row["group_id"], "boundary": row["boundary"],
                    "feature_sha256": canonical(feature),
                    "predicted_benefit": benefit.tolist(),
                    "predicted_protected_nll_change": delta.tolist(),
                    "predicted_accuracy_harm_probability": accuracy_harm.tolist(),
                    "predicted_eos_harm_probability": eos_harm.tolist(), "action": action,
                }) + "\n")
        runs.append({
            "variant": recorded["variant"], "seed": recorded["seed"],
            "checkpoint_sha256": recorded["checkpoint_sha256"],
            "predictions": str(prediction_path), "predictions_sha256": sha256(prediction_path),
            "test_metrics": action_metrics(labels, choices),
            "test_vs_always_write": grouped_bootstrap(
                labels, choices, ["admit_all"] * len(labels), recorded["seed"] + 100
            ),
            "test_vs_never_write": grouped_bootstrap(
                labels, choices, ["reject_all"] * len(labels), recorded["seed"] + 200
            ),
        })
    loss_config = controller_summary["baselines"]["loss_threshold"]["configuration"]
    norm_config = controller_summary["baselines"]["norm_threshold"]["configuration"]
    loss_choices = threshold_policy(labels, x[:, 0].tolist(), loss_config["threshold"], loss_config["direction"])
    norm_scores = x[:, [5 + 5 * index for index in range(7)]].sum(1).tolist()
    norm_choices = threshold_policy(labels, norm_scores, norm_config["threshold"], norm_config["direction"])
    heuristics = numeric_heuristics(features)
    baselines = {
        "always_write": action_metrics(labels, ["admit_all"] * len(labels)),
        "never_write": action_metrics(labels, ["reject_all"] * len(labels)),
        "loss_threshold": {**action_metrics(labels, loss_choices), "configuration": loss_config},
        "norm_threshold": {**action_metrics(labels, norm_choices), "configuration": norm_config},
        **{name: action_metrics(labels, choices) for name, choices in heuristics.items()},
        "safe_oracle": action_metrics(labels, safe_oracle(labels)),
    }
    run_index = {(row["variant"], row["seed"]): row for row in runs}
    releases = {}
    for seed in SEEDS:
        full = run_index["full", seed]
        releases[str(seed)] = release_checks(
            full["test_metrics"], full, run_index["no_support", seed], baselines
        )
    success = all(value["passed"] for value in releases.values())
    all_better_always = all(value["checks"]["better_than_always_write"] for value in releases.values())
    all_better_never = all(value["checks"]["better_than_never_write"] for value in releases.values())
    if success:
        claim = "all_registered_test_success_conditions_met_pending_independent_audit"
    elif all_better_always and not all_better_never:
        claim = "support gate减少了有害更新，但当前native writer尚未产生可验证的净正内容收益"
    else:
        claim = "registered_test_success_conditions_not_met"
    result = {
        "protocol": PROTOCOL + "_sealed_test", "stage": "sealed_test",
        "test_groups": len({row["group_id"] for row in labels}), "test_labels": len(labels),
        "seeds": list(SEEDS), "variants": list(VARIANTS), "runs": runs,
        "baselines": baselines, "test_release": releases,
        "test_success_candidate": success, "claim_boundary": claim,
        "test_opened": True, **controller_dependency, "test_dependency": test_dependency,
        "torch_version": torch.__version__,
    }
    (output / "sealed_test_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
