"""Independent reconstruction of the one-shot V4 sealed-test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.evaluate_native_eos_safe_layerwise_write_gate_v4 import (
    load_checkpoint_prediction,
    release_checks,
    verify_controller_release,
)
from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    PROTOCOL,
    SEEDS,
    action_metrics,
    arrays,
    canonical,
    choose_actions,
    grouped_bootstrap,
    load_stage,
    numeric_heuristics,
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
    label_root = Path(args.label_root)
    controller_root = Path(args.controller_root)
    root = Path(args.evaluation_root)
    summary_path = root / "sealed_test_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol") != PROTOCOL + "_sealed_test" or not summary.get("test_opened"):
        raise ValueError("sealed-test summary identity")
    controller_summary, _, controller_dependency = verify_controller_release(controller_root)
    if any(summary.get(key) != value for key, value in controller_dependency.items()):
        raise ValueError("controller release dependency changed")
    labels, features, test_dependency = load_stage(label_root, "sealed_test")
    if summary.get("test_dependency") != test_dependency:
        raise ValueError("sealed-test label dependency changed")
    x, _, _, _, _ = arrays(labels, features)
    controller_index = {
        (row["variant"], row["seed"]): row for row in controller_summary["runs"]
    }
    reconstructed = []
    for recorded in summary["runs"]:
        key = (recorded["variant"], recorded["seed"])
        controller = controller_index[key]
        prediction = load_checkpoint_prediction(controller, x)
        choices = choose_actions(*prediction)
        metrics = action_metrics(labels, choices)
        vs_always = grouped_bootstrap(labels, choices, ["admit_all"] * len(labels), key[1] + 100)
        vs_never = grouped_bootstrap(labels, choices, ["reject_all"] * len(labels), key[1] + 200)
        if recorded["checkpoint_sha256"] != controller["checkpoint_sha256"] \
                or not close(recorded["test_metrics"], metrics) \
                or not close(recorded["test_vs_always_write"], vs_always) \
                or not close(recorded["test_vs_never_write"], vs_never):
            raise ValueError("sealed-test metric reconstruction")
        prediction_path = Path(recorded["predictions"])
        if sha256(prediction_path) != recorded["predictions_sha256"]:
            raise ValueError("sealed-test prediction hash")
        prediction_rows = [
            json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()
        ]
        if len(prediction_rows) != len(labels):
            raise ValueError("sealed-test prediction coverage")
        for index, (row, label, feature, action) in enumerate(
            zip(prediction_rows, labels, features, choices)
        ):
            expected_identity = (label["group_id"], label["boundary"], canonical(feature), action)
            observed_identity = (row["group_id"], row["boundary"], row["feature_sha256"], row["action"])
            expected_values = [tensor[index].tolist() for tensor in prediction]
            observed_values = [
                row["predicted_benefit"], row["predicted_protected_nll_change"],
                row["predicted_accuracy_harm_probability"], row["predicted_eos_harm_probability"],
            ]
            if observed_identity != expected_identity or not close(observed_values, expected_values):
                raise ValueError("sealed-test prediction reconstruction")
        reconstructed.append({
            "variant": key[0], "seed": key[1], "test_metrics": metrics,
            "test_vs_always_write": vs_always, "test_vs_never_write": vs_never,
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
    if not close(baselines, summary["baselines"]):
        raise ValueError("sealed-test baseline reconstruction")
    run_index = {(row["variant"], row["seed"]): row for row in reconstructed}
    releases = {}
    for seed in SEEDS:
        full = run_index["full", seed]
        releases[str(seed)] = release_checks(
            full["test_metrics"], full, run_index["no_support", seed], baselines
        )
    success = all(value["passed"] for value in releases.values())
    if not close(releases, summary["test_release"]) or success != summary["test_success_candidate"]:
        raise ValueError("sealed-test release reconstruction")
    result = {
        "protocol": PROTOCOL + "_sealed_test_independent_audit", "passed": True,
        "formal_success": success, "summary_sha256": sha256(summary_path),
        "controller_summary_sha256": controller_dependency["controller_summary_sha256"],
        "controller_independent_audit_sha256": controller_dependency[
            "controller_independent_audit_sha256"
        ],
        "test_dependency": test_dependency, "checkpoints": len(reconstructed),
        "test_release": releases, "test_opened": True,
    }
    (root / "sealed_test_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
