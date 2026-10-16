"""Independently audit V11 relation-transfer controllers and release gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v5_residual_training import load_checkpoint
from scripts.audit_qwen_utility_record_controller_v7_cross_source import close
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import SEEDS, predict_state
from tasks.train_qwen_utility_record_controller_v7_cross_source import domain_metric
from tasks.train_qwen_utility_record_controller_v11_relation_transfer import (
    ARMS, AUDIT_PROTOCOL, DRO_MAX_FACTOR, DRO_MIN_FACTOR, PLAN, PROTOCOL,
    _selected_diagnostic, add_arguments, bootstrap_selected_gain, load_inputs,
    v11_gate,
)


def audit(args):
    root = Path(args.training_root); summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    train_domains, development, hashes = load_inputs(args)
    reconstructed = {arm: [] for arm in ARMS}; checkpoints = {arm: [] for arm in ARMS}
    checkpoint_checks = []; dro_checks = []
    for arm in ARMS:
        domains = train_domains if arm != "relation_group_dro_no_fewrel" else {
            key: value for key, value in train_domains.items() if key != "fewrel"}
        for seed in SEEDS:
            saved = next(row for row in summary["arms"][arm] if row["seed"] == seed)
            path = root / arm / f"seed_{seed}" / "controller.pt"
            checkpoint = load_checkpoint(path); checkpoints[arm].append(checkpoint)
            predictions = predict_state(
                checkpoint["state_dict"], development["fewrel_unseen"]["examples"],
                checkpoint["means"], checkpoint["stds"])
            metric = domain_metric(
                development["fewrel_unseen"], predictions,
                checkpoint["selected_config"])
            diagnostic = _selected_diagnostic(
                checkpoint["state_dict"], checkpoint["means"], checkpoint["stds"],
                development, checkpoint["selected_config"])
            row = {**saved, "selected_metric": {
                "config": checkpoint["selected_config"],
                "domains": {"fewrel_unseen": metric}},
                "selected_diagnostic": diagnostic}
            reconstructed[arm].append(row)
            checkpoint_checks.append(
                checkpoint.get("protocol") == PROTOCOL
                and checkpoint.get("arm") == arm and checkpoint.get("seed") == seed
                and tuple(checkpoint.get("feature_names", ())) == FEATURE_NAMES
                and checkpoint.get("input_dim") == len(FEATURE_NAMES)
                and checkpoint.get("training_domains") == sorted(domains)
                and checkpoint.get("upstream_audit_hashes") == hashes
                and checkpoint.get("selected_config") == saved["selected_config"]
                and sha256(path) == saved["checkpoint_sha256"]
                and close(row["selected_metric"], saved["selected_metric"])
                and close(diagnostic, saved["selected_diagnostic"]))
            if arm == "erm_equal":
                dro_checks.append(checkpoint["dro"] is None
                                  and checkpoint["loss_rule"] == "equal_source_erm")
            else:
                weights = checkpoint["dro"]["final_relation_weights"]
                valid = checkpoint["loss_rule"] == "relation_group_dro_benefit_equal_source_risk"
                for source, values in weights.items():
                    valid &= abs(sum(values.values()) - 1) <= 1e-6
                    valid &= min(values.values()) > 0
                    valid &= max(values.values()) / min(values.values()) \
                        <= (DRO_MAX_FACTOR / DRO_MIN_FACTOR) + 1e-6
                dro_checks.append(bool(valid))
    bootstrap = bootstrap_selected_gain(
        development["fewrel_unseen"], checkpoints["relation_group_dro"])
    gate = v11_gate(reconstructed, development, bootstrap, True)
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
    checks = {
        "protocol_plan_and_summary_passed": summary.get("protocol") == PROTOCOL
            and summary.get("plan") == PLAN and summary.get("passed"),
        "support_reconstructed": summary.get("counts") == counts
            and summary.get("development_counts") == dev_counts,
        "upstream_hashes_reconstructed": summary.get("upstream_audit_hashes") == hashes,
        "all_nine_checkpoint_hashes_and_metrics_locked": all(checkpoint_checks),
        "dro_weights_and_loss_rules_valid": all(dro_checks),
        "selected_metrics_reconstructed": all(
            close(observed["selected_metric"], saved["selected_metric"])
            and close(observed["selected_diagnostic"], saved["selected_diagnostic"])
            for arm in ARMS for observed, saved in zip(reconstructed[arm], summary["arms"][arm])),
        "development_gate_reconstructed": close(gate, summary["development_gate"]),
        "training_traces_finite": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for arm in ARMS for row in summary["arms"][arm] for epoch in row["trace"]),
        "old_development_and_tests_sealed": (
            not summary["templama_development_loaded"]
            and not summary["lama_development_loaded"]
            and not summary["internal_test_loaded"]
            and not summary["external_test_loaded"]
            and not summary["longmemeval_loaded"] and not summary["locomo_loaded"]),
    }
    passed = all(checks.values()); release = passed and gate["passed_pending_independent_audit"]
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed,
        "development_release_passed": release, "checks": checks,
        "development_checks": gate["checks"],
        "primary_mean_gain": gate["primary_mean_gain"],
        "primary_seed_gains": gate["primary_seed_gains"],
        "primary_mean_closure": gate["primary_mean_closure"],
        "benefit_average_precision_gain": gate["benefit_average_precision_gain"],
        "fewrel_removal_drop": gate["fewrel_removal_drop"],
        "bootstrap": gate["bootstrap"], "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": {f"{arm}:{seed}": sha256(
            root / arm / f"seed_{seed}" / "controller.pt")
            for arm in ARMS for seed in SEEDS},
        "next_stage": "register_v11_internal_external_tests" if release
                      else "stop_v11_relation_transfer_branch",
        "terminal_status": "v11_audit_passed_released" if release
                           else "v11_audit_passed_gate_failed" if passed
                           else "v11_audit_failed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
