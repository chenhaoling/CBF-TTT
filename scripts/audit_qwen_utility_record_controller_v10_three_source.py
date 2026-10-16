"""Independently audit V10 three-source and FewRel-removal controllers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.audit_qwen_utility_record_controller_v7_cross_source import close
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    SEEDS, predict_state,
)
from tasks.train_qwen_utility_record_controller_v7_cross_source import domain_metric
from tasks.train_qwen_utility_record_controller_v10_three_source import (
    ARMS, AUDIT_PROTOCOL, PLAN, PROTOCOL, _load_inputs, add_arguments,
    extended_gate,
)


def audit(args) -> dict:
    root = Path(args.training_root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    train_domains, development, hashes, v8_summary = _load_inputs(args)
    reconstructed = {arm: [] for arm in ARMS}
    checkpoint_checks = []
    for arm in ARMS:
        expected_domains = sorted(
            train_domains if arm == "three_source_equal"
            else {name: value for name, value in train_domains.items()
                  if name != "fewrel"}
        )
        for seed in SEEDS:
            seed_summary = next(
                row for row in summary["arms"][arm] if row["seed"] == seed
            )
            checkpoint_path = root / arm / f"seed_{seed}" / "controller.pt"
            checkpoint = load_checkpoint(checkpoint_path)
            predictions = {
                name: predict_state(
                    checkpoint["state_dict"], domain["examples"],
                    checkpoint["means"], checkpoint["stds"],
                ) for name, domain in development.items()
            }
            domains = {
                name: domain_metric(
                    domain, predictions[name], checkpoint["selected_config"]
                ) for name, domain in development.items()
            }
            row = {
                **seed_summary,
                "selected_metric": {
                    "config": checkpoint["selected_config"],
                    "domains": domains,
                },
            }
            reconstructed[arm].append(row)
            checkpoint_checks.append(
                checkpoint.get("protocol") == PROTOCOL
                and checkpoint.get("arm") == arm
                and checkpoint.get("seed") == seed
                and tuple(checkpoint.get("feature_names", ())) == FEATURE_NAMES
                and checkpoint.get("input_dim") == len(FEATURE_NAMES)
                and checkpoint.get("training_domains") == expected_domains
                and checkpoint.get("loss_rule")
                == "equal_mean_of_per_domain_weighted_losses"
                and checkpoint.get("upstream_audit_hashes") == hashes
                and checkpoint.get("selected_config")
                == seed_summary["selected_config"]
                and sha256(checkpoint_path) == seed_summary["checkpoint_sha256"]
                and close(row["selected_metric"], seed_summary["selected_metric"])
            )
    gate = extended_gate(
        reconstructed["three_source_equal"],
        reconstructed["fewrel_removed_equal"],
        development, v8_summary, True,
    )
    counts = {
        name: {
            "examples": len(domain["examples"]),
            "benefit": sum(row["benefit_target"] for row in domain["examples"]),
            "risk": sum(row["risk_target"] for row in domain["examples"]),
            "groups": len({row["source_group"] for row in domain["labels"]}),
            "relations": len({row["target_relation"] for row in domain["labels"]}),
        } for name, domain in train_domains.items()
    }
    checks = {
        "protocol_plan_and_summary_passed": (
            summary.get("protocol") == PROTOCOL
            and summary.get("plan") == PLAN and summary.get("passed")
        ),
        "training_counts_reconstructed": summary.get("counts") == counts,
        "upstream_hashes_reconstructed": (
            summary.get("upstream_audit_hashes") == hashes
        ),
        "all_six_checkpoints_hashes_and_states_locked": all(checkpoint_checks),
        "selected_metrics_reconstructed": all(
            close(observed["selected_metric"], saved["selected_metric"])
            for arm in ARMS
            for observed, saved in zip(reconstructed[arm], summary["arms"][arm])
        ),
        "extended_gate_reconstructed": close(gate, summary["development_gate"]),
        "equal_domain_loss_traces_finite": all(
            math.isfinite(epoch["loss"])
            and math.isfinite(epoch["gradient_norm"])
            and set(epoch["per_domain_loss"]) == (
                {"templama", "lama", "fewrel"}
                if arm == "three_source_equal" else {"templama", "lama"}
            )
            and all(math.isfinite(value)
                    for value in epoch["per_domain_loss"].values())
            for arm in ARMS for result in summary["arms"][arm]
            for epoch in result["trace"]
        ),
        "finite_selected_metrics": all(
            math.isfinite(value)
            for arm in ARMS for result in reconstructed[arm]
            for domain in result["selected_metric"]["domains"].values()
            for value in (domain["accuracy"], domain["mean_answer_nll"],
                          domain["gain_over_lifecycle_physics"],
                          domain["oracle_gap_closure"])
        ),
        "heldout_sources_sealed": (
            not summary["templama_test_loaded"]
            and not summary["lama_confirmation_loaded"]
            and not summary["longmemeval_loaded"]
            and not summary["locomo_loaded"]
        ),
    }
    passed = all(checks.values())
    release = passed and gate["passed_pending_independent_audit"]
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed,
        "development_release_passed": release,
        "checks": checks, "development_checks": gate["checks"],
        "mean_gain": gate["mean_gain"],
        "mean_oracle_gap_closure": gate["mean_oracle_gap_closure"],
        "ablation_mean_gain": gate["ablation_mean_gain"],
        "v8_mean_gain": gate["v8_mean_gain"],
        "lama_gain_over_v8": gate["lama_gain_over_v8"],
        "fewrel_removal_lama_drop": gate["fewrel_removal_lama_drop"],
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": {
            f"{arm}:{seed}": sha256(
                root / arm / f"seed_{seed}" / "controller.pt"
            ) for arm in ARMS for seed in SEEDS
        },
        "next_stage": (
            "register_v10_natural_session_evaluation" if release
            else "stop_v10_three_source_iteration"
        ),
        "terminal_status": (
            "v10_three_source_audit_passed_released" if release
            else "v10_three_source_audit_passed_gate_failed" if passed
            else "v10_three_source_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
