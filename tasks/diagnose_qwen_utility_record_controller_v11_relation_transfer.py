"""Frozen no-refit diagnosis of the failed V11 auxiliary AP gate."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_qwen_utility_record_controller_v5_residual_training import load_checkpoint
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    fact_key_projected_actions,
)
from tasks.train_qwen_utility_record_controller_v5_residual import predict_state, residual_actions
from tasks.train_qwen_utility_record_controller_v11_relation_transfer import (
    ARMS, SEEDS, add_arguments, bootstrap_selected_gain, load_inputs,
)


PROTOCOL = "qwen_utility_record_controller_v11_failure_diagnostic"
AUDIT_PROTOCOL = PROTOCOL + "_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V11_FAILURE_DIAGNOSTIC_PLAN.md"
TRAIN_PROTOCOL = "qwen_utility_record_controller_v11_relation_transfer_training"
TRAIN_AUDIT_PROTOCOL = TRAIN_PROTOCOL + "_audit"


def _actions(domain, checkpoint):
    predictions = predict_state(
        checkpoint["state_dict"], domain["examples"],
        checkpoint["means"], checkpoint["stds"])
    raw = residual_actions(
        domain["labels"], domain["features"], predictions,
        checkpoint["selected_config"])
    return fact_key_projected_actions(
        domain["labels"], domain["features"], raw, domain["matches"])


def _arm_summary(rows, bootstrap):
    domain = "fewrel_unseen"
    return {
        "seed_gains": [row["selected_metric"]["domains"][domain]["gain_over_lifecycle_physics"]
                       for row in rows],
        "mean_gain": statistics.fmean(
            row["selected_metric"]["domains"][domain]["gain_over_lifecycle_physics"]
            for row in rows),
        "benefit_average_precision": statistics.fmean(
            row["selected_diagnostic"]["benefit"]["average_precision"] for row in rows),
        "risk_average_precision": statistics.fmean(
            row["selected_diagnostic"]["risk"]["average_precision"] for row in rows),
        "deletion_precision": statistics.fmean(
            row["selected_metric"]["domains"][domain]["residual_deletion_precision"]
            for row in rows),
        "deletion_fraction": statistics.fmean(
            row["selected_metric"]["domains"][domain]["residual_deletion_fraction"]
            for row in rows),
        "bootstrap": bootstrap,
    }


def diagnose(args):
    root = Path(args.training_root); summary_path = root / "summary.json"
    audit_path = Path(args.training_audit)
    summary = json.loads(summary_path.read_text()); audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != TRAIN_PROTOCOL or not summary.get("passed") \
            or audit.get("protocol") != TRAIN_AUDIT_PROTOCOL or not audit.get("passed") \
            or audit.get("development_release_passed"):
        raise ValueError("V11 diagnostic requires the audited failed release")
    if audit.get("summary_sha256") != sha256(summary_path):
        raise ValueError("V11 training summary changed after audit")
    _, development, hashes = load_inputs(args); domain = development["fewrel_unseen"]
    labels = {(row["episode_id"], row["boundary_index"]): row for row in domain["labels"]}
    checkpoints, actions = {}, {}
    for arm in ARMS:
        checkpoints[arm] = [load_checkpoint(
            root / arm / f"seed_{seed}" / "controller.pt") for seed in SEEDS]
        actions[arm] = [_actions(domain, checkpoint) for checkpoint in checkpoints[arm]]
    bootstraps = {
        arm: bootstrap_selected_gain(domain, checkpoints[arm], 202610192 + index)
        for index, arm in enumerate(ARMS)
    }
    arms = {arm: _arm_summary(summary["arms"][arm], bootstraps[arm]) for arm in ARMS}
    comparisons = []
    relation_values, mechanism_values = defaultdict(list), defaultdict(list)
    for seed_index, seed in enumerate(SEEDS):
        erm, dro = actions["erm_equal"][seed_index], actions["relation_group_dro"][seed_index]
        changed = favorable = unfavorable = correct_delta = unrelated_delta = 0
        relation_delta = defaultdict(lambda: [0, 0]); mechanism_delta = defaultdict(lambda: [0, 0])
        for identity in sorted(erm):
            row = labels[identity]; physics = domain["physics_actions"][identity]
            dro_out = row["actions"][dro[identity]]; erm_out = row["actions"][erm[identity]]
            if dro[identity] != erm[identity]:
                changed += 1; delta = dro_out["correct"] - erm_out["correct"]
                favorable += delta > 0; unfavorable += delta < 0; correct_delta += delta
                unrelated_delta += dro_out["unrelated_correct"] - erm_out["unrelated_correct"]
            for key, target in ((row["target_relation"], relation_delta),
                                (row["mechanism"], mechanism_delta)):
                out = row["actions"][dro[identity]]
                base = row["actions"][physics]
                target[key][0] += out["correct"] - base["correct"]
                target[key][1] += base["queries"]
        comparisons.append({"seed": seed, "changed_boundaries": changed,
                            "favorable_for_dro": favorable,
                            "unfavorable_for_dro": unfavorable,
                            "dro_minus_erm_correct": correct_delta,
                            "dro_minus_erm_unrelated_correct": unrelated_delta})
        for key, (delta, queries) in relation_delta.items():
            relation_values[key].append(delta / queries)
        for key, (delta, queries) in mechanism_delta.items():
            mechanism_values[key].append(delta / queries)
    per_relation = {key: statistics.fmean(values) for key, values in sorted(relation_values.items())}
    per_mechanism = {key: statistics.fmean(values) for key, values in sorted(mechanism_values.items())}
    conclusion = (
        "constructed_corpus_effect_supported_robust_loss_not_supported"
        if arms["erm_equal"]["bootstrap"]["relation_cluster_bootstrap_95ci"][0] > 0
        and arms["relation_group_dro"]["bootstrap"]["relation_cluster_bootstrap_95ci"][0] > 0
        and arms["relation_group_dro_no_fewrel"]["mean_gain"] < .002
        else "inconclusive_corpus_effect"
    )
    return {
        "protocol": PROTOCOL, "plan": PLAN, "passed": True,
        "training_summary_sha256": sha256(summary_path),
        "training_audit_sha256": sha256(audit_path),
        "upstream_hashes": hashes, "arms": arms,
        "dro_vs_erm_action_comparison": comparisons,
        "dro_per_relation_gain": per_relation,
        "dro_per_mechanism_gain": per_mechanism,
        "conclusion": conclusion,
        "templama_development_loaded": False, "lama_development_loaded": False,
        "internal_test_loaded": False, "external_test_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "register_new_source_erm_replication",
        "terminal_status": "v11_failure_diagnostic_completed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__); add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = diagnose(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
