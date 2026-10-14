"""Build gated write-time policy labels for the V19 fallback branch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.evaluate_controlled_interference_v18_pilot import arm_index, arm_metrics
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v19_nested_oof_policy_examples_v1"
RULE_PROTOCOL = "v19_causal_rule_explanation_v1"
OUTCOME_PROTOCOL = "controlled_memory_interference_v19_qwen_confirmation"
FEATURE_NAMES = (
    "newest_exact_key", "newest_exact_value", "newest_value_conflict",
    "relative_age", "later_exact_key_fraction", "later_same_value_fraction",
    "later_conflicting_value_fraction", "same_key_group_fraction",
    "same_value_group_fraction", "newest_key_trigram_jaccard",
    "newest_key_common_prefix", "max_later_key_similarity",
    "mean_later_key_similarity", "max_conflicting_later_similarity",
    "useful_repetition", "near_duplicate_key", "normalized_memory_length",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def ngrams(value, width=3):
    text = f"  {value.casefold()}  "
    return {text[index:index + width] for index in range(len(text) - width + 1)}


def trigram_jaccard(left, right):
    left_set, right_set = ngrams(left), ngrams(right)
    return len(left_set & right_set) / len(left_set | right_set)


def common_prefix(left, right):
    count = 0
    for a, b in zip(left.casefold(), right.casefold()):
        if a != b:
            break
        count += 1
    return count / max(1, max(len(left), len(right)))


def component_features(episode, index):
    sessions = episode["sessions"]
    if not 0 <= index < len(sessions) - 1:
        raise ValueError("policy feature requires a historical component")
    component, newest = sessions[index], sessions[-1]
    later = sessions[index + 1:]
    normalizer = max(1, len(sessions) - 1)
    similarities = [trigram_jaccard(component["key_id"], row["key_id"])
                    for row in later]
    conflicting = [value for value, row in zip(similarities, later)
                   if row["value"] != component["value"]]
    same_key_group = sum(row["key_id"] == component["key_id"]
                         for row in sessions[:-1])
    same_value_group = sum(row["value"] == component["value"]
                           for row in sessions[:-1])
    exact_key = component["key_id"] == newest["key_id"]
    exact_value = component["value"] == newest["value"]
    values = (
        float(exact_key), float(exact_value), float(exact_key and not exact_value),
        (len(sessions) - 1 - index) / normalizer,
        sum(row["key_id"] == component["key_id"] for row in later) / normalizer,
        sum(row["value"] == component["value"] for row in later) / normalizer,
        sum(row["value"] != component["value"] for row in later) / normalizer,
        same_key_group / normalizer, same_value_group / normalizer,
        trigram_jaccard(component["key_id"], newest["key_id"]),
        common_prefix(component["key_id"], newest["key_id"]),
        max(similarities), statistics.fmean(similarities),
        max(conflicting) if conflicting else 0.0,
        float(any(row["key_id"] == component["key_id"]
                  and row["value"] == component["value"] for row in later)),
        float(not exact_key and common_prefix(component["key_id"], newest["key_id"]) >= .75),
        len(sessions) / 64.0,
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("invalid V19 policy feature vector")
    return list(values)


def utility_target(retain, deleted):
    delta = deleted["joint_success"] - retain["joint_success"]
    return 1.0 if delta > 0 else (0.0 if delta < 0 else 0.5)


def intervention_rank(retain, deleted):
    target = utility_target(retain, deleted)
    return (target, retain["joint_answer_nll"] - deleted["joint_answer_nll"],
            deleted["locality_accuracy"])


def episode_examples(episode, outcome):
    arms = arm_index(outcome)
    retain = arm_metrics(arms["fixed:retain_all"])
    key_arms = {arm["deleted_key_id"]: arm for arm in outcome["arms"]
                if arm["kind"] == "delete_key"}
    output = []
    for index, component in enumerate(episode["sessions"][:-1]):
        single_arm = arms[f"delete_one:{component['session_id']}"]
        group_arm = key_arms[component["key_id"]]
        single, group = arm_metrics(single_arm), arm_metrics(group_arm)
        later = episode["sessions"][index + 1:]
        exact_conflict = any(row["key_id"] == component["key_id"]
                             and row["value"] != component["value"] for row in later)
        useful_repeat = any(row["key_id"] == component["key_id"]
                            and row["value"] == component["value"] for row in later)
        selected_arm, selected = single_arm, single
        if exact_conflict and intervention_rank(retain, group) > intervention_rank(
                retain, single):
            selected_arm, selected = group_arm, group
        target = utility_target(retain, selected)
        output.append({
            "example_id": f"{episode['episode_id']}::{component['session_id']}",
            "episode_id": episode["episode_id"], "subject": episode["subject"],
            "template_group": episode["template_group"], "fold": episode["fold"],
            "component_session_id": component["session_id"],
            "current_session_id": episode["current_session_id"],
            "features": component_features(episode, index),
            "feature_names": list(FEATURE_NAMES), "target": target,
            "locality_risk_target": int(
                selected["locality_accuracy"] < retain["locality_accuracy"]),
            "conservation_target": int(target <= .5),
            "exact_conflict": int(exact_conflict),
            "useful_repetition": int(useful_repeat),
            "selected_label_arm_id": selected_arm["arm_id"],
            "single_target": utility_target(retain, single),
            "group_target": utility_target(retain, group),
            "selected_joint_nll_gain": (retain["joint_answer_nll"]
                                        - selected["joint_answer_nll"]),
        })
    return output


def verify_release(data_root, outcome_root, rule_root):
    data_root, outcome_root, rule_root = map(Path, (data_root, outcome_root, rule_root))
    manifest = read_json(data_root / "manifest.json")
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    outcome_summary_path, outcome_audit_path = (outcome_root / "summary.json",
                                                outcome_root / "audit.json")
    outcome, outcome_audit = read_json(outcome_summary_path), read_json(outcome_audit_path)
    rule_summary_path, rule_audit_path = rule_root / "summary.json", rule_root / "audit.json"
    rule, rule_audit = read_json(rule_summary_path), read_json(rule_audit_path)
    checks = (
        manifest.get("protocol") == "controlled_memory_interference_v19_confirmation",
        data_audit.get("audit_passed") is True,
        data_audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        outcome.get("protocol") == OUTCOME_PROTOCOL,
        outcome.get("confirmation_gate", {}).get("passed") is True,
        outcome_audit.get("audit_passed") is True,
        outcome_audit.get("summary_sha256") == sha(outcome_summary_path),
        rule.get("protocol") == RULE_PROTOCOL,
        rule.get("explicit_key_explanation_gate", {}).get("passed") is False,
        rule.get("next_stage") == "train_v19_nested_oof_causal_policy",
        rule.get("query_or_answer_features_used") is False,
        rule.get("model_rescored") is False,
        rule_audit.get("audit_passed") is True,
        rule_audit.get("summary_sha256") == sha(rule_summary_path),
        rule_audit.get("recomputed_explanation_gate")
            == rule.get("explicit_key_explanation_gate"),
        (outcome_root / "runner_exit_code.txt").read_text().strip() == "0",
        (rule_root / "runner_exit_code.txt").read_text().strip() == "0",
    )
    if not all(checks):
        raise ValueError("V19 policy labels require the audited fallback release branch")
    return {
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_audit_path),
        "data_sha256": sha(data_root / "confirmation.jsonl"),
        "outcome_summary_sha256": sha(outcome_summary_path),
        "outcome_audit_sha256": sha(outcome_audit_path),
        "rule_summary_sha256": sha(rule_summary_path),
        "rule_audit_sha256": sha(rule_audit_path),
    }


def build(data_root, outcome_root, rule_root, inputs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_release(data_root, outcome_root, rule_root)
    episodes = load_jsonl(Path(data_root) / "confirmation.jsonl")
    outcomes = []
    raw_hashes = {}
    for value in inputs:
        path = Path(value); outcomes.extend(load_jsonl(path)); raw_hashes[path.name] = sha(path)
    outcome_index = {row["episode_id"]: row for row in outcomes}
    if (len(episodes) != 320 or len(outcomes) != 320
            or len(outcome_index) != 320
            or {row["episode_id"] for row in episodes} != set(outcome_index)):
        raise ValueError("invalid V19 policy-label source coverage")
    examples = [example for episode in episodes for example in episode_examples(
        episode, outcome_index[episode["episode_id"]])]
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in examples))
    target_counts = {str(value): sum(row["target"] == value for row in examples)
                     for value in (0.0, 0.5, 1.0)}
    manifest = {
        "protocol": PROTOCOL, "rows": len(examples), "episodes": len(episodes),
        "feature_names": list(FEATURE_NAMES), "feature_dim": len(FEATURE_NAMES),
        "target_grid": [0.0, 0.5, 1.0], "target_counts": target_counts,
        "locality_risk_rows": sum(row["locality_risk_target"] for row in examples),
        "conservation_rows": sum(row["conservation_target"] for row in examples),
        "exact_conflict_rows": sum(row["exact_conflict"] for row in examples),
        "useful_repetition_rows": sum(row["useful_repetition"] for row in examples),
        "by_fold": {str(fold): sum(row["fold"] == fold for row in examples)
                    for fold in range(5)},
        "examples_sha256": sha(path), "raw_rows_sha256": raw_hashes,
        "dependencies": dependencies, "query_features_used": False,
        "answer_features_used": False, "current_update_always_retained": True,
        "external_sets_loaded": False,
        "terminal_status": "frozen_v19_nested_policy_examples",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--outcome-root", required=True)
    parser.add_argument("--rule-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
