"""Evaluate preregistered write-time rules on audited V19 outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import arm_index, arm_metrics
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v19_causal_rule_explanation_v1"
OUTCOME_PROTOCOL = "controlled_memory_interference_v19_qwen_confirmation"
RULES = ("retain_all", "global_reset", "last_1", "last_2",
         "delete_newest_exact", "delete_oldest_exact", "delete_key_exact")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 1927


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_release(data_root, outcome_root):
    data_root, outcome_root = Path(data_root), Path(outcome_root)
    manifest = read_json(data_root / "manifest.json")
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    summary_path, audit_path = outcome_root / "summary.json", outcome_root / "audit.json"
    summary, audit = read_json(summary_path), read_json(audit_path)
    checks = (
        manifest.get("protocol") == "controlled_memory_interference_v19_confirmation",
        manifest.get("primary_locality_budget") == .03,
        manifest.get("confirmation_sha256") == sha(data_root / "confirmation.jsonl"),
        data_audit.get("audit_passed") is True,
        data_audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        summary.get("protocol") == OUTCOME_PROTOCOL,
        summary.get("rows") == 320,
        summary.get("primary_budget") == .03,
        summary.get("terminal_status") in {
            "v19_locality_safe_headroom_confirmed",
            "v19_locality_safe_headroom_not_confirmed"},
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(summary_path),
        audit.get("recomputed_confirmation_gate") == summary.get("confirmation_gate"),
        (outcome_root / "runner_exit_code.txt").read_text().strip() == "0",
        (outcome_root / "terminal_status.txt").read_text().strip()
            == "completed_controlled_interference_v19_qwen",
    )
    if not all(checks):
        raise ValueError("causal rule diagnostic requires the audited terminal V19 outcome")
    return summary, {
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_audit_path),
        "data_sha256": sha(data_root / "confirmation.jsonl"),
        "outcome_summary_sha256": sha(summary_path),
        "outcome_audit_sha256": sha(audit_path),
    }


def choose_rule(episode, row, rule, fixed_policy=None):
    arms = arm_index(row)
    if rule in {"retain_all", "global_reset", "last_1", "last_2"}:
        return arms[f"fixed:{rule}"]
    if rule == "outer_fold_fixed":
        return arms[f"fixed:{fixed_policy}"]
    current = episode["sessions"][-1]
    historical = [session for session in episode["sessions"][:-1]
                  if session["key_id"] == current["key_id"]]
    conflicts = [session for session in historical if session["value"] != current["value"]]
    if not conflicts:
        return arms["fixed:retain_all"]
    if rule == "delete_newest_exact":
        return arms[f"delete_one:{conflicts[-1]['session_id']}"]
    if rule == "delete_oldest_exact":
        return arms[f"delete_one:{conflicts[0]['session_id']}"]
    if rule == "delete_key_exact":
        return arms[f"delete_key:{current['key_id']}"]
    raise ValueError(rule)


def aggregate(episodes, rows, outcome_summary):
    if not len(episodes) == len(rows) == 320:
        raise ValueError("invalid V19 causal-rule coverage")
    episode_index = {row["episode_id"]: row for row in episodes}
    if len(episode_index) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("duplicate V19 causal-rule identity")
    fixed_by_fold = outcome_summary["fixed_selected_by_outer_fold"]
    policies = ("outer_fold_fixed",) + RULES
    records = []
    for row in rows:
        episode = episode_index[row["episode_id"]]
        if any(row[key] != episode[key] for key in (
                "subject", "template_group", "fold", "length", "mechanism")):
            raise ValueError("V19 episode/outcome identity mismatch")
        values = {policy: arm_metrics(choose_rule(
            episode, row, policy, fixed_by_fold[str(row["fold"])])) for policy in policies}
        records.append({"episode_id": row["episode_id"], "subject": row["subject"],
                        "template_group": row["template_group"], "fold": row["fold"],
                        "metrics": values})
    keys = ("joint_success", "generation_joint_success", "update_accuracy",
            "retained_key_accuracy", "locality_accuracy", "joint_answer_nll")
    tables = {policy: {key: statistics.fmean(
        record["metrics"][policy][key] for record in records) for key in keys}
              for policy in policies}
    comparison = []
    for record in records:
        rule = record["metrics"]["delete_key_exact"]
        fixed = record["metrics"]["outer_fold_fixed"]
        comparison.append({"subject": record["subject"],
                           "template_group": record["template_group"],
                           "fold": record["fold"],
                           "joint_gain": rule["joint_success"] - fixed["joint_success"]})
    subject = clustered_interval(comparison, "joint_gain", "subject",
                                 BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    template = clustered_interval(comparison, "joint_gain", "template_group",
                                  BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        record["joint_gain"] for record in comparison if record["fold"] == fold)
               for fold in range(5)}
    oracle = outcome_summary["pareto"]["0.03"]["metrics"]
    fixed = tables["outer_fold_fixed"]
    rule = tables["delete_key_exact"]
    oracle_gap = oracle["joint_success"] - fixed["joint_success"]
    fraction = ((rule["joint_success"] - fixed["joint_success"]) / oracle_gap
                if oracle_gap > 0 else None)
    checks = {
        "exact_rule_locality_accuracy_at_least_97": rule["locality_accuracy"] >= .97,
        "subject_cluster_joint_gain_ci_lower_above_zero": subject["lower_95"] > 0,
        "template_cluster_joint_gain_ci_lower_above_zero": template["lower_95"] > 0,
        "no_outer_fold_has_negative_joint_gain": all(value >= 0 for value in by_fold.values()),
        "exact_rule_closes_at_least_90_percent_of_positive_oracle_gap":
            fraction is not None and fraction >= .90,
    }
    sufficient = all(checks.values())
    return {
        "policies": tables,
        "delete_key_exact_vs_fixed": {
            "joint_gain_micro": rule["joint_success"] - fixed["joint_success"],
            "joint_gain_subject_macro": subject,
            "joint_gain_template_macro": template,
            "by_fold_gain": by_fold,
            "registered_oracle_joint_success": oracle["joint_success"],
            "registered_oracle_gap": oracle_gap,
            "fraction_oracle_gap_closed": fraction,
        },
        "explicit_key_explanation_gate": {"checks": checks, "passed": sufficient},
        "next_stage": "build_v20_semantic_alias_confirmation" if sufficient
            else ("train_v19_nested_oof_causal_policy"
                  if outcome_summary["confirmation_gate"]["passed"]
                  else "stop_synthetic_selective_forgetting_claim"),
    }


def run(data_root, outcome_root, inputs, output):
    summary, dependencies = verify_release(data_root, outcome_root)
    episodes = load_jsonl(Path(data_root) / "confirmation.jsonl")
    rows, hashes = [], {}
    for value in inputs:
        path = Path(value); rows.extend(load_jsonl(path)); hashes[path.name] = sha(path)
    table = aggregate(episodes, rows, summary)
    result = {"protocol": PROTOCOL, **table, "rows": len(rows),
              "raw_rows_sha256": hashes, "dependencies": dependencies,
              "query_or_answer_features_used": False,
              "current_update_always_retained": True,
              "model_rescored": False,
              "terminal_status": "completed_v19_causal_rule_diagnostic"}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--outcome-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
