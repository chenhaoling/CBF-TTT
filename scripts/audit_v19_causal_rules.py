"""Independent audit for the preregistered V19 causal-rule diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_controlled_interference_v18_formal_outcome import metrics
from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v19_causal_rule_explanation_v1"
POLICIES = ("outer_fold_fixed", "retain_all", "global_reset", "last_1", "last_2",
            "delete_newest_exact", "delete_oldest_exact", "delete_key_exact")
METRICS = ("joint_success", "generation_joint_success", "update_accuracy",
           "retained_key_accuracy", "locality_accuracy", "joint_answer_nll")
BOOTSTRAP_SEED = 1927
BOOTSTRAP_DRAWS = 20_000


def read_json(path):
    return json.loads(Path(path).read_text())


def select(episode, row, policy, fixed):
    arms = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(arms) != len(row["arms"]):
        raise ValueError("duplicate causal-rule arm")
    if policy == "outer_fold_fixed":
        return arms[f"fixed:{fixed}"]
    if policy in {"retain_all", "global_reset", "last_1", "last_2"}:
        return arms[f"fixed:{policy}"]
    newest = episode["sessions"][-1]
    conflicts = [item for item in episode["sessions"][:-1]
                 if item["key_id"] == newest["key_id"] and item["value"] != newest["value"]]
    if not conflicts:
        return arms["fixed:retain_all"]
    if policy == "delete_newest_exact":
        return arms[f"delete_one:{conflicts[-1]['session_id']}"]
    if policy == "delete_oldest_exact":
        return arms[f"delete_one:{conflicts[0]['session_id']}"]
    if policy == "delete_key_exact":
        return arms[f"delete_key:{newest['key_id']}"]
    raise ValueError(policy)


def recompute(episodes, rows, outcome):
    by_episode = {row["episode_id"]: row for row in episodes}
    fixed_by_fold = outcome["fixed_selected_by_outer_fold"]
    records = []
    for row in rows:
        episode = by_episode[row["episode_id"]]
        values = {policy: metrics(select(
            episode, row, policy, fixed_by_fold[str(row["fold"])])) for policy in POLICIES}
        records.append({"subject": row["subject"], "template_group": row["template_group"],
                        "fold": row["fold"], "metrics": values})
    tables = {policy: {key: statistics.fmean(
        row["metrics"][policy][key] for row in records) for key in METRICS}
              for policy in POLICIES}
    gains = [{"subject": row["subject"], "template_group": row["template_group"],
              "fold": row["fold"], "joint_gain":
              row["metrics"]["delete_key_exact"]["joint_success"]
              - row["metrics"]["outer_fold_fixed"]["joint_success"]} for row in records]
    subject = clustered_interval(gains, "joint_gain", "subject", BOOTSTRAP_SEED,
                                 BOOTSTRAP_DRAWS)
    template = clustered_interval(gains, "joint_gain", "template_group",
                                  BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["joint_gain"] for row in gains if row["fold"] == fold) for fold in range(5)}
    oracle = outcome["pareto"]["0.03"]["metrics"]
    fixed, rule = tables["outer_fold_fixed"], tables["delete_key_exact"]
    gap = oracle["joint_success"] - fixed["joint_success"]
    fraction = (rule["joint_success"] - fixed["joint_success"]) / gap if gap > 0 else None
    checks = {
        "exact_rule_locality_accuracy_at_least_97": rule["locality_accuracy"] >= .97,
        "subject_cluster_joint_gain_ci_lower_above_zero": subject["lower_95"] > 0,
        "template_cluster_joint_gain_ci_lower_above_zero": template["lower_95"] > 0,
        "no_outer_fold_has_negative_joint_gain": all(value >= 0 for value in by_fold.values()),
        "exact_rule_closes_at_least_90_percent_of_positive_oracle_gap":
            fraction is not None and fraction >= .90,
    }
    passed = all(checks.values())
    return {
        "policies": tables,
        "delete_key_exact_vs_fixed": {
            "joint_gain_micro": rule["joint_success"] - fixed["joint_success"],
            "joint_gain_subject_macro": subject,
            "joint_gain_template_macro": template,
            "by_fold_gain": by_fold,
            "registered_oracle_joint_success": oracle["joint_success"],
            "registered_oracle_gap": gap,
            "fraction_oracle_gap_closed": fraction,
        },
        "explicit_key_explanation_gate": {"checks": checks, "passed": passed},
        "next_stage": "build_v20_semantic_alias_confirmation" if passed
            else ("train_v19_nested_oof_causal_policy"
                  if outcome["confirmation_gate"]["passed"]
                  else "stop_synthetic_selective_forgetting_claim"),
    }


def audit(data_root, outcome_root, inputs, summary_path, output):
    data_root, outcome_root, summary_path = map(Path, (data_root, outcome_root, summary_path))
    result = read_json(summary_path)
    outcome_path, outcome_audit_path = (outcome_root / "summary.json",
                                        outcome_root / "audit.json")
    outcome, outcome_audit = read_json(outcome_path), read_json(outcome_audit_path)
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    episodes = load_jsonl(data_root / "confirmation.jsonl")
    rows, hashes = [], {}
    for value in inputs:
        path = Path(value); rows.extend(load_jsonl(path)); hashes[path.name] = sha(path)
    dependencies = {
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_audit_path),
        "data_sha256": sha(data_root / "confirmation.jsonl"),
        "outcome_summary_sha256": sha(outcome_path),
        "outcome_audit_sha256": sha(outcome_audit_path),
    }
    table = recompute(episodes, rows, outcome)
    checks = {
        "audited_terminal_v19_dependencies": (
            data_audit.get("audit_passed") is True
            and outcome_audit.get("audit_passed") is True
            and outcome_audit.get("summary_sha256") == sha(outcome_path)
            and (outcome_root / "runner_exit_code.txt").read_text().strip() == "0"
            and (outcome_root / "terminal_status.txt").read_text().strip()
                == "completed_controlled_interference_v19_qwen"),
        "complete_unique_source_coverage_and_hashes": (
            len(episodes) == len({row["episode_id"] for row in episodes}) == 320
            and len(rows) == len({row["episode_id"] for row in rows}) == 320
            and {row["episode_id"] for row in episodes}
                == {row["episode_id"] for row in rows}
            and result.get("raw_rows_sha256") == hashes),
        "independent_rules_metrics_intervals_and_gate_recomputation": all(
            result.get(key) == value for key, value in table.items()),
        "causal_access_and_terminal_locks": (
            result.get("protocol") == PROTOCOL and result.get("rows") == 320
            and result.get("dependencies") == dependencies
            and result.get("query_or_answer_features_used") is False
            and result.get("current_update_always_retained") is True
            and result.get("model_rescored") is False
            and result.get("terminal_status") == "completed_v19_causal_rule_diagnostic"),
    }
    audit_result = {"protocol": f"{PROTOCOL}_audit",
                    "audit_passed": all(checks.values()), "checks": checks,
                    "summary_sha256": sha(summary_path), "raw_rows_sha256": hashes,
                    "recomputed_explanation_gate": table["explicit_key_explanation_gate"],
                    "model_rescored": False}
    write_json(output, audit_result)
    if not audit_result["audit_passed"]:
        raise RuntimeError("V19 causal-rule diagnostic audit failed")
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--outcome-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
