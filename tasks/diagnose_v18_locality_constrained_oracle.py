"""Post-hoc V18 Pareto diagnostic under frozen aggregate locality budgets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index, arm_metrics, select_best_fixed,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_locality_constrained_oracle_posthoc_v1"
BUDGETS = (0.0, 0.01, 0.03, 0.05, 0.10)
PRIMARY_BUDGET = 0.03
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2011


def candidates(row):
    return row["arms"]


def _rank(total, failures):
    return (total["joint_success"], total["update_accuracy"],
            total["retained_key_accuracy"], total["locality_accuracy"],
            total["generation_joint_success"], -total["joint_answer_nll"], -failures)


def constrained_selection(rows, maximum_failures):
    zero = {"joint_success": 0, "update_accuracy": 0.0,
            "retained_key_accuracy": 0.0, "locality_accuracy": 0,
            "generation_joint_success": 0, "joint_answer_nll": 0.0}
    dp = {0: (zero, [])}
    for row in rows:
        next_dp = {}
        for failures, (total, chosen) in dp.items():
            for arm in candidates(row):
                value = arm_metrics(arm); extra = 1 - value["locality_accuracy"]
                new_failures = failures + extra
                if new_failures > maximum_failures:
                    continue
                updated = {key: total[key] + value[key] for key in zero}
                candidate = (updated, chosen + [(arm["arm_id"], value)])
                previous = next_dp.get(new_failures)
                if previous is None or _rank(updated, new_failures) > _rank(
                        previous[0], new_failures):
                    next_dp[new_failures] = candidate
        if not next_dp:
            raise RuntimeError("locality-constrained oracle has no feasible state")
        dp = next_dp
    failures, (total, chosen) = max(
        dp.items(), key=lambda item: _rank(item[1][0], item[0]))
    return failures, total, chosen


def aggregate(rows):
    if len(rows) != 320 or len({row["episode_id"] for row in rows}) != 320:
        raise ValueError("V18-LC requires complete unique formal rows")
    fixed_by_fold = {str(fold): select_best_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(5)}
    fixed = []
    for row in rows:
        policy = fixed_by_fold[str(row["fold"])]
        fixed.append(arm_metrics(arm_index(row)[f"fixed:{policy}"]))
    fixed_metrics = {key: statistics.fmean(value[key] for value in fixed) for key in (
        "joint_success", "update_accuracy", "retained_key_accuracy", "locality_accuracy",
        "generation_joint_success", "joint_answer_nll")}
    pareto, selections = {}, {}
    for budget in BUDGETS:
        maximum = math.floor(budget * len(rows) + 1e-12)
        failures, total, chosen = constrained_selection(rows, maximum)
        metrics = {key: value / len(rows) for key, value in total.items()}
        records = []
        for row, fixed_value, (arm_id, selected) in zip(rows, fixed, chosen):
            records.append({"episode_id": row["episode_id"], "subject": row["subject"],
                            "template_group": row["template_group"], "fold": row["fold"],
                            "mechanism": row["mechanism"], "length": row["length"],
                            "arm_id": arm_id, "joint_gain": selected["joint_success"]
                                - fixed_value["joint_success"],
                            "selected": selected, "fixed": fixed_value})
        subject = clustered_interval(records, "joint_gain", "subject",
                                     BOOTSTRAP_SEED + int(budget * 100), BOOTSTRAP_DRAWS)
        template = clustered_interval(records, "joint_gain", "template_group",
                                      BOOTSTRAP_SEED + 100 + int(budget * 100),
                                      BOOTSTRAP_DRAWS)
        by_fold = {str(fold): statistics.fmean(
            row["joint_gain"] for row in records if row["fold"] == fold) for fold in range(5)}
        key = str(budget)
        pareto[key] = {"budget_rate": budget, "maximum_locality_failures": maximum,
                       "observed_locality_failures": failures, "metrics": metrics,
                       "joint_gain_micro": metrics["joint_success"]
                           - fixed_metrics["joint_success"],
                       "joint_gain_subject_macro": subject,
                       "joint_gain_template_macro": template, "by_fold_gain": by_fold,
                       "action_counts": {kind: sum(
                           (arm_id.startswith("fixed:retain_all") and kind == "retain_all")
                           or (arm_id.startswith("fixed:") and kind == "other_fixed")
                           or (arm_id.startswith("delete_one:") and kind == "delete_one")
                           or (arm_id.startswith("delete_key:") and kind == "delete_key")
                           for arm_id, _ in chosen)
                                         for kind in ("retain_all", "other_fixed",
                                                      "delete_one", "delete_key")}}
        selections[key] = [{"episode_id": row["episode_id"], "arm_id": arm_id}
                           for row, (arm_id, _) in zip(rows, chosen)]
    primary = pareto[str(PRIMARY_BUDGET)]
    checks = {
        "subject_macro_joint_gain_at_least_5_points":
            primary["joint_gain_subject_macro"]["mean"] >= .05,
        "subject_cluster_joint_gain_ci_lower_above_zero":
            primary["joint_gain_subject_macro"]["lower_95"] > 0,
        "template_macro_joint_gain_at_least_5_points":
            primary["joint_gain_template_macro"]["mean"] >= .05,
        "template_cluster_joint_gain_ci_lower_above_zero":
            primary["joint_gain_template_macro"]["lower_95"] > 0,
        "oracle_update_accuracy_at_least_90": primary["metrics"]["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90":
            primary["metrics"]["retained_key_accuracy"] >= .90,
        "oracle_locality_accuracy_at_least_97": primary["metrics"]["locality_accuracy"] >= .97,
        "no_outer_fold_has_negative_joint_gain":
            all(value >= 0 for value in primary["by_fold_gain"].values()),
    }
    return {"fixed_selected_by_outer_fold": fixed_by_fold, "outer_fold_fixed": fixed_metrics,
            "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
            "pareto": pareto, "selections": selections,
            "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
            "original_v18_status_unchanged": "formal_no_sufficient_headroom",
            "stage_b_released": False,
            "next_stage": "build_independent_v19_locality_confirmation"
                if all(checks.values()) else "stop_selective_forgetting_headroom_claim"}


def run(formal_root, inputs, output):
    formal_root = Path(formal_root); summary = json.loads((formal_root / "summary.json").read_text())
    audit = json.loads((formal_root / "audit.json").read_text())
    if not (summary.get("terminal_status") == "formal_no_sufficient_headroom"
            and audit.get("audit_passed") is True
            and audit.get("summary_sha256") == sha(formal_root / "summary.json")
            and audit.get("recomputed_headroom_gate", {}).get("passed") is False
            and (formal_root / "runner_exit_code.txt").read_text().strip() == "0"):
        raise ValueError("V18-LC requires the audited terminal V18 formal result")
    rows, hashes = [], {}
    for value in inputs:
        path = Path(value); rows.extend(load_jsonl(path)); hashes[path.name] = sha(path)
    table = aggregate(rows)
    result = {"protocol": PROTOCOL, **table, "rows": len(rows),
              "raw_rows_sha256": hashes, "formal_summary_sha256": sha(
                  formal_root / "summary.json"),
              "formal_audit_sha256": sha(formal_root / "audit.json"),
              "model_rescored": False, "posthoc": True, "external_sets_loaded": False,
              "terminal_status": "completed_v18_locality_constrained_diagnostic"}
    write_json(output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
