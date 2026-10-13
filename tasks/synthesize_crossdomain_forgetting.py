"""Deterministic synthesis of frozen CounterFact and held-out ZsRE evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KINDS = ("primary_initial", "hard_retain", "primary_update")


def _integer_count(events: int, accuracy: float) -> int:
    value = round(events * accuracy)
    if abs(value / events - accuracy) > 1e-12:
        raise ValueError("reported accuracy is not an integer event count")
    return value


def _domain(name: str, metric: dict, comparison: dict, status: str) -> dict:
    first_seed = next(iter(metric["per_seed"].values()))
    counts = {}
    for kind in KINDS:
        events = first_seed["by_event_kind"][kind]["events"]
        accuracy = metric["by_event_kind"][kind]["accuracy"]
        counts[kind] = {
            "events": events,
            "correct": _integer_count(events, accuracy),
            "accuracy": accuracy,
        }
    return {
        "name": name,
        "data_status": status,
        "sessions": first_seed["sessions"],
        "actions": first_seed["actions"],
        "query_accuracy": metric["mean_accuracy"],
        "action_accuracy": metric["mean_action_accuracy"],
        "by_event_kind": counts,
        "baseline_query_gain": comparison["query_accuracy_gain"],
        "baseline_query_bootstrap_95ci": comparison["query_bootstrap_95ci"],
        "baseline_action_gain": comparison["action_accuracy_gain"],
        "baseline_action_bootstrap_95ci": comparison["action_bootstrap_95ci"],
    }


def synthesize(counterfact: dict, zsre: dict) -> dict:
    if counterfact.get("protocol") != "causal_onpolicy_scope_v13":
        raise ValueError("unexpected CounterFact protocol")
    if counterfact.get("data_status") != "observed_development_only":
        raise ValueError("CounterFact status must remain development-only")
    if counterfact.get("terminal_status") != "completed_causal_onpolicy_scope_v13":
        raise ValueError("CounterFact run is incomplete")
    if zsre.get("protocol") != "zsre_counterfactual_forgetting_final_test_v1":
        raise ValueError("unexpected ZsRE protocol")
    if not zsre.get("test_loaded") or not zsre.get("test_scored"):
        raise ValueError("ZsRE held-out test was not completed")
    if zsre.get("terminal_status") != "completed_one_time_heldout_test":
        raise ValueError("ZsRE run is incomplete")

    cf = _domain(
        "CounterFact",
        counterfact["arms"]["onpolicy_mlp"],
        counterfact["onpolicy_vs_fixed_retain"],
        "grouped_oof_observed_development",
    )
    zr = _domain(
        "ZsRE",
        zsre["test"]["controller"],
        zsre["test"]["comparison"],
        "one_time_subject_disjoint_heldout_test",
    )
    pooled = {}
    for kind in KINDS:
        events = cf["by_event_kind"][kind]["events"] + zr["by_event_kind"][kind]["events"]
        correct = cf["by_event_kind"][kind]["correct"] + zr["by_event_kind"][kind]["correct"]
        pooled[kind] = {"events": events, "correct": correct, "accuracy": correct / events}
    total_events = sum(row["events"] for row in pooled.values())
    total_correct = sum(row["correct"] for row in pooled.values())
    differences = {
        "overall_action_abs": abs(cf["action_accuracy"] - zr["action_accuracy"]),
        "hard_retain_abs": abs(
            cf["by_event_kind"]["hard_retain"]["accuracy"]
            - zr["by_event_kind"]["hard_retain"]["accuracy"]
        ),
        "primary_update_abs": abs(
            cf["by_event_kind"]["primary_update"]["accuracy"]
            - zr["by_event_kind"]["primary_update"]["accuracy"]
        ),
    }
    checks = {
        "positive_query_gain_both_domains": all(
            row["baseline_query_bootstrap_95ci"][0] > 0 for row in (cf, zr)
        ),
        "positive_action_gain_both_domains": all(
            row["baseline_action_bootstrap_95ci"][0] > 0 for row in (cf, zr)
        ),
        "overall_action_within_one_point": differences["overall_action_abs"] <= .01,
        "hard_retain_within_one_point": differences["hard_retain_abs"] <= .01,
        "primary_update_within_one_point": differences["primary_update_abs"] <= .01,
    }
    return {
        "protocol": "crossdomain_forgetting_replication_synthesis_v1",
        "domains": {"counterfact": cf, "zsre": zr},
        "descriptive_pooled_counts": {
            "actions": total_events,
            "correct": total_correct,
            "accuracy": total_correct / total_events,
            "by_event_kind": pooled,
            "statistical_role": "descriptive_only; domain-specific clustered intervals remain primary",
        },
        "absolute_domain_differences": differences,
        "replication_checks": checks,
        "replicated": all(checks.values()),
        "claim_boundary": {
            "counterfact": "observed development-only grouped OOF",
            "zsre": "one-time held-out subject-disjoint test",
            "query_metrics_comparable": False,
            "native_dense_fast_weight_proven": False,
            "longmemeval_scored": False,
            "locomo_scored": False,
        },
        "next_stage": "natural_contextual_lowrank_sequential_validation",
        "terminal_status": "completed_crossdomain_replication_synthesis",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfact", required=True)
    parser.add_argument("--zsre", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counterfact = json.loads(Path(args.counterfact).read_text())
    zsre = json.loads(Path(args.zsre).read_text())
    result = synthesize(counterfact, zsre)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
