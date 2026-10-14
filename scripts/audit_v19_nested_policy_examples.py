"""Audit the gated V19 nested-OOF fallback policy examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.audit_controlled_interference_v18_formal_outcome import metrics
from tasks.build_v19_nested_policy_examples import (
    FEATURE_NAMES, PROTOCOL, verify_release,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def grams(value):
    text = f"  {value.casefold()}  "
    return {text[index:index + 3] for index in range(len(text) - 2)}


def similarity(left, right):
    a, b = grams(left), grams(right)
    return len(a & b) / len(a | b)


def prefix(left, right):
    count = 0
    for a, b in zip(left.casefold(), right.casefold()):
        if a != b:
            break
        count += 1
    return count / max(1, len(left), len(right))


def features(episode, index):
    sessions = episode["sessions"]
    component, newest = sessions[index], sessions[-1]
    later = sessions[index + 1:]
    scale = max(1, len(sessions) - 1)
    sims = [similarity(component["key_id"], row["key_id"]) for row in later]
    conflicts = [value for value, row in zip(sims, later)
                 if row["value"] != component["value"]]
    exact_key = component["key_id"] == newest["key_id"]
    exact_value = component["value"] == newest["value"]
    return [
        float(exact_key), float(exact_value), float(exact_key and not exact_value),
        (len(sessions) - 1 - index) / scale,
        sum(row["key_id"] == component["key_id"] for row in later) / scale,
        sum(row["value"] == component["value"] for row in later) / scale,
        sum(row["value"] != component["value"] for row in later) / scale,
        sum(row["key_id"] == component["key_id"] for row in sessions[:-1]) / scale,
        sum(row["value"] == component["value"] for row in sessions[:-1]) / scale,
        similarity(component["key_id"], newest["key_id"]),
        prefix(component["key_id"], newest["key_id"]), max(sims),
        statistics.fmean(sims), max(conflicts) if conflicts else 0.0,
        float(any(row["key_id"] == component["key_id"]
                  and row["value"] == component["value"] for row in later)),
        float(not exact_key and prefix(component["key_id"], newest["key_id"]) >= .75),
        len(sessions) / 64.0,
    ]


def target(retain, deleted):
    delta = deleted["joint_success"] - retain["joint_success"]
    return 1.0 if delta > 0 else (0.0 if delta < 0 else .5)


def intervention_rank(retain, deleted):
    return (target(retain, deleted),
            retain["joint_answer_nll"] - deleted["joint_answer_nll"],
            deleted["locality_accuracy"])


def recompute_episode(episode, outcome):
    arms = {row["arm_id"]: row for row in outcome["arms"]}
    if len(arms) != len(outcome["arms"]):
        raise ValueError("duplicate V19 policy-label arm")
    retain = metrics(arms["fixed:retain_all"])
    key_arms = {row["deleted_key_id"]: row for row in outcome["arms"]
                if row["kind"] == "delete_key"}
    rows = []
    for index, component in enumerate(episode["sessions"][:-1]):
        single_arm = arms[f"delete_one:{component['session_id']}"]
        group_arm = key_arms[component["key_id"]]
        single, group = metrics(single_arm), metrics(group_arm)
        later = episode["sessions"][index + 1:]
        exact = any(row["key_id"] == component["key_id"]
                    and row["value"] != component["value"] for row in later)
        useful = any(row["key_id"] == component["key_id"]
                     and row["value"] == component["value"] for row in later)
        chosen_arm, chosen = single_arm, single
        if exact and intervention_rank(retain, group) > intervention_rank(retain, single):
            chosen_arm, chosen = group_arm, group
        utility = target(retain, chosen)
        rows.append({
            "example_id": f"{episode['episode_id']}::{component['session_id']}",
            "episode_id": episode["episode_id"], "subject": episode["subject"],
            "template_group": episode["template_group"], "fold": episode["fold"],
            "component_session_id": component["session_id"],
            "current_session_id": episode["current_session_id"],
            "features": features(episode, index), "feature_names": list(FEATURE_NAMES),
            "target": utility,
            "locality_risk_target": int(
                chosen["locality_accuracy"] < retain["locality_accuracy"]),
            "conservation_target": int(utility <= .5),
            "exact_conflict": int(exact), "useful_repetition": int(useful),
            "selected_label_arm_id": chosen_arm["arm_id"],
            "single_target": target(retain, single), "group_target": target(retain, group),
            "selected_joint_nll_gain": (retain["joint_answer_nll"]
                                        - chosen["joint_answer_nll"]),
        })
    return rows


def audit(data_root, outcome_root, rule_root, inputs, examples_root, output):
    examples_root = Path(examples_root)
    dependencies = verify_release(data_root, outcome_root, rule_root)
    manifest = json.loads((examples_root / "manifest.json").read_text())
    observed = load_jsonl(examples_root / "examples.jsonl")
    episodes = load_jsonl(Path(data_root) / "confirmation.jsonl")
    outcomes, raw_hashes = [], {}
    for value in inputs:
        path = Path(value); outcomes.extend(load_jsonl(path)); raw_hashes[path.name] = sha(path)
    outcome_index = {row["episode_id"]: row for row in outcomes}
    expected = [item for episode in episodes for item in recompute_episode(
        episode, outcome_index[episode["episode_id"]])]
    finite = all(len(row.get("features", [])) == len(FEATURE_NAMES)
                 and all(math.isfinite(value) for value in row["features"])
                 and math.isfinite(row.get("selected_joint_nll_gain", math.nan))
                 for row in observed)
    counts = {
        "target_counts": {str(value): sum(row["target"] == value for row in observed)
                          for value in (0.0, .5, 1.0)},
        "locality_risk_rows": sum(row["locality_risk_target"] for row in observed),
        "conservation_rows": sum(row["conservation_target"] for row in observed),
        "exact_conflict_rows": sum(row["exact_conflict"] for row in observed),
        "useful_repetition_rows": sum(row["useful_repetition"] for row in observed),
        "by_fold": {str(fold): sum(row["fold"] == fold for row in observed)
                    for fold in range(5)},
    }
    checks = {
        "audited_v19_fallback_release": manifest.get("dependencies") == dependencies,
        "complete_unique_source_and_example_coverage": (
            len(episodes) == len(outcomes) == 320
            and len(outcome_index) == 320
            and len(observed) == len(expected)
            and len({row["example_id"] for row in observed}) == len(observed)),
        "independent_features_labels_and_label_arm_recomputation": observed == expected,
        "finite_write_time_feature_vectors": finite,
        "manifest_counts_hashes_and_access_locks": (
            manifest.get("protocol") == PROTOCOL
            and manifest.get("rows") == len(observed)
            and manifest.get("episodes") == 320
            and manifest.get("feature_names") == list(FEATURE_NAMES)
            and manifest.get("feature_dim") == len(FEATURE_NAMES)
            and manifest.get("target_grid") == [0.0, .5, 1.0]
            and manifest.get("target_counts") == counts["target_counts"]
            and all(manifest.get(key) == value for key, value in counts.items()
                    if key != "target_counts")
            and manifest.get("examples_sha256") == sha(examples_root / "examples.jsonl")
            and manifest.get("raw_rows_sha256") == raw_hashes
            and manifest.get("query_features_used") is False
            and manifest.get("answer_features_used") is False
            and manifest.get("current_update_always_retained") is True
            and manifest.get("external_sets_loaded") is False
            and manifest.get("terminal_status") == "frozen_v19_nested_policy_examples"),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "manifest_sha256": sha(examples_root / "manifest.json"),
        "examples_sha256": sha(examples_root / "examples.jsonl"),
        "rows": len(observed), "counts": counts, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V19 nested policy example audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--outcome-root", required=True)
    parser.add_argument("--rule-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
