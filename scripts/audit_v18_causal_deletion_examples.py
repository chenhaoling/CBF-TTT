"""Independent audit of V18 write-time causal deletion examples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.build_v18_causal_deletion_examples import (
    FEATURE_NAMES, PROTOCOL, verify_physical_release,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.evaluate_v18_lowrank_fastmemory_bridge import PRIMARY_SCALE, arm_metrics
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def _index(root):
    rows = []
    for path in sorted(Path(root).glob("shard[01].jsonl")):
        rows.extend(load_jsonl(path))
    result = {row["episode_id"]: row for row in rows}
    if len(rows) != len(result) or len(rows) != 320:
        raise ValueError("invalid audited shard coverage")
    return result


def _sim(table, left, right):
    return table["||".join(sorted((left, right)))]


def _features(episode, table, index):
    sessions = episode["sessions"]; component, newest = sessions[index], sessions[-1]
    newer = sessions[index + 1:]; denominator = max(1, len(sessions) - 1)
    sims = [_sim(table, component["key_id"], row["key_id"]) for row in newer]
    conflicts = [value for value, row in zip(sims, newer)
                 if row["value"] != component["value"]]
    return [
        _sim(table, component["key_id"], newest["key_id"]),
        float(component["key_id"] == newest["key_id"]),
        float(component["value"] == newest["value"]),
        float(component["key_id"] == newest["key_id"]
              and component["value"] != newest["value"]),
        (len(sessions) - 1 - index) / denominator,
        sum(row["key_id"] == component["key_id"] for row in newer) / denominator,
        sum(row["value"] == component["value"] for row in newer) / denominator,
        sum(row["key_id"] == component["key_id"] for row in sessions[:-1]) / denominator,
        sum(row["value"] == component["value"] for row in sessions[:-1]) / denominator,
        max(sims), statistics.fmean(sims), max(conflicts) if conflicts else -1.0,
        len(conflicts) / denominator,
    ]


def _target(retain, deleted):
    delta = deleted["joint_success"] - retain["joint_success"]
    return 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5


def audit(data_root, formal_root, bridge_root, physical_root, examples_root, output):
    dependencies = verify_physical_release(
        data_root, formal_root, bridge_root, physical_root)
    data_root, examples_root = Path(data_root), Path(examples_root)
    episodes = load_jsonl(data_root / "formal.jsonl")
    bridge, physical = _index(bridge_root), _index(physical_root)
    rows = load_jsonl(examples_root / "examples.jsonl")
    manifest = json.loads((examples_root / "manifest.json").read_text())
    expected_ids, contents_ok = [], True
    cursor = 0
    for episode in episodes:
        brow, prow = bridge[episode["episode_id"]], physical[episode["episode_id"]]
        arms = {arm["arm_id"]: arm for arm in brow["arms"]}
        retain = arm_metrics(arms["fixed:retain_all"], PRIMARY_SCALE)
        key_arms = {arm["deleted_key_id"]: arm for arm in brow["arms"]
                    if arm["kind"] == "delete_key"}
        for index, component in enumerate(episode["sessions"][:-1]):
            row = rows[cursor] if cursor < len(rows) else {}; cursor += 1
            identifier = f"{episode['episode_id']}::{component['session_id']}"
            expected_ids.append(identifier)
            single = arm_metrics(arms[f"delete_one:{component['session_id']}"], PRIMARY_SCALE)
            group = arm_metrics(key_arms[component["key_id"]], PRIMARY_SCALE)
            single_target, group_target = _target(retain, single), _target(retain, group)
            later = episode["sessions"][index + 1:]
            conflict = any(value["key_id"] == component["key_id"]
                           and value["value"] != component["value"] for value in later)
            repeat = any(value["key_id"] == component["key_id"]
                         and value["value"] == component["value"] for value in later)
            target = max(single_target, group_target) if conflict else single_target
            contents_ok &= (
                row.get("example_id") == identifier
                and row.get("episode_id") == episode["episode_id"]
                and row.get("subject") == episode["subject"]
                and row.get("template_group") == episode["template_group"]
                and row.get("fold") == episode["fold"]
                and row.get("component_session_id") == component["session_id"]
                and row.get("component_key_id") == component["key_id"]
                and row.get("current_session_id") == episode["current_session_id"]
                and row.get("feature_names") == list(FEATURE_NAMES)
                and row.get("features") == _features(episode, prow["key_similarities"], index)
                and row.get("single_target") == single_target
                and row.get("group_target") == group_target
                and row.get("target") == target
                and row.get("exact_conflict") == int(conflict)
                and row.get("useful_repetition") == int(repeat)
                and row.get("conservation_target") == int(not conflict)
                and math.isclose(row.get("single_joint_nll_gain", math.nan),
                                 retain["joint_answer_nll"] - single["joint_answer_nll"])
                and math.isclose(row.get("group_joint_nll_gain", math.nan),
                                 retain["joint_answer_nll"] - group["joint_answer_nll"]))
    target_counts = {str(value): sum(row["target"] == value for row in rows)
                     for value in (0.0, 0.5, 1.0)}
    checks = {
        "positive_physical_release_and_dependencies": manifest.get("dependencies") == dependencies,
        "complete_ordered_historical_component_coverage": (
            len(rows) == len(expected_ids) == 9280
            and [row.get("example_id") for row in rows] == expected_ids
            and len(set(expected_ids)) == len(expected_ids)),
        "independent_features_targets_and_group_credit": contents_ok,
        "three_point_target_and_fold_counts": (
            manifest.get("target_grid") == [0.0, 0.5, 1.0]
            and manifest.get("target_counts") == target_counts
            and manifest.get("by_fold") == {str(fold): sum(row["fold"] == fold for row in rows)
                                             for fold in range(5)}),
        "write_time_no_query_feature_contract": (
            manifest.get("feature_names") == list(FEATURE_NAMES)
            and manifest.get("feature_dim") == len(FEATURE_NAMES)
            and not any("query" in name or "length" in name for name in FEATURE_NAMES)
            and manifest.get("query_features_used") is False
            and manifest.get("current_update_always_retained") is True),
        "counts_hash_and_access_locks": (
            manifest.get("protocol") == PROTOCOL and manifest.get("rows") == 9280
            and manifest.get("episodes") == 320
            and manifest.get("exact_conflicts") == sum(row["exact_conflict"] for row in rows)
            and manifest.get("useful_repetitions") == sum(
                row["useful_repetition"] for row in rows)
            and manifest.get("conservation_examples") == sum(
                row["conservation_target"] for row in rows)
            and manifest.get("examples_sha256") == sha(examples_root / "examples.jsonl")
            and manifest.get("external_sets_loaded") is False),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "manifest_sha256": sha(examples_root / "manifest.json"),
              "examples_sha256": sha(examples_root / "examples.jsonl"),
              "rows": len(rows), "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 causal deletion example audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
