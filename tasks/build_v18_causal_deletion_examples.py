"""Build write-time causal deletion-utility examples from audited V18 arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.evaluate_v18_lowrank_fastmemory_bridge import PRIMARY_SCALE, arm_metrics
from tasks.evaluate_v18_physical_conflict_rules import read_json, verify_bridge_release
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_causal_deletion_examples_v1"
FEATURE_NAMES = (
    "newest_key_cosine", "newest_exact_key", "newest_value_agreement",
    "newest_value_conflict", "relative_age", "newer_exact_key_fraction",
    "newer_same_value_fraction", "same_key_group_fraction",
    "same_value_group_fraction", "max_newer_key_cosine", "mean_newer_key_cosine",
    "max_conflicting_newer_cosine", "conflicting_newer_fraction",
)


def similarity(table, left, right):
    return table["||".join(sorted((left, right)))]


def component_features(episode, similarity_table, index):
    sessions = episode["sessions"]
    if not 0 <= index < len(sessions) - 1:
        raise ValueError("deletion feature requires a historical component")
    component, newest = sessions[index], sessions[-1]
    newer = sessions[index + 1:]
    normalizer = max(1, len(sessions) - 1)
    newer_sims = [similarity(similarity_table, component["key_id"], row["key_id"])
                  for row in newer]
    conflicting = [value for value, row in zip(newer_sims, newer)
                   if row["value"] != component["value"]]
    exact_key_newer = sum(row["key_id"] == component["key_id"] for row in newer)
    same_value_newer = sum(row["value"] == component["value"] for row in newer)
    same_key_group = sum(row["key_id"] == component["key_id"] for row in sessions[:-1])
    same_value_group = sum(row["value"] == component["value"] for row in sessions[:-1])
    values = (
        similarity(similarity_table, component["key_id"], newest["key_id"]),
        float(component["key_id"] == newest["key_id"]),
        float(component["value"] == newest["value"]),
        float(component["key_id"] == newest["key_id"]
              and component["value"] != newest["value"]),
        (len(sessions) - 1 - index) / normalizer,
        exact_key_newer / normalizer,
        same_value_newer / normalizer,
        same_key_group / normalizer,
        same_value_group / normalizer,
        max(newer_sims), statistics.fmean(newer_sims),
        max(conflicting) if conflicting else -1.0,
        len(conflicting) / normalizer,
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("invalid causal deletion feature vector")
    return list(values)


def utility_target(retain, deleted):
    delta = deleted["joint_success"] - retain["joint_success"]
    if delta > 0:
        return 1.0
    if delta < 0:
        return 0.0
    return 0.5


def episode_examples(episode, bridge_row, physical_row):
    arm_index = {arm["arm_id"]: arm for arm in bridge_row["arms"]}
    retain = arm_metrics(arm_index["fixed:retain_all"], PRIMARY_SCALE)
    key_arm = {arm["deleted_key_id"]: arm for arm in bridge_row["arms"]
               if arm["kind"] == "delete_key"}
    output = []
    for index, component in enumerate(episode["sessions"][:-1]):
        single = arm_metrics(arm_index[f"delete_one:{component['session_id']}"], PRIMARY_SCALE)
        group = arm_metrics(key_arm[component["key_id"]], PRIMARY_SCALE)
        single_target, group_target = utility_target(retain, single), utility_target(retain, group)
        features = component_features(episode, physical_row["key_similarities"], index)
        later = episode["sessions"][index + 1:]
        exact_conflict = any(row["key_id"] == component["key_id"]
                             and row["value"] != component["value"] for row in later)
        useful_repeat = any(row["key_id"] == component["key_id"]
                            and row["value"] == component["value"] for row in later)
        target = max(single_target, group_target) if exact_conflict else single_target
        output.append({
            "example_id": f"{episode['episode_id']}::{component['session_id']}",
            "episode_id": episode["episode_id"], "subject": episode["subject"],
            "template_group": episode["template_group"], "fold": episode["fold"],
            "component_session_id": component["session_id"],
            "component_key_id": component["key_id"], "current_session_id":
                episode["current_session_id"],
            "features": features, "feature_names": list(FEATURE_NAMES),
            "single_target": single_target, "group_target": group_target,
            "target": target,
            "single_joint_nll_gain": retain["joint_answer_nll"]
                - single["joint_answer_nll"],
            "group_joint_nll_gain": retain["joint_answer_nll"]
                - group["joint_answer_nll"],
            "exact_conflict": int(exact_conflict),
            "useful_repetition": int(useful_repeat),
            "conservation_target": int(not exact_conflict),
        })
    return output


def verify_physical_release(data_root, formal_root, bridge_root, physical_root):
    dependencies = verify_bridge_release(data_root, formal_root, bridge_root)
    physical_root = Path(physical_root)
    summary, audit = read_json(physical_root / "summary.json"), read_json(
        physical_root / "audit.json")
    checks = (
        summary.get("protocol") == "v18_physical_conflict_rules_v1",
        summary.get("terminal_status") == "physical_conflict_rules_evaluated",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(physical_root / "summary.json"),
        (physical_root / "runner_exit_code.txt").read_text().strip() == "0",
        (physical_root / "terminal_status.txt").read_text().strip()
            == "completed_v18_physical_conflict_rules",
    )
    if not all(checks):
        raise ValueError("causal deletion examples require audited physical-rule results")
    shards = sorted(physical_root.glob("shard[01].jsonl"))
    if len(shards) != 2:
        raise ValueError("causal deletion examples require both physical-rule shards")
    return dependencies | {"physical_summary_sha256": sha(physical_root / "summary.json"),
                           "physical_audit_sha256": sha(physical_root / "audit.json"),
                           "physical_raw_sha256": {path.name: sha(path) for path in shards}}


def _index_shards(root):
    rows = []
    for path in sorted(Path(root).glob("shard[01].jsonl")):
        rows.extend(load_jsonl(path))
    index = {row["episode_id"]: row for row in rows}
    if len(rows) != len(index) or len(rows) != 320:
        raise ValueError("invalid two-shard episode coverage")
    return index


def build(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_physical_release(
        args.data_root, args.formal_root, args.bridge_root, args.physical_root)
    episodes = load_jsonl(Path(args.data_root) / "formal.jsonl")
    bridge, physical = _index_shards(args.bridge_root), _index_shards(args.physical_root)
    examples = [example for episode in episodes for example in episode_examples(
        episode, bridge[episode["episode_id"]], physical[episode["episode_id"]])]
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in examples))
    target_counts = {str(value): sum(row["target"] == value for row in examples)
                     for value in (0.0, 0.5, 1.0)}
    manifest = {"protocol": PROTOCOL, "rows": len(examples),
                "episodes": len(episodes), "feature_names": list(FEATURE_NAMES),
                "feature_dim": len(FEATURE_NAMES), "target_grid": [0.0, 0.5, 1.0],
                "target_counts": target_counts,
                "by_fold": {str(fold): sum(row["fold"] == fold for row in examples)
                            for fold in range(5)},
                "exact_conflicts": sum(row["exact_conflict"] for row in examples),
                "useful_repetitions": sum(row["useful_repetition"] for row in examples),
                "conservation_examples": sum(row["conservation_target"] for row in examples),
                "examples_sha256": sha(path), "dependencies": dependencies,
                "query_features_used": False, "current_update_always_retained": True,
                "external_sets_loaded": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
