"""Score-free failure localization for on-policy natural forgetting V3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    chain_map,
    checkpoint_label,
    record_for,
)
from tasks.evaluate_qwen_natural_forgetting_online_v3 import ARMS, PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


DIAGNOSED = (
    "factor_ensemble", "no_schedule_ensemble", "benefit_only_ensemble",
    "factor_seed_9401", "factor_seed_9402", "factor_seed_9403",
)


def decision_class(row):
    deleted = row["deleted"]
    desired_local = row["desired_source_action"] == "local"
    if not desired_local and not deleted:
        return "retain_true_negative"
    if not desired_local and deleted:
        return "retain_false_delete"
    if desired_local and not deleted:
        return "local_missed_delete"
    if deleted[0]["chain_id"] == row["incoming_chain_id"]:
        return "local_correct_delete"
    return "local_wrong_top_delete"


def load_artifacts(root, num_shards):
    root = Path(root); rows = []; traces = []
    for shard in range(num_shards):
        row_path = root / f"rows_shard{shard}.jsonl"
        trace_path = root / f"decisions_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        if resource["rows_sha256"] != sha(row_path) \
                or resource["decisions_sha256"] != sha(trace_path):
            raise ValueError("V3 trace hash mismatch")
        rows.extend(json.loads(line) for line in row_path.read_text().splitlines() if line)
        traces.extend(json.loads(line) for line in trace_path.read_text().splitlines() if line)
    if len(rows) != 5120 or len(traces) != 64:
        raise ValueError("incomplete V3 traces")
    return rows, traces


def reconstruct_snapshots(session, trace):
    chains = chain_map(session); states = {arm: [] for arm in ARMS}; snapshots = {}
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        for arm in ARMS:
            decision = trace["decisions"][arm][step]
            memory = states[arm]
            if decision["action"] == "scope_top1_a0":
                memory = [row for index, row in enumerate(memory) if index != decision["top"]]
            states[arm] = [*memory, {**record_for(chain, version), "strength": 1.0}]
        phase = checkpoint_label(step, session)
        if phase is not None:
            snapshots[phase] = {arm: [dict(row) for row in state]
                                for arm, state in states.items()}
    return snapshots


def decision_metrics(traces, arm):
    rows = [row for trace in traces for row in trace["decisions"][arm]]
    counts = Counter(decision_class(row) for row in rows)
    by_event = {}
    for kind in sorted({row["event_kind"] for row in rows}):
        part = [row for row in rows if row["event_kind"] == kind]
        by_event[kind] = dict(sorted(Counter(decision_class(row) for row in part).items()))
    probabilities = {}
    for desired in ("local", "retain"):
        part = [row for row in rows if row["desired_source_action"] == desired
                and row["probabilities"]]
        if part:
            benefits = [statistics.fmean(value[0] for value in row["probabilities"].values())
                        for row in part]
            risks = [statistics.fmean(value[1] for value in row["probabilities"].values())
                     for row in part]
            probabilities[desired] = {
                "events": len(part), "mean_benefit": statistics.fmean(benefits),
                "mean_risk": statistics.fmean(risks),
                "mean_action": statistics.fmean(
                    row["action"] == "scope_top1_a0" for row in part),
            }
    return {"classes": dict(sorted(counts.items())), "by_event_kind": by_event,
            "probability_by_desired_action": probabilities}


def error_metrics(rows, sessions, traces, arm):
    snapshots = {session["id"]: reconstruct_snapshots(session, traces[session["id"]])
                 for session in sessions}
    added = []; recovered = []
    for row in rows:
        retain = row["arms"]["retain_all"]["correct"]
        current = row["arms"][arm]["correct"]
        if retain and not current:
            added.append(row)
        elif current and not retain:
            recovered.append(row)
    def strata(part, added_error):
        values = Counter()
        for row in part:
            value = row["arms"][arm]
            memory = snapshots[row["session_id"]][row["phase"]][arm]
            present = any(record["chain_id"] == row["chain_id"]
                          and record["version"] == row["active_version"] for record in memory)
            values[f"surface:{row['surface']}"] += 1
            values[f"chain_kind:{row['chain_kind']}"] += 1
            values[f"phase:{row['phase']}"] += 1
            if added_error:
                category = ("active_record_missing" if not present else
                            "stale_read" if value["stale_selection"] else
                            "wrong_chain_read" if value["wrong_chain_selection"] else
                            "defer" if value["defer"] else "other_readout_change")
                values[f"cause:{category}"] += 1
        return dict(sorted(values.items()))
    return {"added": len(added), "recovered": len(recovered),
            "net": len(recovered) - len(added),
            "added_strata": strata(added, True),
            "recovered_strata": strata(recovered, False)}


def agreement(traces, left, right):
    pairs = [(a, b) for trace in traces.values()
             for a, b in zip(trace["decisions"][left], trace["decisions"][right])]
    table = Counter((a["action"], b["action"]) for a, b in pairs)
    return {f"{a}|{b}": value for (a, b), value in sorted(table.items())}


def analyze(root, data_root, num_shards=2):
    summary = json.loads((Path(root) / "summary.json").read_text())
    if summary["protocol"] != PROTOCOL:
        raise ValueError("unexpected V3 protocol")
    rows, traces_list = load_artifacts(root, num_shards)
    _, sessions = load(data_root, "dev")
    traces = {row["session_id"]: row for row in traces_list}
    output = {
        "protocol": PROTOCOL + "_diagnosis",
        "decision_metrics": {arm: decision_metrics(traces_list, arm) for arm in DIAGNOSED},
        "paired_error_metrics": {
            arm: error_metrics(rows, sessions, traces, arm) for arm in DIAGNOSED},
        "agreement": {
            "factor_vs_no_schedule": agreement(traces, "factor_ensemble",
                                                "no_schedule_ensemble"),
            "factor_vs_source": agreement(traces, "factor_ensemble", "source_action_rule"),
            "no_schedule_vs_source": agreement(traces, "no_schedule_ensemble",
                                                 "source_action_rule"),
        },
        "summary_sha256": sha(Path(root) / "summary.json"),
        "development_only_exploratory": True,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args(); result = analyze(args.root, args.data_root, args.num_shards)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
