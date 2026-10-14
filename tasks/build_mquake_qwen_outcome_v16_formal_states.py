"""Freeze causal memory states for every V16 formal policy before Qwen scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_outcome_v16_formal_states"
POLICIES = (
    "no_memory", "retain_all", "global_reset", "last_1", "last_2",
    "matched_random", "v14", "v15", "oracle",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def trace_index(path):
    rows = read_jsonl(path)
    result = {row["episode_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate trace episode")
    return result


def replay_trace(episode, trace, query_step):
    if (trace["episode_id"] != episode["episode_id"]
            or trace["subject"] != episode["primary_subject_hash"]
            or trace["length"] != episode["length"]
            or len(trace["decisions"]) != len(episode["sessions"])):
        raise ValueError("trace identity mismatch")
    if not 0 <= query_step < len(episode["sessions"]):
        raise ValueError("invalid query step")
    memory = []
    for step, event in enumerate(episode["sessions"][:query_step + 1]):
        decision = trace["decisions"][step]
        if decision["step"] != step or decision["event_kind"] != event["event_kind"]:
            raise ValueError("trace step mismatch")
        deleted = decision["deleted"]
        if deleted is not None:
            matches = [index for index, value in enumerate(memory)
                       if value["session_id"] == deleted]
            if len(matches) != 1:
                raise ValueError("trace deletes a non-visible or duplicate session")
            memory.pop(matches[0])
        memory.append(event)
    return memory


def policy_states(episode, sample, traces):
    query_step = sample["query_step"]
    visible = episode["sessions"][:query_step + 1]
    if not visible or visible[-1]["session_id"] != sample["primary_current_session_id"]:
        raise ValueError("formal query point is not primary-current")
    old = sample["primary_old_session_id"]
    values = {
        "no_memory": [],
        "retain_all": visible,
        "global_reset": visible[-1:],
        "last_1": visible[-1:],
        "last_2": visible[-2:],
        "matched_random": replay_trace(episode, traces["matched_random"], query_step),
        "v14": replay_trace(episode, traces["v14"], query_step),
        "v15": replay_trace(episode, traces["v15"], query_step),
        "oracle": [row for row in visible if row["session_id"] != old],
    }
    if tuple(values) != POLICIES:
        raise RuntimeError("formal policy order changed")
    current = sample["primary_current_session_id"]
    for name, memory in values.items():
        identifiers = [row["session_id"] for row in memory]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate retained session")
        if name != "no_memory" and current not in identifiers:
            raise ValueError("formal policy lost the current session at query time")
        if any(row not in visible for row in memory):
            raise ValueError("formal state contains a future session")
    if old in {row["session_id"] for row in values["oracle"]}:
        raise RuntimeError("oracle did not delete the primary-old session")
    return {name: [row["session_id"] for row in memory]
            for name, memory in values.items()}


def verify_sources(sample_root, v14_root, v15_root, pilot_root):
    sample_root, v14_root, v15_root, pilot_root = map(
        Path, (sample_root, v14_root, v15_root, pilot_root))
    sample = read_json(sample_root / "manifest.json")
    v14_summary, v14_audit = read_json(v14_root / "summary.json"), read_json(v14_root.parent / "audit.json")
    v15_summary, v15_audit = read_json(v15_root / "summary.json"), read_json(v15_root.parent / "audit.json")
    pilot_summary, pilot_audit = read_json(pilot_root / "summary.json"), read_json(pilot_root / "audit.json")
    if (sample.get("protocol") != "mquake_qwen_outcome_v16"
            or sample.get("formal_sha256") != sha(sample_root / "formal.jsonl")
            or sample.get("model_scored") is not False
            or sample.get("external_sets_loaded") is not False):
        raise ValueError("invalid frozen V16 sample")
    if (v14_summary.get("protocol") != "mquake_online_margin_v14"
            or v14_summary.get("terminal_status") != "completed_mquake_online_margin_v14"
            or v14_audit.get("audit_passed") is not True
            or v14_audit.get("summary_sha256") != sha(v14_root / "summary.json")):
        raise ValueError("invalid V14 dependency")
    if (v15_summary.get("protocol") != "mquake_jackknife_consensus_v15"
            or v15_summary.get("terminal_status") != "completed_mquake_jackknife_consensus_v15"
            or v15_summary.get("jackknife_consensus_supported") is not True
            or v15_audit.get("audit_passed") is not True
            or v15_audit.get("summary_sha256") != sha(v15_root / "summary.json")):
        raise ValueError("formal state freezing requires supported audited V15")
    if (pilot_summary.get("protocol") != "mquake_qwen_outcome_v16_pilot"
            or pilot_summary.get("terminal_status") != "pilot_passed"
            or pilot_summary.get("pilot_gate", {}).get("passed") is not True
            or pilot_summary.get("formal_scored") is not False
            or pilot_summary.get("external_sets_loaded") is not False
            or pilot_audit.get("audit_passed") is not True
            or pilot_audit.get("summary_sha256") != sha(pilot_root / "summary.json")):
        raise ValueError("formal state freezing requires a passed audited pilot")
    for root in (v14_root, v15_root):
        training = read_json(root / "training.json")
        if training["raw_rows_sha256"] != {
                path.name: sha(path) for path in root.glob("*.jsonl")}:
            raise ValueError("OOF trace hash mismatch")
    return {
        "sample_manifest_sha256": sha(sample_root / "manifest.json"),
        "formal_sample_sha256": sample["formal_sha256"],
        "v14_summary_sha256": sha(v14_root / "summary.json"),
        "v14_audit_sha256": sha(v14_root.parent / "audit.json"),
        "v15_summary_sha256": sha(v15_root / "summary.json"),
        "v15_audit_sha256": sha(v15_root.parent / "audit.json"),
        "pilot_summary_sha256": sha(pilot_root / "summary.json"),
        "pilot_audit_sha256": sha(pilot_root / "audit.json"),
    }


def build(data_root, sample_root, v14_root, v15_root, pilot_root, output):
    data_root, sample_root, v14_root, v15_root, output = map(
        Path, (data_root, sample_root, v14_root, v15_root, output))
    dependencies = verify_sources(sample_root, v14_root, v15_root, pilot_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    samples = read_jsonl(sample_root / "formal.jsonl")
    trace_sets = {
        "v14": trace_index(v14_root / "oof_controller.jsonl"),
        "v15": trace_index(v15_root / "oof_controller.jsonl"),
        "matched_random": trace_index(v15_root / "oof_random_matched.jsonl"),
    }
    if any(set(values) != set(episode_index) for values in trace_sets.values()):
        raise ValueError("OOF traces do not cover the complete source corpus")
    rows = []
    for sample in samples:
        episode = episode_index[sample["episode_id"]]
        traces = {name: values[sample["episode_id"]] for name, values in trace_sets.items()}
        rows.append({
            "episode_id": sample["episode_id"], "subject": sample["subject"],
            "fold": sample["fold"], "length": sample["length"],
            "query_step": sample["query_step"],
            "hard_relation_matched": sample["hard_relation_matched"],
            "policies": policy_states(episode, sample, traces),
        })
    states_path = output / "formal_states.jsonl"
    with states_path.open("x") as sink:
        sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "protocol": PROTOCOL, "rows": len(rows), "policies": list(POLICIES),
        "query_time": "immediately after primary_current",
        "matched_random_source": "V15 exact deletion-count matched OOF trace",
        "global_reset_equals_last_1_at_registered_query_time": True,
        "dependencies": dependencies,
        "formal_states_sha256": sha(states_path),
        "model_scored": False, "external_sets_loaded": False,
        "mquake_cf_used": False, "longmemeval_used": False,
        "locomo_used": False, "mmlu_used": False, "ruler_used": False,
        "terminal_status": "frozen_after_passed_v16_pilot_before_formal_scoring",
    }
    if len(rows) != 272 or len({row["episode_id"] for row in rows}) != 272:
        raise RuntimeError("formal state coverage mismatch")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--v14-root", required=True)
    parser.add_argument("--v15-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
