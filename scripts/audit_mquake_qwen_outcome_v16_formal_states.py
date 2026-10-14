"""Independently reconstruct and audit all frozen V16 formal memory states."""

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


def trace_map(path):
    rows = read_jsonl(path)
    result = {row["episode_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate OOF trace")
    return result


def trace_state(episode, trace, query_step):
    if (trace.get("episode_id") != episode["episode_id"]
            or trace.get("subject") != episode["primary_subject_hash"]
            or trace.get("length") != episode["length"]
            or len(trace.get("decisions", ())) != len(episode["sessions"])):
        raise ValueError("OOF trace identity mismatch")
    memory = []
    for step in range(query_step + 1):
        event, decision = episode["sessions"][step], trace["decisions"][step]
        if decision.get("step") != step or decision.get("event_kind") != event["event_kind"]:
            raise ValueError("OOF trace chronology mismatch")
        deleted = decision.get("deleted")
        if deleted is not None:
            positions = [index for index, row in enumerate(memory)
                         if row["session_id"] == deleted]
            if len(positions) != 1:
                raise ValueError("OOF trace deletion is not causal")
            memory.pop(positions[0])
        memory.append(event)
    return [row["session_id"] for row in memory]


def expected_states(episode, sample, v14, v15, random):
    step = sample["query_step"]
    visible = episode["sessions"][:step + 1]
    visible_ids = [row["session_id"] for row in visible]
    if (not visible or visible_ids[-1] != sample["primary_current_session_id"]
            or sample["primary_old_session_id"] not in visible_ids):
        raise ValueError("sample query identity mismatch")
    old = sample["primary_old_session_id"]
    return {
        "no_memory": [],
        "retain_all": visible_ids,
        "global_reset": visible_ids[-1:],
        "last_1": visible_ids[-1:],
        "last_2": visible_ids[-2:],
        "matched_random": trace_state(episode, random, step),
        "v14": trace_state(episode, v14, step),
        "v15": trace_state(episode, v15, step),
        "oracle": [identifier for identifier in visible_ids if identifier != old],
    }


def audit(data_root, sample_root, v14_root, v15_root, pilot_root, state_root, output):
    data_root, sample_root, v14_root, v15_root, pilot_root, state_root = map(
        Path, (data_root, sample_root, v14_root, v15_root, pilot_root, state_root))
    manifest = read_json(state_root / "manifest.json")
    rows = read_jsonl(state_root / "formal_states.jsonl")
    samples = read_jsonl(sample_root / "formal.jsonl")
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    v14 = trace_map(v14_root / "oof_controller.jsonl")
    v15 = trace_map(v15_root / "oof_controller.jsonl")
    random = trace_map(v15_root / "oof_random_matched.jsonl")
    sample_index = {row["episode_id"]: row for row in samples}

    identities_ok = len(rows) == len(samples) == len(sample_index) == 272
    states_ok = True
    safety_ok = True
    for row, sample in zip(rows, samples):
        identities_ok &= (
            row["episode_id"] == sample["episode_id"]
            and all(row[key] == sample[key] for key in
                    ("subject", "fold", "length", "query_step", "hard_relation_matched"))
            and tuple(row["policies"]) == POLICIES
        )
        episode = episode_index[row["episode_id"]]
        expected = expected_states(
            episode, sample, v14[row["episode_id"]],
            v15[row["episode_id"]], random[row["episode_id"]])
        states_ok &= row["policies"] == expected
        visible = {value["session_id"]
                   for value in episode["sessions"][:sample["query_step"] + 1]}
        current, old = sample["primary_current_session_id"], sample["primary_old_session_id"]
        safety_ok &= all(
            len(values) == len(set(values)) and set(values) <= visible
            and (name == "no_memory" or current in values)
            for name, values in row["policies"].items())
        safety_ok &= old not in row["policies"]["oracle"]

    v14_summary, v14_audit = read_json(v14_root / "summary.json"), read_json(v14_root.parent / "audit.json")
    v15_summary, v15_audit = read_json(v15_root / "summary.json"), read_json(v15_root.parent / "audit.json")
    pilot_summary, pilot_audit = read_json(pilot_root / "summary.json"), read_json(pilot_root / "audit.json")
    dependencies = {
        "sample_manifest_sha256": sha(sample_root / "manifest.json"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
        "v14_summary_sha256": sha(v14_root / "summary.json"),
        "v14_audit_sha256": sha(v14_root.parent / "audit.json"),
        "v15_summary_sha256": sha(v15_root / "summary.json"),
        "v15_audit_sha256": sha(v15_root.parent / "audit.json"),
        "pilot_summary_sha256": sha(pilot_root / "summary.json"),
        "pilot_audit_sha256": sha(pilot_root / "audit.json"),
    }
    raw_hashes_ok = True
    for root in (v14_root, v15_root):
        raw_hashes_ok &= read_json(root / "training.json")["raw_rows_sha256"] == {
            path.name: sha(path) for path in root.glob("*.jsonl")}
    checks = {
        "protocol_policy_order_and_state_hash": manifest.get("protocol") == PROTOCOL
            and manifest.get("policies") == list(POLICIES)
            and manifest.get("formal_states_sha256") == sha(state_root / "formal_states.jsonl"),
        "ordered_formal_sample_coverage_and_identities": identities_ok,
        "independent_nine_policy_reconstruction": states_ok,
        "causal_visibility_unique_sessions_and_current_retention": safety_ok,
        "complete_oof_trace_coverage": set(v14) == set(v15) == set(random) == set(episode_index),
        "v14_terminal_and_audit": v14_summary.get("terminal_status")
            == "completed_mquake_online_margin_v14" and v14_audit.get("audit_passed") is True,
        "v15_supported_terminal_and_audit": v15_summary.get("terminal_status")
            == "completed_mquake_jackknife_consensus_v15"
            and v15_summary.get("jackknife_consensus_supported") is True
            and v15_audit.get("audit_passed") is True,
        "pilot_passed_terminal_and_audit": pilot_summary.get("terminal_status") == "pilot_passed"
            and pilot_summary.get("pilot_gate", {}).get("passed") is True
            and pilot_audit.get("audit_passed") is True,
        "dependency_hashes": manifest.get("dependencies") == dependencies,
        "raw_oof_hashes": raw_hashes_ok,
        "no_model_or_external_access": manifest.get("model_scored") is False
            and manifest.get("external_sets_loaded") is False
            and not any(manifest.get(key) for key in
                        ("mquake_cf_used", "longmemeval_used", "locomo_used",
                         "mmlu_used", "ruler_used")),
        "registered_duplicate_baseline_disclosed":
            manifest.get("global_reset_equals_last_1_at_registered_query_time") is True
            and all(row["policies"]["global_reset"] == row["policies"]["last_1"]
                    for row in rows),
    }
    result = {
        "protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "rows": len(rows),
        "manifest_sha256": sha(state_root / "manifest.json"),
        "formal_states_sha256": sha(state_root / "formal_states.jsonl"),
        "model_scored": False, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V16 formal state audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--v14-root", required=True)
    parser.add_argument("--v15-root", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
