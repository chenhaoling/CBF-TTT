"""Independently audit exhaustive V17 causal headroom memory states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_states"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def reconstruct(episode, sample):
    step = sample["query_step"]
    visible = episode["sessions"][:step + 1]
    identifiers = [row["session_id"] for row in visible]
    current, old = sample["primary_current_session_id"], sample["primary_old_session_id"]
    if (not visible or identifiers[-1] != current or old not in identifiers
            or len(identifiers) != len(set(identifiers))):
        raise ValueError("independent headroom identity mismatch")
    fixed = {"no_memory": [], "retain_all": identifiers,
             "global_reset": identifiers[-1:], "last_1": identifiers[-1:],
             "last_2": identifiers[-2:]}
    deletions = [{"deleted_session_id": value,
                  "memory_session_ids": identifiers[:index] + identifiers[index + 1:]}
                 for index, value in enumerate(identifiers[:-1])]
    return fixed, deletions


def audit(data_root, sample_root, v15_root, state_root, output):
    data_root, sample_root, v15_root, state_root = map(
        Path, (data_root, sample_root, v15_root, state_root))
    manifest = read_json(state_root / "manifest.json")
    sample_manifest = read_json(sample_root / "manifest.json")
    summary = read_json(v15_root / "summary.json")
    v15_audit = read_json(v15_root.parent / "audit.json")
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    identities_ok = states_ok = causal_ok = True
    counts, hashes = {}, {}
    for split, expected in (("pilot", 40), ("formal", 272)):
        samples = read_jsonl(sample_root / f"{split}.jsonl")
        rows = read_jsonl(state_root / f"{split}_states.jsonl")
        hashes[f"{split}_states.jsonl"] = sha(state_root / f"{split}_states.jsonl")
        if len(samples) != len(rows) or len(rows) != expected:
            identities_ok = False
        candidate_count = 0
        for row, sample in zip(rows, samples):
            identities_ok &= all(row[key] == sample[key] for key in
                ("episode_id", "subject", "fold", "length", "query_step",
                 "hard_relation_matched"))
            identities_ok &= tuple(row["fixed"]) == FIXED
            episode = episode_index[row["episode_id"]]
            fixed, deletions = reconstruct(episode, sample)
            states_ok &= row["fixed"] == fixed and row["delete_one"] == deletions
            candidate_count += len(deletions)
            visible = {event["session_id"] for event in
                       episode["sessions"][:sample["query_step"] + 1]}
            current = sample["primary_current_session_id"]
            causal_ok &= all(
                candidate["deleted_session_id"] in visible
                and candidate["deleted_session_id"] != current
                and current in candidate["memory_session_ids"]
                and len(candidate["memory_session_ids"])
                    == len(set(candidate["memory_session_ids"]))
                and set(candidate["memory_session_ids"]) < visible
                for candidate in row["delete_one"])
        counts[split] = {"rows": len(rows), "single_deletion_states": candidate_count,
                         "fixed_states": len(rows) * len(FIXED)}
    dependencies = {
        "sample_manifest_sha256": sha(sample_root / "manifest.json"),
        "pilot_sample_sha256": sha(sample_root / "pilot.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
        "v15_summary_sha256": sha(v15_root / "summary.json"),
        "v15_audit_sha256": sha(v15_root.parent / "audit.json"),
    }
    checks = {
        "protocol_terminal_and_registered_fixed_order": manifest.get("protocol") == PROTOCOL
            and manifest.get("terminal_status")
                == "frozen_after_audited_unsupported_v15_before_headroom_scoring"
            and manifest.get("fixed_policies") == list(FIXED),
        "ordered_sample_coverage_and_identity": identities_ok,
        "independent_fixed_and_all_single_deletion_reconstruction": states_ok,
        "causal_historical_only_deletion_and_current_retention": causal_ok,
        "counts_and_state_hashes": manifest.get("counts") == counts
            and manifest.get("state_sha256") == hashes,
        "frozen_sample_hashes_and_pre_model_lock":
            sample_manifest.get("pilot_sha256") == sha(sample_root / "pilot.jsonl")
            and sample_manifest.get("formal_sha256") == sha(sample_root / "formal.jsonl")
            and sample_manifest.get("model_scored") is False,
        "terminal_audited_unsupported_v15": summary.get("protocol")
            == "mquake_jackknife_consensus_v15"
            and summary.get("terminal_status") == "completed_mquake_jackknife_consensus_v15"
            and summary.get("jackknife_consensus_supported") is False
            and v15_audit.get("audit_passed") is True
            and v15_audit.get("summary_sha256") == sha(v15_root / "summary.json"),
        "dependency_hashes": manifest.get("dependencies") == dependencies,
        "registered_intervention_and_oracle_tie_break": manifest.get("intervention")
            == "delete exactly one visible historical session before query or use a fixed arm"
            and manifest.get("current_session_deletion_allowed") is False
            and manifest.get("causal_oracle_tie_break")
                == "primary accuracy, lower primary NLL, retain-all, chronological deletion order",
        "no_model_or_external_access": manifest.get("model_scored") is False
            and manifest.get("external_sets_loaded") is False,
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "manifest_sha256": sha(state_root / "manifest.json"),
              "state_sha256": hashes, "counts": counts,
              "model_scored": False, "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V17 causal headroom state audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--v15-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
