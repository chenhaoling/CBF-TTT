"""Freeze fixed and exhaustive single-deletion states after an unsupported V15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_causal_headroom_v17_states"
FIXED_POLICIES = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def freeze_states(episode, sample):
    step = sample["query_step"]
    if not 0 <= step < len(episode["sessions"]):
        raise ValueError("invalid headroom query step")
    visible = episode["sessions"][:step + 1]
    identifiers = [row["session_id"] for row in visible]
    current = sample["primary_current_session_id"]
    old = sample["primary_old_session_id"]
    if (not visible or identifiers[-1] != current or old not in identifiers
            or len(identifiers) != len(set(identifiers))):
        raise ValueError("headroom sample/session identity mismatch")
    fixed = {
        "no_memory": [], "retain_all": identifiers,
        "global_reset": identifiers[-1:], "last_1": identifiers[-1:],
        "last_2": identifiers[-2:],
    }
    deletions = [{"deleted_session_id": identifier,
                  "memory_session_ids": [value for value in identifiers if value != identifier]}
                 for identifier in identifiers[:-1]]
    if tuple(fixed) != FIXED_POLICIES or len(deletions) != len(visible) - 1:
        raise RuntimeError("headroom state order changed")
    if not any(row["deleted_session_id"] == old for row in deletions):
        raise RuntimeError("primary-old deletion candidate missing")
    if any(current not in row["memory_session_ids"] for row in deletions):
        raise RuntimeError("single deletion removed current memory")
    return fixed, deletions


def verify_dependencies(sample_root, v15_root):
    sample_root, v15_root = Path(sample_root), Path(v15_root)
    sample = read_json(sample_root / "manifest.json")
    summary = read_json(v15_root / "summary.json")
    audit_path = v15_root.parent / "audit.json"
    audit = read_json(audit_path)
    if (sample.get("protocol") != "mquake_qwen_outcome_v16"
            or sample.get("pilot_sha256") != sha(sample_root / "pilot.jsonl")
            or sample.get("formal_sha256") != sha(sample_root / "formal.jsonl")
            or sample.get("model_scored") is not False
            or sample.get("external_sets_loaded") is not False):
        raise ValueError("invalid frozen V16 sample dependency")
    if (summary.get("protocol") != "mquake_jackknife_consensus_v15"
            or summary.get("terminal_status") != "completed_mquake_jackknife_consensus_v15"
            or summary.get("jackknife_consensus_supported") is not False
            or audit.get("audit_passed") is not True
            or audit.get("summary_sha256") != sha(v15_root / "summary.json")):
        raise ValueError("V17 states require terminal audited but unsupported V15")
    return {
        "sample_manifest_sha256": sha(sample_root / "manifest.json"),
        "pilot_sample_sha256": sha(sample_root / "pilot.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl"),
        "v15_summary_sha256": sha(v15_root / "summary.json"),
        "v15_audit_sha256": sha(audit_path),
    }


def build(data_root, sample_root, v15_root, output):
    data_root, sample_root, output = map(Path, (data_root, sample_root, output))
    dependencies = verify_dependencies(sample_root, v15_root)
    output.mkdir(parents=True, exist_ok=False)
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    if len(episode_index) != len(episodes):
        raise ValueError("duplicate source episode")
    counts, hashes = {}, {}
    for split, expected in (("pilot", 40), ("formal", 272)):
        samples = read_jsonl(sample_root / f"{split}.jsonl")
        if len(samples) != expected or len({row["episode_id"] for row in samples}) != expected:
            raise ValueError(f"{split} sample coverage mismatch")
        rows, candidates = [], 0
        for sample in samples:
            episode = episode_index[sample["episode_id"]]
            fixed, delete_one = freeze_states(episode, sample)
            candidates += len(delete_one)
            rows.append({
                "episode_id": sample["episode_id"], "subject": sample["subject"],
                "fold": sample["fold"], "length": sample["length"],
                "query_step": sample["query_step"],
                "hard_relation_matched": sample["hard_relation_matched"],
                "fixed": fixed, "delete_one": delete_one,
            })
        path = output / f"{split}_states.jsonl"
        with path.open("x") as sink:
            sink.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        counts[split] = {"rows": len(rows), "single_deletion_states": candidates,
                         "fixed_states": len(rows) * len(FIXED_POLICIES)}
        hashes[path.name] = sha(path)
    manifest = {
        "protocol": PROTOCOL, "fixed_policies": list(FIXED_POLICIES),
        "intervention": "delete exactly one visible historical session before query or use a fixed arm",
        "current_session_deletion_allowed": False,
        "causal_oracle_tie_break": "primary accuracy, lower primary NLL, retain-all, chronological deletion order",
        "counts": counts, "state_sha256": hashes, "dependencies": dependencies,
        "model_scored": False, "external_sets_loaded": False,
        "terminal_status": "frozen_after_audited_unsupported_v15_before_headroom_scoring",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--v15-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
