"""Independently audit frozen MQuAKE-T temporal transfer V7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.evaluate_mquake_temporal_transfer_v7 import transfer_gate


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, data_root, source_root, v3_root, v5_root, raw_source):
    root, data_root, source_root, v3_root, v5_root, raw_source = map(
        Path, (root, data_root, source_root, v3_root, v5_root, raw_source))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    records = read_jsonl(root / "dev_controller.jsonl")
    candidates = read_jsonl(root / "dev_candidate_rows.jsonl")
    controls = {name: read_jsonl(root / f"dev_{name}.jsonl")
                for name in ("retain_all", "random_matched", "oracle")}
    metric = aggregate(records)
    positives = [row for row in candidates if row["label"]]
    shortlist = statistics.fmean(row["shortlist_correct"] for row in positives)
    candidate = statistics.fmean(row["candidate_correct"] for row in positives)
    versus_random = paired_bootstrap(records, controls["random_matched"], 33104)
    gate = transfer_gate(metric, shortlist, candidate, versus_random)
    aggregates_ok = (metric == summary["controller"]
                     and {name: aggregate(rows) for name, rows in controls.items()} == summary["controls"]
                     and shortlist == summary["shortlist_recall"]
                     and candidate == summary["candidate_top1"]
                     and versus_random == summary["primary_vs_random"]
                     and gate == summary["transfer_gate"])
    expected = data_manifest["counts"]["dev"]["episodes"]
    identities_ok = (len(records) == len({row["episode_id"] for row in records}) == expected
                     and len(positives) == expected)
    controller_deletions = sum(decision["deleted"] is not None
                               for row in records for decision in row["decisions"])
    random_deletions = sum(decision["deleted"] is not None
                           for row in controls["random_matched"] for decision in row["decisions"])
    hashes_ok = (training["raw_rows_sha256"]
                 == {path.name: sha(path) for path in root.glob("*.jsonl")})
    source_ok = (data_manifest["source_sha256"] == sha(raw_source)
                 == manifest["mquake_source_sha256"]
                 and manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["v3_hashes"] == {
                     "summary": sha(v3_root / "summary.json"),
                     "training": sha(v3_root / "training.json"),
                     "checkpoint": sha(v3_root / "scope_candidate_rankers.pt")}
                 and training["v5_hashes"] == {
                     "summary": sha(v5_root / "summary.json"),
                     "training": sha(v5_root / "training.json"),
                     "checkpoint": sha(v5_root / "crossfit_gates.pt")})
    locks_ok = (summary["protocol"] == manifest["protocol"] == "mquake_temporal_transfer_v7"
                and summary["evaluation_mode"] == "frozen_zero_shot_transfer"
                and not summary["mquake_train_or_tune_loaded"]
                and not manifest["mquake_train_or_tune_loaded"]
                and manifest["frozen_v3_resolver"] and manifest["frozen_v5_controller"]
                and not manifest["threshold_tuning_on_mquake"]
                and manifest["subject_disjoint_dev"]
                and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
                and not manifest["locomo_used"] and not summary["locomo_scored"]
                and len(manifest["git_commit"]) == 40)
    split_subjects = {}
    for split in ("train", "tune", "dev"):
        values = read_jsonl(data_root / f"{split}.jsonl")
        split_subjects[split] = {row["primary_subject_hash"] for row in values}
    split_ok = (data_manifest["subject_disjoint"]
                and not split_subjects["train"] & split_subjects["tune"]
                and not split_subjects["train"] & split_subjects["dev"]
                and not split_subjects["tune"] & split_subjects["dev"])
    checks = {"aggregates_and_preregistered_gate": aggregates_ok,
              "dev_episode_identities": identities_ok,
              "matched_random_deletions": (controller_deletions == random_deletions
                                            == training["controller_deletions"]),
              "raw_output_hashes": hashes_ok, "source_and_checkpoint_hashes": source_ok,
              "entity_disjoint_splits": split_ok, "claim_and_data_access_locks": locks_ok}
    return {"protocol": "mquake_temporal_transfer_v7_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v3-root", required=True)
    parser.add_argument("--v5-root", required=True); parser.add_argument("--raw-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.v3_root, args.v5_root,
                   args.raw_source)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
