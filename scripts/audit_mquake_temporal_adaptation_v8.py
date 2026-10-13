"""Independently audit MQuAKE-T adapted forgetting V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_crossfit_calibrated_forgetting_v5 import selected_threshold
from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.train_mquake_temporal_adaptation_v8 import (
    candidate_metrics,
    development_gate,
    tune_gate,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, data_root, source_root, raw_source):
    root, data_root, source_root, raw_source = map(
        Path, (root, data_root, source_root, raw_source))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    tune = read_jsonl(root / "tune_controller.jsonl")
    tune_candidates = read_jsonl(root / "tune_candidate_rows.jsonl")
    tune_metric = aggregate(tune); tune_candidate = candidate_metrics(tune_candidates)
    prerequisite = tune_gate(tune_metric, tune_candidate)
    tune_ok = (tune_metric == summary["tune"]
               and tune_candidate == summary["tune_candidate"]
               and prerequisite == summary["tune_prerequisite"])
    calibration_ok = all(
        selected_threshold(row["grid"]) == row["threshold"]
        for row in training["calibration"].values())

    dev_ok = True; random_ok = True
    if prerequisite["passed"]:
        dev = read_jsonl(root / "dev_controller.jsonl")
        candidates = read_jsonl(root / "dev_candidate_rows.jsonl")
        controls = {name: read_jsonl(root / f"dev_{name}.jsonl")
                    for name in ("retain_all", "random_matched", "oracle")}
        metric = aggregate(dev); candidate = candidate_metrics(candidates)
        versus_random = paired_bootstrap(dev, controls["random_matched"], 34105)
        gate = development_gate(metric, candidate, versus_random)
        dev_ok = (summary["development_scored"] and metric == summary["development"]
                  and candidate == summary["development_candidate"]
                  and {name: aggregate(rows) for name, rows in controls.items()} == summary["controls"]
                  and versus_random == summary["primary_vs_random"]
                  and gate == summary["development_gate"]
                  and len(dev) == data_manifest["counts"]["dev"]["episodes"])
        controller_deletions = sum(decision["deleted"] is not None
                                   for row in dev for decision in row["decisions"])
        random_deletions = sum(decision["deleted"] is not None
                               for row in controls["random_matched"] for decision in row["decisions"])
        random_ok = (controller_deletions == random_deletions
                     == training["matched_random_deletions"])
        expected_next = "preregister_locomo_confirmation" if gate["passed"] \
            else "stop_mquake_dev_no_retuning"
    else:
        absent = ("dev_controller.jsonl", "dev_candidate_rows.jsonl", "dev_retain_all.jsonl",
                  "dev_random_matched.jsonl", "dev_oracle.jsonl")
        dev_ok = (not summary["development_scored"]
                  and all(not (root / name).exists() for name in absent))
        expected_next = "stop_mquake_tune"

    split_subjects = {}
    for split in ("train", "tune", "dev"):
        rows = read_jsonl(data_root / f"{split}.jsonl")
        split_subjects[split] = {row["primary_subject_hash"] for row in rows}
    split_ok = (not split_subjects["train"] & split_subjects["tune"]
                and not split_subjects["train"] & split_subjects["dev"]
                and not split_subjects["tune"] & split_subjects["dev"])
    expected_tune = data_manifest["counts"]["tune"]["episodes"]
    identities_ok = (len(tune) == len({row["episode_id"] for row in tune}) == expected_tune
                     and sum(row["label"] for row in tune_candidates) == expected_tune)
    source_ok = (data_manifest["source_sha256"] == sha(raw_source)
                 == manifest["mquake_source_sha256"]
                 and manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["checkpoint_sha256"] == sha(root / "mquake_adapted_forgetting.pt"))
    locks_ok = (summary["protocol"] == manifest["protocol"]
                == "mquake_temporal_adaptation_v8"
                and manifest["candidate_epoch"] == manifest["gate_epoch"] == 50
                and not manifest["refit_after_calibration"]
                and manifest["development_labels_loaded_conditionally"]
                and manifest["payload_excluded_from_candidate_ranking"]
                and not manifest["candidate_age_feature"] and not manifest["unit_position_feature"]
                and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
                and not manifest["locomo_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == expected_next and len(manifest["git_commit"]) == 40)
    checks = {"tune_aggregates_and_prerequisite": tune_ok,
              "threshold_calibration_selection": calibration_ok,
              "conditional_development": dev_ok, "matched_random_deletions": random_ok,
              "tune_episode_identities": identities_ok, "entity_disjoint_splits": split_ok,
              "raw_output_hashes": training["raw_rows_sha256"]
              == {path.name: sha(path) for path in root.glob("*.jsonl")},
              "source_and_checkpoint_hashes": source_ok, "protocol_and_claim_locks": locks_ok}
    return {"protocol": "mquake_temporal_adaptation_v8_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_tune_prerequisite": prerequisite}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--raw-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.raw_source)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
