"""Independently audit MQuAKE robust cross-fit calibration V11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.train_mquake_anchor_forgetting_v10 import candidate_metrics
from tasks.train_mquake_robust_calibration_v11 import (
    FOLDS,
    GATE_SEEDS,
    oof_gate,
    subject_folds,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def selected_threshold(grid):
    rows = []
    for key, metric in grid.items():
        threshold = float(key)
        feasible = (metric["current_retention"] >= .99
                    and metric["non_target_retention"] >= .99
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .99)
        safety = min(metric["current_retention"], metric["non_target_retention"],
                     metric["by_event_kind"]["hard_retain"]["accuracy"])
        rows.append((int(feasible), metric["pair_state_accuracy"], metric["stale_deletion"],
                     safety, -abs(threshold - .9), threshold))
    feasible_rows = [row for row in rows if row[0]]
    return (max(feasible_rows) if feasible_rows else max(
        rows, key=lambda row: (row[3], row[1], row[2], row[4])))[-1]


def audit(root, data_root, source_root, v8_root):
    import torch

    root, data_root, source_root, v8_root = map(
        Path, (root, data_root, source_root, v8_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    records = read_jsonl(root / "oof_controller.jsonl")
    candidates = read_jsonl(root / "oof_candidate_rows.jsonl")
    random_records = read_jsonl(root / "oof_random_matched.jsonl")
    metric = aggregate(records); candidate = candidate_metrics(candidates)
    comparison = paired_bootstrap(records, random_records, 36192)
    artifact = torch.load(root / "mquake_robust_crossfit_gates.pt",
                          map_location="cpu", weights_only=False)
    fold_metrics = {str(fold): aggregate(
        [row for row in records if artifact["folds"][row["subject"]] == fold])
        for fold in range(FOLDS)}
    gate = oof_gate(metric, candidate, fold_metrics, comparison)
    recompute_ok = (metric == summary["oof"] and candidate == summary["oof_candidate"]
                    and comparison == summary["oof_vs_random"]
                    and fold_metrics == summary["fold_metrics"] and gate == summary["oof_gate"])
    calibration_ok = all(
        selected_threshold(training["folds"][str(fold)]["models"][str(seed)]["grid"])
        == training["folds"][str(fold)]["models"][str(seed)]["threshold"]
        for fold in range(FOLDS) for seed in GATE_SEEDS)
    train, data_manifest = load_split(data_root, "train"); tune, _ = load_split(data_root, "tune")
    episodes = train + tune; expected_folds, expected_loads = subject_folds(episodes)
    identities_ok = (len(records) == len({row["episode_id"] for row in records}) == len(episodes)
                     and len(candidates) == sum(row["length"] - 1 for row in episodes)
                     and len({row["subject"] for row in records}) == len(expected_folds)
                     and artifact["folds"] == expected_folds
                     and training["fold_loads"] == expected_loads)
    controller_deletions = sum(decision["deleted"] is not None
                               for row in records for decision in row["decisions"])
    random_deletions = sum(decision["deleted"] is not None
                           for row in random_records for decision in row["decisions"])
    v8_checkpoint = v8_root / "mquake_adapted_forgetting.pt"
    source_ok = (manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["checkpoint_sha256"] == sha(root / "mquake_robust_crossfit_gates.pt")
                 and training["v8_hashes"] == {
                     "summary": sha(v8_root / "summary.json"),
                     "training": sha(v8_root / "training.json"),
                     "checkpoint": sha(v8_checkpoint)})
    locks_ok = (summary["protocol"] == manifest["protocol"]
                == "mquake_robust_calibration_v11"
                and manifest["folds"] == 5 and manifest["gate_epoch"] == 50
                and manifest["calibration_safety_floor"] == .99
                and manifest["candidate_policy"] == "anchor_4_then_v8_ranker"
                and manifest["shortlist"] == 4
                and manifest["nested_train_calibrate_validate"]
                and not manifest["refit_after_calibration"]
                and not manifest["v10_development_used"] and not summary["v10_development_loaded"]
                and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
                and not manifest["locomo_used"] and not summary["locomo_scored"]
                and len(manifest["git_commit"]) == 40)
    checks = {"oof_metrics_comparison_and_gate": recompute_ok,
              "robust_threshold_selection": calibration_ok,
              "subject_fold_assignment_and_identities": identities_ok,
              "matched_random_deletions": controller_deletions == random_deletions
              == training["matched_random_deletions"],
              "raw_output_hashes": training["raw_rows_sha256"]
              == {path.name: sha(path) for path in root.glob("*.jsonl")},
              "source_v8_checkpoint_hashes": source_ok,
              "protocol_and_access_locks": locks_ok}
    return {"protocol": "mquake_robust_calibration_v11_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_oof_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v8-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root, args.data_root, args.source_root, args.v8_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
