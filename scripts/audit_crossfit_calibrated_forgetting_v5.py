"""Independently recompute cross-fit calibrated forgetting V5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_crossfit_calibrated_forgetting_v5 import (
    FOLDS,
    GATE_SEEDS,
    aggregate,
    load_v1,
    oof_gate,
    paired_bootstrap,
    release_gate,
)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def selected_threshold(grid):
    rows = []
    for threshold_key, metric in grid.items():
        threshold = float(threshold_key)
        feasible = (metric["current_retention"] >= .98
                    and metric["non_target_retention"] >= .97
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97)
        rows.append((int(feasible), metric["pair_state_accuracy"],
                     metric["by_event_kind"]["primary_update"]["accuracy"],
                     metric["non_target_retention"], -abs(threshold - .9), threshold))
    feasible = [row for row in rows if row[0]]
    return max(feasible)[-1] if feasible else max(row[-1] for row in rows)


def audit(root, data_root, source_root, v1_root, v3_root):
    root, data_root, source_root, v1_root, v3_root = map(
        Path, (root, data_root, source_root, v1_root, v3_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    oof_records = read_jsonl(root / "oof_validation.jsonl")
    expected_calibration = (data_manifest["counts"]["train"]["episodes"]
                            + data_manifest["counts"]["tune"]["episodes"])
    oof_ids = {row["episode_id"] for row in oof_records}
    oof_identity = len(oof_records) == len(oof_ids) == expected_calibration
    oof_metric = aggregate(oof_records); prerequisite = oof_gate(oof_metric)
    oof_ok = oof_metric == summary["oof"] and prerequisite == summary["oof_gate"]
    calibration_ok = all(
        selected_threshold(training["folds"][str(fold)]["models"][str(seed)]["grid"])
        == training["folds"][str(fold)]["models"][str(seed)]["threshold"]
        for fold in range(FOLDS) for seed in GATE_SEEDS)
    fold_validation_ok = all(
        aggregate([row for row in oof_records
                   if training["folds"][str(fold)]["validation"]["episodes"]
                   and row["subject"] in {
                       subject for subject, value in __import__("torch").load(
                           root / "crossfit_gates.pt", map_location="cpu", weights_only=False
                       )["folds"].items() if value == fold
                   }]) == training["folds"][str(fold)]["validation"]
        for fold in range(FOLDS))
    # Load once more only for assignment integrity; artifacts themselves are hash-covered.
    artifact = __import__("torch").load(root / "crossfit_gates.pt", map_location="cpu",
                                        weights_only=False)
    fold_counts = [sum(value == fold for value in artifact["folds"].values()) for fold in range(FOLDS)]
    folds_ok = len(artifact["folds"]) == 267 and max(fold_counts) - min(fold_counts) <= 1

    dev_ok = True; random_ok = True
    if prerequisite["passed"]:
        dev = read_jsonl(root / "dev_controller.jsonl")
        controls = {name: read_jsonl(root / f"dev_{name}.jsonl")
                    for name in ("retain_all", "random_matched", "oracle")}
        candidates = read_jsonl(root / "dev_candidate_rows.jsonl")
        dev_metric = aggregate(dev)
        positives = [row for row in candidates if row["label"]]
        shortlist = statistics.fmean(row["shortlist_correct"] for row in positives)
        candidate = statistics.fmean(row["candidate_correct"] for row in positives)
        v1_records, v1_hashes = load_v1(v1_root)
        versus_v1 = paired_bootstrap(dev, v1_records, 32601)
        versus_random = paired_bootstrap(dev, controls["random_matched"], 32602)
        gate = release_gate(dev_metric, shortlist, candidate, versus_v1, versus_random)
        dev_ok = (summary["development_scored"] and dev_metric == summary["development"]
                  and {name: aggregate(rows) for name, rows in controls.items()} == summary["controls"]
                  and shortlist == summary["shortlist_recall"] and candidate == summary["candidate_top1"]
                  and versus_v1 == summary["versus_v1"] and versus_random == summary["versus_random"]
                  and gate == summary["development_gate"] and training["v1_hashes"] == v1_hashes)
        controller_deletions = sum(decision["deleted"] is not None
                                   for row in dev for decision in row["decisions"])
        random_deletions = sum(decision["deleted"] is not None
                               for row in controls["random_matched"] for decision in row["decisions"])
        random_ok = controller_deletions == random_deletions == training["random_matched_deletions"]
        expected_next = "preregister_locomo" if gate["passed"] else "stop_crossfit_dev"
    else:
        absent = ("dev_controller.jsonl", "dev_candidate_rows.jsonl", "dev_retain_all.jsonl",
                  "dev_random_matched.jsonl", "dev_oracle.jsonl")
        dev_ok = not summary["development_scored"] and all(not (root / name).exists() for name in absent)
        expected_next = "stop_crossfit_oof"
    v3_hashes = {"summary": sha(v3_root / "summary.json"),
                 "training": sha(v3_root / "training.json"),
                 "checkpoint": sha(v3_root / "scope_candidate_rankers.pt")}
    source_ok = (manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["checkpoint_sha256"] == sha(root / "crossfit_gates.pt")
                 and training["v3_hashes"] == v3_hashes)
    locks_ok = (manifest["nested_train_calibrate_validate"]
                and not manifest["refit_after_calibration"]
                and not manifest["ranker_logits_in_gate_feature"]
                and manifest["payload_excluded_from_candidate_ranking"]
                and not manifest["candidate_age_feature"] and not manifest["unit_position_feature"]
                and not manifest["longmemeval_used"] and not manifest["locomo_used"]
                and not summary["longmemeval_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == expected_next and len(manifest["git_commit"]) == 40)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    checks = {"raw_hashes": raw_hashes == training["raw_rows_sha256"],
              "oof_identities": oof_identity, "oof_aggregates_and_gate": oof_ok,
              "fold_calibration_selection": calibration_ok,
              "fold_validation_aggregates": fold_validation_ok,
              "fold_assignment": folds_ok, "conditional_development": dev_ok,
              "matched_random": random_ok, "source_v3_checkpoint_hashes": source_ok,
              "protocol_and_claim_locks": locks_ok}
    return {"protocol": "crossfit_calibrated_forgetting_v5_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "oof_episodes": len(oof_records), "recomputed_oof_gate": prerequisite}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v1-root", required=True)
    parser.add_argument("--v3-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.data_root, args.source_root, args.v1_root, args.v3_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
