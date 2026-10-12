"""Independent audit for subject-excluded augmented forgetting gate V3."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, load_calibration
from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS, paired_comparison, summarize_seed_metrics
from tasks.train_zsre_forgetting_controller_v2 import THRESHOLDS
from tasks.train_zsre_forgetting_controller_v3 import (
    AUXILIARY_WEIGHT, PROTOCOL, auxiliary_examples, load_contextual_calibration, subject_hash,
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root, data_root, context_root, source_root, scope_root):
    root, data_root, context_root, source, scope = map(
        Path, (root, data_root, context_root, source_root, scope_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    hard_sessions, _ = load_calibration(data_root)
    context_sessions, _ = load_contextual_calibration(context_root)
    auxiliary = auxiliary_examples(context_sessions)
    context_subjects = {row["subject"] for row in auxiliary}
    session_subjects = {session["id"]: {chain["subject"] for chain in session["chains"]}
                        for session in hard_sessions}
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "context_hashes": manifest["context_manifest_sha256"] == sha(context_root / "manifest.json")
                          and training["context_manifest_sha256"] == sha(context_root / "manifest.json")
                          and training["context_calibration_sha256"] == sha(
                              context_root / "calibration.jsonl"),
        "nested_loso": manifest["split"] == "calibration_nested_loso"
                       and len(manifest["outer_folds"]) == len(set(manifest["outer_folds"])) == 7,
        "seeds": tuple(manifest["controller_seeds"]) == CONTROLLER_SEEDS
                 and tuple(manifest["readout_seeds"]) == GATE_SEEDS,
        "threshold_grid": tuple(manifest["thresholds"]) == THRESHOLDS,
        "auxiliary_protocol": manifest["outer_subjects_excluded_from_auxiliary"] is True
                              and manifest["auxiliary_weight"] == AUXILIARY_WEIGHT
                              and training["auxiliary_examples"] == 801
                              and training["auxiliary_subjects"] == 267,
        "no_identity_feature": manifest["chain_identity_in_features"] is False,
        "calibration_only": not summary["dev_loaded"] and not summary["test_loaded"]
                            and not summary["test_scored"] and not manifest["dev_loaded"]
                            and not manifest["test_loaded"] and not manifest["test_scored"],
        "no_heldout_raw": not list(root.glob("dev_*.jsonl")) and not list(root.glob("test_*.jsonl")),
        "event_count": training["events_with_candidates"] == 75 and training["feature_dim"] == 13,
        "checkpoint_hash": training["checkpoint_sha256"] == sha(root / "forgetting_augmented_gate.pt"),
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_training_sha256"] == sha(source / "training.json")
                         and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
        "scope_hashes": training["scope_training_sha256"] == sha(scope / "training.json")
                        and training["scope_checkpoint_sha256"] == sha(scope / "scope_gates.pt"),
    }
    controller_records, controller_metrics = {}, {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"calibration_loso_augmented_gate_seed{seed}.jsonl"; rows = read_rows(path)
        controller_records[str(seed)] = rows; controller_metrics[str(seed)] = aggregate(rows)
        checks[f"controller_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
        checks[f"outer_once_{seed}"] = sorted(row["session_id"] for row in rows) == sorted(
            manifest["outer_folds"])
    baseline_records, baseline_metrics = {}, {}
    for seed in GATE_SEEDS:
        path = root / f"calibration_cosine_local_seed{seed}.jsonl"; rows = read_rows(path)
        baseline_records[str(seed)] = rows; baseline_metrics[str(seed)] = aggregate(rows)
        checks[f"baseline_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
    checks["controller_recomputed"] = summarize_seed_metrics(controller_metrics) == summary["controller"]
    checks["baseline_recomputed"] = summarize_seed_metrics(baseline_metrics) == summary["cosine_local"]
    checks["comparison_recomputed"] = paired_comparison(
        controller_records, baseline_records, 8901) == summary["comparison"]
    for outer_id, fold in training["folds"].items():
        outer_train, inner_train = set(fold["outer_train_sessions"]), set(fold["inner_train_sessions"])
        tune = fold["inner_tune_session"]
        outer_subjects = session_subjects[outer_id]; inner_excluded = outer_subjects | session_subjects[tune]
        expected_outer_aux = sum(row["subject"] not in outer_subjects for row in auxiliary)
        expected_inner_aux = sum(row["subject"] not in inner_excluded for row in auxiliary)
        checks[f"fold_partition_{outer_id}"] = (outer_train == inner_train | {tune}
                                                and outer_id not in outer_train and len(inner_train) == 5)
        checks[f"heldout_subject_hash_{outer_id}"] = fold["heldout_subject_hash"] == subject_hash(outer_subjects)
        checks[f"auxiliary_exclusion_{outer_id}"] = (
            fold["outer_auxiliary_examples"] == expected_outer_aux
            and fold["inner_auxiliary_examples"] == expected_inner_aux
            and outer_subjects <= context_subjects)
    return {"protocol": PROTOCOL, "audit_passed": all(checks.values()), "checks": checks,
            "dev_loaded": False, "test_loaded": False, "test_scored": False,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "manifest_sha256": sha(root / "training_manifest.json"),
            "checkpoint_sha256": sha(root / "forgetting_augmented_gate.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--context-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.data_root, args.context_root, args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
