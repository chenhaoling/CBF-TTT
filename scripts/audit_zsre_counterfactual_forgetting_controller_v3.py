"""Independent audit for efficient action-OOF ZsRE forgetting calibration."""

import argparse
import json
from pathlib import Path

from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.train_zsre_forgetting_controller import CONTROLLER_SEEDS, paired_comparison, summarize_seed_metrics
from tasks.train_zsre_forgetting_controller_v2 import THRESHOLDS
from tasks.train_zsre_counterfactual_forgetting_controller_v3 import (
    FIXED_EPOCHS, INNER_FOLDS, PROTOCOL, TRAINING_SEED,
)


EXPECTED_CALIBRATION = {"subjects": 258, "primary_chains": 258, "auxiliary_chains": 258,
                        "primary_versions": 516, "hard_retain_events": 258,
                        "local_transitions": 258, "sessions": 65, "events": 774}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root, data_root, source_root, scope_root):
    root, data_root, source, scope = map(Path, (root, data_root, source_root, scope_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "data_protocol": manifest["data_protocol"] == data_manifest["protocol"] == DATA_PROTOCOL,
        "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
        "data_counts": data_manifest["counts"]["calibration"] == EXPECTED_CALIBRATION,
        "nested_oof": manifest["split"] == "calibration_nested_action_oof"
                      and manifest["inner_folds"] == training["inner_folds"] == INNER_FOLDS
                      and manifest["fixed_epochs"] == training["fixed_epochs"] == FIXED_EPOCHS
                      and manifest["training_seed"] == TRAINING_SEED
                      and len(manifest["outer_folds"]) == len(set(manifest["outer_folds"])) == 65,
        "seeds": tuple(manifest["controller_seeds"]) == CONTROLLER_SEEDS
                 and tuple(manifest["readout_seeds"]) == GATE_SEEDS,
        "threshold_grid": tuple(manifest["thresholds"]) == THRESHOLDS,
        "action_objective": manifest["selection_objective"] == "oracle_state_action_accuracy",
        "no_identity_feature": manifest["chain_identity_in_features"] is False,
        "calibration_only": not summary["dev_loaded"] and not summary["test_loaded"]
                            and not summary["test_scored"] and not manifest["dev_loaded"]
                            and not manifest["test_loaded"] and not manifest["test_scored"],
        "no_heldout_raw": not list(root.glob("dev_*.jsonl")) and not list(root.glob("test_*.jsonl")),
        "event_count": training["events_with_candidates"] == 709 and training["feature_dim"] == 13,
        "checkpoint_hash": training["checkpoint_sha256"] == sha(
            root / "counterfactual_forgetting_action_oof_gate.pt"),
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_training_sha256"] == sha(source / "training.json")
                         and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
        "scope_hashes": training["scope_training_sha256"] == sha(scope / "training.json")
                        and training["scope_checkpoint_sha256"] == sha(scope / "scope_gates.pt"),
    }
    controller_records, controller_metrics = {}, {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"calibration_loso_action_oof_gate_seed{seed}.jsonl"; rows = read_rows(path)
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
        controller_records, baseline_records, 9201) == summary["comparison"]
    checks["paired_units"] = summary["comparison"]["paired_subject_units"] == 258
    for outer_id, fold in training["folds"].items():
        outer_train = set(fold["outer_train_sessions"])
        validation_sets = [set(row["validation_sessions"]) for row in fold["inner_folds"]]
        train_sets = [set(row["train_sessions"]) for row in fold["inner_folds"]]
        union = set().union(*validation_sets)
        checks[f"outer_partition_{outer_id}"] = outer_id not in outer_train and len(outer_train) == 64
        checks[f"inner_partition_{outer_id}"] = (
            len(validation_sets) == INNER_FOLDS and union == outer_train
            and sum(map(len, validation_sets)) == len(union)
            and all(train == outer_train - validation
                    for train, validation in zip(train_sets, validation_sets)))
    return {"protocol": PROTOCOL, "audit_passed": all(checks.values()), "checks": checks,
            "dev_loaded": False, "test_loaded": False, "test_scored": False,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "manifest_sha256": sha(root / "training_manifest.json"),
            "checkpoint_sha256": sha(root / "counterfactual_forgetting_action_oof_gate.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
