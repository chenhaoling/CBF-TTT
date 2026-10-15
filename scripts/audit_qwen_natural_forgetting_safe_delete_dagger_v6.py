#!/usr/bin/env python3
"""Audit V6 DAgger collection or the retrained cross-fit safe-delete head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead, NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import (
    PLAN,
    PROTOCOL,
    FoldSafeDeleteController,
    load_v5,
    merged_corpus,
    oof_metrics,
    state_id,
)


def load_dagger_rows(root):
    rows = []
    for shard in range(2):
        rows.extend(json.loads(line) for line in
                    (root / f"dagger_rows_shard{shard}.jsonl").read_text().splitlines()
                    if line)
    rows.sort(key=lambda row: (row["session_id"], row["event_step"]))
    return rows


def audit_collect(args):
    root = Path(args.root)
    summary = json.loads((root / "dagger_summary.json").read_text())
    rows = load_dagger_rows(root)
    _, _, artifact = load_v5(args.safe_delete_root)
    controllers = {fold: FoldSafeDeleteController(artifact, fold) for fold in range(5)}
    transition_ok = probability_ok = label_ok = identity_ok = heldout_ok = True
    by_session = {}
    for row in rows:
        by_session.setdefault(row["session_id"], []).append(row)
    for session_id, session_rows in by_session.items():
        memory = []
        for expected_step, row in enumerate(session_rows):
            transition_ok &= row["event_step"] == expected_step and row["memory"] == memory
            identity_ok &= row["state_id"] == state_id(session_id, expected_step, memory)
            top = row["top"]
            expected_label = int(
                row["desired_source_action"] == "local" and top is not None
                and memory[int(top)][0] == row["incoming_chain_id"]
            )
            label_ok &= row["safe_delete"] == expected_label
            controller = controllers[row["outer_fold"]]
            full = [0.] * 13
            for name, value in zip(NO_SCHEDULE, row["features"]):
                from tasks.train_qwen_natural_forgetting_factor_gate_v2 import ENGINEERED
                full[ENGINEERED.index(name)] = value
            probability = controller.probability(full)
            probability_ok &= abs(probability - row["v5_probability"]) <= 1e-7
            expected_action = bool(memory) and probability >= controller.threshold
            probability_ok &= (row["v5_action"] == "scope_top1_a0") == expected_action
            heldout_ok &= session_id not in artifact["folds"][str(row["outer_fold"])][
                "train_sessions"]
            if expected_action:
                memory.pop(int(top))
            memory.append([row["incoming_chain_id"], row["incoming_version"]])
    resource_ok = True
    for shard in range(2):
        resource = json.loads((root / f"dagger_resource_shard{shard}.json").read_text())
        resource_ok &= resource["rows_sha256"] == sha(root / f"dagger_rows_shard{shard}.jsonl")
        resource_ok &= resource["frozen_inputs"] == {
            "safe_delete_summary": sha(Path(args.safe_delete_root) / "result" / "summary.json"),
            "safe_delete_audit": sha(Path(args.safe_delete_root) / "execution_audit.json"),
            "safe_delete_checkpoint": sha(Path(args.safe_delete_root) / "result" /
                                           "safe_delete_crossfit.pt"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
            "reader_checkpoint": sha(Path(args.v5_root) / "result" /
                                     "constrained_scope_gate.pt"),
            "context_transforms": sha(Path(args.context_root) / "transforms.pt"),
            "scope_artifact": sha(Path(args.scope_root) / "scope_gates.pt"),
        }
    checks = {
        "complete_65_session_774_event_rollout": len(by_session) == 65 and len(rows) == 774,
        "state_transitions_reconstruct": transition_ok,
        "physical_labels_reconstruct": label_ok,
        "state_identities_reconstruct": identity_ok,
        "v5_probabilities_and_actions_replay": probability_ok,
        "each_behavior_controller_excludes_its_session": heldout_ok,
        "frozen_input_and_shard_hashes_match": resource_ok and
            summary["plan_sha256"] == sha(PLAN),
        "collection_summary_checks_pass": summary["training_authorized"],
        "heldout_sources_remained_sealed": summary["calibration_loaded"] and
            not summary["development_loaded"] and not summary["test_loaded"] and
            not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "dagger_collect_audit", "checks": checks,
            "passed": all(checks.values()), "rows": len(rows), "sessions": len(by_session)}


def audit_train(args):
    import torch

    root = Path(args.root)
    result_root = root / "result"
    summary = json.loads((result_root / "summary.json").read_text())
    rebuilt = merged_corpus(args.safe_delete_root, root)
    stored = [json.loads(line) for line in
              (result_root / "safe_delete_dagger_corpus.jsonl").read_text().splitlines()
              if line]
    features = torch.load(result_root / "safe_delete_dagger_features.pt",
                          map_location="cpu", weights_only=False)
    artifact = torch.load(result_root / "safe_delete_dagger_crossfit.pt",
                          map_location="cpu", weights_only=False)
    oof = [json.loads(line) for line in
           (result_root / "oof_predictions.jsonl").read_text().splitlines() if line]
    corpus_ok = rebuilt == stored and [row["state_id"] for row in rebuilt] == [
        row["state_id"] for row in oof]
    tensors_ok = tuple(features["feature_names"]) == tuple(NO_SCHEDULE) and torch.equal(
        features["features"], torch.tensor([row["features"] for row in rebuilt],
                                           dtype=torch.float32)) and torch.equal(
        features["targets"], torch.tensor([row["safe_delete"] for row in rebuilt],
                                          dtype=torch.float32))
    max_error = 0.0
    action_mismatches = 0
    leaks = 0
    for outer in range(5):
        fold = artifact["folds"][str(outer)]
        validation = [index for index, row in enumerate(rebuilt)
                      if row["outer_fold"] == outer]
        validation_sessions = {rebuilt[index]["session_id"] for index in validation}
        leaks += len(set(fold["train_sessions"]) & validation_sessions)
        predictions = []
        for seed in SEEDS:
            checkpoint = fold["checkpoints"][str(seed)]
            model = BinaryHead.create(len(NO_SCHEDULE), fold["config"]["hidden"], seed)
            model.load_state_dict(checkpoint["state"])
            model.eval()
            with torch.no_grad():
                predictions.append(model(((features["features"][validation] -
                    checkpoint["mean"]) / checkpoint["std"])).squeeze(-1).sigmoid())
        ensemble = torch.stack(predictions).mean(0)
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            max_error = max(max_error, abs(probability - oof[index]["probability"]))
            action_mismatches += int((probability >= fold["threshold"]) !=
                                     (oof[index]["action"] == "scope_top1_a0"))
    recomputed = oof_metrics(rebuilt, oof)
    hashes_ok = all([
        summary["hashes"]["plan"] == sha(PLAN),
        summary["hashes"]["corpus"] == sha(result_root / "safe_delete_dagger_corpus.jsonl"),
        summary["hashes"]["features"] == sha(result_root / "safe_delete_dagger_features.pt"),
        summary["hashes"]["checkpoint"] == sha(result_root /
                                                "safe_delete_dagger_crossfit.pt"),
        summary["hashes"]["oof"] == sha(result_root / "oof_predictions.jsonl"),
        summary["hashes"]["inner_search"] == sha(result_root / "inner_search.json"),
        summary["hashes"]["v5_summary"] == sha(Path(args.safe_delete_root) /
                                               "result" / "summary.json"),
        summary["hashes"]["v5_audit"] == sha(Path(args.safe_delete_root) /
                                             "execution_audit.json"),
        summary["hashes"]["dagger_summary"] == sha(root / "dagger_summary.json"),
        summary["hashes"]["dagger_audit"] == sha(root / "dagger_execution_audit.json"),
    ])
    checks = {
        "merged_corpus_reconstruction_exact": corpus_ok,
        "feature_and_target_tensors_exact": tensors_ok,
        "no_train_validation_session_overlap": leaks == 0,
        "checkpoint_probabilities_match": max_error <= 1e-7,
        "checkpoint_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": recomputed == summary["metrics"],
        "artifact_hashes_match": hashes_ok,
        "same_registered_features": tuple(summary["feature_names"]) == tuple(NO_SCHEDULE),
        "heldout_sources_remained_sealed": summary["calibration_loaded"] and
            not summary["development_loaded"] and not summary["test_loaded"] and
            not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL, "stage": "offline_crossfit_audit", "checks": checks,
            "passed": all(checks.values()), "rows": len(rebuilt),
            "max_probability_error": max_error, "action_mismatches": action_mismatches,
            "session_leaks": leaks, "recomputed_metrics": recomputed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--safe-delete-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--v5-root")
    parser.add_argument("--context-root")
    parser.add_argument("--scope-root")
    args = parser.parse_args()
    result = audit_collect(args) if args.command == "collect" else audit_train(args)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
