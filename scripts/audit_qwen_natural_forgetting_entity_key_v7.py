#!/usr/bin/env python3
"""Audit V7 prompt-derived entity features and cross-fit predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead, NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import oof_metrics
from tasks.train_qwen_natural_forgetting_entity_key_v7 import (
    ENTITY_FEATURES,
    FEATURE_NAMES,
    PLAN,
    PROTOCOL,
    augment_corpus,
)


def audit(args):
    import torch

    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((root / "summary.json").read_text())
    rebuilt = augment_corpus(args.source_root, args.data_root)
    stored = [json.loads(line) for line in
              (root / "entity_key_corpus.jsonl").read_text().splitlines() if line]
    features = torch.load(root / "entity_key_features.pt", map_location="cpu",
                          weights_only=False)
    artifact = torch.load(root / "entity_key_crossfit.pt", map_location="cpu",
                          weights_only=False)
    oof = [json.loads(line) for line in
           (root / "oof_predictions.jsonl").read_text().splitlines() if line]
    corpus_ok = rebuilt == stored and [row["state_id"] for row in rebuilt] == [
        row["state_id"] for row in oof]
    tensors_ok = tuple(features["feature_names"]) == tuple(FEATURE_NAMES) and torch.equal(
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
            model = BinaryHead.create(len(FEATURE_NAMES), fold["config"]["hidden"], seed)
            model.load_state_dict(checkpoint["state"])
            model.eval()
            with torch.no_grad():
                predictions.append(model(((features["features"][validation] -
                    checkpoint["mean"]) / checkpoint["std"])).squeeze(-1).sigmoid())
        ensemble = torch.stack(predictions).mean(0)
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            max_error = max(max_error, abs(probability - oof[index]["probability"]))
            delete = probability >= fold["threshold"] and (
                not fold["use_entity_anchor_barrier"] or
                rebuilt[index]["entity_anchor_barrier"])
            action_mismatches += int(delete !=
                                     (oof[index]["action"] == "scope_top1_a0"))
    recomputed = oof_metrics(rebuilt, oof)
    source = Path(args.source_root)
    hashes_ok = all([
        summary["hashes"]["plan"] == sha(PLAN),
        summary["hashes"]["corpus"] == sha(root / "entity_key_corpus.jsonl"),
        summary["hashes"]["features"] == sha(root / "entity_key_features.pt"),
        summary["hashes"]["checkpoint"] == sha(root / "entity_key_crossfit.pt"),
        summary["hashes"]["oof"] == sha(root / "oof_predictions.jsonl"),
        summary["hashes"]["inner_search"] == sha(root / "inner_search.json"),
        summary["hashes"]["source_summary"] == sha(source / "result" / "summary.json"),
        summary["hashes"]["source_audit"] == sha(source / "execution_audit.json"),
        summary["hashes"]["source_corpus"] == sha(source / "result" /
                                                  "safe_delete_dagger_corpus.jsonl"),
        summary["hashes"]["data_manifest"] == sha(Path(args.data_root) / "manifest.json"),
    ])
    feature_isolation = tuple(summary["feature_names"][:10]) == tuple(NO_SCHEDULE) and \
        tuple(summary["feature_names"][10:]) == ENTITY_FEATURES and \
        not {"subject", "chain_id", "source_action", "event_kind"} & \
        set(summary["feature_names"])
    checks = {
        "prompt_feature_corpus_reconstruction_exact": corpus_ok,
        "feature_and_target_tensors_exact": tensors_ok,
        "no_train_validation_session_overlap": leaks == 0,
        "checkpoint_probabilities_match": max_error <= 1e-7,
        "checkpoint_barrier_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": recomputed == summary["metrics"],
        "artifact_hashes_match": hashes_ok,
        "privileged_fields_excluded": feature_isolation,
        "heldout_sources_remained_sealed": summary["calibration_loaded"] and
            not summary["development_loaded"] and not summary["test_loaded"] and
            not summary["external_benchmarks_loaded"],
    }
    result = {
        "protocol": PROTOCOL, "checks": checks, "passed": all(checks.values()),
        "rows": len(rebuilt), "max_probability_error": max_error,
        "action_mismatches": action_mismatches, "session_leaks": leaks,
        "recomputed_metrics": recomputed, "summary_sha256": sha(root / "summary.json"),
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
