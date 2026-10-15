#!/usr/bin/env python3
"""Audit V10 state-equivalent labels and cross-fit predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS
from tasks.train_qwen_natural_forgetting_entity_key_v7 import FEATURE_NAMES
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead, NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import oof_metrics
from tasks.train_qwen_natural_forgetting_state_equivalent_v10 import (
    PLAN,
    PROTOCOL,
    relabel_corpus,
)


def audit(args):
    import torch

    root = Path(args.root); output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((root / "summary.json").read_text())
    rebuilt = relabel_corpus(args.source_root, args.data_root)
    stored = [json.loads(line) for line in
              (root / "state_equivalent_corpus.jsonl").read_text().splitlines() if line]
    features = torch.load(root / "state_equivalent_features.pt", map_location="cpu",
                          weights_only=False)
    artifact = torch.load(root / "state_equivalent_crossfit.pt", map_location="cpu",
                          weights_only=False)
    oof = [json.loads(line) for line in
           (root / "oof_predictions.jsonl").read_text().splitlines() if line]
    corpus_ok = rebuilt == stored and [row["state_id"] for row in rebuilt] == [
        row["state_id"] for row in oof]
    tensors_ok = tuple(features["feature_names"]) == tuple(FEATURE_NAMES) \
        and torch.equal(features["features"], torch.tensor(
            [row["features"] for row in rebuilt], dtype=torch.float32)) \
        and torch.equal(features["targets"], torch.tensor(
            [row["safe_delete"] for row in rebuilt], dtype=torch.float32))
    max_error, action_mismatches, leaks = 0.0, 0, 0
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
            model.load_state_dict(checkpoint["state"]); model.eval()
            with torch.no_grad():
                predictions.append(model((features["features"][validation]
                    - checkpoint["mean"]) / checkpoint["std"])
                    .squeeze(-1).sigmoid())
        ensemble = torch.stack(predictions).mean(0)
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            max_error = max(max_error, abs(probability - oof[index]["probability"]))
            delete = probability >= fold["threshold"] and (
                not fold["use_entity_anchor_barrier"]
                or rebuilt[index]["entity_anchor_barrier"])
            action_mismatches += int(delete !=
                                     (oof[index]["action"] == "scope_top1_a0"))
    recomputed = oof_metrics(rebuilt, oof)
    source = Path(args.source_root)
    hashes = summary["hashes"]
    hashes_ok = all((
        hashes["plan"] == sha(PLAN),
        hashes["corpus"] == sha(root / "state_equivalent_corpus.jsonl"),
        hashes["features"] == sha(root / "state_equivalent_features.pt"),
        hashes["checkpoint"] == sha(root / "state_equivalent_crossfit.pt"),
        hashes["oof"] == sha(root / "oof_predictions.jsonl"),
        hashes["inner_search"] == sha(root / "inner_search.json"),
        hashes["source_summary"] == sha(source / "result" / "summary.json"),
        hashes["source_audit"] == sha(source / "execution_audit.json"),
        hashes["source_corpus"] == sha(source / "result" / "entity_key_corpus.jsonl"),
        hashes["data_manifest"] == sha(Path(args.data_root) / "manifest.json"),
    ))
    feature_isolation = tuple(summary["feature_names"][:10]) == tuple(NO_SCHEDULE) \
        and not {"source_row_sha256", "target", "chain_id", "event_kind"} \
        & set(summary["feature_names"])
    components = summary["label_components"]
    label_components_ok = components == {
        "lifecycle_safe": sum(row["lifecycle_safe_delete"] for row in rebuilt),
        "duplicate_equivalent": sum(row["duplicate_equivalent"] for row in rebuilt),
        "duplicate_only": sum(row["duplicate_equivalent"]
                              and not row["lifecycle_safe_delete"] for row in rebuilt),
    }
    checks = {
        "state_equivalent_corpus_reconstruction_exact": corpus_ok,
        "feature_and_target_tensors_exact": tensors_ok,
        "label_components_exact": label_components_ok,
        "no_train_validation_session_overlap": leaks == 0,
        "checkpoint_probabilities_match": max_error <= 2e-7,
        "checkpoint_barrier_actions_match": action_mismatches == 0,
        "aggregate_metrics_match": recomputed == summary["metrics"],
        "artifact_hashes_match": hashes_ok,
        "label_only_metadata_excluded": feature_isolation,
        "heldout_sources_remained_sealed": summary["calibration_loaded"]
            and not summary["development_loaded"] and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
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
    result = audit(parser.parse_args())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
