"""Subject-level cross-fit robust calibration for MQuAKE anchor forgetting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossfit_calibrated_forgetting_v5 import CalibratedEnsemble, GATE_SEEDS
from tasks.train_factorized_causal_forgetting_v4 import (
    PairEvidence,
    THRESHOLDS,
    aggregate,
    evaluate,
    oracle_rows,
    paired_bootstrap,
    payload,
)
from tasks.train_factorized_scope_candidate_v3 import scope_key
from tasks.train_long_session_sparse_gate_v1 import (
    Controller,
    embedding_matrices,
    fit_snapshots,
    matched_random_schedules,
)
from tasks.train_mquake_anchor_forgetting_v10 import (
    AnchorResolver,
    candidate_metrics,
    compact,
    prepare_episodes,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_robust_calibration_v11"
FOLDS = 5
EPOCH = 50


def subject_folds(episodes):
    """Assign whole subjects while greedily balancing episode counts."""
    counts = Counter(row["primary_subject_hash"] for row in episodes)
    loads = [0] * FOLDS; assignment = {}
    ordered = sorted(counts.items(), key=lambda item: (
        -item[1], hashlib.sha256(f"{PROTOCOL}|{item[0]}".encode()).hexdigest()))
    for subject, count in ordered:
        fold = min(range(FOLDS), key=lambda value: (loads[value], value))
        assignment[subject] = fold; loads[fold] += count
    return assignment, loads


def robust_threshold(artifact, episodes, evidence, resolver):
    rows = []
    for threshold in THRESHOLDS:
        metric = aggregate(evaluate(
            episodes, evidence, "controller", resolver, Controller(artifact, threshold)))
        feasible = (metric["current_retention"] >= .99
                    and metric["non_target_retention"] >= .99
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .99)
        safety_floor = min(metric["current_retention"], metric["non_target_retention"],
                           metric["by_event_kind"]["hard_retain"]["accuracy"])
        rows.append((int(feasible), metric["pair_state_accuracy"], metric["stale_deletion"],
                     safety_floor, -abs(threshold - .9), threshold, metric))
    feasible_rows = [row for row in rows if row[0]]
    selected = max(feasible_rows) if feasible_rows else max(
        rows, key=lambda row: (row[3], row[1], row[2], row[4]))
    return selected[5], {str(row[5]): row[6] for row in rows}, bool(feasible_rows)


def oof_gate(metric, candidate, fold_metrics, comparison):
    checks = {
        "candidate_top1_at_least_99": candidate["candidate_top1"] >= .99,
        "current_retention_at_least_99": metric["current_retention"] >= .99,
        "non_target_retention_at_least_97": metric["non_target_retention"] >= .97,
        "hard_retain_at_least_97": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97,
        "stale_deletion_at_least_90": metric["stale_deletion"] >= .90,
        "pair_state_at_least_90": metric["pair_state_accuracy"] >= .90,
        "minimum_fold_pair_at_least_85": min(
            row["pair_state_accuracy"] for row in fold_metrics.values()) >= .85,
        "random_gain_lower_bound_above_zero": comparison["bootstrap_95ci"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, data_manifest = load_split(args.data_root, "train")
    tune, _ = load_split(args.data_root, "tune")
    episodes = train + tune
    texts = load_texts(args.data_root, data_manifest)
    scopes = {identifier: scope_key(text) for identifier, text in texts.items()}
    payloads = sorted({value for text in texts.values() if (value := payload(text))})
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    v8 = Path(args.v8_root); v8_training = json.loads((v8 / "training.json").read_text())
    v8_checkpoint = v8 / "mquake_adapted_forgetting.pt"
    if v8_training["checkpoint_sha256"] != sha(v8_checkpoint):
        raise ValueError("V8 checkpoint mismatch")
    candidate_artifacts = torch.load(v8_checkpoint, map_location="cpu", weights_only=False)[
        "candidate_artifacts"]

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())), "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, names, transforms, source_summary["selected_config"]["regularizer"])
    scope_matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw; torch.cuda.empty_cache()
    evidence = PairEvidence(scope_matrices, embeddings, index, texts, scopes)
    resolver = AnchorResolver(evidence, list(candidate_artifacts.values()))
    prepare_episodes(resolver, episodes)
    all_rows = oracle_rows(episodes, evidence, resolver)
    folds, fold_loads = subject_folds(episodes)

    fold_details, oof_records, oof_candidates, all_members = {}, [], [], []
    for fold in range(FOLDS):
        validation = [row for row in episodes if folds[row["primary_subject_hash"]] == fold]
        calibration = [row for row in episodes
                       if folds[row["primary_subject_hash"]] == (fold + 1) % FOLDS]
        train_subjects = {subject for subject, value in folds.items()
                          if value not in {fold, (fold + 1) % FOLDS}}
        fit_rows = [row for row in all_rows if row["subject"] in train_subjects]
        members, model_details = [], {}
        for seed in GATE_SEEDS:
            artifact = fit_snapshots(fit_rows, seed + 100 * fold, EPOCH)[EPOCH]
            threshold, grid, feasible = robust_threshold(
                artifact, calibration, evidence, resolver)
            members.append((artifact, threshold)); all_members.append((artifact, threshold))
            model_details[str(seed)] = {"threshold": threshold,
                                        "calibration_feasible": feasible, "grid": grid}
        controller = CalibratedEnsemble(members)
        records = evaluate(validation, evidence, "controller", resolver, controller,
                           seed=36100 + fold)
        validation_rows = [row for row in all_rows
                           if folds[row["subject"]] == fold]
        oof_records.extend(records); oof_candidates.extend(validation_rows)
        fold_details[str(fold)] = {"models": model_details,
                                   "validation": aggregate(records),
                                   "validation_candidate": candidate_metrics(validation_rows),
                                   "validation_subjects": len({row["subject"] for row in records}),
                                   "validation_episodes": len(validation),
                                   "calibration_episodes": len(calibration),
                                   "fit_rows": len(fit_rows)}
        print(f"stage=fold_complete fold={fold}", flush=True)

    metric = aggregate(oof_records); candidate = candidate_metrics(oof_candidates)
    deletion_count = sum(decision["deleted"] is not None
                         for row in oof_records for decision in row["decisions"])
    schedules = matched_random_schedules(episodes, deletion_count, 36190)
    random_records = evaluate(episodes, evidence, "random_matched",
                              schedules=schedules, seed=36191)
    comparison = paired_bootstrap(oof_records, random_records, 36192)
    fold_metrics = {fold: details["validation"] for fold, details in fold_details.items()}
    gate = oof_gate(metric, candidate, fold_metrics, comparison)
    (root / "oof_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof_records))
    (root / "oof_candidate_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in compact(oof_candidates)))
    (root / "oof_random_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in random_records))

    summary = {"protocol": PROTOCOL, "candidate_policy": "anchor_4_then_v8_ranker",
               "oof": metric, "oof_candidate": candidate,
               "oof_vs_random": comparison, "fold_metrics": fold_metrics,
               "oof_gate": gate, "v10_development_loaded": False,
               "mquake_cf_used": False, "longmemeval_used": False, "locomo_scored": False,
               "next_stage": "build_mquake_cf_confirmation" if gate["passed"]
                              else "stop_threshold_tuning_redesign_safety_gate",
               "terminal_status": "completed_mquake_robust_calibration_v11"}
    checkpoint = root / "mquake_robust_crossfit_gates.pt"
    torch.save({"members": all_members, "folds": folds,
                "candidate_artifacts": candidate_artifacts}, checkpoint)
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": all_rows[0]["feature"].numel(),
                "fold_loads": fold_loads, "folds": fold_details,
                "fold_assignment_sha256": hashlib.sha256(
                    json.dumps(folds, sort_keys=True).encode()).hexdigest(),
                "embedding_hashes": embedding_hashes,
                "matched_random_deletions": deletion_count,
                "raw_rows_sha256": {path.name: sha(path) for path in root.glob("*.jsonl")},
                "checkpoint_sha256": sha(checkpoint),
                "v8_hashes": {"summary": sha(v8 / "summary.json"),
                              "training": sha(v8 / "training.json"),
                              "checkpoint": sha(v8_checkpoint)}}
    manifest = {"protocol": PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "folds": FOLDS, "gate_epoch": EPOCH, "gate_seeds": list(GATE_SEEDS),
                "thresholds": list(THRESHOLDS),
                "calibration_safety_floor": .99,
                "candidate_policy": "anchor_4_then_v8_ranker", "shortlist": 4,
                "nested_train_calibrate_validate": True, "refit_after_calibration": False,
                "v10_development_used": False, "mquake_cf_used": False,
                "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v8-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
