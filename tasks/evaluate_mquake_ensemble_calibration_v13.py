"""Calibrate the final raw-logit V12 ensemble once, without retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_factorized_causal_forgetting_v4 import (
    PairEvidence,
    aggregate,
    evaluate,
    matched_random_schedules,
    oracle_rows,
    paired_bootstrap,
    payload,
)
from tasks.train_factorized_scope_candidate_v3 import scope_key
from tasks.train_long_session_sparse_gate_v1 import Controller, embedding_matrices
from tasks.train_mquake_anchor_forgetting_v10 import prepare_episodes
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_subject_risk_v12 import (
    PAIR_CACHE_LIMIT,
    WideAnchorResolver,
    candidate_metrics,
    cluster_metrics,
    release_gate,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_ensemble_calibration_v13"
ENSEMBLE_THRESHOLDS = (
    .001, .002, .005, .01, .02, .03, .04, .05, .075, .1, .15,
    .2, .3, .4, .5, .6, .7, .8, .9, .95, .975, .99,
)
SAFETY_MACRO_FLOOR = .985
SAFETY_LCB_FLOOR = .97
EXPECTED_V12_HASHES = {
    "summary.json": "76b61afa5e10affea808e06a9ea82cd489f943f65db379e962c2f3b618aa4cc4",
    "training.json": "8444427160e6fe8d9721efbf527fa9569782d057df4f58cdfbb1948e5f53392e",
    "training_manifest.json": "0450f440f0e39ef6a486979d508ebf8b94db90514e47b124e118d55a790d08d5",
    "mquake_subject_risk_crossfit.pt":
        "d61386baebac1bd02352db5eeb1a0fe7ef00a894c130f1b408536bc5c995d41c",
}


def median_logit_probability(probabilities):
    """Combine raw member scores without applying member-specific thresholds."""
    logits = []
    for probability in probabilities:
        probability = min(max(float(probability), 1e-7), 1 - 1e-7)
        logits.append(math.log(probability / (1 - probability)))
    value = statistics.median(logits)
    return 1 / (1 + math.exp(-max(-30., min(30., value))))


class RawMedianEnsemble:
    """Median raw-logit ensemble with one deploy-time threshold."""

    def __init__(self, artifacts, threshold):
        self.members = [Controller(artifact, .5) for artifact in artifacts]
        self.threshold = threshold

    def probability(self, feature):
        return median_logit_probability(
            [controller.probability(feature) for controller in self.members]
        )


def select_threshold_from_rows(rows):
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["cluster"]["pair_macro"],
                row["metric"]["pair_state_accuracy"],
                row["metric"]["stale_deletion"],
                row["cluster"]["safety_lcb95"],
                -abs(row["threshold"] - .5),
            ),
        )
    else:
        selected = max(
            rows,
            key=lambda row: (
                row["cluster"]["safety_lcb95"],
                row["cluster"]["safety_macro"],
                row["cluster"]["pair_macro"],
                row["metric"]["pair_state_accuracy"],
                -abs(row["threshold"] - .5),
            ),
        )
    return selected, bool(feasible)


def ensemble_threshold(artifacts, episodes, evidence, resolver):
    controller = RawMedianEnsemble(artifacts, ENSEMBLE_THRESHOLDS[0])
    rows = []
    for threshold in ENSEMBLE_THRESHOLDS:
        controller.threshold = threshold
        records = evaluate(episodes, evidence, "controller", resolver, controller)
        metric = aggregate(records)
        cluster = cluster_metrics(records)
        feasible = (
            metric["current_retention"] >= .97
            and metric["non_target_retention"] >= .97
            and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97
            and cluster["safety_macro"] >= SAFETY_MACRO_FLOOR
            and cluster["safety_lcb95"] >= SAFETY_LCB_FLOOR
        )
        rows.append({"threshold": threshold, "metric": metric, "cluster": cluster,
                     "feasible": feasible})
    selected, has_feasible = select_threshold_from_rows(rows)
    grid = {
        str(row["threshold"]): {
            "metric": row["metric"], "cluster": row["cluster"],
            "feasible": row["feasible"],
        }
        for row in rows
    }
    return selected["threshold"], grid, has_feasible


def _load_and_verify_v12(root):
    import torch

    root = Path(root)
    actual = {name: sha(root / name) for name in EXPECTED_V12_HASHES}
    if actual != EXPECTED_V12_HASHES:
        raise ValueError(f"V12 artifact hash mismatch: {actual}")
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    if summary["protocol"] != "mquake_subject_risk_v12" \
            or summary["terminal_status"] != "completed_mquake_subject_risk_v12":
        raise ValueError("V12 source is not a completed subject-risk run")
    if training["checkpoint_sha256"] != actual["mquake_subject_risk_crossfit.pt"]:
        raise ValueError("V12 checkpoint hash is not internally consistent")
    checkpoint = torch.load(
        root / "mquake_subject_risk_crossfit.pt", map_location="cpu", weights_only=False
    )
    return summary, training, manifest, checkpoint, actual


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    train, data_manifest = load_split(args.data_root, "train")
    tune, _ = load_split(args.data_root, "tune")
    episodes = train + tune
    texts = load_texts(args.data_root, data_manifest)
    scopes = {identifier: scope_key(text) for identifier, text in texts.items()}
    payloads = sorted({value for text in texts.values() if (value := payload(text))})

    v12_summary, v12_training, v12_manifest, v12_checkpoint, v12_hashes = \
        _load_and_verify_v12(args.v12_root)
    folds, fold_loads = subject_folds(episodes)
    if folds != v12_checkpoint["folds"] or fold_loads != v12_training["fold_loads"]:
        raise ValueError("V12 fold assignment mismatch")
    if v12_manifest["data_manifest_sha256"] != sha(Path(args.data_root) / "manifest.json"):
        raise ValueError("V12 data manifest mismatch")
    if args.model != v12_manifest["model"] or args.tokenizer != v12_manifest["tokenizer"]:
        raise ValueError("V13 must use the exact V12 model and tokenizer paths")

    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    if {
        "summary": sha(source / "summary.json"),
        "training": sha(source / "training.json"),
        "transforms": sha(source / "transforms.pt"),
    } != {
        "summary": v12_manifest["source_summary_sha256"],
        "training": v12_manifest["source_training_sha256"],
        "transforms": v12_manifest["source_transforms_sha256"],
    }:
        raise ValueError("V13 contextual source differs from V12")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())),
             "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, names, torch.load(source / "transforms.pt", map_location="cpu",
                               weights_only=False)["transforms"],
        source_summary["selected_config"]["regularizer"],
    )
    scope_matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw
    torch.cuda.empty_cache()
    print("stage=embeddings_complete", flush=True)
    evidence = PairEvidence(
        scope_matrices, embeddings, index, texts, scopes,
        cache_limit=PAIR_CACHE_LIMIT,
    )

    fold_details = {}
    oof_records, oof_candidates = [], []
    selected_thresholds = {}
    for fold in range(FOLDS):
        validation = [row for row in episodes if folds[row["primary_subject_hash"]] == fold]
        calibration = [row for row in episodes
                       if folds[row["primary_subject_hash"]] == (fold + 1) % FOLDS]
        candidate_artifacts = list(v12_checkpoint["candidate_members"][str(fold)].values())
        resolver = WideAnchorResolver(evidence, candidate_artifacts)
        prepare_episodes(resolver, calibration + validation)
        gate_artifacts = [artifact for artifact, _ in
                          v12_checkpoint["gate_members"][str(fold)]]
        threshold, grid, feasible = ensemble_threshold(
            gate_artifacts, calibration, evidence, resolver
        )
        controller = RawMedianEnsemble(gate_artifacts, threshold)
        records = evaluate(
            validation, evidence, "controller", resolver, controller,
            seed=38100 + fold,
        )
        candidate_rows = oracle_rows(validation, evidence, resolver)
        fold_details[str(fold)] = {
            "threshold": threshold,
            "calibration_feasible": feasible,
            "grid": grid,
            "validation": aggregate(records),
            "validation_cluster": cluster_metrics(records),
            "validation_candidate": candidate_metrics(candidate_rows),
            "validation_subjects": len({row["subject"] for row in records}),
            "validation_episodes": len(validation),
            "calibration_episodes": len(calibration),
            "frozen_gate_members": len(gate_artifacts),
        }
        selected_thresholds[str(fold)] = threshold
        oof_records.extend(records)
        oof_candidates.extend(candidate_rows)
        print(f"stage=fold_complete fold={fold} threshold={threshold}", flush=True)
        del resolver, candidate_rows

    metric = aggregate(oof_records)
    cluster = cluster_metrics(oof_records)
    candidate = candidate_metrics(oof_candidates)
    deletion_count = sum(
        decision["deleted"] is not None
        for row in oof_records for decision in row["decisions"]
    )
    schedules = matched_random_schedules(episodes, deletion_count, 38190)
    random_records = evaluate(
        episodes, evidence, "random_matched", schedules=schedules, seed=38191
    )
    comparison = paired_bootstrap(oof_records, random_records, 38192)
    fold_metrics = {
        fold: details["validation"] for fold, details in fold_details.items()
    }
    gate = release_gate(metric, candidate, cluster, fold_metrics, comparison)
    (root / "oof_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof_records)
    )
    compact_keys = (
        "episode_id", "subject", "length", "step", "event_kind", "label",
        "candidate_correct", "shortlist_correct",
    )
    (root / "oof_candidate_rows.jsonl").write_text(
        "".join(json.dumps({key: row[key] for key in compact_keys}) + "\n"
                for row in oof_candidates)
    )
    (root / "oof_random_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in random_records)
    )

    pair_checks = (
        gate["checks"]["stale_deletion_at_least_90"]
        and gate["checks"]["pair_state_at_least_90"]
        and gate["checks"]["subject_macro_pair_at_least_85"]
        and gate["checks"]["minimum_fold_pair_at_least_85"]
    )
    safety_checks = (
        gate["checks"]["current_retention_at_least_99"]
        and gate["checks"]["non_target_retention_at_least_97"]
        and gate["checks"]["hard_retain_at_least_97"]
        and gate["checks"]["cluster_safety_lcb_at_least_95"]
    )
    summary = {
        "protocol": PROTOCOL,
        "intervention": "single_final_threshold_on_raw_median_gate_logits",
        "oof": metric,
        "oof_cluster": cluster,
        "oof_candidate": candidate,
        "oof_vs_random": comparison,
        "fold_metrics": fold_metrics,
        "oof_gate": gate,
        "calibration_hypothesis_supported": pair_checks and safety_checks,
        "v12_weights_frozen": True,
        "v10_development_loaded": False,
        "mquake_cf_used": False,
        "longmemeval_used": False,
        "locomo_scored": False,
        "next_stage": "candidate_only_relation_scope" if pair_checks and safety_checks
                      else "stop_scalar_calibration_and_diagnose_replay",
        "terminal_status": "completed_mquake_ensemble_calibration_v13",
    }
    calibration_checkpoint = root / "mquake_ensemble_calibration_crossfit.pt"
    torch.save({
        "folds": folds,
        "selected_thresholds": selected_thresholds,
        "v12_checkpoint_sha256": v12_hashes["mquake_subject_risk_crossfit.pt"],
    }, calibration_checkpoint)
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "fold_loads": fold_loads,
        "folds": fold_details,
        "embedding_hashes": embedding_hashes,
        "matched_random_deletions": deletion_count,
        "source_v12_hashes": v12_hashes,
        "pair_cache": {"limit": evidence.cache_limit, "entries": len(evidence.cache),
                       "hits": evidence.hits, "misses": evidence.misses,
                       "evictions": evidence.evictions},
        "raw_rows_sha256": {path.name: sha(path) for path in root.glob("*.jsonl")},
        "checkpoint_sha256": sha(calibration_checkpoint),
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "source_summary_sha256": sha(source / "summary.json"),
        "source_training_sha256": sha(source / "training.json"),
        "source_transforms_sha256": sha(source / "transforms.pt"),
        "v12_hashes": v12_hashes,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "task_source_sha256": sha(Path(__file__)),
        "folds": FOLDS,
        "ensemble_thresholds": list(ENSEMBLE_THRESHOLDS),
        "safety_macro_floor": SAFETY_MACRO_FLOOR,
        "safety_lcb_floor": SAFETY_LCB_FLOOR,
        "pair_cache_limit": PAIR_CACHE_LIMIT,
        "raw_logit_median": True,
        "member_thresholds_ignored": True,
        "single_final_threshold": True,
        "causal_calibration_replay": True,
        "nested_calibrate_validate": True,
        "model_fitting": False,
        "refit_after_calibration": False,
        "v10_development_used": False,
        "mquake_cf_used": False,
        "longmemeval_used": False,
        "locomo_used": False,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--v12-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
