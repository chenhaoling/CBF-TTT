"""Train nested subject-jackknife gates with calibrated consensus forgetting."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from tasks.evaluate_mquake_ensemble_calibration_v13 import _load_and_verify_v12
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
from tasks.train_long_session_sparse_gate_v1 import embedding_matrices
from tasks.train_mquake_anchor_forgetting_v10 import prepare_episodes
from tasks.train_mquake_online_margin_v14 import (
    EPOCH,
    ListwiseEnsemble,
    SAFETY_LCB_FLOOR,
    SAFETY_MACRO_FLOOR,
    SCORE_SCALE_FLOOR,
    fit_listwise_gate,
    relative_margin,
)
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_subject_risk_v12 import (
    PAIR_CACHE_LIMIT,
    WideAnchorResolver,
    candidate_metrics,
    cluster_metrics,
    release_gate,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_jackknife_consensus_v15"
INNER_MEMBERS = 3
QUORUM = 3
FPR_TARGETS = (.0005, .001, .0025, .005)


def gate_seed(outer_fold, calibration_fold, member):
    return 42000 + 1000 * outer_fold + 100 * calibration_fold + member


def empirical_threshold(values, target):
    values = sorted(values)
    if not values:
        raise ValueError("non-executable calibration scores are empty")
    allowed = math.floor(target * len(values))
    if allowed == 0:
        return math.nextafter(values[-1], math.inf)
    return values[len(values) - allowed]


def retain_trace(episode, evidence, resolver, ensemble):
    memory, history, rows = [], [], []
    episode_scores = evidence.prepare_episode(episode)
    for step, event in enumerate(episode["sessions"]):
        if memory:
            selected, feature, _ = resolver.resolve(
                memory, event, step, episode["length"], episode_scores
            )
            selected_id = memory[selected]["session_id"]
            score = ensemble.score(feature)
            value = relative_margin(score, history)
            rows.append({
                "margin": value,
                "executable": (event["desired"] == "local"
                               and selected_id == event["target_session_id"]),
            })
            history.append(score)
        memory.append({**event, "position": step})
    return rows


def calibration_candidates(episodes, evidence, resolver, ensemble):
    rows = [value for episode in episodes
            for value in retain_trace(episode, evidence, resolver, ensemble)
            if value["margin"] is not None]
    negatives = [row["margin"] for row in rows if not row["executable"]]
    return [(target, empirical_threshold(negatives, target))
            for target in FPR_TARGETS]


def replay_single(episode, evidence, resolver, ensemble, threshold):
    memory, decisions, history = [], [], []
    episode_scores = evidence.prepare_episode(episode)
    old_id = next(row["session_id"] for row in episode["sessions"]
                  if row["event_kind"] == "primary_old")
    current_id = next(row["session_id"] for row in episode["sessions"]
                      if row["event_kind"] == "primary_update")
    for step, event in enumerate(episode["sessions"]):
        deleted = selected_id = None
        score = margin = 0.
        history_size = len(history)
        if memory:
            selected, feature, _ = resolver.resolve(
                memory, event, step, episode["length"], episode_scores
            )
            selected_id = memory[selected]["session_id"]
            score = ensemble.score(feature)
            value = relative_margin(score, history)
            margin = 0. if value is None else value
            if value is not None and value >= threshold:
                deleted = memory.pop(selected)["session_id"]
            history.append(score)
        correct = (event["desired"] == "retain" and deleted is None) or (
            event["desired"] == "local" and deleted == event["target_session_id"]
        )
        decisions.append({
            "step": step, "desired": event["desired"],
            "event_kind": event["event_kind"], "deleted": deleted,
            "top": selected_id, "correct": int(correct), "score": score,
            "margin": margin, "threshold": threshold,
            "history_size": history_size,
        })
        memory.append({**event, "position": step})
    retained = {row["session_id"] for row in memory}
    non_target = {row["session_id"] for row in episode["sessions"]} - {old_id, current_id}
    return {
        "episode_id": episode["episode_id"],
        "subject": episode["primary_subject_hash"],
        "length": episode["length"], "policy": "inner_calibration",
        "decisions": decisions,
        "pair_state_correct": int(old_id not in retained and current_id in retained),
        "stale_deleted": int(old_id not in retained),
        "current_retained": int(current_id in retained),
        "non_target_retention": len(non_target & retained) / len(non_target),
        "memory_fraction": len(memory) / episode["length"],
    }


def select_inner_threshold(episodes, evidence, resolver, artifacts):
    ensemble = ListwiseEnsemble(artifacts)
    candidates = calibration_candidates(episodes, evidence, resolver, ensemble)
    rows = []
    for target, threshold in candidates:
        records = [replay_single(row, evidence, resolver, ensemble, threshold)
                   for row in episodes]
        metric = aggregate(records); cluster = cluster_metrics(records)
        feasible = (
            metric["current_retention"] >= .97
            and metric["non_target_retention"] >= .97
            and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97
            and cluster["safety_macro"] >= SAFETY_MACRO_FLOOR
            and cluster["safety_lcb95"] >= SAFETY_LCB_FLOOR
        )
        rows.append({"target_fpr": target, "threshold": threshold,
                     "metric": metric, "cluster": cluster, "feasible": feasible})
    feasible = [row for row in rows if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            row["metric"]["stale_deletion"], row["cluster"]["safety_lcb95"],
            -row["target_fpr"],
        ))
    else:
        selected = max(rows, key=lambda row: (
            row["cluster"]["safety_lcb95"], row["cluster"]["safety_macro"],
            row["cluster"]["pair_macro"], row["metric"]["pair_state_accuracy"],
            -row["target_fpr"],
        ))
    return selected["threshold"], selected["target_fpr"], rows, bool(feasible)


def replay_consensus(episode, evidence, resolver, voters):
    memory, decisions = [], []
    histories = [[] for _ in voters]
    episode_scores = evidence.prepare_episode(episode)
    old_id = next(row["session_id"] for row in episode["sessions"]
                  if row["event_kind"] == "primary_old")
    current_id = next(row["session_id"] for row in episode["sessions"]
                      if row["event_kind"] == "primary_update")
    for step, event in enumerate(episode["sessions"]):
        deleted = selected_id = None
        scores, margins, votes = [], [], []
        history_sizes = [len(history) for history in histories]
        if memory:
            selected, feature, _ = resolver.resolve(
                memory, event, step, episode["length"], episode_scores
            )
            selected_id = memory[selected]["session_id"]
            for position, (ensemble, threshold) in enumerate(voters):
                score = ensemble.score(feature)
                value = relative_margin(score, histories[position])
                scores.append(score); margins.append(0. if value is None else value)
                votes.append(bool(value is not None and value >= threshold))
                histories[position].append(score)
            if sum(votes) >= QUORUM:
                deleted = memory.pop(selected)["session_id"]
        correct = (event["desired"] == "retain" and deleted is None) or (
            event["desired"] == "local" and deleted == event["target_session_id"]
        )
        decisions.append({
            "step": step, "desired": event["desired"],
            "event_kind": event["event_kind"], "deleted": deleted,
            "top": selected_id, "correct": int(correct),
            "scores": scores, "margins": margins,
            "thresholds": [threshold for _, threshold in voters],
            "votes": votes, "vote_count": sum(votes), "quorum": QUORUM,
            "history_sizes": history_sizes,
            "probability": sum(votes) / len(voters),
        })
        memory.append({**event, "position": step})
    retained = {row["session_id"] for row in memory}
    non_target = {row["session_id"] for row in episode["sessions"]} - {old_id, current_id}
    return {
        "episode_id": episode["episode_id"],
        "subject": episode["primary_subject_hash"],
        "length": episode["length"], "policy": "jackknife_consensus",
        "decisions": decisions,
        "pair_state_correct": int(old_id not in retained and current_id in retained),
        "stale_deleted": int(old_id not in retained),
        "current_retained": int(current_id in retained),
        "non_target_retention": len(non_target & retained) / len(non_target),
        "memory_fraction": len(memory) / episode["length"],
    }


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
    _, v12_training, v12_manifest, v12_checkpoint, v12_hashes = \
        _load_and_verify_v12(args.v12_root)
    folds, fold_loads = subject_folds(episodes)
    if folds != v12_checkpoint["folds"] or fold_loads != v12_training["fold_loads"]:
        raise ValueError("V12 fold assignment mismatch")
    if v12_manifest["data_manifest_sha256"] != sha(Path(args.data_root) / "manifest.json"):
        raise ValueError("V12 data manifest mismatch")
    if args.model != v12_manifest["model"] or args.tokenizer != v12_manifest["tokenizer"]:
        raise ValueError("V15 must use the exact V12 model and tokenizer paths")

    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    expected_source = {
        "summary": v12_manifest["source_summary_sha256"],
        "training": v12_manifest["source_training_sha256"],
        "transforms": v12_manifest["source_transforms_sha256"],
    }
    actual_source = {"summary": sha(source / "summary.json"),
                     "training": sha(source / "training.json"),
                     "transforms": sha(source / "transforms.pt")}
    if actual_source != expected_source:
        raise ValueError("V15 contextual source differs from V12")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())),
             "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    transforms = torch.load(source / "transforms.pt", map_location="cpu",
                            weights_only=False)["transforms"]
    embeddings, embedding_hashes = apply(
        raw, names, transforms, source_summary["selected_config"]["regularizer"]
    )
    scope_matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw; torch.cuda.empty_cache()
    print("stage=embeddings_complete", flush=True)
    evidence = PairEvidence(scope_matrices, embeddings, index, texts, scopes,
                            cache_limit=PAIR_CACHE_LIMIT)

    fold_details, checkpoints = {}, {}
    oof_records, oof_candidates = [], []
    for outer in range(FOLDS):
        validation = [row for row in episodes
                      if folds[row["primary_subject_hash"]] == outer]
        candidate_artifacts = list(
            v12_checkpoint["candidate_members"][str(outer)].values()
        )
        resolver = WideAnchorResolver(evidence, candidate_artifacts)
        prepare_episodes(resolver, episodes)
        all_rows = oracle_rows(episodes, evidence, resolver)
        rows_by_episode = defaultdict(list)
        for row in all_rows:
            rows_by_episode[row["episode_id"]].append(row)
        voters, inner_details, inner_checkpoints = [], {}, {}
        for calibration_fold in range(FOLDS):
            if calibration_fold == outer:
                continue
            fit_folds = set(range(FOLDS)) - {outer, calibration_fold}
            fit_ids = {row["episode_id"] for row in episodes
                       if folds[row["primary_subject_hash"]] in fit_folds}
            fit_rows = [row for identifier in sorted(fit_ids)
                        for row in rows_by_episode[identifier]]
            calibration = [row for row in episodes
                           if folds[row["primary_subject_hash"]] == calibration_fold]
            artifacts = [fit_listwise_gate(
                fit_rows, gate_seed(outer, calibration_fold, member)
            ) for member in range(INNER_MEMBERS)]
            threshold, target_fpr, grid, feasible = select_inner_threshold(
                calibration, evidence, resolver, artifacts
            )
            voters.append((ListwiseEnsemble(artifacts), threshold))
            inner_checkpoints[str(calibration_fold)] = artifacts
            inner_details[str(calibration_fold)] = {
                "fit_folds": sorted(fit_folds), "fit_episodes": len(fit_ids),
                "fit_rows": len(fit_rows), "calibration_episodes": len(calibration),
                "threshold": threshold, "target_fpr": target_fpr,
                "calibration_feasible": feasible, "grid": grid,
                "models": {str(artifact["seed"]): {
                    "train_loss": artifact["train_loss"],
                    "score_center": artifact["score_center"],
                    "score_scale": artifact["score_scale"],
                } for artifact in artifacts},
            }
            print(f"stage=inner_complete outer={outer} calibration={calibration_fold} "
                  f"threshold={threshold} target_fpr={target_fpr}", flush=True)
        records = [replay_consensus(row, evidence, resolver, voters)
                   for row in validation]
        candidate_rows = [row for episode in validation
                          for row in rows_by_episode[episode["episode_id"]]]
        fold_details[str(outer)] = {
            "validation": aggregate(records),
            "validation_cluster": cluster_metrics(records),
            "validation_candidate": candidate_metrics(candidate_rows),
            "validation_subjects": len({row["subject"] for row in records}),
            "validation_episodes": len(validation), "inner": inner_details,
        }
        checkpoints[str(outer)] = inner_checkpoints
        oof_records.extend(records); oof_candidates.extend(candidate_rows)
        print(f"stage=outer_complete outer={outer}", flush=True)
        del resolver, all_rows, rows_by_episode, voters, candidate_rows

    metric = aggregate(oof_records); cluster = cluster_metrics(oof_records)
    candidate = candidate_metrics(oof_candidates)
    deletion_count = sum(decision["deleted"] is not None
                         for row in oof_records for decision in row["decisions"])
    schedules = matched_random_schedules(episodes, deletion_count, 42190)
    random_records = evaluate(
        episodes, evidence, "random_matched", schedules=schedules, seed=42191
    )
    comparison = paired_bootstrap(oof_records, random_records, 42192)
    fold_metrics = {key: value["validation"] for key, value in fold_details.items()}
    gate = release_gate(metric, candidate, cluster, fold_metrics, comparison)
    pair_checks = all(gate["checks"][key] for key in (
        "stale_deletion_at_least_90", "pair_state_at_least_90",
        "subject_macro_pair_at_least_85", "minimum_fold_pair_at_least_85",
    ))
    safety_checks = all(gate["checks"][key] for key in (
        "current_retention_at_least_99", "non_target_retention_at_least_97",
        "hard_retain_at_least_97", "cluster_safety_lcb_at_least_95",
    ))
    (root / "oof_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof_records))
    compact_keys = ("episode_id", "subject", "length", "step", "event_kind",
                    "label", "candidate_correct", "shortlist_correct")
    (root / "oof_candidate_rows.jsonl").write_text(
        "".join(json.dumps({key: row[key] for key in compact_keys}) + "\n"
                for row in oof_candidates))
    (root / "oof_random_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in random_records))
    summary = {
        "protocol": PROTOCOL,
        "intervention": "nested_subject_jackknife_calibration_with_three_of_four_consensus",
        "oof": metric, "oof_cluster": cluster, "oof_candidate": candidate,
        "oof_vs_random": comparison, "fold_metrics": fold_metrics,
        "oof_gate": gate,
        "jackknife_consensus_supported": pair_checks and safety_checks,
        "v10_development_loaded": False, "mquake_cf_used": False,
        "longmemeval_used": False, "locomo_scored": False,
        "next_stage": ("real_fast_weight_memory_integration" if pair_checks and safety_checks
                       else "stop_listwise_scalar_gate_feature_family"),
        "terminal_status": "completed_mquake_jackknife_consensus_v15",
    }
    checkpoint = root / "mquake_jackknife_consensus_crossfit.pt"
    torch.save({"folds": folds, "inner_gate_members": checkpoints,
                "selected_thresholds": {
                    outer: {inner: details["threshold"]
                            for inner, details in fold["inner"].items()}
                    for outer, fold in fold_details.items()
                },
                "v12_checkpoint_sha256": v12_hashes["mquake_subject_risk_crossfit.pt"]},
               checkpoint)
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "fold_loads": fold_loads, "folds": fold_details,
        "embedding_hashes": embedding_hashes,
        "matched_random_deletions": deletion_count,
        "source_v12_hashes": v12_hashes,
        "pair_cache": {"limit": evidence.cache_limit, "entries": len(evidence.cache),
                       "hits": evidence.hits, "misses": evidence.misses,
                       "evictions": evidence.evictions},
        "raw_rows_sha256": {path.name: sha(path) for path in root.glob("*.jsonl")},
        "checkpoint_sha256": sha(checkpoint),
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "source_summary_sha256": sha(source / "summary.json"),
        "source_training_sha256": sha(source / "training.json"),
        "source_transforms_sha256": sha(source / "transforms.pt"),
        "v12_hashes": v12_hashes, "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              text=True).strip(),
        "task_source_sha256": sha(Path(__file__)), "folds": FOLDS,
        "inner_members": INNER_MEMBERS, "quorum": QUORUM,
        "gate_epoch": EPOCH, "fpr_targets": list(FPR_TARGETS),
        "safety_macro_floor": SAFETY_MACRO_FLOOR,
        "safety_lcb_floor": SAFETY_LCB_FLOOR,
        "score_scale_floor": SCORE_SCALE_FLOOR,
        "pair_cache_limit": PAIR_CACHE_LIMIT,
        "subject_equal_episode_listwise_loss": True,
        "member_fit_iqr_normalization": True,
        "online_history": "median_all_prior_action_scores",
        "online_history_uses_only_prior_scores": True,
        "online_history_reset_each_episode": True,
        "first_scored_action_cannot_delete": True,
        "inner_models_refit_after_calibration": False,
        "position_feature": False, "length_feature": False,
        "subject_identity_feature": False, "relation_identity_feature": False,
        "future_score_feature": False, "nested_subject_jackknife": True,
        "v10_development_used": False, "mquake_cf_used": False,
        "longmemeval_used": False, "locomo_used": False,
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
