"""Train a subject-balanced listwise Gate with a causal online score margin."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
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
from tasks.train_long_session_sparse_gate_v1 import SparseGate, embedding_matrices
from tasks.train_mquake_anchor_forgetting_v10 import prepare_episodes
from tasks.evaluate_mquake_ensemble_calibration_v13 import _load_and_verify_v12
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_subject_risk_v12 import (
    PAIR_CACHE_LIMIT,
    WideAnchorResolver,
    candidate_metrics,
    cluster_metrics,
    equal_subject_weights,
    release_gate,
)
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_online_margin_v14"
EPOCH = 50
GATE_SEEDS = (39101, 39102, 39103)
MARGIN_THRESHOLDS = (
    -1., -.5, 0., .25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.,
    4., 5., 6., 8., 10., 12., 15., 20.,
)
SAFETY_MACRO_FLOOR = .985
SAFETY_LCB_FLOOR = .97
SCORE_SCALE_FLOOR = .1


def grouped_indices(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["episode_id"]].append(index)
    output = []
    for identifier, indices in groups.items():
        indices.sort(key=lambda value: rows[value]["step"])
        positives = [position for position, value in enumerate(indices)
                     if rows[value]["label"]]
        if len(positives) != 1:
            raise ValueError(f"episode {identifier} must have exactly one update")
        subjects = {rows[value]["subject"] for value in indices}
        if len(subjects) != 1:
            raise ValueError(f"episode {identifier} spans subjects")
        output.append({"episode_id": identifier, "indices": indices,
                       "target": positives[0], "subject": subjects.pop()})
    output.sort(key=lambda row: row["episode_id"])
    return output


def fit_listwise_gate(rows, seed, epoch=EPOCH):
    import torch

    groups = grouped_indices(rows)
    x = torch.stack([row["feature"] for row in rows])
    feature_weights = torch.tensor(
        equal_subject_weights([row["subject"] for row in rows]), dtype=torch.float32
    )
    total = feature_weights.sum()
    mean = (x * feature_weights[:, None]).sum(0) / total
    variance = ((x - mean).square() * feature_weights[:, None]).sum(0) / total
    std = variance.sqrt().clamp_min(1e-4)
    maximum = max(len(group["indices"]) for group in groups)
    values = torch.zeros(len(groups), maximum, x.shape[-1])
    mask = torch.zeros(len(groups), maximum, dtype=torch.bool)
    targets = torch.tensor([group["target"] for group in groups], dtype=torch.long)
    for position, group in enumerate(groups):
        length = len(group["indices"])
        values[position, :length] = (x[group["indices"]] - mean) / std
        mask[position, :length] = True
    group_weights = torch.tensor(
        equal_subject_weights([group["subject"] for group in groups]),
        dtype=torch.float32,
    )
    gate = SparseGate(x.shape[-1], seed)
    optimizer = torch.optim.AdamW(gate.model.parameters(), lr=.01, weight_decay=.1)
    loss = None
    for _ in range(epoch):
        optimizer.zero_grad(set_to_none=True)
        scores = gate(values).masked_fill(~mask, -float("inf"))
        per_episode = torch.nn.functional.cross_entropy(
            scores, targets, reduction="none"
        )
        loss = (per_episode * group_weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.model.parameters(), 1.)
        optimizer.step()
    with torch.no_grad():
        fit_scores = gate((x - mean) / std)
        center = torch.quantile(fit_scores, .5)
        scale = (torch.quantile(fit_scores, .75)
                 - torch.quantile(fit_scores, .25)).clamp_min(SCORE_SCALE_FLOOR)
    return {
        "state": copy.deepcopy(gate.model.state_dict()),
        "mean": mean,
        "std": std,
        "dim": x.shape[-1],
        "seed": seed,
        "epoch": epoch,
        "train_loss": float(loss.detach()),
        "score_center": float(center),
        "score_scale": float(scale),
        "episodes": len(groups),
        "subject_equal_listwise_loss": True,
    }


class ListwiseMember:
    def __init__(self, artifact):
        self.artifact = artifact
        self.gate = SparseGate(artifact["dim"], artifact["seed"])
        self.gate.model.load_state_dict(artifact["state"])
        self.gate.model.eval()

    def score(self, feature):
        import torch

        with torch.no_grad():
            raw = self.gate(((feature - self.artifact["mean"])
                             / self.artifact["std"]).unsqueeze(0))[0]
        return (float(raw) - self.artifact["score_center"]) \
            / self.artifact["score_scale"]


class ListwiseEnsemble:
    def __init__(self, artifacts):
        self.members = [ListwiseMember(artifact) for artifact in artifacts]

    def score(self, feature):
        return statistics.median(member.score(feature) for member in self.members)


def relative_margin(score, history):
    if not history:
        return None
    return score - statistics.median(history)


def replay_online(episode, evidence, resolver, ensemble, threshold, seed=0):
    memory, decisions, score_history = [], [], []
    episode_scores = evidence.prepare_episode(episode)
    old_id = next(row["session_id"] for row in episode["sessions"]
                  if row["event_kind"] == "primary_old")
    current_id = next(row["session_id"] for row in episode["sessions"]
                      if row["event_kind"] == "primary_update")
    del seed  # The registered controller is deterministic.
    for step, event in enumerate(episode["sessions"]):
        deleted = None
        selected_id = None
        score = baseline = margin = 0.
        history_size = len(score_history)
        if memory:
            selected, feature, _ = resolver.resolve(
                memory, event, step, episode["length"], episode_scores
            )
            selected_id = memory[selected]["session_id"]
            score = ensemble.score(feature)
            value = relative_margin(score, score_history)
            baseline = score if value is None else score - value
            margin = 0. if value is None else value
            if value is not None and value >= threshold:
                deleted = memory.pop(selected)["session_id"]
            score_history.append(score)
        correct = (event["desired"] == "retain" and deleted is None) or (
            event["desired"] == "local" and deleted == event["target_session_id"]
        )
        probability = 1 / (1 + math.exp(-max(-30., min(30., margin - threshold))))
        decisions.append({
            "step": step,
            "desired": event["desired"],
            "event_kind": event["event_kind"],
            "deleted": deleted,
            "top": selected_id,
            "correct": int(correct),
            "probability": probability,
            "score": score,
            "baseline": baseline,
            "margin": margin,
            "threshold": threshold,
            "history_size": history_size,
        })
        memory.append({**event, "position": step})
    retained = {row["session_id"] for row in memory}
    non_target = {row["session_id"] for row in episode["sessions"]} - {old_id, current_id}
    return {
        "episode_id": episode["episode_id"],
        "subject": episode["primary_subject_hash"],
        "length": episode["length"],
        "policy": "online_listwise_margin",
        "decisions": decisions,
        "pair_state_correct": int(old_id not in retained and current_id in retained),
        "stale_deleted": int(old_id not in retained),
        "current_retained": int(current_id in retained),
        "non_target_retention": len(non_target & retained) / len(non_target),
        "memory_fraction": len(memory) / episode["length"],
    }


def evaluate_online(episodes, evidence, resolver, ensemble, threshold, seed=0):
    return [replay_online(row, evidence, resolver, ensemble, threshold, seed)
            for row in episodes]


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
                -abs(row["threshold"] - 2.),
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
                -abs(row["threshold"] - 2.),
            ),
        )
    return selected, bool(feasible)


def calibrate_margin(artifacts, episodes, evidence, resolver):
    ensemble = ListwiseEnsemble(artifacts)
    rows = []
    for threshold in MARGIN_THRESHOLDS:
        records = evaluate_online(episodes, evidence, resolver, ensemble, threshold)
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
    return selected["threshold"], {
        str(row["threshold"]): {
            "metric": row["metric"], "cluster": row["cluster"],
            "feasible": row["feasible"],
        }
        for row in rows
    }, has_feasible


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
    _, v12_training, v12_manifest, v12_checkpoint, v12_hashes = \
        _load_and_verify_v12(args.v12_root)
    folds, fold_loads = subject_folds(episodes)
    if folds != v12_checkpoint["folds"] or fold_loads != v12_training["fold_loads"]:
        raise ValueError("V12 fold assignment mismatch")
    if v12_manifest["data_manifest_sha256"] != sha(Path(args.data_root) / "manifest.json"):
        raise ValueError("V12 data manifest mismatch")
    if args.model != v12_manifest["model"] or args.tokenizer != v12_manifest["tokenizer"]:
        raise ValueError("V14 must use the exact V12 model and tokenizer paths")

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
        raise ValueError("V14 contextual source differs from V12")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())),
             "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    transforms = torch.load(
        source / "transforms.pt", map_location="cpu", weights_only=False
    )["transforms"]
    embeddings, embedding_hashes = apply(
        raw, names, transforms, source_summary["selected_config"]["regularizer"]
    )
    scope_matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw
    torch.cuda.empty_cache()
    print("stage=embeddings_complete", flush=True)
    evidence = PairEvidence(
        scope_matrices, embeddings, index, texts, scopes,
        cache_limit=PAIR_CACHE_LIMIT,
    )

    fold_details, gate_checkpoints = {}, {}
    oof_records, oof_candidates = [], []
    for fold in range(FOLDS):
        validation = [row for row in episodes if folds[row["primary_subject_hash"]] == fold]
        calibration = [row for row in episodes
                       if folds[row["primary_subject_hash"]] == (fold + 1) % FOLDS]
        fit_subjects = {subject for subject, value in folds.items()
                        if value not in {fold, (fold + 1) % FOLDS}}
        fit_episodes = [row for row in episodes
                        if row["primary_subject_hash"] in fit_subjects]
        candidate_artifacts = list(v12_checkpoint["candidate_members"][str(fold)].values())
        resolver = WideAnchorResolver(evidence, candidate_artifacts)
        prepare_episodes(resolver, episodes)
        fit_rows = oracle_rows(fit_episodes, evidence, resolver)
        artifacts = [fit_listwise_gate(fit_rows, seed + 100 * fold)
                     for seed in GATE_SEEDS]
        gate_checkpoints[str(fold)] = artifacts
        threshold, grid, feasible = calibrate_margin(
            artifacts, calibration, evidence, resolver
        )
        ensemble = ListwiseEnsemble(artifacts)
        records = evaluate_online(
            validation, evidence, resolver, ensemble, threshold, seed=40100 + fold
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
            "fit_episodes": len(fit_episodes),
            "fit_rows": len(fit_rows),
            "models": {str(artifact["seed"]): {
                "train_loss": artifact["train_loss"],
                "score_center": artifact["score_center"],
                "score_scale": artifact["score_scale"],
            } for artifact in artifacts},
        }
        oof_records.extend(records)
        oof_candidates.extend(candidate_rows)
        print(f"stage=fold_complete fold={fold} threshold={threshold}", flush=True)
        del resolver, fit_rows, candidate_rows

    metric = aggregate(oof_records)
    cluster = cluster_metrics(oof_records)
    candidate = candidate_metrics(oof_candidates)
    deletion_count = sum(
        decision["deleted"] is not None
        for row in oof_records for decision in row["decisions"]
    )
    schedules = matched_random_schedules(episodes, deletion_count, 40190)
    random_records = evaluate(
        episodes, evidence, "random_matched", schedules=schedules, seed=40191
    )
    comparison = paired_bootstrap(oof_records, random_records, 40192)
    fold_metrics = {key: value["validation"] for key, value in fold_details.items()}
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
    pair_checks = all(gate["checks"][key] for key in (
        "stale_deletion_at_least_90", "pair_state_at_least_90",
        "subject_macro_pair_at_least_85", "minimum_fold_pair_at_least_85",
    ))
    safety_checks = all(gate["checks"][key] for key in (
        "current_retention_at_least_99", "non_target_retention_at_least_97",
        "hard_retain_at_least_97", "cluster_safety_lcb_at_least_95",
    ))
    summary = {
        "protocol": PROTOCOL,
        "intervention": "subject_equal_episode_listwise_gate_with_online_median_margin",
        "oof": metric,
        "oof_cluster": cluster,
        "oof_candidate": candidate,
        "oof_vs_random": comparison,
        "fold_metrics": fold_metrics,
        "oof_gate": gate,
        "relative_margin_hypothesis_supported": pair_checks and safety_checks,
        "v10_development_loaded": False,
        "mquake_cf_used": False,
        "longmemeval_used": False,
        "locomo_scored": False,
        "next_stage": "candidate_only_relation_scope" if pair_checks and safety_checks
                      else "stop_gate_feature_family_for_relation_abstention",
        "terminal_status": "completed_mquake_online_margin_v14",
    }
    checkpoint = root / "mquake_online_margin_crossfit.pt"
    torch.save({"folds": folds, "gate_members": gate_checkpoints,
                "selected_thresholds": {key: value["threshold"]
                                        for key, value in fold_details.items()},
                "v12_checkpoint_sha256": v12_hashes["mquake_subject_risk_crossfit.pt"]},
               checkpoint)
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
        "checkpoint_sha256": sha(checkpoint),
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
        "gate_seeds": list(GATE_SEEDS),
        "gate_epoch": EPOCH,
        "margin_thresholds": list(MARGIN_THRESHOLDS),
        "safety_macro_floor": SAFETY_MACRO_FLOOR,
        "safety_lcb_floor": SAFETY_LCB_FLOOR,
        "score_scale_floor": SCORE_SCALE_FLOOR,
        "pair_cache_limit": PAIR_CACHE_LIMIT,
        "subject_equal_episode_listwise_loss": True,
        "member_fit_iqr_normalization": True,
        "online_baseline": "median_all_prior_action_scores",
        "online_history_uses_only_prior_scores": True,
        "online_history_reset_each_episode": True,
        "first_scored_action_cannot_delete": True,
        "position_feature": False,
        "length_feature": False,
        "subject_identity_feature": False,
        "future_score_feature": False,
        "nested_train_calibrate_validate": True,
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
