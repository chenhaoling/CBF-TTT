"""Subject-balanced, cluster-risk-controlled MQuAKE-T forgetting."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
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
    aggregate,
    evaluate,
    matched_random_schedules,
    oracle_rows,
    paired_bootstrap,
    payload,
)
from tasks.train_factorized_scope_candidate_v3 import SEEDS as CANDIDATE_SEEDS, scope_key
from tasks.train_long_session_candidate_v2 import Ranker, build_groups, padded
from tasks.train_long_session_sparse_gate_v1 import Controller, SparseGate, embedding_matrices
from tasks.train_mquake_anchor_forgetting_v10 import AnchorResolver, compact, prepare_episodes
from tasks.train_mquake_robust_calibration_v11 import FOLDS, subject_folds
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_subject_risk_v12"
EPOCH = 50
SHORTLIST = 16
RISK_THRESHOLDS = tuple(value / 20 for value in range(1, 20)) + (.975, .99, .995)
SAFETY_MACRO_FLOOR = .985
SAFETY_LCB_FLOOR = .97
Z_ONE_SIDED_95 = 1.645
PAIR_CACHE_LIMIT = 1_000_000


class WideAnchorResolver(AnchorResolver):
    """The V10 resolver with a fixed, preregistered wider anchor shortlist."""

    def resolve(self, memory, incoming, step, length, episode_scores):
        import torch

        del length, episode_scores
        anchor = self.anchor_tables[incoming["episode_id"]]
        positions = [row["position"] for row in memory]
        scores = anchor[torch.tensor(positions), step]
        shortlist = sorted(
            range(len(memory)), key=lambda index: (-float(scores[index]), index)
        )[:SHORTLIST]
        pairs = [self.evidence.pair(memory[index]["units"], incoming["units"])
                 for index in shortlist]
        features = torch.stack([row["scope_feature"] for row in pairs])
        member_scores = []
        with torch.no_grad():
            for ranker, artifact in self.members:
                member_scores.append(ranker((features - artifact["mean"]) / artifact["std"]))
        learned = torch.stack(member_scores).mean(0)
        order = sorted(
            range(len(shortlist)), key=lambda index: (-float(learned[index]), shortlist[index])
        )
        selected_local = order[0]
        selected = shortlist[selected_local]
        score_order = sorted(
            range(len(memory)), key=lambda index: (-float(scores[index]), index)
        )
        second = float(scores[score_order[1]]) if len(score_order) > 1 else -1.
        pair = pairs[selected_local]
        feature = torch.cat(
            (
                pair["scope_feature"],
                torch.tensor(
                    [float(scores[selected]), float(scores[score_order[0]]) - second]
                ),
                pair["payload_feature"],
            )
        )
        return selected, feature, {
            "shortlist_positions": [memory[index]["position"] for index in shortlist],
            "selected_position": memory[selected]["position"],
            "scope_score": float(scores[selected]),
        }


def equal_subject_weights(subjects):
    """Return mean-one weights with exactly equal total mass per subject."""
    counts = Counter(subjects)
    raw = [1 / counts[subject] for subject in subjects]
    scale = len(raw) / sum(raw)
    return [value * scale for value in raw]


def fit_balanced_ranker(groups, seed):
    import torch

    values, mask, targets, mean, std = padded(groups)
    weights = torch.tensor(
        equal_subject_weights([row["subject"] for row in groups]), dtype=torch.float32
    )
    ranker = Ranker(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=.01, weight_decay=.1)
    loss = None
    for _ in range(EPOCH):
        optimizer.zero_grad(set_to_none=True)
        scores = ranker(values).masked_fill(~mask, -float("inf"))
        per_group = torch.nn.functional.cross_entropy(scores, targets, reduction="none")
        loss = (per_group * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        optimizer.step()
    return {
        "state": copy.deepcopy(ranker.model.state_dict()),
        "mean": mean,
        "std": std,
        "dim": values.shape[-1],
        "seed": seed,
        "epoch": EPOCH,
        "train_loss": float(loss.detach()),
        "subject_equal_loss": True,
    }


def fit_balanced_gate(rows, seed):
    import torch

    x = torch.stack([row["feature"] for row in rows])
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    weights = torch.tensor(
        equal_subject_weights([row["subject"] for row in rows]), dtype=torch.float32
    )
    total = weights.sum()
    mean = (x * weights[:, None]).sum(0) / total
    variance = ((x - mean).square() * weights[:, None]).sum(0) / total
    std = variance.sqrt().clamp_min(1e-4)
    positive = (weights * y).sum().clamp_min(1e-6)
    negative = (weights * (1 - y)).sum().clamp_min(1e-6)
    pos_weight = (negative / positive).sqrt()
    gate = SparseGate(x.shape[-1], seed)
    optimizer = torch.optim.AdamW(gate.model.parameters(), lr=.01, weight_decay=.1)
    loss = None
    for _ in range(EPOCH):
        optimizer.zero_grad(set_to_none=True)
        logits = gate((x - mean) / std)
        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction="none"
        )
        loss = (per_row * weights).sum() / total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.model.parameters(), 1.)
        optimizer.step()
    return {
        "state": copy.deepcopy(gate.model.state_dict()),
        "mean": mean,
        "std": std,
        "dim": x.shape[-1],
        "epoch": EPOCH,
        "seed": seed,
        "train_loss": float(loss.detach()),
        "subject_equal_loss": True,
        "weighted_pos_weight": float(pos_weight),
    }


def lower_confidence(values):
    if not values:
        raise ValueError("empty subject values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return max(0., mean - Z_ONE_SIDED_95 * standard_error)


def cluster_metrics(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["subject"]].append(row)
    pair, safety = [], []
    subject_rows = {}
    for subject, rows in grouped.items():
        metric = aggregate(rows)
        pair_value = metric["pair_state_accuracy"]
        safety_value = min(
            metric["current_retention"],
            metric["non_target_retention"],
            metric["by_event_kind"]["hard_retain"]["accuracy"],
        )
        pair.append(pair_value)
        safety.append(safety_value)
        subject_rows[subject] = {"episodes": len(rows), "pair": pair_value,
                                 "safety": safety_value}
    return {
        "subjects": len(grouped),
        "pair_macro": statistics.fmean(pair),
        "pair_lcb95": lower_confidence(pair),
        "safety_macro": statistics.fmean(safety),
        "safety_lcb95": lower_confidence(safety),
        "minimum_subject_pair": min(pair),
        "minimum_subject_safety": min(safety),
        "subject_rows": subject_rows,
    }


def candidate_metrics(rows):
    positives = [row for row in rows if row["label"]]
    grouped = defaultdict(list)
    for row in positives:
        grouped[row["subject"]].append(row)
    subject_top1 = [statistics.fmean(row["candidate_correct"] for row in values)
                    for values in grouped.values()]
    subject_shortlist = [statistics.fmean(row["shortlist_correct"] for row in values)
                         for values in grouped.values()]
    return {
        "positives": len(positives),
        "subjects": len(grouped),
        "shortlist_recall": statistics.fmean(row["shortlist_correct"] for row in positives),
        "candidate_top1": statistics.fmean(row["candidate_correct"] for row in positives),
        "subject_macro_shortlist": statistics.fmean(subject_shortlist),
        "subject_macro_top1": statistics.fmean(subject_top1),
        "minimum_subject_shortlist": min(subject_shortlist),
        "minimum_subject_top1": min(subject_top1),
    }


def risk_threshold(artifact, episodes, evidence, resolver):
    rows = []
    for threshold in RISK_THRESHOLDS:
        records = evaluate(
            episodes, evidence, "controller", resolver, Controller(artifact, threshold)
        )
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
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        selected = max(
            feasible_rows,
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
    grid = {str(row["threshold"]): {
        "metric": row["metric"], "cluster": row["cluster"],
        "feasible": row["feasible"]} for row in rows}
    return selected["threshold"], grid, bool(feasible_rows)


def release_gate(metric, candidate, cluster, fold_metrics, comparison):
    checks = {
        "shortlist_recall_at_least_99": candidate["shortlist_recall"] >= .99,
        "candidate_top1_at_least_99": candidate["candidate_top1"] >= .99,
        "candidate_subject_macro_top1_at_least_97": candidate["subject_macro_top1"] >= .97,
        "current_retention_at_least_99": metric["current_retention"] >= .99,
        "non_target_retention_at_least_97": metric["non_target_retention"] >= .97,
        "hard_retain_at_least_97": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97,
        "cluster_safety_lcb_at_least_95": cluster["safety_lcb95"] >= .95,
        "stale_deletion_at_least_90": metric["stale_deletion"] >= .90,
        "pair_state_at_least_90": metric["pair_state_accuracy"] >= .90,
        "subject_macro_pair_at_least_85": cluster["pair_macro"] >= .85,
        "minimum_fold_pair_at_least_85": min(
            row["pair_state_accuracy"] for row in fold_metrics.values()
        ) >= .85,
        "random_gain_lower_bound_above_zero": comparison["bootstrap_95ci"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


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
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(
        source / "transforms.pt", map_location="cpu", weights_only=False
    )["transforms"]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scopes.values())),
             "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
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
    all_groups = build_groups(episodes, scope_matrices, index, scopes, "train_tune")
    groups_by_id = {row["episode_id"]: row for row in all_groups}
    folds, fold_loads = subject_folds(episodes)

    fold_details = {}
    oof_records, oof_candidates = [], []
    candidate_checkpoints, gate_checkpoints = {}, {}
    for fold in range(FOLDS):
        validation = [row for row in episodes if folds[row["primary_subject_hash"]] == fold]
        calibration = [row for row in episodes
                       if folds[row["primary_subject_hash"]] == (fold + 1) % FOLDS]
        fit_subjects = {subject for subject, value in folds.items()
                        if value not in {fold, (fold + 1) % FOLDS}}
        fit_episodes = [row for row in episodes
                        if row["primary_subject_hash"] in fit_subjects]
        fit_groups = [groups_by_id[row["episode_id"]] for row in fit_episodes]
        candidate_artifacts = {
            str(seed): fit_balanced_ranker(fit_groups, seed + 100 * fold)
            for seed in CANDIDATE_SEEDS
        }
        candidate_checkpoints[str(fold)] = candidate_artifacts
        resolver = WideAnchorResolver(evidence, list(candidate_artifacts.values()))
        prepare_episodes(resolver, episodes)
        fit_rows = oracle_rows(fit_episodes, evidence, resolver)
        validation_rows = oracle_rows(validation, evidence, resolver)
        members, model_details = [], {}
        for seed in GATE_SEEDS:
            artifact = fit_balanced_gate(fit_rows, seed + 100 * fold)
            threshold, grid, feasible = risk_threshold(
                artifact, calibration, evidence, resolver
            )
            members.append((artifact, threshold))
            model_details[str(seed)] = {
                "threshold": threshold,
                "calibration_feasible": feasible,
                "grid": grid,
            }
        gate_checkpoints[str(fold)] = members
        controller = CalibratedEnsemble(members)
        records = evaluate(
            validation, evidence, "controller", resolver, controller,
            seed=37100 + fold,
        )
        fold_details[str(fold)] = {
            "models": model_details,
            "validation": aggregate(records),
            "validation_cluster": cluster_metrics(records),
            "validation_candidate": candidate_metrics(validation_rows),
            "validation_subjects": len({row["subject"] for row in records}),
            "validation_episodes": len(validation),
            "calibration_episodes": len(calibration),
            "fit_episodes": len(fit_episodes),
            "fit_rows": len(fit_rows),
        }
        oof_records.extend(records)
        oof_candidates.extend(validation_rows)
        print(f"stage=fold_complete fold={fold}", flush=True)
        del resolver, fit_rows, validation_rows

    metric = aggregate(oof_records)
    cluster = cluster_metrics(oof_records)
    candidate = candidate_metrics(oof_candidates)
    deletion_count = sum(
        decision["deleted"] is not None
        for row in oof_records for decision in row["decisions"]
    )
    schedules = matched_random_schedules(episodes, deletion_count, 37190)
    random_records = evaluate(
        episodes, evidence, "random_matched", schedules=schedules, seed=37191
    )
    comparison = paired_bootstrap(oof_records, random_records, 37192)
    fold_metrics = {
        fold: details["validation"] for fold, details in fold_details.items()
    }
    gate = release_gate(metric, candidate, cluster, fold_metrics, comparison)
    (root / "oof_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof_records)
    )
    (root / "oof_candidate_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in compact(oof_candidates))
    )
    (root / "oof_random_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in random_records)
    )

    summary = {
        "protocol": PROTOCOL,
        "candidate_policy": "nested_subject_balanced_anchor_16",
        "oof": metric,
        "oof_cluster": cluster,
        "oof_candidate": candidate,
        "oof_vs_random": comparison,
        "fold_metrics": fold_metrics,
        "oof_gate": gate,
        "v10_development_loaded": False,
        "mquake_cf_used": False,
        "longmemeval_used": False,
        "locomo_scored": False,
        "next_stage": "run_registered_v12_ablations" if gate["passed"]
                      else "stop_feature_family_or_add_relation_explicit_scope",
        "terminal_status": "completed_mquake_subject_risk_v12",
    }
    checkpoint = root / "mquake_subject_risk_crossfit.pt"
    torch.save({"folds": folds, "candidate_members": candidate_checkpoints,
                "gate_members": gate_checkpoints}, checkpoint)
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "feature_dim": next(iter(gate_checkpoints["0"]))[0]["dim"],
        "fold_loads": fold_loads,
        "folds": fold_details,
        "embedding_hashes": embedding_hashes,
        "matched_random_deletions": deletion_count,
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
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "task_source_sha256": sha(Path(__file__)),
        "folds": FOLDS,
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "gate_seeds": list(GATE_SEEDS),
        "candidate_epoch": EPOCH,
        "gate_epoch": EPOCH,
        "shortlist": SHORTLIST,
        "risk_thresholds": list(RISK_THRESHOLDS),
        "safety_macro_floor": SAFETY_MACRO_FLOOR,
        "safety_lcb_floor": SAFETY_LCB_FLOOR,
        "one_sided_z": Z_ONE_SIDED_95,
        "pair_cache_limit": PAIR_CACHE_LIMIT,
        "nested_train_calibrate_validate": True,
        "candidate_subject_equal_loss": True,
        "gate_subject_equal_loss": True,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
