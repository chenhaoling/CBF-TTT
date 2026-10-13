"""Cross-fit calibrated ensemble for factorized causal forgetting."""

from __future__ import annotations

import argparse
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
from tasks.train_factorized_causal_forgetting_v4 import (
    SHORTLIST,
    THRESHOLDS,
    PairEvidence,
    aggregate,
    embedding_matrices,
    evaluate,
    load_data,
    load_texts,
    load_v1,
    matched_random_schedules,
    oracle_rows,
    paired_bootstrap,
    payload,
    release_gate,
    scope_key,
)
from tasks.train_factorized_scope_candidate_v3 import SEEDS as CANDIDATE_SEEDS
from tasks.train_long_session_candidate_v2 import Ranker
from tasks.train_long_session_sparse_gate_v1 import Controller, fit_snapshots


PROTOCOL = "crossfit_calibrated_forgetting_v5"
GATE_SEEDS = (32101, 32102, 32103)
FOLDS = 5
EPOCH = 50


def subject_folds(episodes):
    subjects = sorted({row["primary_subject_hash"] for row in episodes},
                      key=lambda value: hashlib.sha256(f"{PROTOCOL}|{value}".encode()).hexdigest())
    return {subject: position % FOLDS for position, subject in enumerate(subjects)}


class EnsembleResolver:
    def __init__(self, evidence, artifacts):
        self.evidence, self.members = evidence, []
        for artifact in artifacts:
            ranker = Ranker(artifact["dim"], artifact["seed"])
            ranker.model.load_state_dict(artifact["state"]); ranker.model.eval()
            self.members.append((ranker, artifact))

    def resolve(self, memory, incoming, step, length, episode_scores):
        import torch

        positions = [row["position"] for row in memory]
        scope_scores = episode_scores[torch.tensor(positions), step]
        shortlist = sorted(range(len(memory)), key=lambda i: (-float(scope_scores[i]), i))[:SHORTLIST]
        pairs = [self.evidence.pair(memory[i]["units"], incoming["units"]) for i in shortlist]
        features = torch.stack([row["scope_feature"] for row in pairs])
        member_scores = []
        with torch.no_grad():
            for ranker, artifact in self.members:
                member_scores.append(ranker((features - artifact["mean"]) / artifact["std"]))
        scores = torch.stack(member_scores).mean(0)
        order = sorted(range(len(shortlist)), key=lambda i: (-float(scores[i]), shortlist[i]))
        selected = shortlist[order[0]]; pair = pairs[order[0]]
        scope_order = sorted(range(len(memory)), key=lambda i: (-float(scope_scores[i]), i))
        scope_second = float(scope_scores[scope_order[1]]) if len(scope_order) > 1 else -1.
        # Absolute listwise logits are intentionally excluded from gate calibration.
        feature = torch.cat((pair["scope_feature"], torch.tensor([
            float(scope_scores[scope_order[0]]), float(scope_scores[scope_order[0]]) - scope_second,
        ]), pair["payload_feature"]))
        return selected, feature, {
            "shortlist_positions": [memory[i]["position"] for i in shortlist],
            "selected_position": memory[selected]["position"],
            "scope_score": float(scope_scores[selected]),
        }


class CalibratedEnsemble:
    threshold = .5

    def __init__(self, members):
        self.members = [(Controller(artifact, threshold), threshold)
                        for artifact, threshold in members]

    def probability(self, feature):
        margins = []
        for controller, threshold in self.members:
            probability = min(max(controller.probability(feature), 1e-7), 1 - 1e-7)
            margins.append(math.log(probability / (1 - probability))
                           - math.log(threshold / (1 - threshold)))
        margin = statistics.median(margins)
        return 1 / (1 + math.exp(-max(-30., min(30., margin))))


def calibration_threshold(artifact, episodes, evidence, resolver):
    rows = []
    for threshold in THRESHOLDS:
        metric = aggregate(evaluate(
            episodes, evidence, "controller", resolver, Controller(artifact, threshold)))
        feasible = (metric["current_retention"] >= .98
                    and metric["non_target_retention"] >= .97
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97)
        rows.append((int(feasible), metric["pair_state_accuracy"],
                     metric["by_event_kind"]["primary_update"]["accuracy"],
                     metric["non_target_retention"], -abs(threshold - .9), threshold, metric))
    feasible = [row for row in rows if row[0]]
    selected = max(feasible) if feasible else next(row for row in rows if row[5] == max(THRESHOLDS))
    return selected[5], {str(row[5]): row[6] for row in rows}, bool(feasible)


def oof_gate(metric):
    checks = {"current_retention_at_least_97": metric["current_retention"] >= .97,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "stale_deletion_at_least_85": metric["stale_deletion"] >= .85,
              "pair_state_at_least_85": metric["pair_state_accuracy"] >= .85}
    return {"checks": checks, "passed": all(checks.values())}


def run(args, protocol=PROTOCOL, fit_gate_fn=fit_snapshots, objective="binary_bce"):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    fit, manifest = load_data(args.data_root, "train")
    tune, _ = load_data(args.data_root, "tune"); calibration = fit + tune
    full_texts = load_texts(args.data_root, manifest)
    scope_texts = {identifier: scope_key(text) for identifier, text in full_texts.items()}
    payloads = sorted({value for text in full_texts.values() if (value := payload(text))})
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    v3 = Path(args.v3_root); v3_summary = json.loads((v3 / "summary.json").read_text())
    v3_training = json.loads((v3 / "training.json").read_text())
    if not v3_summary["release_gate"]["passed"]:
        raise ValueError("V3 candidate resolver did not pass")
    candidate_artifacts = torch.load(v3 / "scope_candidate_rankers.pt", map_location="cpu",
                                     weights_only=False)
    if v3_training["checkpoint_sha256"] != sha(v3 / "scope_candidate_rankers.pt"):
        raise ValueError("V3 checkpoint mismatch")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scope_texts.values())), "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, names, transforms, source_summary["selected_config"]["regularizer"])
    scope_matrices, index = embedding_matrices(embeddings, scope_texts, scope_texts)
    del model, raw; torch.cuda.empty_cache(); print("stage=embeddings_complete", flush=True)
    evidence = PairEvidence(scope_matrices, embeddings, index, full_texts, scope_texts)
    resolver = EnsembleResolver(evidence, [candidate_artifacts[str(seed)] for seed in CANDIDATE_SEEDS])
    folds = subject_folds(calibration)
    all_rows = oracle_rows(calibration, evidence, resolver)

    members, fold_details, oof_records = [], {}, []
    for fold in range(FOLDS):
        validation = [row for row in calibration if folds[row["primary_subject_hash"]] == fold]
        calibration_fold = [row for row in calibration
                            if folds[row["primary_subject_hash"]] == (fold + 1) % FOLDS]
        train_subjects = {subject for subject, value in folds.items()
                          if value not in {fold, (fold + 1) % FOLDS}}
        train_rows = [row for row in all_rows if row["subject"] in train_subjects]
        fold_members, fold_details[str(fold)] = [], {"models": {}}
        for seed in GATE_SEEDS:
            artifact = fit_gate_fn(train_rows, seed + 100 * fold, EPOCH)[EPOCH]
            threshold, grid, feasible = calibration_threshold(
                artifact, calibration_fold, evidence, resolver)
            fold_members.append((artifact, threshold)); members.append((artifact, threshold))
            fold_details[str(fold)]["models"][str(seed)] = {
                "threshold": threshold, "calibration_feasible": feasible, "grid": grid}
        records = evaluate(validation, evidence, "controller", resolver,
                           CalibratedEnsemble(fold_members), seed=fold)
        oof_records.extend(records)
        fold_details[str(fold)]["validation"] = aggregate(records)
        print(f"stage=fold_complete fold={fold}", flush=True)
    oof_metric = aggregate(oof_records); prerequisite = oof_gate(oof_metric)
    (root / "oof_validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in oof_records))

    summary = {"protocol": protocol, "oof": oof_metric, "oof_gate": prerequisite,
               "development_scored": False, "longmemeval_used": False, "locomo_scored": False,
               "next_stage": "stop_crossfit_oof", "terminal_status": f"completed_{protocol}_oof"}
    dev_records = controls = None
    if prerequisite["passed"]:
        dev, _ = load_data(args.data_root, "dev")
        ensemble = CalibratedEnsemble(members)
        dev_records = evaluate(dev, evidence, "controller", resolver, ensemble, seed=32501)
        (root / "dev_controller.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in dev_records))
        dev_rows = oracle_rows(dev, evidence, resolver)
        (root / "dev_candidate_rows.jsonl").write_text("".join(json.dumps({key: row[key] for key in (
            "episode_id", "subject", "length", "step", "event_kind", "label",
            "candidate_correct", "shortlist_correct"
        )}) + "\n" for row in dev_rows))
        positives = [row for row in dev_rows if row["label"]]
        shortlist = statistics.fmean(row["shortlist_correct"] for row in positives)
        candidate = statistics.fmean(row["candidate_correct"] for row in positives)
        deletion_count = sum(decision["deleted"] is not None
                             for row in dev_records for decision in row["decisions"])
        schedules = matched_random_schedules(dev, deletion_count, 32502)
        controls = {"retain_all": evaluate(dev, evidence, "retain_all"),
                    "random_matched": evaluate(dev, evidence, "random_matched",
                                               schedules=schedules, seed=32503),
                    "oracle": evaluate(dev, evidence, "oracle")}
        for name, records in controls.items():
            (root / f"dev_{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records))
        v1_records, v1_hashes = load_v1(args.v1_root)
        versus_v1 = paired_bootstrap(dev_records, v1_records, 32601)
        versus_random = paired_bootstrap(dev_records, controls["random_matched"], 32602)
        dev_metric = aggregate(dev_records)
        gate = release_gate(dev_metric, shortlist, candidate, versus_v1, versus_random)
        summary.update({"development_scored": True, "development": dev_metric,
                        "shortlist_recall": shortlist, "candidate_top1": candidate,
                        "controls": {name: aggregate(rows) for name, rows in controls.items()},
                        "versus_v1": versus_v1, "versus_random": versus_random,
                        "development_gate": gate,
                        "next_stage": "preregister_locomo" if gate["passed"] else "stop_crossfit_dev",
                        "terminal_status": f"completed_{protocol}"})
    else:
        v1_hashes = None; deletion_count = None

    checkpoint = root / "crossfit_gates.pt"
    torch.save({"members": members, "folds": folds}, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": all_rows[0]["feature"].numel(), "folds": fold_details,
                "fold_assignment_sha256": hashlib.sha256(json.dumps(
                    folds, sort_keys=True).encode()).hexdigest(),
                "embedding_hashes": embedding_hashes, "v1_hashes": v1_hashes,
                "v3_hashes": {"summary": sha(v3 / "summary.json"),
                              "training": sha(v3 / "training.json"),
                              "checkpoint": sha(v3 / "scope_candidate_rankers.pt")},
                "random_matched_deletions": deletion_count,
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint)}
    training_manifest = {"protocol": protocol,
                         "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                         "source_summary_sha256": sha(source / "summary.json"),
                         "source_training_sha256": sha(source / "training.json"),
                         "source_transforms_sha256": sha(source / "transforms.pt"),
                         "model": args.model, "tokenizer": args.tokenizer,
                         "loading_info": model_meta["loading_info"],
                         "git_commit": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], text=True).strip(),
                         "folds": FOLDS, "epoch": EPOCH, "gate_seeds": list(GATE_SEEDS),
                         "objective": objective,
                         "candidate_seeds": list(CANDIDATE_SEEDS),
                         "thresholds": list(THRESHOLDS), "shortlist": SHORTLIST,
                         "nested_train_calibrate_validate": True, "refit_after_calibration": False,
                         "ranker_logits_in_gate_feature": False,
                         "payload_excluded_from_candidate_ranking": True,
                         "candidate_age_feature": False, "unit_position_feature": False,
                         "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", training_manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v1-root", required=True); parser.add_argument("--v3-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
