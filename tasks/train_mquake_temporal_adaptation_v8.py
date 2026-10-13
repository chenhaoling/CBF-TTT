"""Train MQuAKE-T factorized forgetting on train, calibrate on tune, gate dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from tasks.build_mquake_temporal_sessions import PROTOCOL as DATA_PROTOCOL
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossfit_calibrated_forgetting_v5 import (
    CalibratedEnsemble,
    EnsembleResolver,
    GATE_SEEDS,
    calibration_threshold,
)
from tasks.train_factorized_causal_forgetting_v4 import (
    PairEvidence,
    aggregate,
    evaluate,
    oracle_rows,
    paired_bootstrap,
    payload,
)
from tasks.train_factorized_scope_candidate_v3 import SEEDS as CANDIDATE_SEEDS, scope_key
from tasks.train_long_session_candidate_v2 import build_groups, fit_snapshots as fit_rankers
from tasks.train_long_session_sparse_gate_v1 import (
    embedding_matrices,
    fit_snapshots as fit_gates,
    matched_random_schedules,
)


PROTOCOL = "mquake_temporal_adaptation_v8"
EPOCH = 50


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_split(root, split):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL or not manifest.get("subject_disjoint")
            or not manifest.get("real_temporal_updates") or manifest.get("mquake_cf_used")
            or manifest.get("longmemeval_used") or manifest.get("locomo_used")):
        raise ValueError("invalid MQuAKE-T session manifest")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError(f"MQuAKE-T {split} hash mismatch")
    return read_jsonl(path), manifest


def load_texts(root, manifest):
    path = Path(root) / "texts.json"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("MQuAKE-T text hash mismatch")
    return json.loads(path.read_text())


def candidate_metrics(rows):
    positives = [row for row in rows if row["label"]]
    return {"shortlist_recall": statistics.fmean(row["shortlist_correct"] for row in positives),
            "candidate_top1": statistics.fmean(row["candidate_correct"] for row in positives)}


def tune_gate(metric, candidate):
    checks = {"shortlist_recall_at_least_90": candidate["shortlist_recall"] >= .90,
              "candidate_top1_at_least_80": candidate["candidate_top1"] >= .80,
              "current_retention_at_least_97": metric["current_retention"] >= .97,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "pair_state_at_least_70": metric["pair_state_accuracy"] >= .70}
    return {"checks": checks, "passed": all(checks.values())}


def development_gate(metric, candidate, versus_random):
    checks = {"shortlist_recall_at_least_90": candidate["shortlist_recall"] >= .90,
              "candidate_top1_at_least_85": candidate["candidate_top1"] >= .85,
              "current_retention_at_least_97": metric["current_retention"] >= .97,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "stale_deletion_at_least_80": metric["stale_deletion"] >= .80,
              "pair_state_at_least_80": metric["pair_state_accuracy"] >= .80,
              "every_length_pair_at_least_70": min(
                  row["pair_state_accuracy"] for row in metric["by_length"].values()) >= .70,
              "random_gain_lower_bound_above_zero": versus_random["bootstrap_95ci"][0] > 0}
    return {"checks": checks, "passed": all(checks.values())}


def compact_candidate_rows(rows):
    keys = ("episode_id", "subject", "length", "step", "event_kind", "label",
            "candidate_correct", "shortlist_correct")
    return [{key: row[key] for key in keys} for row in rows]


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, data_manifest = load_split(args.data_root, "train")
    tune, _ = load_split(args.data_root, "tune")
    full_texts = load_texts(args.data_root, data_manifest)
    scope_texts = {identifier: scope_key(text) for identifier, text in full_texts.items()}
    payloads = sorted({value for text in full_texts.values() if (value := payload(text))})
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]

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

    train_groups = build_groups(train, scope_matrices, index, scope_texts, "train")
    candidate_artifacts = {str(seed): fit_rankers(train_groups, seed, EPOCH)[EPOCH]
                           for seed in CANDIDATE_SEEDS}
    resolver = EnsembleResolver(
        evidence, [candidate_artifacts[str(seed)] for seed in CANDIDATE_SEEDS])
    print("stage=candidate_fit_complete", flush=True)

    train_rows = oracle_rows(train, evidence, resolver)
    tune_rows = oracle_rows(tune, evidence, resolver)
    tune_candidate = candidate_metrics(tune_rows)
    members, calibration = [], {}
    for seed in GATE_SEEDS:
        artifact = fit_gates(train_rows, seed, EPOCH)[EPOCH]
        threshold, grid, feasible = calibration_threshold(
            artifact, tune, evidence, resolver)
        members.append((artifact, threshold))
        calibration[str(seed)] = {"threshold": threshold,
                                  "calibration_feasible": feasible, "grid": grid}
        print(f"stage=gate_calibrated seed={seed} threshold={threshold}", flush=True)
    controller = CalibratedEnsemble(members)
    tune_records = evaluate(tune, evidence, "controller", resolver, controller, seed=34101)
    tune_metric = aggregate(tune_records); prerequisite = tune_gate(tune_metric, tune_candidate)
    (root / "tune_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tune_records))
    (root / "tune_candidate_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in compact_candidate_rows(tune_rows)))

    summary = {"protocol": PROTOCOL, "tune": tune_metric,
               "tune_candidate": tune_candidate, "tune_prerequisite": prerequisite,
               "development_scored": False, "mquake_cf_used": False,
               "longmemeval_used": False, "locomo_scored": False,
               "next_stage": "stop_mquake_tune",
               "terminal_status": "completed_mquake_temporal_adaptation_v8_tune"}
    deletion_count = None
    if prerequisite["passed"]:
        # Dev labels are loaded only after all train/tune models and thresholds are frozen.
        dev, _ = load_split(args.data_root, "dev")
        dev_rows = oracle_rows(dev, evidence, resolver); dev_candidate = candidate_metrics(dev_rows)
        records = evaluate(dev, evidence, "controller", resolver, controller, seed=34102)
        metric = aggregate(records)
        deletion_count = sum(decision["deleted"] is not None
                             for row in records for decision in row["decisions"])
        schedules = matched_random_schedules(dev, deletion_count, 34103)
        controls = {"retain_all": evaluate(dev, evidence, "retain_all"),
                    "random_matched": evaluate(dev, evidence, "random_matched",
                                               schedules=schedules, seed=34104),
                    "oracle": evaluate(dev, evidence, "oracle")}
        versus_random = paired_bootstrap(records, controls["random_matched"], 34105)
        gate = development_gate(metric, dev_candidate, versus_random)
        (root / "dev_controller.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
        (root / "dev_candidate_rows.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in compact_candidate_rows(dev_rows)))
        for name, rows in controls.items():
            (root / f"dev_{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
        summary.update({"development_scored": True, "development": metric,
                        "development_candidate": dev_candidate,
                        "controls": {name: aggregate(rows) for name, rows in controls.items()},
                        "primary_vs_random": versus_random, "development_gate": gate,
                        "next_stage": ("preregister_locomo_confirmation" if gate["passed"]
                                       else "stop_mquake_dev_no_retuning"),
                        "terminal_status": "completed_mquake_temporal_adaptation_v8"})

    checkpoint = root / "mquake_adapted_forgetting.pt"
    torch.save({"candidate_artifacts": candidate_artifacts, "gate_members": members}, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": train_rows[0]["feature"].numel(),
                "calibration": calibration, "embedding_hashes": embedding_hashes,
                "candidate_train_groups": len(train_groups), "gate_train_rows": len(train_rows),
                "matched_random_deletions": deletion_count,
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "mquake_source_sha256": data_manifest["source_sha256"],
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "candidate_seeds": list(CANDIDATE_SEEDS), "gate_seeds": list(GATE_SEEDS),
                "candidate_epoch": EPOCH, "gate_epoch": EPOCH,
                "refit_after_calibration": False, "development_labels_loaded_conditionally": True,
                "payload_excluded_from_candidate_ranking": True,
                "candidate_age_feature": False, "unit_position_feature": False,
                "mquake_cf_used": False, "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
