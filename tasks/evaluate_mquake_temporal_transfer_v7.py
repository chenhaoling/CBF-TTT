"""Evaluate frozen ZsRE factorized forgetting on entity-disjoint MQuAKE-T dev."""

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
from tasks.train_crossfit_calibrated_forgetting_v5 import CalibratedEnsemble, EnsembleResolver
from tasks.train_factorized_causal_forgetting_v4 import (
    PairEvidence,
    aggregate,
    evaluate,
    oracle_rows,
    paired_bootstrap,
    payload,
)
from tasks.train_factorized_scope_candidate_v3 import SEEDS as CANDIDATE_SEEDS, scope_key
from tasks.train_long_session_sparse_gate_v1 import embedding_matrices, matched_random_schedules


PROTOCOL = "mquake_temporal_transfer_v7"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_dev(root):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL or not manifest.get("subject_disjoint")
            or not manifest.get("real_temporal_updates") or manifest.get("locomo_used")
            or manifest.get("longmemeval_used") or manifest.get("mquake_cf_used")):
        raise ValueError("invalid MQuAKE-T session manifest")
    path = root / "dev.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("MQuAKE-T dev hash mismatch")
    texts = root / "texts.json"
    if sha(texts) != manifest["files_sha256"][texts.name]:
        raise ValueError("MQuAKE-T text hash mismatch")
    return read_jsonl(path), json.loads(texts.read_text()), manifest


def transfer_gate(metric, shortlist, candidate, versus_random):
    checks = {
        "shortlist_recall_at_least_90": shortlist >= .90,
        "candidate_top1_at_least_80": candidate >= .80,
        "current_retention_at_least_97": metric["current_retention"] >= .97,
        "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
        "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
        "stale_deletion_at_least_70": metric["stale_deletion"] >= .70,
        "pair_state_at_least_70": metric["pair_state_accuracy"] >= .70,
        "random_gain_lower_bound_above_zero": versus_random["bootstrap_95ci"][0] > 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    dev, full_texts, data_manifest = load_dev(args.data_root)
    scope_texts = {identifier: scope_key(text) for identifier, text in full_texts.items()}
    payloads = sorted({value for text in full_texts.values() if (value := payload(text))})

    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]

    v3 = Path(args.v3_root); v5 = Path(args.v5_root)
    v3_summary = json.loads((v3 / "summary.json").read_text())
    v3_training = json.loads((v3 / "training.json").read_text())
    if not v3_summary["release_gate"]["passed"]:
        raise ValueError("V3 resolver was not released")
    candidate_artifacts = torch.load(v3 / "scope_candidate_rankers.pt", map_location="cpu",
                                     weights_only=False)
    if v3_training["checkpoint_sha256"] != sha(v3 / "scope_candidate_rankers.pt"):
        raise ValueError("V3 checkpoint mismatch")
    v5_summary = json.loads((v5 / "summary.json").read_text())
    v5_training = json.loads((v5 / "training.json").read_text())
    if v5_summary.get("protocol") != "crossfit_calibrated_forgetting_v5":
        raise ValueError("unexpected forgetting checkpoint protocol")
    gate_artifact = torch.load(v5 / "crossfit_gates.pt", map_location="cpu", weights_only=False)
    if v5_training["checkpoint_sha256"] != sha(v5 / "crossfit_gates.pt"):
        raise ValueError("V5 checkpoint mismatch")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = {"entity": sorted(set(scope_texts.values())), "value": sorted({"unknown", *payloads})}
    raw = extract(model, tokenizer, names, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, names, transforms, source_summary["selected_config"]["regularizer"])
    scope_matrices, index = embedding_matrices(embeddings, scope_texts, scope_texts)
    del model, raw; torch.cuda.empty_cache()

    evidence = PairEvidence(scope_matrices, embeddings, index, full_texts, scope_texts)
    resolver = EnsembleResolver(
        evidence, [candidate_artifacts[str(seed)] for seed in CANDIDATE_SEEDS])
    controller = CalibratedEnsemble(gate_artifact["members"])
    candidate_rows = oracle_rows(dev, evidence, resolver)
    positives = [row for row in candidate_rows if row["label"]]
    shortlist = statistics.fmean(row["shortlist_correct"] for row in positives)
    candidate = statistics.fmean(row["candidate_correct"] for row in positives)

    records = evaluate(dev, evidence, "controller", resolver, controller, seed=33101)
    metric = aggregate(records)
    deletion_count = sum(decision["deleted"] is not None
                         for row in records for decision in row["decisions"])
    schedules = matched_random_schedules(dev, deletion_count, 33102)
    controls = {
        "retain_all": evaluate(dev, evidence, "retain_all"),
        "random_matched": evaluate(dev, evidence, "random_matched", schedules=schedules, seed=33103),
        "oracle": evaluate(dev, evidence, "oracle"),
    }
    versus_random = paired_bootstrap(records, controls["random_matched"], 33104)
    gate = transfer_gate(metric, shortlist, candidate, versus_random)

    (root / "dev_controller.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    (root / "dev_candidate_rows.jsonl").write_text("".join(json.dumps({key: row[key] for key in (
        "episode_id", "subject", "length", "step", "event_kind", "label",
        "candidate_correct", "shortlist_correct")}) + "\n" for row in candidate_rows))
    for name, rows in controls.items():
        (root / f"dev_{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))

    summary = {"protocol": PROTOCOL, "evaluation_mode": "frozen_zero_shot_transfer",
               "dataset": "MQuAKE-T", "shortlist_recall": shortlist,
               "candidate_top1": candidate, "controller": metric,
               "controls": {name: aggregate(rows) for name, rows in controls.items()},
               "primary_vs_random": versus_random, "transfer_gate": gate,
               "mquake_train_or_tune_loaded": False, "mquake_cf_used": False,
               "longmemeval_used": False, "locomo_scored": False,
               "next_stage": ("preregister_locomo_confirmation" if gate["passed"]
                              else "fit_mquake_train_calibrate_tune_keep_dev_locked"),
               "terminal_status": "completed_mquake_temporal_transfer_v7"}
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "dev_episodes": len(dev), "dev_subjects": len({row["subject"] for row in records}),
                "feature_dim": candidate_rows[0]["feature"].numel(),
                "embedding_hashes": embedding_hashes,
                "controller_deletions": deletion_count,
                "raw_rows_sha256": raw_hashes,
                "v3_hashes": {"summary": sha(v3 / "summary.json"),
                              "training": sha(v3 / "training.json"),
                              "checkpoint": sha(v3 / "scope_candidate_rankers.pt")},
                "v5_hashes": {"summary": sha(v5 / "summary.json"),
                              "training": sha(v5 / "training.json"),
                              "checkpoint": sha(v5 / "crossfit_gates.pt")}}
    manifest = {"protocol": PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "mquake_source_sha256": data_manifest["source_sha256"],
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "frozen_v3_resolver": True, "frozen_v5_controller": True,
                "threshold_tuning_on_mquake": False, "mquake_train_or_tune_loaded": False,
                "subject_disjoint_dev": True, "payload_excluded_from_candidate_ranking": True,
                "candidate_age_feature": False, "unit_position_feature": False,
                "mquake_cf_used": False, "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v3-root", required=True); parser.add_argument("--v5-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
