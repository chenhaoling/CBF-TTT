"""Diagnose MQuAKE tune candidate ranks under frozen V8 scope resolvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_factorized_scope_candidate_v3 import scope_key
from tasks.train_long_session_candidate_v2 import Ranker, build_groups
from tasks.train_long_session_sparse_gate_v1 import embedding_matrices
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_candidate_diagnostic_v9"
WIDTHS = (4, 8, 16, 32)


def rank(values, target):
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return order.index(target) + 1


def record_metrics(records, key, recall_key=None):
    result = {"episodes": len(records),
              "top1": statistics.fmean(row[key] == 1 for row in records),
              "mrr": statistics.fmean(1 / row[key] for row in records),
              "mean_rank": statistics.fmean(row[key] for row in records)}
    if recall_key is not None:
        result["target_recall"] = statistics.fmean(row[recall_key] for row in records)
    result["by_length"] = {str(length): {
        "episodes": len(selected := [row for row in records if row["length"] == length]),
        "top1": statistics.fmean(row[key] == 1 for row in selected),
        **({"target_recall": statistics.fmean(row[recall_key] for row in selected)}
           if recall_key is not None else {})}
        for length in (16, 32, 48, 64)}
    return result


def select_arm(metrics):
    eligible = []
    for name, row in metrics.items():
        recall = row.get("target_recall", 1.)
        if row["top1"] >= .80 and recall >= .90:
            if name == "learned_full":
                width, preference = 10_000, 0
            else:
                kind, width_text = name.rsplit("_", 1)
                width = int(width_text); preference = int(kind == "anchor")
            eligible.append((row["top1"], -width, preference, name))
    selected = max(eligible)[-1] if eligible else None
    return {"eligible": sorted(row[-1] for row in eligible), "selected": selected,
            "passed": selected is not None}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    tune, data_manifest = load_split(args.data_root, "tune")
    texts = load_texts(args.data_root, data_manifest)
    scopes = {identifier: scope_key(text) for identifier, text in texts.items()}
    source = Path(args.source_root)
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    v8 = Path(args.v8_root); v8_training = json.loads((v8 / "training.json").read_text())
    checkpoint = v8 / "mquake_adapted_forgetting.pt"
    if v8_training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("V8 checkpoint mismatch")
    candidate_artifacts = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "candidate_artifacts"]

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = sorted(set(scopes.values()))
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms,
        source_summary["selected_config"]["regularizer"])
    matrices, index = embedding_matrices(embeddings, scopes, scopes)
    del model, raw; torch.cuda.empty_cache()
    groups = build_groups(tune, matrices, index, scopes, "tune")

    rankers = []
    for artifact in candidate_artifacts.values():
        ranker_model = Ranker(artifact["dim"], artifact["seed"])
        ranker_model.model.load_state_dict(artifact["state"]); ranker_model.model.eval()
        rankers.append((ranker_model, artifact))
    learned_scores = []
    with torch.no_grad():
        for row in groups:
            scores = [ranker_model(
                (row["features"] - artifact["mean"]) / artifact["std"])
                for ranker_model, artifact in rankers]
            learned_scores.append(torch.stack(scores).mean(0))

    records = []
    for row, learned in zip(groups, learned_scores):
        target = row["target"]
        controls = {name: row["control_scores"][name]
                    for name in ("max_mean", "anchor_only")}
        output = {"episode_id": row["episode_id"], "subject": row["subject"],
                  "length": row["length"], "candidates": len(learned),
                  "rank_learned_full": rank(learned, target),
                  "rank_max_mean_raw": rank(controls["max_mean"], target),
                  "rank_anchor_raw": rank(controls["anchor_only"], target)}
        for kind, scores in (("max", controls["max_mean"]),
                             ("anchor", controls["anchor_only"])):
            base_order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
            for width in WIDTHS:
                shortlist = base_order[:min(width, len(base_order))]
                output[f"recall_{kind}_{width}"] = int(target in shortlist)
                predicted = max(shortlist, key=lambda i: (float(learned[i]), -i))
                output[f"rank_{kind}_{width}"] = 1 if predicted == target else 2
        records.append(output)
    (root / "tune_candidate_diagnostic.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records))
    metrics = {"learned_full": record_metrics(records, "rank_learned_full")}
    controls = {"max_mean_raw": record_metrics(records, "rank_max_mean_raw"),
                "anchor_raw": record_metrics(records, "rank_anchor_raw")}
    for kind in ("max", "anchor"):
        for width in WIDTHS:
            metrics[f"{kind}_{width}"] = record_metrics(
                records, f"rank_{kind}_{width}", f"recall_{kind}_{width}")
    gate = select_arm(metrics)
    summary = {"protocol": PROTOCOL, "split": "tune", "metrics": metrics,
               "controls": controls, "release_gate": gate,
               "development_loaded": False, "mquake_cf_used": False,
               "longmemeval_used": False, "locomo_scored": False,
               "next_stage": ("train_causal_with_selected_candidate" if gate["passed"]
                              else "redesign_memory_scope_key"),
               "terminal_status": "completed_mquake_candidate_diagnostic_v9"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "embedding_hashes": embedding_hashes,
                "raw_rows_sha256": {"tune_candidate_diagnostic.jsonl": sha(
                    root / "tune_candidate_diagnostic.jsonl")},
                "v8_hashes": {"summary": sha(v8 / "summary.json"),
                              "training": sha(v8 / "training.json"),
                              "checkpoint": sha(checkpoint)}}
    manifest = {"protocol": PROTOCOL,
                "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                "source_summary_sha256": sha(source / "summary.json"),
                "source_training_sha256": sha(source / "training.json"),
                "source_transforms_sha256": sha(source / "transforms.pt"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "widths": list(WIDTHS), "explicit_scope_anchor": True,
                "development_loaded": False, "mquake_cf_used": False,
                "longmemeval_used": False, "locomo_used": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v8-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
