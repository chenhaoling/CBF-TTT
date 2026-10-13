"""Train the factorized prompt-scope long-session candidate resolver."""

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
from tasks.train_long_session_candidate_v2 import (
    CONTROLS,
    EPOCHS,
    RANK,
    bootstrap,
    build_groups,
    control,
    evaluate_artifact,
    fit_snapshots,
    mean_metric,
    select_epoch,
)
from tasks.train_long_session_sparse_gate_v1 import (
    EMBEDDING_SEEDS,
    embedding_matrices,
    load_data,
    load_texts,
)


PROTOCOL = "factorized_scope_candidate_v3"
SEEDS = (30101, 30102, 30103)


def scope_key(text):
    value = text.split("\nValue:", 1)[0]
    prefix = "Memory statement: "
    if not value.startswith(prefix):
        raise ValueError("unexpected sparse-corpus unit serialization")
    value = value[len(prefix):].strip()
    if not value:
        raise ValueError("empty scope key")
    return value


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_v2(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    if summary.get("protocol") != "long_session_candidate_v2" \
            or summary.get("release_gate", {}).get("passed") \
            or summary.get("next_stage") != "stop_candidate_v2":
        raise ValueError("unexpected V2 baseline status")
    paths = {"learned": root / "dev_learned_seed29101.jsonl",
             "full_max": root / "dev_max_mean.jsonl"}
    if any(training["raw_rows_sha256"].get(path.name) != sha(path) for path in paths.values()):
        raise ValueError("V2 baseline record hash mismatch")
    return {key: read_jsonl(path) for key, path in paths.items()}, {
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "record_sha256": {key: sha(path) for key, path in paths.items()},
    }


def release_gate(mean, per_seed, versus_v2, scope_max, full_max):
    checks = {
        "mean_top1_at_least_95": mean["top1"] >= .95,
        "minimum_seed_top1_at_least_93": min(row["top1"] for row in per_seed.values()) >= .93,
        "every_length_mean_top1_at_least_93": min(
            row["top1"] for row in mean["by_length"].values()) >= .93,
        "mean_mrr_at_least_97": mean["mrr"] >= .97,
        "gain_over_v2_lower_bound_above_15": versus_v2["bootstrap_95ci"][0] > .15,
        "scope_max_beats_full_max_by_30": scope_max["top1"] - full_max["top1"] >= .30,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, manifest = load_data(args.data_root, "train")
    tune, _ = load_data(args.data_root, "tune"); dev, _ = load_data(args.data_root, "dev")
    full_texts = load_texts(args.data_root, manifest)
    scope_texts = {identifier: scope_key(text) for identifier, text in full_texts.items()}
    v2_records, v2_hashes = load_v2(args.v2_root)
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    print("stage=load_model", flush=True)
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = sorted(set(scope_texts.values()))
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms,
        source_summary["selected_config"]["regularizer"])
    matrices, index = embedding_matrices(embeddings, scope_texts, scope_texts)
    del model, raw; torch.cuda.empty_cache(); print("stage=scope_embeddings_complete", flush=True)

    train_groups = build_groups(train, matrices, index, scope_texts, "train")
    tune_groups = build_groups(tune, matrices, index, scope_texts, "tune")
    combined = train_groups + tune_groups; selections = {}
    for seed in SEEDS:
        epoch, grid = select_epoch(train_groups, tune_groups, seed)
        selections[str(seed)] = {"epoch": epoch, "tune_grid": grid}
        print(f"stage=selection_complete seed={seed} epoch={epoch}", flush=True)

    dev_groups = build_groups(dev, matrices, index, scope_texts, "dev")
    artifacts, seed_metrics, seed_records = {}, {}, {}
    for seed in SEEDS:
        epoch = selections[str(seed)]["epoch"]
        artifact = fit_snapshots(combined, seed + 1000, epoch)[epoch]
        metric, records = evaluate_artifact(dev_groups, artifact, f"scope_learned_seed{seed}")
        artifacts[str(seed)] = artifact; seed_metrics[str(seed)] = metric
        seed_records[str(seed)] = records
        (root / f"dev_scope_learned_seed{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))

    controls, control_records = {}, {}
    for name in CONTROLS:
        metric, records = control(dev_groups, name)
        controls[f"scope_{name}"] = metric; control_records[name] = records
        (root / f"dev_scope_{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
    mean = mean_metric(seed_metrics)
    versus_v2 = bootstrap(seed_records[str(SEEDS[0])], v2_records["learned"], 30201)
    scope_max, _ = control(dev_groups, "max_mean")
    full_max_top1 = statistics.fmean(row["correct"] for row in v2_records["full_max"])
    full_max = {"top1": full_max_top1}
    gate = release_gate(mean, seed_metrics, versus_v2, scope_max, full_max)
    checkpoint = root / "scope_candidate_rankers.pt"; torch.save(artifacts, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    summary = {
        "protocol": PROTOCOL,
        "scope_serialization": "prompt_before_value_delimiter",
        "learned": mean,
        "learned_per_seed": seed_metrics,
        "controls": controls,
        "full_text_v1_max_top1": full_max_top1,
        "primary_vs_v2_learned": versus_v2,
        "release_gate": gate,
        "longmemeval_used": False,
        "locomo_scored": False,
        "next_stage": "preregister_factorized_sparse_gate" if gate["passed"] else "stop_candidate_v3",
        "terminal_status": "completed_factorized_scope_candidate_v3",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "feature_dim": train_groups[0]["features"].shape[-1],
        "unique_scope_keys": len(names),
        "selections": selections,
        "embedding_hashes": embedding_hashes,
        "v2_hashes": v2_hashes,
        "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": sha(checkpoint),
    }
    training_manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "source_summary_sha256": sha(source / "summary.json"),
        "source_training_sha256": sha(source / "training.json"),
        "source_transforms_sha256": sha(source / "transforms.pt"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": list(SEEDS), "embedding_seeds": list(EMBEDDING_SEEDS),
        "epochs": list(EPOCHS), "rank": RANK,
        "payload_excluded_from_candidate_key": True,
        "candidate_age_feature": False, "unit_position_feature": False,
        "longmemeval_used": False, "locomo_used": False,
    }
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", training_manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--v2-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
