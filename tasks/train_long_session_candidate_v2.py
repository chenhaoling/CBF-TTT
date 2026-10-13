"""Train a hierarchical listwise stale-session candidate resolver."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from tasks.build_long_session_sparse_corpus import LENGTHS, UNITS_PER_SESSION
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_long_session_sparse_gate_v1 import (
    EMBEDDING_SEEDS,
    PAIR_LEVELS,
    embedding_matrices,
    load_data,
    load_texts,
)
from tasks.train_zsre_forgetting_controller import lexical_features
from tasks.train_zsre_forgetting_controller_v2 import relation_tokens, set_similarity


PROTOCOL = "long_session_candidate_v2"
SEEDS = (29101, 29102, 29103)
EPOCHS = (10, 25, 50, 100, 200)
RANK = 16
CONTROLS = ("max_mean", "top4_mean", "mean_all", "consensus_max", "anchor_only")


def update_group(episode, matrices, index, texts):
    import torch

    step = next(i for i, row in enumerate(episode["sessions"])
                if row["event_kind"] == "primary_update")
    incoming = episode["sessions"][step]
    memory = episode["sessions"][:step]
    target = next(i for i, row in enumerate(memory)
                  if row["session_id"] == incoming["target_session_id"])
    memory_indices = torch.tensor([[index[value] for value in row["units"]] for row in memory])
    incoming_indices = torch.tensor([index[value] for value in incoming["units"]])
    similarities = []
    for seed in EMBEDDING_SEEDS:
        matrix = matrices[str(seed)]
        similarities.append(torch.einsum(
            "mud,vd->muv", matrix[memory_indices], matrix[incoming_indices]))
    stacked = torch.stack(similarities)  # projection, candidate, old unit, new unit
    features, control_scores = [], {name: [] for name in CONTROLS}
    for candidate, old in enumerate(memory):
        values = stacked[:, candidate]
        seed_summary, maxima, top4 = [], [], []
        for projection in values:
            flat = projection.flatten(); ordered = flat.sort(descending=True).values
            maxima.append(float(ordered[0])); top4.append(float(ordered[:4].mean()))
            seed_summary.extend((float(ordered[0]), float(ordered[:2].mean()),
                                 float(ordered[:4].mean()), float(flat.mean()),
                                 float(flat.std(unbiased=False))))
        mean_matrix = values.mean(0); ordered = mean_matrix.flatten().sort(descending=True).values
        best = int(mean_matrix.flatten().argmax()); left, right = divmod(best, mean_matrix.shape[1])
        left_text, right_text = texts[old["units"][left]], texts[incoming["units"][right]]
        argmax = [int(value.flatten().argmax()) for value in values]
        feature = [
            *seed_summary,
            statistics.fmean(maxima),
            statistics.pstdev(maxima),
            float(len(set(argmax)) == 1),
            float(ordered[0] - ordered[1]),
            *(float((mean_matrix >= level).float().mean()) for level in PAIR_LEVELS),
            *lexical_features(left_text, right_text),
            *set_similarity(relation_tokens(left_text), relation_tokens(right_text)),
            len(old["units"]) / UNITS_PER_SESSION,
            len(incoming["units"]) / UNITS_PER_SESSION,
        ]
        features.append(torch.tensor(feature, dtype=torch.float32))
        control_scores["max_mean"].append(statistics.fmean(maxima))
        control_scores["top4_mean"].append(statistics.fmean(top4))
        control_scores["mean_all"].append(float(values.mean()))
        control_scores["consensus_max"].append(
            statistics.fmean(maxima) - statistics.pstdev(maxima))
        control_scores["anchor_only"].append(float(values[:, 0, 0].mean()))
    return {
        "episode_id": episode["episode_id"],
        "subject": episode["primary_subject_hash"],
        "length": episode["length"],
        "features": torch.stack(features),
        "target": target,
        "control_scores": {key: torch.tensor(value) for key, value in control_scores.items()},
    }


def build_groups(episodes, matrices, index, texts, split):
    groups = []
    for position, episode in enumerate(episodes, 1):
        groups.append(update_group(episode, matrices, index, texts))
        if position % 250 == 0:
            print(f"stage=groups split={split} completed={position}", flush=True)
    return groups


def padded(groups, mean=None, std=None):
    import torch

    dim = groups[0]["features"].shape[-1]
    width = max(row["features"].shape[0] for row in groups)
    values = torch.zeros(len(groups), width, dim)
    mask = torch.zeros(len(groups), width, dtype=torch.bool)
    targets = torch.tensor([row["target"] for row in groups], dtype=torch.long)
    for i, row in enumerate(groups):
        count = row["features"].shape[0]
        values[i, :count] = row["features"]; mask[i, :count] = True
    if mean is None:
        flat = values[mask]; mean = flat.mean(0); std = flat.std(0, unbiased=False).clamp_min(1e-4)
    return (values - mean) / std, mask, targets, mean, std


class Ranker:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(dim, RANK), torch.nn.Tanh(),
                                         torch.nn.Linear(RANK, 1))
        torch.nn.init.normal_(self.model[0].weight, std=.02); torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.normal_(self.model[2].weight, std=.02); torch.nn.init.zeros_(self.model[2].bias)

    def __call__(self, values): return self.model(values).squeeze(-1)


def score_metrics(groups, scores, arm):
    records = []
    for row, values in zip(groups, scores):
        values = values[:row["features"].shape[0]].tolist()
        ranking = sorted(range(len(values)), key=lambda i: (-values[i], i))
        rank = ranking.index(row["target"]) + 1
        records.append({"episode_id": row["episode_id"], "subject": row["subject"],
                        "length": row["length"], "arm": arm, "candidates": len(values),
                        "target": row["target"], "prediction": ranking[0], "rank": rank,
                        "correct": int(rank == 1), "reciprocal_rank": 1 / rank})
    metric = {"episodes": len(records),
              "top1": statistics.fmean(row["correct"] for row in records),
              "mrr": statistics.fmean(row["reciprocal_rank"] for row in records),
              "mean_rank": statistics.fmean(row["rank"] for row in records),
              "by_length": {}}
    for length in LENGTHS:
        selected = [row for row in records if row["length"] == length]
        metric["by_length"][str(length)] = {
            "episodes": len(selected),
            "top1": statistics.fmean(row["correct"] for row in selected),
            "mrr": statistics.fmean(row["reciprocal_rank"] for row in selected),
            "mean_rank": statistics.fmean(row["rank"] for row in selected),
        }
    return metric, records


def evaluate_artifact(groups, artifact, arm):
    import torch

    values, mask, _, _, _ = padded(groups, artifact["mean"], artifact["std"])
    ranker = Ranker(artifact["dim"], artifact["seed"])
    ranker.model.load_state_dict(artifact["state"]); ranker.model.eval()
    with torch.no_grad():
        scores = ranker(values); scores[~mask] = -float("inf")
    return score_metrics(groups, scores, arm)


def fit_snapshots(groups, seed, epochs=None):
    import torch

    values, mask, targets, mean, std = padded(groups)
    ranker = Ranker(values.shape[-1], seed)
    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=.01, weight_decay=.1)
    limit = epochs or max(EPOCHS); output = {}
    for epoch in range(1, limit + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = ranker(values); scores = scores.masked_fill(~mask, -float("inf"))
        loss = torch.nn.functional.cross_entropy(scores, targets)
        loss.backward(); torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        optimizer.step()
        if epoch in EPOCHS or epoch == limit:
            output[epoch] = {"state": copy.deepcopy(ranker.model.state_dict()),
                             "mean": mean, "std": std, "dim": values.shape[-1],
                             "seed": seed, "epoch": epoch, "train_loss": float(loss)}
    return output


def select_epoch(train_groups, tune_groups, seed):
    snapshots = fit_snapshots(train_groups, seed); candidates, grid = [], {}
    for epoch, artifact in snapshots.items():
        metric, _ = evaluate_artifact(tune_groups, artifact, f"tune_seed{seed}")
        grid[str(epoch)] = metric
        worst = min(row["top1"] for row in metric["by_length"].values())
        candidates.append((metric["top1"], worst, metric["mrr"], -epoch, epoch))
    return max(candidates)[-1], grid


def control(groups, name):
    scores = [row["control_scores"][name] for row in groups]
    return score_metrics(groups, scores, name)


def mean_metric(metrics):
    result = {key: statistics.fmean(row[key] for row in metrics.values())
              for key in ("top1", "mrr", "mean_rank")}
    result["by_length"] = {str(length): {key: statistics.fmean(
        row["by_length"][str(length)][key] for row in metrics.values())
        for key in ("top1", "mrr", "mean_rank")} for length in LENGTHS}
    return result


def bootstrap(left, right, seed, draws=10000):
    left_by = {row["episode_id"]: row for row in left}
    right_by = {row["episode_id"]: row for row in right}
    subjects = sorted({row["subject"] for row in left})
    grouped = {subject: [identifier for identifier, row in left_by.items()
                         if row["subject"] == subject] for subject in subjects}
    rng = random.Random(seed); values = []
    for _ in range(draws):
        selected = [subjects[rng.randrange(len(subjects))] for _ in subjects]
        values.append(statistics.fmean(
            left_by[identifier]["correct"] - right_by[identifier]["correct"]
            for subject in selected for identifier in grouped[subject]))
    values.sort()
    gain = statistics.fmean(row["correct"] for row in left) \
        - statistics.fmean(row["correct"] for row in right)
    return {"gain": gain, "subject_clusters": len(subjects),
            "bootstrap_95ci": [values[int(.025 * draws)], values[int(.975 * draws) - 1]]}


def release_gate(mean, per_seed, comparison):
    checks = {
        "mean_top1_at_least_95": mean["top1"] >= .95,
        "minimum_seed_top1_at_least_93": min(row["top1"] for row in per_seed.values()) >= .93,
        "every_length_mean_top1_at_least_93": min(
            row["top1"] for row in mean["by_length"].values()) >= .93,
        "mean_mrr_at_least_97": mean["mrr"] >= .97,
        "gain_over_v1_lower_bound_above_20": comparison["bootstrap_95ci"][0] > .20,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, manifest = load_data(args.data_root, "train")
    tune, _ = load_data(args.data_root, "tune"); dev, _ = load_data(args.data_root, "dev")
    texts = load_texts(args.data_root, manifest)
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    print("stage=load_model", flush=True)
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = sorted(texts.values())
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms,
        source_summary["selected_config"]["regularizer"])
    matrices, index = embedding_matrices(embeddings, texts, texts)
    del model, raw; torch.cuda.empty_cache(); print("stage=embeddings_complete", flush=True)

    train_groups = build_groups(train, matrices, index, texts, "train")
    tune_groups = build_groups(tune, matrices, index, texts, "tune")
    combined = train_groups + tune_groups
    selections = {}
    for seed in SEEDS:
        epoch, grid = select_epoch(train_groups, tune_groups, seed)
        selections[str(seed)] = {"epoch": epoch, "tune_grid": grid}
        print(f"stage=selection_complete seed={seed} epoch={epoch}", flush=True)

    # Development labels are materialized only after every fit/tune choice is frozen.
    dev_groups = build_groups(dev, matrices, index, texts, "dev")
    artifacts, seed_metrics, seed_records = {}, {}, {}
    for seed in SEEDS:
        epoch = selections[str(seed)]["epoch"]
        artifact = fit_snapshots(combined, seed + 1000, epoch)[epoch]
        metric, records = evaluate_artifact(dev_groups, artifact, f"learned_seed{seed}")
        artifacts[str(seed)] = artifact; seed_metrics[str(seed)] = metric
        seed_records[str(seed)] = records
        (root / f"dev_learned_seed{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
        print(f"stage=seed_complete seed={seed}", flush=True)

    controls, control_records = {}, {}
    for name in CONTROLS:
        metric, records = control(dev_groups, name)
        controls[name] = metric; control_records[name] = records
        (root / f"dev_{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
    mean = mean_metric(seed_metrics)
    comparison = bootstrap(seed_records[str(SEEDS[0])], control_records["max_mean"], 29201)
    gate = release_gate(mean, seed_metrics, comparison)
    checkpoint = root / "candidate_rankers.pt"; torch.save(artifacts, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    summary = {
        "protocol": PROTOCOL,
        "data_status": "constructed_subject_disjoint_development_v2",
        "learned": mean,
        "learned_per_seed": seed_metrics,
        "controls": controls,
        "primary_vs_v1_max_cosine": comparison,
        "release_gate": gate,
        "longmemeval_used": False,
        "locomo_scored": False,
        "next_stage": "preregister_sparse_gate_v2" if gate["passed"] else "stop_candidate_v2",
        "terminal_status": "completed_long_session_candidate_v2",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "feature_dim": train_groups[0]["features"].shape[-1],
        "selections": selections,
        "embedding_hashes": embedding_hashes,
        "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": sha(checkpoint),
    }
    training_manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "source_summary_sha256": sha(source / "summary.json"),
        "source_training_sha256": sha(source / "training.json"),
        "source_transforms_sha256": sha(source / "transforms.pt"),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seeds": list(SEEDS),
        "embedding_seeds": list(EMBEDDING_SEEDS),
        "epochs": list(EPOCHS),
        "rank": RANK,
        "candidate_age_feature": False,
        "unit_position_feature": False,
        "longmemeval_used": False,
        "locomo_used": False,
    }
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", training_manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
