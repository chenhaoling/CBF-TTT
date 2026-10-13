"""Train and evaluate a sparse long-session deletion gate on constructed ZsRE."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

from tasks.build_long_session_sparse_corpus import (
    LENGTHS,
    PROTOCOL as DATA_PROTOCOL,
    UNITS_PER_SESSION,
)
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_forgetting_controller import lexical_features
from tasks.train_zsre_forgetting_controller_v2 import relation_tokens, set_similarity


PROTOCOL = "long_session_sparse_gate_v1"
SEEDS = (28101, 28102, 28103)
EMBEDDING_SEEDS = (5101, 5102, 5103)
EPOCHS = (10, 25, 50, 100)
THRESHOLDS = tuple(value / 20 for value in range(1, 20))
COSINE_THRESHOLDS = tuple(value / 20 for value in range(6, 20))
PAIR_LEVELS = (.3, .5, .7, .9)
RANK = 16


def load_data(root, split):
    root = Path(root); manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("protocol") != DATA_PROTOCOL or manifest.get("longmemeval_used") \
            or manifest.get("locomo_used") or not manifest.get("subject_disjoint"):
        raise ValueError("invalid sparse corpus manifest")
    path = root / f"{split}.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("sparse corpus split hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def load_texts(root, manifest):
    path = Path(root) / "texts.json"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("sparse corpus text hash mismatch")
    return json.loads(path.read_text())


def embedding_matrices(embeddings, text_ids, texts):
    import torch

    ordered = sorted(text_ids); index = {value: position for position, value in enumerate(ordered)}
    matrices = {}
    for seed in EMBEDDING_SEEDS:
        table = embeddings[str(seed)]
        matrices[str(seed)] = torch.stack([table.get("entity", texts[value]) for value in ordered])
    return matrices, index


class Evidence:
    def __init__(self, matrices, index, texts):
        import torch

        self.matrices, self.index, self.texts = matrices, index, texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.score_matrices = {key: value.to(device) for key, value in matrices.items()}
        self.cache, self.episode_scores = OrderedDict(), {}
        self.cache_hits = self.cache_misses = self.cache_evictions = 0

    def prepare_episode(self, episode):
        """Precompute all session-pair scores once for repeated causal replays."""
        import torch

        identifier = episode["episode_id"]
        if identifier in self.episode_scores:
            return self.episode_scores[identifier]
        device = next(iter(self.score_matrices.values())).device
        indices = torch.tensor([
            [self.index[value] for value in session["units"]]
            for session in episode["sessions"]
        ], dtype=torch.long, device=device)
        per_seed = []
        for seed in EMBEDDING_SEEDS:
            values = self.score_matrices[str(seed)][indices]
            similarities = torch.einsum("aud,bvd->abuv", values, values)
            per_seed.append(similarities.amax(dim=(2, 3)))
        scores = torch.stack(per_seed).mean(0).cpu()
        self.episode_scores[identifier] = scores
        return scores

    def pair(self, left_units, right_units):
        import torch

        key = (tuple(left_units), tuple(right_units))
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.cache_misses += 1
        left = [self.index[value] for value in left_units]
        right = [self.index[value] for value in right_units]
        score_matrices, maxima, argmax = [], [], []
        for seed in EMBEDDING_SEEDS:
            values = self.matrices[str(seed)][left] @ self.matrices[str(seed)][right].T
            score_matrices.append(values); maxima.append(float(values.max()))
            argmax.append(int(values.argmax()))
        mean_matrix = torch.stack(score_matrices).mean(0); flat = mean_matrix.flatten()
        best = int(flat.argmax()); best_left, best_right = divmod(best, mean_matrix.shape[1])
        agreement = len(set(argmax)) == 1
        fractions = [float((mean_matrix >= level).float().mean()) for level in PAIR_LEVELS]
        left_text, right_text = self.texts[left_units[best_left]], self.texts[right_units[best_right]]
        base = [*maxima, statistics.fmean(maxima), statistics.pstdev(maxima),
                float(agreement), *fractions, *lexical_features(left_text, right_text),
                *set_similarity(relation_tokens(left_text), relation_tokens(right_text)),
                len(left_units) / UNITS_PER_SESSION, len(right_units) / UNITS_PER_SESSION]
        result = {"score": statistics.fmean(maxima), "base": base,
                  "best_left": left_units[best_left], "best_right": right_units[best_right]}
        self.cache[key] = result
        if len(self.cache) > 250_000:
            self.cache.popitem(last=False); self.cache_evictions += 1
        return result


    def top(self, memory, incoming, step, episode_length, episode_scores):
        import torch

        positions = torch.tensor([row["position"] for row in memory], dtype=torch.long)
        scores = episode_scores[positions, step]
        top = int(scores.argmax())
        if len(memory) > 1:
            remaining = scores.clone(); remaining[top] = -float("inf")
            second = float(remaining.max())
        else:
            second = -1.
        pair = self.pair(memory[top]["units"], incoming["units"])
        age = (step - memory[top]["position"]) / episode_length
        feature = torch.tensor([*pair["base"], pair["score"] - second,
                                age, math.log1p(len(memory)) / math.log(65.)], dtype=torch.float32)
        return top, feature, pair


def oracle_rows(episodes, evidence):
    rows = []
    for episode in episodes:
        episode_scores = evidence.prepare_episode(episode)
        memory = []
        for step, event in enumerate(episode["sessions"]):
            if memory:
                top, feature, pair = evidence.top(
                    memory, event, step, episode["length"], episode_scores)
                label = float(event["desired"] == "local")
                target = event["target_session_id"]
                rows.append({"episode_id": episode["episode_id"],
                             "subject": episode["primary_subject_hash"],
                             "length": episode["length"], "step": step,
                             "event_kind": event["event_kind"], "label": label,
                             "feature": feature, "pair_score": pair["score"],
                             "candidate_correct": int(not label or memory[top]["session_id"] == target)})
            if event["desired"] == "local":
                memory = [row for row in memory if row["session_id"] != event["target_session_id"]]
            memory.append({**event, "position": step})
    return rows


class SparseGate:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(dim, RANK), torch.nn.Tanh(),
                                         torch.nn.Linear(RANK, 1))
        torch.nn.init.normal_(self.model[0].weight, std=.02); torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.normal_(self.model[2].weight, std=.02); torch.nn.init.zeros_(self.model[2].bias)

    def __call__(self, values): return self.model(values).squeeze(-1)


def fit_snapshots(rows, seed, max_epoch=None):
    import torch

    x = torch.stack([row["feature"] for row in rows]); y = torch.tensor([row["label"] for row in rows])
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-4)
    model = SparseGate(x.shape[-1], seed); optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=.01, weight_decay=.1)
    positive = y.sum().clamp_min(1); negative = (1 - y).sum().clamp_min(1)
    pos_weight = (negative / positive).sqrt()
    epochs = max_epoch or max(EPOCHS); snapshots = {}
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = model((x - mean) / std)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.); optimizer.step()
        if epoch in EPOCHS or epoch == epochs:
            snapshots[epoch] = {"state": copy.deepcopy(model.model.state_dict()),
                                "mean": mean, "std": std, "dim": x.shape[-1],
                                "epoch": epoch, "seed": seed}
    return snapshots


class Controller:
    def __init__(self, artifact, threshold):
        self.artifact, self.threshold = artifact, threshold
        self.gate = SparseGate(artifact["dim"], artifact["seed"])
        self.gate.model.load_state_dict(artifact["state"]); self.gate.model.eval()

    def probability(self, feature):
        import torch
        with torch.no_grad():
            return float(self.gate(((feature - self.artifact["mean"])
                                    / self.artifact["std"]).unsqueeze(0)).sigmoid()[0])


def replay_episode(episode, evidence, policy, controller=None, cosine_threshold=None,
                   random_schedule=None, random_seed=0):
    memory, decisions = [], []
    episode_scores = evidence.prepare_episode(episode) if policy in {
        "fixed_top", "cosine", "controller"
    } else None
    old_id = next(row["session_id"] for row in episode["sessions"]
                  if row["event_kind"] == "primary_old")
    current_id = next(row["session_id"] for row in episode["sessions"]
                      if row["event_kind"] == "primary_update")
    rng = random.Random(random_seed)
    for step, event in enumerate(episode["sessions"]):
        deleted = None; probability = 0.; top_id = None
        if memory:
            should_delete = False
            top = 0
            if policy in {"fixed_top", "cosine", "controller"}:
                top, feature, pair = evidence.top(
                    memory, event, step, episode["length"], episode_scores)
                top_id = memory[top]["session_id"]
            if policy == "fixed_top":
                should_delete = True
            elif policy == "cosine":
                should_delete = pair["score"] >= cosine_threshold
            elif policy == "controller":
                probability = controller.probability(feature); should_delete = probability >= controller.threshold
            elif policy == "random_matched":
                should_delete = bool(random_schedule[step])
                if should_delete: top = rng.randrange(len(memory)); top_id = memory[top]["session_id"]
            elif policy == "oracle":
                matches = [index for index, row in enumerate(memory)
                           if row["session_id"] == event["target_session_id"]]
                should_delete = bool(matches); top = matches[0] if matches else top
                top_id = memory[top]["session_id"] if matches else top_id
            if should_delete:
                deleted = memory.pop(top)["session_id"]
        correct = (event["desired"] == "retain" and deleted is None) or (
            event["desired"] == "local" and deleted == event["target_session_id"])
        decisions.append({"step": step, "desired": event["desired"], "event_kind": event["event_kind"],
                          "deleted": deleted, "top": top_id, "correct": int(correct),
                          "probability": probability})
        memory.append({**event, "position": step})
    retained = {row["session_id"] for row in memory}
    non_target = {row["session_id"] for row in episode["sessions"]} - {old_id, current_id}
    return {"episode_id": episode["episode_id"], "subject": episode["primary_subject_hash"],
            "length": episode["length"], "policy": policy, "decisions": decisions,
            "pair_state_correct": int(old_id not in retained and current_id in retained),
            "stale_deleted": int(old_id not in retained), "current_retained": int(current_id in retained),
            "non_target_retention": len(non_target & retained) / len(non_target),
            "memory_fraction": len(memory) / episode["length"]}


def aggregate(records):
    decisions = [decision for row in records for decision in row["decisions"]]
    by_kind = {}
    for kind in ("primary_old", "hard_retain", "distractor", "primary_update"):
        selected = [row for row in decisions if row["event_kind"] == kind]
        by_kind[kind] = {"events": len(selected),
                         "accuracy": statistics.fmean(row["correct"] for row in selected)}
    result = {"episodes": len(records), "actions": len(decisions),
              "action_accuracy": statistics.fmean(row["correct"] for row in decisions),
              "pair_state_accuracy": statistics.fmean(row["pair_state_correct"] for row in records),
              "stale_deletion": statistics.fmean(row["stale_deleted"] for row in records),
              "current_retention": statistics.fmean(row["current_retained"] for row in records),
              "non_target_retention": statistics.fmean(row["non_target_retention"] for row in records),
              "memory_fraction": statistics.fmean(row["memory_fraction"] for row in records),
              "by_event_kind": by_kind, "by_length": {}}
    for length in LENGTHS:
        selected = [row for row in records if row["length"] == length]
        if not selected:
            continue
        result["by_length"][str(length)] = {
            "episodes": len(selected),
            "pair_state_accuracy": statistics.fmean(row["pair_state_correct"] for row in selected),
            "current_retention": statistics.fmean(row["current_retained"] for row in selected),
            "non_target_retention": statistics.fmean(row["non_target_retention"] for row in selected)}
    return result


def evaluate(episodes, evidence, policy, controller=None, cosine_threshold=None, seed=0,
             schedules=None):
    output = []
    for episode in episodes:
        schedule = None if schedules is None else schedules[episode["episode_id"]]
        output.append(replay_episode(episode, evidence, policy, controller, cosine_threshold,
                                     schedule, int(hashlib.sha256(
                                         f"{PROTOCOL}|{seed}|{episode['episode_id']}".encode()
                                     ).hexdigest()[:16], 16)))
    return output


def matched_random_schedules(episodes, deletion_count, seed):
    """Choose an exact matched number of deletion times without reading labels."""
    candidates = []
    for episode in episodes:
        for step in range(1, episode["length"]):
            score = hashlib.sha256(
                f"{PROTOCOL}|matched-random|{seed}|{episode['episode_id']}|{step}".encode()
            ).hexdigest()
            candidates.append((score, episode["episode_id"], step))
    if deletion_count > len(candidates):
        raise ValueError("matched random deletion count exceeds eligible events")
    chosen = {(identifier, step) for _, identifier, step in sorted(candidates)[:deletion_count]}
    return {episode["episode_id"]: [
        (episode["episode_id"], step) in chosen for step in range(episode["length"])
    ] for episode in episodes}


def select_controller(train_rows, tune_episodes, evidence, seed):
    snapshots = fit_snapshots(train_rows, seed); candidates, grid = [], {}
    for epoch, artifact in snapshots.items():
        grid[str(epoch)] = {}
        for threshold in THRESHOLDS:
            controller = Controller(artifact, threshold)
            records = evaluate(tune_episodes, evidence, "controller", controller)
            metric = aggregate(records); grid[str(epoch)][str(threshold)] = metric
            constraints = (metric["by_event_kind"]["primary_update"]["accuracy"] >= .85
                           and metric["current_retention"] >= .95
                           and metric["non_target_retention"] >= .95)
            candidates.append((int(constraints), metric["pair_state_accuracy"],
                               metric["by_event_kind"]["primary_update"]["accuracy"],
                               metric["non_target_retention"], -epoch, -abs(threshold - .5),
                               epoch, threshold))
    selected = max(candidates)
    return selected[-2], selected[-1], grid


def select_cosine(tune_episodes, evidence):
    rows = []
    for threshold in COSINE_THRESHOLDS:
        metric = aggregate(evaluate(tune_episodes, evidence, "cosine", cosine_threshold=threshold))
        constraints = metric["current_retention"] >= .95 and metric["non_target_retention"] >= .95
        rows.append((int(constraints), metric["pair_state_accuracy"], metric["stale_deletion"],
                     -abs(threshold - .5), threshold, metric))
    selected = max(rows)
    return selected[4], {str(row[4]): row[5] for row in rows}


def paired_bootstrap(left, right, seed, draws=10000):
    left_by = {row["episode_id"]: row for row in left}; right_by = {row["episode_id"]: row for row in right}
    subjects = sorted({row["subject"] for row in left})
    by_subject = {subject: [identifier for identifier, row in left_by.items() if row["subject"] == subject]
                  for subject in subjects}
    rng = random.Random(seed); values = []
    for _ in range(draws):
        chosen = [subjects[rng.randrange(len(subjects))] for _ in subjects]
        diffs = [left_by[identifier]["pair_state_correct"] - right_by[identifier]["pair_state_correct"]
                 for subject in chosen for identifier in by_subject[subject]]
        values.append(statistics.fmean(diffs))
    values.sort(); gain = statistics.fmean(row["pair_state_correct"] for row in left) \
        - statistics.fmean(row["pair_state_correct"] for row in right)
    return {"gain": gain, "subject_clusters": len(subjects),
            "bootstrap_95ci": [values[int(.025 * draws)], values[int(.975 * draws) - 1]]}


def development_gate(metric, candidate_accuracy, comparison):
    checks = {"candidate_top1_at_least_95": candidate_accuracy >= .95,
              "current_retention_at_least_95": metric["current_retention"] >= .95,
              "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
              "stale_deletion_at_least_90": metric["stale_deletion"] >= .90,
              "pair_state_at_least_90": metric["pair_state_accuracy"] >= .90,
              "every_length_pair_at_least_85": min(
                  row["pair_state_accuracy"] for row in metric["by_length"].values()) >= .85,
              "beats_random_by_five_points_confidently": comparison["bootstrap_95ci"][0] > .05}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, manifest = load_data(args.data_root, "train"); tune, _ = load_data(args.data_root, "tune")
    dev, _ = load_data(args.data_root, "dev"); texts = load_texts(args.data_root, manifest)
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    print("stage=load_model", flush=True)
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    names = sorted(texts.values()); raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]},
                                                  args.batch_size)
    print(f"stage=embedded_texts count={len(names)}", flush=True)
    embeddings, embedding_hashes = apply(raw, {"entity": names, "value": ["unknown"]}, transforms,
                                         source_summary["selected_config"]["regularizer"])
    matrices, index = embedding_matrices(embeddings, texts, texts); evidence = Evidence(matrices, index, texts)
    del model, raw; torch.cuda.empty_cache()

    print("stage=oracle_rows split=train", flush=True)
    train_rows = oracle_rows(train, evidence); tune_rows = oracle_rows(tune, evidence)
    print("stage=oracle_rows split=tune complete", flush=True)
    cosine_threshold, cosine_grid = select_cosine(tune, evidence)
    selections, artifacts, seed_records, seed_metrics = {}, {}, {}, {}
    combined_rows = train_rows + tune_rows
    for seed in SEEDS:
        epoch, threshold, grid = select_controller(train_rows, tune, evidence, seed)
        artifact = fit_snapshots(combined_rows, seed + 1000, epoch)[epoch]
        controller = Controller(artifact, threshold)
        records = evaluate(dev, evidence, "controller", controller, seed=seed)
        selections[str(seed)] = {"epoch": epoch, "threshold": threshold, "tune_grid": grid}
        artifacts[str(seed)] = artifact; seed_records[str(seed)] = records
        seed_metrics[str(seed)] = aggregate(records)
        path = root / f"dev_controller_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"stage=controller_complete seed={seed}", flush=True)

    primary_seed = str(SEEDS[0]); primary = seed_records[primary_seed]
    deletion_count = sum(decision["deleted"] is not None
                         for row in primary for decision in row["decisions"])
    schedules = matched_random_schedules(dev, deletion_count, 28191)
    controls = {
        "retain_all": evaluate(dev, evidence, "retain_all"),
        "fixed_top": evaluate(dev, evidence, "fixed_top"),
        "cosine": evaluate(dev, evidence, "cosine", cosine_threshold=cosine_threshold),
        "random_matched": evaluate(dev, evidence, "random_matched", seed=28191, schedules=schedules),
        "oracle": evaluate(dev, evidence, "oracle"),
    }
    for name, records in controls.items():
        (root / f"dev_{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    dev_rows = oracle_rows(dev, evidence)
    candidate_path = root / "dev_candidate_rows.jsonl"
    candidate_path.write_text("".join(json.dumps({key: row[key] for key in (
        "episode_id", "subject", "length", "step", "event_kind", "label",
        "pair_score", "candidate_correct"
    )}) + "\n" for row in dev_rows))
    candidate_accuracy = statistics.fmean(
        row["candidate_correct"] for row in dev_rows if row["label"])
    mean_metric = {}
    for key in ("action_accuracy", "pair_state_accuracy", "stale_deletion", "current_retention",
                "non_target_retention", "memory_fraction"):
        mean_metric[key] = statistics.fmean(metric[key] for metric in seed_metrics.values())
    mean_metric["by_event_kind"] = {kind: {"accuracy": statistics.fmean(
        metric["by_event_kind"][kind]["accuracy"] for metric in seed_metrics.values())}
        for kind in ("primary_old", "hard_retain", "distractor", "primary_update")}
    mean_metric["by_length"] = {str(length): {key: statistics.fmean(
        metric["by_length"][str(length)][key] for metric in seed_metrics.values())
        for key in ("pair_state_accuracy", "current_retention", "non_target_retention")}
        for length in LENGTHS}
    comparison = paired_bootstrap(primary, controls["random_matched"], 28201)
    gate = development_gate(mean_metric, candidate_accuracy, comparison)
    checkpoint = root / "sparse_session_gates.pt"; torch.save(artifacts, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    summary = {"protocol": PROTOCOL, "data_status": "constructed_subject_disjoint_development",
               "candidate_top1": candidate_accuracy, "controller": mean_metric,
               "controller_per_seed": seed_metrics,
               "controls": {name: aggregate(records) for name, records in controls.items()},
               "controller_vs_random": comparison, "development_gate": gate,
               "longmemeval_used_for_fit_or_selection": False, "locomo_scored": False,
               "next_stage": "preregister_locomo" if gate["passed"] else "stop_sparse_session_gate_v1",
               "terminal_status": "completed_long_session_sparse_gate_v1"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": train_rows[0]["feature"].numel(), "selections": selections,
                "cosine_threshold": cosine_threshold, "cosine_grid": cosine_grid,
                "embedding_hashes": embedding_hashes, "evidence_cache_entries": len(evidence.cache),
                "evidence_episode_score_tables": len(evidence.episode_scores),
                "evidence_cache_hits": evidence.cache_hits,
                "evidence_cache_misses": evidence.cache_misses,
                "evidence_cache_evictions": evidence.cache_evictions,
                "random_matched_deletions": deletion_count,
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint)}
    training_manifest = {"protocol": PROTOCOL, "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                         "source_summary_sha256": sha(source / "summary.json"),
                         "source_training_sha256": sha(source / "training.json"),
                         "source_transforms_sha256": sha(source / "transforms.pt"),
                         "model": args.model, "tokenizer": args.tokenizer,
                         "git_commit": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], text=True).strip(),
                         "loading_info": model_meta["loading_info"], "seeds": list(SEEDS),
                         "embedding_seeds": list(EMBEDDING_SEEDS), "epochs": list(EPOCHS),
                         "thresholds": list(THRESHOLDS), "rank": RANK,
                         "longmemeval_used": False, "locomo_used": False,
                         "chain_identity_model_feature": False}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", training_manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__": main()
