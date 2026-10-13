"""Train a causal high-specificity gate over frozen factorized scope retrieval."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

from tasks.build_long_session_sparse_corpus import LENGTHS, UNITS_PER_SESSION
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_factorized_scope_candidate_v3 import SEEDS as CANDIDATE_SEEDS, scope_key
from tasks.train_long_session_candidate_v2 import Ranker, build_groups, fit_snapshots as fit_rankers
from tasks.train_long_session_sparse_gate_v1 import (
    EMBEDDING_SEEDS,
    PAIR_LEVELS,
    Controller,
    aggregate,
    development_gate as _unused_development_gate,
    embedding_matrices,
    fit_snapshots,
    load_data,
    load_texts,
    matched_random_schedules,
    paired_bootstrap,
)
from tasks.train_zsre_forgetting_controller import lexical_features
from tasks.train_zsre_forgetting_controller_v2 import relation_tokens, set_similarity


PROTOCOL = "factorized_causal_forgetting_v4"
GATE_SEEDS = (31101, 31102, 31103)
EPOCHS = (10, 25, 50, 100)
THRESHOLDS = (.50, .60, .70, .75, .80, .85, .90, .925, .95, .975, .99, .995)
SCOPE_THRESHOLDS = tuple(value / 40 for value in range(20, 40))
SHORTLIST = 4


def payload(text):
    marker = "\nValue:"
    if marker not in text:
        return None
    value = text.split(marker, 1)[1].strip()
    return value or None


class PairEvidence:
    def __init__(self, scope_matrices, value_embeddings, index, full_texts, scope_texts,
                 cache_limit=400_000):
        import torch

        self.scope_matrices = scope_matrices
        self.value_embeddings = value_embeddings
        self.index = index; self.full_texts = full_texts; self.scope_texts = scope_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.score_matrices = {key: value.to(device) for key, value in scope_matrices.items()}
        if cache_limit <= 0:
            raise ValueError("cache_limit must be positive")
        self.episode_scores, self.cache = {}, OrderedDict()
        self.cache_limit = int(cache_limit)
        self.hits = self.misses = self.evictions = 0

    def prepare_episode(self, episode):
        import torch

        identifier = episode["episode_id"]
        if identifier in self.episode_scores:
            return self.episode_scores[identifier]
        device = next(iter(self.score_matrices.values())).device
        indices = torch.tensor([[self.index[value] for value in row["units"]]
                                for row in episode["sessions"]], device=device)
        values = []
        for seed in EMBEDDING_SEEDS:
            embedded = self.score_matrices[str(seed)][indices]
            values.append(torch.einsum("aud,bvd->abuv", embedded, embedded).amax(dim=(2, 3)))
        result = torch.stack(values).mean(0).cpu()
        self.episode_scores[identifier] = result
        return result

    def pair(self, left_units, right_units):
        import torch

        key = (tuple(left_units), tuple(right_units))
        if key in self.cache:
            self.hits += 1; self.cache.move_to_end(key); return self.cache[key]
        self.misses += 1
        left_indices = [self.index[value] for value in left_units]
        right_indices = [self.index[value] for value in right_units]
        matrices, maxima, argmax = [], [], []
        seed_summary = []
        for seed in EMBEDDING_SEEDS:
            values = self.scope_matrices[str(seed)][left_indices] \
                @ self.scope_matrices[str(seed)][right_indices].T
            matrices.append(values); flat = values.flatten(); ordered = flat.sort(descending=True).values
            maxima.append(float(ordered[0])); argmax.append(int(flat.argmax()))
            seed_summary.extend((float(ordered[0]), float(ordered[:2].mean()),
                                 float(ordered[:4].mean()), float(flat.mean()),
                                 float(flat.std(unbiased=False))))
        mean_matrix = torch.stack(matrices).mean(0); flat = mean_matrix.flatten()
        ordered = flat.sort(descending=True).values; best = int(flat.argmax())
        left, right = divmod(best, mean_matrix.shape[1])
        left_scope = self.scope_texts[left_units[left]]; right_scope = self.scope_texts[right_units[right]]
        scope_feature = [
            *seed_summary, statistics.fmean(maxima), statistics.pstdev(maxima),
            float(len(set(argmax)) == 1), float(ordered[0] - ordered[1]),
            *(float((mean_matrix >= level).float().mean()) for level in PAIR_LEVELS),
            *lexical_features(left_scope, right_scope),
            *set_similarity(relation_tokens(left_scope), relation_tokens(right_scope)),
            len(left_units) / UNITS_PER_SESSION, len(right_units) / UNITS_PER_SESSION,
        ]
        left_value = payload(self.full_texts[left_units[left]])
        right_value = payload(self.full_texts[right_units[right]])
        both = left_value is not None and right_value is not None
        if both:
            cosines = [float(self.value_embeddings[str(seed)].get("value", left_value)
                             @ self.value_embeddings[str(seed)].get("value", right_value))
                       for seed in EMBEDDING_SEEDS]
            lexical = lexical_features(left_value, right_value)
        else:
            cosines, lexical = [0.] * len(EMBEDDING_SEEDS), (0.,) * 6
        payload_feature = [*cosines, statistics.fmean(cosines), statistics.pstdev(cosines),
                           *lexical, float(both), float(both and left_value == right_value)]
        result = {"scope_feature": torch.tensor(scope_feature, dtype=torch.float32),
                  "payload_feature": torch.tensor(payload_feature, dtype=torch.float32),
                  "scope_score": statistics.fmean(maxima),
                  "best_left": left_units[left], "best_right": right_units[right]}
        self.cache[key] = result
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False); self.evictions += 1
        return result


class FrozenResolver:
    def __init__(self, evidence, artifact):
        self.evidence, self.artifact = evidence, artifact
        self.ranker = Ranker(artifact["dim"], artifact["seed"])
        self.ranker.model.load_state_dict(artifact["state"]); self.ranker.model.eval()

    def resolve(self, memory, incoming, step, length, episode_scores):
        import torch

        positions = [row["position"] for row in memory]
        scope_scores = episode_scores[torch.tensor(positions), step]
        shortlist = sorted(range(len(memory)), key=lambda i: (-float(scope_scores[i]), i))[:SHORTLIST]
        pairs = [self.evidence.pair(memory[i]["units"], incoming["units"]) for i in shortlist]
        features = torch.stack([row["scope_feature"] for row in pairs])
        normalized = (features - self.artifact["mean"]) / self.artifact["std"]
        with torch.no_grad(): ranker_scores = self.ranker(normalized)
        order = sorted(range(len(shortlist)), key=lambda i: (-float(ranker_scores[i]), shortlist[i]))
        selected_local = order[0]; selected = shortlist[selected_local]
        ranker_second = float(ranker_scores[order[1]]) if len(order) > 1 else -1.
        scope_order = sorted(range(len(memory)), key=lambda i: (-float(scope_scores[i]), i))
        scope_second = float(scope_scores[scope_order[1]]) if len(scope_order) > 1 else -1.
        pair = pairs[selected_local]
        gate_feature = torch.cat((pair["scope_feature"], torch.tensor([
            float(ranker_scores[selected_local]), float(ranker_scores[selected_local]) - ranker_second,
            float(scope_scores[scope_order[0]]), float(scope_scores[scope_order[0]]) - scope_second,
        ]), pair["payload_feature"]))
        return selected, gate_feature, {
            "shortlist_positions": [memory[i]["position"] for i in shortlist],
            "selected_position": memory[selected]["position"],
            "scope_score": float(scope_scores[selected]),
        }


def oracle_rows(episodes, evidence, resolver):
    rows = []
    for episode in episodes:
        episode_scores = evidence.prepare_episode(episode); memory = []
        for step, event in enumerate(episode["sessions"]):
            if memory:
                selected, feature, meta = resolver.resolve(
                    memory, event, step, episode["length"], episode_scores)
                target = event["target_session_id"]
                target_position = next((row["position"] for row in memory
                                        if row["session_id"] == target), None)
                label = float(event["desired"] == "local")
                rows.append({"episode_id": episode["episode_id"],
                             "subject": episode["primary_subject_hash"], "length": episode["length"],
                             "step": step, "event_kind": event["event_kind"], "label": label,
                             "feature": feature,
                             "candidate_correct": int(not label or memory[selected]["session_id"] == target),
                             "shortlist_correct": int(not label or target_position in meta["shortlist_positions"])})
            if event["desired"] == "local":
                memory = [row for row in memory if row["session_id"] != event["target_session_id"]]
            memory.append({**event, "position": step})
    return rows


def replay(episode, evidence, policy, resolver=None, controller=None, threshold=None,
           random_schedule=None, random_seed=0):
    memory, decisions = [], []
    episode_scores = evidence.prepare_episode(episode) if policy in {"controller", "scope"} else None
    old_id = next(row["session_id"] for row in episode["sessions"] if row["event_kind"] == "primary_old")
    current_id = next(row["session_id"] for row in episode["sessions"] if row["event_kind"] == "primary_update")
    rng = random.Random(random_seed)
    for step, event in enumerate(episode["sessions"]):
        deleted = None; probability = 0.; selected_id = None
        if memory:
            should_delete = False; selected = 0
            if policy == "controller":
                selected, feature, _ = resolver.resolve(
                    memory, event, step, episode["length"], episode_scores)
                selected_id = memory[selected]["session_id"]
                probability = controller.probability(feature)
                should_delete = probability >= controller.threshold
            elif policy == "scope":
                positions = [row["position"] for row in memory]
                scores = episode_scores[positions, step]
                selected = int(scores.argmax()); selected_id = memory[selected]["session_id"]
                probability = float(scores[selected]); should_delete = probability >= threshold
            elif policy == "random_matched":
                should_delete = bool(random_schedule[step])
                if should_delete: selected = rng.randrange(len(memory)); selected_id = memory[selected]["session_id"]
            elif policy == "oracle":
                matches = [i for i, row in enumerate(memory)
                           if row["session_id"] == event["target_session_id"]]
                should_delete = bool(matches); selected = matches[0] if matches else 0
                selected_id = memory[selected]["session_id"] if matches else None
            if should_delete: deleted = memory.pop(selected)["session_id"]
        correct = (event["desired"] == "retain" and deleted is None) or (
            event["desired"] == "local" and deleted == event["target_session_id"])
        decisions.append({"step": step, "desired": event["desired"],
                          "event_kind": event["event_kind"], "deleted": deleted,
                          "top": selected_id, "correct": int(correct), "probability": probability})
        memory.append({**event, "position": step})
    retained = {row["session_id"] for row in memory}
    non_target = {row["session_id"] for row in episode["sessions"]} - {old_id, current_id}
    return {"episode_id": episode["episode_id"], "subject": episode["primary_subject_hash"],
            "length": episode["length"], "policy": policy, "decisions": decisions,
            "pair_state_correct": int(old_id not in retained and current_id in retained),
            "stale_deleted": int(old_id not in retained), "current_retained": int(current_id in retained),
            "non_target_retention": len(non_target & retained) / len(non_target),
            "memory_fraction": len(memory) / episode["length"]}


def evaluate(episodes, evidence, policy, resolver=None, controller=None, threshold=None,
             schedules=None, seed=0):
    output = []
    for episode in episodes:
        schedule = None if schedules is None else schedules[episode["episode_id"]]
        random_seed = int(hashlib.sha256(
            f"{PROTOCOL}|{seed}|{episode['episode_id']}".encode()).hexdigest()[:16], 16)
        output.append(replay(episode, evidence, policy, resolver, controller, threshold,
                             schedule, random_seed))
    return output


def select_gate(train_rows, tune, evidence, resolver, seed):
    snapshots = fit_snapshots(train_rows, seed); candidates, grid = [], {}
    for epoch, artifact in snapshots.items():
        grid[str(epoch)] = {}
        for threshold in THRESHOLDS:
            records = evaluate(tune, evidence, "controller", resolver,
                               Controller(artifact, threshold))
            metric = aggregate(records); grid[str(epoch)][str(threshold)] = metric
            safe = (metric["current_retention"] >= .98
                    and metric["non_target_retention"] >= .97
                    and metric["by_event_kind"]["hard_retain"]["accuracy"] >= .97
                    and metric["by_event_kind"]["primary_update"]["accuracy"] >= .85)
            candidates.append((int(safe), metric["pair_state_accuracy"],
                               metric["by_event_kind"]["primary_update"]["accuracy"],
                               metric["non_target_retention"], -epoch,
                               -abs(threshold - .9), epoch, threshold))
    selected = max(candidates)
    return selected[-2], selected[-1], grid


def select_scope(tune, evidence):
    rows = []
    for threshold in SCOPE_THRESHOLDS:
        metric = aggregate(evaluate(tune, evidence, "scope", threshold=threshold))
        safe = metric["current_retention"] >= .98 and metric["non_target_retention"] >= .97
        rows.append((int(safe), metric["pair_state_accuracy"], metric["stale_deletion"],
                     -abs(threshold - .8), threshold, metric))
    selected = max(rows)
    return selected[4], {str(row[4]): row[5] for row in rows}


def mean_metrics(metrics):
    result = {key: statistics.fmean(row[key] for row in metrics.values())
              for key in ("action_accuracy", "pair_state_accuracy", "stale_deletion",
                          "current_retention", "non_target_retention", "memory_fraction")}
    result["by_event_kind"] = {kind: {"accuracy": statistics.fmean(
        row["by_event_kind"][kind]["accuracy"] for row in metrics.values())}
        for kind in ("primary_old", "hard_retain", "distractor", "primary_update")}
    result["by_length"] = {str(length): {key: statistics.fmean(
        row["by_length"][str(length)][key] for row in metrics.values())
        for key in ("pair_state_accuracy", "current_retention", "non_target_retention")}
        for length in LENGTHS}
    return result


def release_gate(metric, shortlist_recall, candidate_top1, versus_v1, versus_random):
    checks = {
        "shortlist_recall_at_least_99": shortlist_recall >= .99,
        "candidate_top1_at_least_95": candidate_top1 >= .95,
        "current_retention_at_least_97": metric["current_retention"] >= .97,
        "non_target_retention_at_least_95": metric["non_target_retention"] >= .95,
        "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
        "stale_deletion_at_least_90": metric["stale_deletion"] >= .90,
        "pair_state_at_least_90": metric["pair_state_accuracy"] >= .90,
        "every_length_pair_at_least_85": min(
            row["pair_state_accuracy"] for row in metric["by_length"].values()) >= .85,
        "v1_gain_lower_bound_above_25": versus_v1["bootstrap_95ci"][0] > .25,
        "random_gain_lower_bound_above_10": versus_random["bootstrap_95ci"][0] > .10,
    }
    return {"checks": checks, "passed": all(checks.values())}


def load_v1(root):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    path = root / "dev_controller_seed28101.jsonl"
    if training["raw_rows_sha256"].get(path.name) != sha(path):
        raise ValueError("V1 baseline record hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], {
        "training_sha256": sha(root / "training.json"), "record_sha256": sha(path)}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    train, manifest = load_data(args.data_root, "train")
    tune, _ = load_data(args.data_root, "tune"); dev, _ = load_data(args.data_root, "dev")
    full_texts = load_texts(args.data_root, manifest)
    scope_texts = {identifier: scope_key(text) for identifier, text in full_texts.items()}
    payloads = sorted({value for text in full_texts.values() if (value := payload(text))})
    v1_records, v1_hashes = load_v1(args.v1_root)
    v3 = Path(args.v3_root); v3_summary = json.loads((v3 / "summary.json").read_text())
    v3_training = json.loads((v3 / "training.json").read_text())
    v3_manifest = json.loads((v3 / "training_manifest.json").read_text())
    if not v3_summary["release_gate"]["passed"] or v3_summary["next_stage"] != "preregister_factorized_sparse_gate":
        raise ValueError("V3 did not release causal gate")
    candidate_artifacts = torch.load(v3 / "scope_candidate_rankers.pt", map_location="cpu",
                                     weights_only=False)
    if v3_training["checkpoint_sha256"] != sha(v3 / "scope_candidate_rankers.pt"):
        raise ValueError("V3 checkpoint mismatch")
    source = Path(args.source_root); source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("contextual transform mismatch")
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

    # Reconstruct fit-only candidate rankers using frozen V3 epoch selections.
    train_candidate_groups = build_groups(train, scope_matrices, index, scope_texts, "candidate_train")
    tune_candidate_groups = build_groups(tune, scope_matrices, index, scope_texts, "candidate_tune")
    fit_candidate_artifacts = {}
    for candidate_seed in CANDIDATE_SEEDS:
        epoch = v3_training["selections"][str(candidate_seed)]["epoch"]
        fit_candidate_artifacts[str(candidate_seed)] = fit_rankers(
            train_candidate_groups, candidate_seed, epoch)[epoch]

    selections, fit_rows = {}, {}
    for candidate_seed, gate_seed in zip(CANDIDATE_SEEDS, GATE_SEEDS):
        resolver = FrozenResolver(evidence, fit_candidate_artifacts[str(candidate_seed)])
        rows = oracle_rows(train, evidence, resolver); fit_rows[str(gate_seed)] = rows
        epoch, threshold, grid = select_gate(rows, tune, evidence, resolver, gate_seed)
        selections[str(gate_seed)] = {"candidate_seed": candidate_seed, "epoch": epoch,
                                     "threshold": threshold, "tune_grid": grid}
        print(f"stage=selection_complete gate_seed={gate_seed}", flush=True)
    scope_threshold, scope_grid = select_scope(tune, evidence)

    seed_records, seed_metrics, gate_artifacts = {}, {}, {}
    dev_oracle_rows = {}
    for candidate_seed, gate_seed in zip(CANDIDATE_SEEDS, GATE_SEEDS):
        resolver = FrozenResolver(evidence, candidate_artifacts[str(candidate_seed)])
        combined_rows = oracle_rows(train + tune, evidence, resolver)
        selected = selections[str(gate_seed)]
        artifact = fit_snapshots(combined_rows, gate_seed + 1000, selected["epoch"])[selected["epoch"]]
        controller = Controller(artifact, selected["threshold"])
        records = evaluate(dev, evidence, "controller", resolver, controller, seed=gate_seed)
        seed_records[str(gate_seed)] = records; seed_metrics[str(gate_seed)] = aggregate(records)
        gate_artifacts[str(gate_seed)] = artifact
        dev_oracle_rows[str(gate_seed)] = oracle_rows(dev, evidence, resolver)
        (root / f"dev_controller_seed{gate_seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))

    primary = seed_records[str(GATE_SEEDS[0])]
    deletion_count = sum(decision["deleted"] is not None for row in primary for decision in row["decisions"])
    schedules = matched_random_schedules(dev, deletion_count, 31201)
    controls = {
        "retain_all": evaluate(dev, evidence, "retain_all"),
        "scope": evaluate(dev, evidence, "scope", threshold=scope_threshold),
        "random_matched": evaluate(dev, evidence, "random_matched", schedules=schedules, seed=31202),
        "oracle": evaluate(dev, evidence, "oracle"),
    }
    for name, records in controls.items():
        (root / f"dev_{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    mean = mean_metrics(seed_metrics)
    primary_oracle = dev_oracle_rows[str(GATE_SEEDS[0])]
    candidate_path = root / "dev_candidate_rows_seed31101.jsonl"
    candidate_path.write_text("".join(json.dumps({key: row[key] for key in (
        "episode_id", "subject", "length", "step", "event_kind", "label",
        "candidate_correct", "shortlist_correct"
    )}) + "\n" for row in primary_oracle))
    positives = [row for row in primary_oracle if row["label"]]
    shortlist_recall = statistics.fmean(row["shortlist_correct"] for row in positives)
    candidate_top1 = statistics.fmean(row["candidate_correct"] for row in positives)
    versus_v1 = paired_bootstrap(primary, v1_records, 31301)
    versus_random = paired_bootstrap(primary, controls["random_matched"], 31302)
    gate = release_gate(mean, shortlist_recall, candidate_top1, versus_v1, versus_random)
    checkpoint = root / "factorized_sparse_gates.pt"; torch.save(gate_artifacts, checkpoint)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    summary = {"protocol": PROTOCOL, "shortlist_recall": shortlist_recall,
               "candidate_top1": candidate_top1, "controller": mean,
               "controller_per_seed": seed_metrics,
               "controls": {name: aggregate(rows) for name, rows in controls.items()},
               "primary_vs_v1": versus_v1, "primary_vs_random": versus_random,
               "release_gate": gate, "longmemeval_used": False, "locomo_scored": False,
               "next_stage": "preregister_locomo" if gate["passed"] else "stop_factorized_gate_v4",
               "terminal_status": "completed_factorized_causal_forgetting_v4"}
    training = {"seconds": time.perf_counter() - started,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "feature_dim": dev_oracle_rows[str(GATE_SEEDS[0])][0]["feature"].numel(),
                "selections": selections, "scope_threshold": scope_threshold,
                "scope_grid": scope_grid, "embedding_hashes": embedding_hashes,
                "v1_hashes": v1_hashes,
                "v3_hashes": {"summary": sha(v3 / "summary.json"),
                              "training": sha(v3 / "training.json"),
                              "manifest": sha(v3 / "training_manifest.json"),
                              "checkpoint": sha(v3 / "scope_candidate_rankers.pt")},
                "evidence_cache": {"entries": len(evidence.cache), "hits": evidence.hits,
                                   "misses": evidence.misses, "evictions": evidence.evictions,
                                   "episode_tables": len(evidence.episode_scores)},
                "random_matched_deletions": deletion_count,
                "raw_rows_sha256": raw_hashes, "checkpoint_sha256": sha(checkpoint)}
    training_manifest = {"protocol": PROTOCOL,
                         "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
                         "source_summary_sha256": sha(source / "summary.json"),
                         "source_training_sha256": sha(source / "training.json"),
                         "source_transforms_sha256": sha(source / "transforms.pt"),
                         "model": args.model, "tokenizer": args.tokenizer,
                         "loading_info": model_meta["loading_info"],
                         "git_commit": subprocess.check_output(
                             ["git", "rev-parse", "HEAD"], text=True).strip(),
                         "candidate_seeds": list(CANDIDATE_SEEDS), "gate_seeds": list(GATE_SEEDS),
                         "epochs": list(EPOCHS), "thresholds": list(THRESHOLDS),
                         "shortlist": SHORTLIST, "payload_excluded_from_candidate_ranking": True,
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
