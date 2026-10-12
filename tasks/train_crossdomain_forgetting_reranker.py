"""Develop a residual candidate ranker and separate safe-delete gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.build_counterfact_zero_shot_forgetting import PROTOCOL as COUNTERFACT_PROTOCOL
from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as ZSRE_PROTOCOL
from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_counterfactual_forgetting_controller import load_split
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS,
    MAX_KEYS,
    oracle_event_groups,
    paired_comparison,
    pair_feature,
    replay_controller,
    summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "crossdomain_forgetting_reranker_v1"
FOLDS = 5
EPOCHS = (10, 25, 50, 100)
THRESHOLDS = tuple(index / 20 for index in range(1, 20))
COSINE_TEMPERATURE = .07


def tag_sessions(sessions, prefix):
    """Namespace observed ZsRE identities without changing any model-visible text."""
    output = copy.deepcopy(sessions)
    for session in output:
        session["id"] = f"{prefix}:{session['id']}"
        mapping = {chain["id"]: f"{prefix}:{chain['id']}" for chain in session["chains"]}
        for chain in session["chains"]:
            chain["id"] = mapping[chain["id"]]
            if "parent_chain_id" in chain:
                chain["parent_chain_id"] = mapping[chain["parent_chain_id"]]
        for event in session["events"]:
            event["chain_id"] = mapping[event["chain_id"]]
    return output


def deterministic_folds(session_ids, folds=FOLDS):
    ranked = sorted(session_ids, key=lambda value: hashlib.sha256(
        f"{PROTOCOL}|fold|{value}".encode()).hexdigest())
    return {value: index % folds for index, value in enumerate(ranked)}


def candidate_counts(groups):
    return __import__("torch").tensor([len(group["keys"]) for group in groups])


def targets(groups):
    return __import__("torch").tensor([group["target"] for group in groups])


def feature_tensor(groups, scorer):
    import torch

    rows = []
    for group in groups:
        values = [pair_feature(key, group["query"], scorer) for key in group["keys"]]
        values.extend(torch.zeros_like(values[0]) for _ in range(MAX_KEYS - len(values)))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def valid_mask(counts, width=MAX_KEYS):
    import torch

    return torch.arange(width, device=counts.device)[None, :] < counts[:, None]


def fit_normalizer(values, counts, indices):
    selected = values[indices]
    valid = valid_mask(counts[indices])
    flattened = selected[valid]
    return flattened.mean(0), flattened.std(0, unbiased=False).clamp_min(1e-4)


class ResidualRanker:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def parameters(self):
        return self.linear.parameters()

    def state_dict(self):
        return self.linear.state_dict()

    def load_state_dict(self, state):
        self.linear.load_state_dict(state)

    def logits(self, values, cosines, counts):
        scores = cosines / COSINE_TEMPERATURE + self.linear(values).squeeze(-1)
        return scores.masked_fill(~valid_mask(counts, values.shape[1]), float("-inf"))


def rank_metrics(groups, logits):
    import torch

    target = targets(groups).to(logits.device)
    local = target < MAX_KEYS
    if not local.any():
        raise ValueError("candidate metric requires local-update groups")
    selected_logits, selected_target = logits[local], target[local]
    return {
        "local_events": int(local.sum()),
        "top1_accuracy": float(selected_logits.argmax(-1).eq(selected_target).float().mean()),
        "nll": float(torch.nn.functional.cross_entropy(selected_logits, selected_target)),
    }


def cosine_values(values, pair_dim):
    return values[:, :, pair_dim:pair_dim + 3].mean(-1)


def train_ranker(groups, values, cosines, counts, train_indices, tune_indices, seed):
    import torch

    mean, std = fit_normalizer(values, counts, train_indices)
    train_groups = [groups[index] for index in train_indices]
    tune_groups = [groups[index] for index in tune_indices]
    train_target = targets(train_groups); train_local = train_target < MAX_KEYS
    device = torch.device("cuda")
    train_values = ((values[train_indices] - mean) / std).to(device)
    tune_values = ((values[tune_indices] - mean) / std).to(device)
    train_cosines = cosines[train_indices].to(device); tune_cosines = cosines[tune_indices].to(device)
    train_counts = counts[train_indices].to(device); tune_counts = counts[tune_indices].to(device)
    train_target = train_target.to(device); train_local = train_local.to(device)
    head = ResidualRanker(values.shape[-1], seed); head.linear.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    candidates, curve = [], []
    for epoch in range(1, max(EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(train_values, train_cosines, train_counts)
        loss = torch.nn.functional.cross_entropy(logits[train_local], train_target[train_local])
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
        if epoch in EPOCHS:
            with torch.no_grad():
                train_metric = rank_metrics(train_groups, head.logits(
                    train_values, train_cosines, train_counts))
                tune_metric = rank_metrics(tune_groups, head.logits(
                    tune_values, tune_cosines, tune_counts))
            state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            curve.append({"epoch": epoch, "train": train_metric, "tune": tune_metric})
            candidates.append((tune_metric["top1_accuracy"], -tune_metric["nll"],
                               -epoch, state))
    selected = max(candidates, key=lambda row: row[:3])
    head.linear.cpu(); head.load_state_dict(selected[3])
    return head, mean, std, -selected[2], curve


def gate_features(values, logits, counts):
    import torch

    top = logits.argmax(-1)
    chosen = values[torch.arange(len(values)), top]
    geometry = []
    for row, count in zip(logits, counts.tolist()):
        valid = row[:count]
        top_values = torch.topk(valid, min(2, count)).values
        margin = top_values[0] - top_values[1] if count > 1 else top_values[0] + 1
        probability = torch.softmax(valid, -1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
        geometry.append(torch.stack((top_values[0], margin, entropy,
                                     torch.tensor(float(count)).log())))
    return torch.cat((chosen, torch.stack(geometry).to(chosen.device)), dim=-1), top


def action_metrics(groups, top, probability, threshold):
    target = targets(groups).to(top.device)
    delete = probability >= threshold
    correct = (~delete & target.eq(MAX_KEYS)) | (delete & top.eq(target))
    by_kind = {}
    for kind in ("primary_initial", "hard_retain", "primary_update"):
        mask = __import__("torch").tensor([group["event_kind"] == kind for group in groups],
                                          device=top.device)
        by_kind[kind] = float(correct[mask].float().mean())
    return {"accuracy": float(correct.float().mean()), "by_event_kind": by_kind}


def train_gate(groups, raw_values, normalized_values, rank_logits, counts,
               train_indices, tune_indices, seed):
    import torch

    all_gate, all_top = gate_features(normalized_values, rank_logits, counts)
    train_gate = all_gate[train_indices]; tune_gate = all_gate[tune_indices]
    gate_mean = train_gate.mean(0); gate_std = train_gate.std(0, unbiased=False).clamp_min(1e-4)
    train_groups = [groups[index] for index in train_indices]
    tune_groups = [groups[index] for index in tune_indices]
    train_target = targets(train_groups)
    train_top = all_top[train_indices]
    # A deletion target is positive only when the ranker points to the correct old
    # memory. This prevents a high conflict score from authorizing known-wrong deletion.
    safe_delete = train_target.lt(MAX_KEYS) & train_top.eq(train_target)
    device = torch.device("cuda")
    train_x = ((train_gate - gate_mean) / gate_std).to(device)
    tune_x = ((tune_gate - gate_mean) / gate_std).to(device)
    safe_delete = safe_delete.float().to(device)
    tune_top = all_top[tune_indices].to(device)
    head = BinaryActionHead(train_x.shape[-1], seed); head.linear.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    candidates, curve = [], []
    for epoch in range(1, max(EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True); logits = head(train_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe_delete)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
        if epoch in EPOCHS:
            with torch.no_grad():
                probability = head(tune_x).sigmoid()
            grid = {threshold: action_metrics(tune_groups, tune_top, probability, threshold)
                    for threshold in THRESHOLDS}
            threshold = max(THRESHOLDS, key=lambda value: (
                grid[value]["accuracy"],
                min(grid[value]["by_event_kind"]["hard_retain"],
                    grid[value]["by_event_kind"]["primary_update"]),
                -abs(value - .5)))
            metric = grid[threshold]
            state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            curve.append({"epoch": epoch, "threshold": threshold, "tune": metric})
            candidates.append((metric["accuracy"],
                               min(metric["by_event_kind"]["hard_retain"],
                                   metric["by_event_kind"]["primary_update"]),
                               -epoch, -abs(threshold - .5), threshold, state))
    selected = max(candidates, key=lambda row: row[:4])
    head.linear.cpu(); head.load_state_dict(selected[5])
    return head, gate_mean, gate_std, selected[4], -selected[2], curve


def fit_policy(groups, values, cosines, counts, train_indices, tune_indices, pair_dim, seed):
    ranker, mean, std, rank_epoch, rank_curve = train_ranker(
        groups, values, cosines, counts, train_indices, tune_indices, seed)
    normalized = (values - mean) / std
    with __import__("torch").no_grad():
        rank_logits = ranker.logits(normalized, cosines, counts)
    gate, gate_mean, gate_std, threshold, gate_epoch, gate_curve = train_gate(
        groups, values, normalized, rank_logits, counts,
        train_indices, tune_indices, seed + 100)
    return {
        "ranker": ranker.state_dict(), "rank_mean": mean, "rank_std": std,
        "gate": gate.state_dict(), "gate_mean": gate_mean, "gate_std": gate_std,
        "threshold": threshold, "rank_epoch": rank_epoch, "gate_epoch": gate_epoch,
        "feature_dim": values.shape[-1], "pair_dim": pair_dim, "seed": seed,
        "rank_curve": rank_curve, "gate_curve": gate_curve,
    }


class DecomposedController:
    def __init__(self, scorer, artifact):
        self.scorer, self.artifact = scorer, artifact
        self.ranker = ResidualRanker(artifact["feature_dim"], artifact["seed"])
        self.ranker.load_state_dict(artifact["ranker"])
        self.gate = BinaryActionHead(artifact["feature_dim"] + 4, artifact["seed"] + 100)
        self.gate.load_state_dict(artifact["gate"])

    def predict(self, query, keys):
        import torch

        if not keys:
            return {"prediction": 0, "probability": 0.0}
        values = torch.stack([pair_feature(key, query, self.scorer) for key in keys])
        padded = values.unsqueeze(0)
        counts = torch.tensor([len(keys)])
        cosines = cosine_values(padded, self.artifact["pair_dim"])
        normalized = (padded - self.artifact["rank_mean"]) / self.artifact["rank_std"]
        with torch.no_grad():
            logits = self.ranker.logits(normalized, cosines, counts)
            gate_raw, top = gate_features(normalized, logits, counts)
            gate_value = (gate_raw - self.artifact["gate_mean"]) / self.artifact["gate_std"]
            probability = float(self.gate(gate_value).sigmoid()[0])
        prediction = int(top[0]) if probability >= self.artifact["threshold"] else len(keys)
        return {"prediction": prediction, "probability": probability}


def evaluate_records(sessions, scorer, artifact_for_session, split, root, stem):
    records_by_seed, metrics_by_seed, raw_hashes = {}, {}, {}
    for replicate, seed in enumerate(CONTROLLER_SEEDS):
        records = []
        for session in sessions:
            controller = DecomposedController(scorer, artifact_for_session(session["id"]))
            records.append(replay_controller(
                session, scorer, GATE_SEEDS[replicate], seed, controller, split=split))
        records_by_seed[str(seed)] = records; metrics_by_seed[str(seed)] = aggregate(records)
        path = root / f"{stem}_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)
    return records_by_seed, summarize_seed_metrics(metrics_by_seed), raw_hashes


def oracle_state_metrics(groups, values, cosines, counts, indices, artifact_for_group):
    import torch

    candidate_correct, action_correct = [], []
    by_kind = {kind: [] for kind in ("primary_initial", "hard_retain", "primary_update")}
    for index in indices:
        artifact = artifact_for_group(groups[index])
        ranker = ResidualRanker(artifact["feature_dim"], artifact["seed"])
        ranker.load_state_dict(artifact["ranker"])
        gate = BinaryActionHead(artifact["feature_dim"] + 4, artifact["seed"] + 100)
        gate.load_state_dict(artifact["gate"])
        row_values = values[index:index + 1]
        normalized = (row_values - artifact["rank_mean"]) / artifact["rank_std"]
        with torch.no_grad():
            logits = ranker.logits(normalized, cosines[index:index + 1], counts[index:index + 1])
            gate_raw, top = gate_features(normalized, logits, counts[index:index + 1])
            probability = float(gate((gate_raw - artifact["gate_mean"])
                                     / artifact["gate_std"]).sigmoid()[0])
        target = groups[index]["target"]; delete = probability >= artifact["threshold"]
        correct = (not delete and target == MAX_KEYS) or (delete and int(top[0]) == target)
        action_correct.append(correct); by_kind[groups[index]["event_kind"]].append(correct)
        if target < MAX_KEYS:
            candidate_correct.append(int(top[0]) == target)
    return {"candidate_top1_accuracy": statistics.fmean(candidate_correct),
            "action_accuracy": statistics.fmean(action_correct),
            "by_event_kind": {kind: {"accuracy": statistics.fmean(rows), "events": len(rows)}
                              for kind, rows in by_kind.items()}}


def load_previous(root):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    output = {}
    for seed in CONTROLLER_SEEDS:
        path = root / f"transfer_test_controller_seed{seed}.jsonl"
        if training["raw_rows_sha256"][path.name] != sha(path):
            raise ValueError("previous CounterFact controller trace hash mismatch")
        output[str(seed)] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return output, json.loads((root / "summary.json").read_text())


def development_gate(metrics, oracle_metrics, fixed_retain_query):
    checks = {
        "candidate_top1": oracle_metrics["candidate_top1_accuracy"] >= .75,
        "overall_action": metrics["mean_action_accuracy"] >= .85,
        "hard_retain": metrics["by_event_kind"]["hard_retain"]["accuracy"] >= .85,
        "primary_update": metrics["by_event_kind"]["primary_update"]["accuracy"] >= .75,
        "query_above_fixed_retain": metrics["mean_accuracy"] > fixed_retain_query,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    zsre = []
    for split in ("calibration", "dev", "test"):
        zsre.extend(tag_sessions(load_split(args.zsre_root, split), f"zsre_{split}"))
    counterfact, counterfact_manifest = load_transfer(args.counterfact_root)
    if counterfact_manifest["protocol"] != COUNTERFACT_PROTOCOL:
        raise ValueError("CounterFact protocol mismatch")
    zsre_manifest = json.loads((Path(args.zsre_root) / "manifest.json").read_text())
    if zsre_manifest["protocol"] != ZSRE_PROTOCOL:
        raise ValueError("ZsRE protocol mismatch")

    all_sessions = [*zsre, *counterfact]
    groups = oracle_event_groups(all_sessions)
    zsre_ids = {session["id"] for session in zsre}; counterfact_ids = {
        session["id"] for session in counterfact}
    zsre_train_ids = {value for value in zsre_ids if not value.startswith("zsre_test:")}
    zsre_tune_ids = zsre_ids - zsre_train_ids
    names, pairs = all_names(all_sessions)

    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms["transforms"],
        source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    values = feature_tensor(groups, scorer); counts = candidate_counts(groups)
    pair_dim = next(iter(pair_features.values())).numel(); cosines = cosine_values(values, pair_dim)
    del model; torch.cuda.empty_cache()

    zsre_train = [index for index, group in enumerate(groups)
                  if group["session_id"] in zsre_train_ids]
    zsre_tune = [index for index, group in enumerate(groups)
                 if group["session_id"] in zsre_tune_ids]
    counterfact_indices = [index for index, group in enumerate(groups)
                           if group["session_id"] in counterfact_ids]
    zero_artifact = fit_policy(
        groups, values, cosines, counts, zsre_train, zsre_tune, pair_dim, 10101)
    torch.save(zero_artifact, root / "zsre_only_policy.pt")
    zero_records, zero_metrics, raw_hashes = evaluate_records(
        counterfact, scorer, lambda _: zero_artifact, "counterfact_observed_zero_shot",
        root, "counterfact_zsre_only")
    zero_oracle = oracle_state_metrics(
        groups, values, cosines, counts, counterfact_indices, lambda _: zero_artifact)

    fold_map = deterministic_folds(counterfact_ids); fold_artifacts = {}
    for outer in range(FOLDS):
        tune_fold = (outer + 1) % FOLDS
        train_ids = zsre_ids | {value for value, fold in fold_map.items()
                                if fold not in (outer, tune_fold)}
        tune_ids = {value for value, fold in fold_map.items() if fold == tune_fold}
        train_indices = [index for index, group in enumerate(groups)
                         if group["session_id"] in train_ids]
        tune_indices = [index for index, group in enumerate(groups)
                        if group["session_id"] in tune_ids]
        artifact = fit_policy(
            groups, values, cosines, counts, train_indices, tune_indices,
            pair_dim, 10201 + outer)
        fold_artifacts[outer] = artifact
        torch.save(artifact, root / f"mixed_oof_fold{outer}.pt")
    artifact_for_session = lambda session_id: fold_artifacts[fold_map[session_id]]
    oof_records, oof_metrics, oof_hashes = evaluate_records(
        counterfact, scorer, artifact_for_session, "counterfact_observed_mixed_oof",
        root, "counterfact_mixed_oof")
    raw_hashes.update(oof_hashes)
    oof_oracle = oracle_state_metrics(
        groups, values, cosines, counts, counterfact_indices,
        lambda group: fold_artifacts[fold_map[group["session_id"]]])

    previous_records, previous_summary = load_previous(args.previous_root)
    comparison = paired_comparison(oof_records, previous_records, 10301)
    fixed_retain_query = previous_summary["policies"]["fixed_retain"]["mean_accuracy"]
    gate = development_gate(oof_metrics, oof_oracle, fixed_retain_query)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    summary = {
        "protocol": PROTOCOL,
        "data_status": "observed_development_only",
        "zsre_only_transfer": {"metrics": zero_metrics, "oracle_state": zero_oracle},
        "mixed_counterfact_oof": {"metrics": oof_metrics, "oracle_state": oof_oracle},
        "previous_frozen_controller": previous_summary["policies"]["learned"],
        "fixed_retain": previous_summary["policies"]["fixed_retain"],
        "mixed_oof_vs_previous": comparison,
        "development_gate": gate,
        "terminal_status": "completed_crossdomain_reranker_development",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "sessions": {"zsre": len(zsre), "counterfact": len(counterfact)},
        "groups": {"total": len(groups), "counterfact": len(counterfact_indices)},
        "unique_texts": len(names), "unique_pairs": len(pairs),
        "feature_dim": values.shape[-1], "pair_dim": pair_dim,
        "counterfact_folds": fold_map,
        "zero_shot_selected": {key: zero_artifact[key] for key in
                               ("rank_epoch", "gate_epoch", "threshold")},
        "oof_selected": {str(fold): {key: artifact[key] for key in
                                     ("rank_epoch", "gate_epoch", "threshold")}
                         for fold, artifact in fold_artifacts.items()},
        "embedding_hashes": embedding_hashes,
        "pair_features_sha256": table_hash(pair_features),
        "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": {
            "zsre_only_policy.pt": sha(root / "zsre_only_policy.pt"),
            **{f"mixed_oof_fold{fold}.pt": sha(root / f"mixed_oof_fold{fold}.pt")
               for fold in range(FOLDS)},
        },
    }
    manifest = {
        "protocol": PROTOCOL,
        "zsre_protocol": ZSRE_PROTOCOL,
        "counterfact_protocol": COUNTERFACT_PROTOCOL,
        "zsre_manifest_sha256": sha(Path(args.zsre_root) / "manifest.json"),
        "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
        "previous_summary_sha256": sha(Path(args.previous_root) / "summary.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "folds": FOLDS, "epochs": list(EPOCHS), "thresholds": list(THRESHOLDS),
        "cosine_temperature": COSINE_TEMPERATURE,
        "current_write_gate": 1,
        "chain_identity_in_features": False,
        "counterfact_status": "observed_development_only",
        "longmemeval_scored": False, "locomo_scored": False,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-root", required=True)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
