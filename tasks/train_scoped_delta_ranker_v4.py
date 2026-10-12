"""Train a factorized relation/scope ranker on scoped-delta supervision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from tasks.build_scoped_delta_corpus import MAX_KEYS, PROTOCOL as CORPUS_PROTOCOL
from tasks.evaluate_zsre_contextual_readout import SEEDS as BI_SEEDS, apply, extract
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import (
    COSINE_TEMPERATURE, FOLDS, deterministic_folds,
)
from tasks.train_zsre_forgetting_controller import lexical_features, pair_feature
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features


PROTOCOL = "scoped_delta_ranker_v4"
ARMS = ("structured_full", "no_relation", "no_subject")
SEEDS = (13101, 13102, 13103)
CHECKPOINT_EPOCHS = (5, 10, 25, 50)
RELATION_PAIR_TEMPLATE = (
    "Old memory relation pattern: {key}\n"
    "Current input relation pattern: {query}\n"
    "Same factual relation:"
)
MARGIN = .25
MARGIN_WEIGHT = .5


def load_corpus(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != CORPUS_PROTOCOL:
        raise ValueError("scoped-delta corpus protocol mismatch")
    path = root / "groups.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("scoped-delta corpus hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line], manifest


def local_groups(groups):
    rows = [group for group in groups if group["target"] < MAX_KEYS]
    if any(group["desired_action"] != "local" for group in rows):
        raise ValueError("candidate stage may only consume local decisions")
    return rows


def all_visible_texts(groups):
    names, pairs = set(), set()
    for group in groups:
        names.update((group["query"], group["query_relation"]))
        for candidate in group["candidates"]:
            names.update((candidate["prompt"], candidate["relation_prompt"]))
            pairs.add(PAIR_TEMPLATE.format(key=candidate["prompt"], query=group["query"]))
            pairs.add(RELATION_PAIR_TEMPLATE.format(
                key=candidate["relation_prompt"], query=group["query_relation"]))
    return sorted(names), sorted(pairs)


def relation_numeric(group, candidate, embeddings):
    import torch

    cosines = []
    for seed in BI_SEEDS:
        table = embeddings[str(seed)]
        cosines.append(float(table.get("entity", candidate["relation_prompt"])
                             @ table.get("entity", group["query_relation"])))
    lexical = lexical_features(candidate["relation_prompt"], group["query_relation"])
    return torch.tensor([*cosines, statistics.fmean(cosines),
                         statistics.pstdev(cosines), *lexical])


def raw_feature(group, candidate, scorer, pair_features, embeddings):
    import torch

    base = pair_feature(candidate["prompt"], group["query"], scorer)
    relation_text = RELATION_PAIR_TEMPLATE.format(
        key=candidate["relation_prompt"], query=group["query_relation"])
    relation = torch.cat((pair_features[relation_text],
                          relation_numeric(group, candidate, embeddings)))
    subject = torch.tensor([float(candidate["subject"].casefold()
                                  == group["query_subject"].casefold())])
    return base, relation, subject


def feature_tables(groups, scorer, pair_features, embeddings):
    import torch

    arm_rows = {arm: [] for arm in ARMS}; cosine_rows = []
    base_dim = relation_dim = None
    for group in groups:
        per_arm = {arm: [] for arm in ARMS}; per_cosine = []
        for candidate in group["candidates"]:
            base, relation, subject = raw_feature(
                group, candidate, scorer, pair_features, embeddings)
            base_dim, relation_dim = base.numel(), relation.numel()
            per_arm["structured_full"].append(torch.cat((base, relation, subject)))
            per_arm["no_relation"].append(torch.cat((base, subject)))
            per_arm["no_subject"].append(torch.cat((base, relation)))
            pair_dim = next(iter(scorer.pair_features.values())).numel()
            per_cosine.append(base[pair_dim:pair_dim + 3].mean())
        for arm in ARMS:
            per_arm[arm].extend(torch.zeros_like(per_arm[arm][0])
                                for _ in range(MAX_KEYS - len(per_arm[arm])))
            arm_rows[arm].append(torch.stack(per_arm[arm]))
        per_cosine.extend(torch.tensor(float("-inf"))
                          for _ in range(MAX_KEYS - len(per_cosine)))
        cosine_rows.append(torch.stack(per_cosine))
    return ({arm: torch.stack(rows) for arm, rows in arm_rows.items()},
            torch.stack(cosine_rows), {"base_dim": base_dim, "relation_dim": relation_dim})


def counts(groups):
    return __import__("torch").tensor([len(group["candidates"]) for group in groups])


def targets(groups):
    return __import__("torch").tensor([group["target"] for group in groups])


def valid_mask(candidate_counts, width=MAX_KEYS):
    import torch
    return torch.arange(width, device=candidate_counts.device)[None, :] < candidate_counts[:, None]


def fit_normalizer(values, candidate_counts, indices):
    selected = values[indices]
    valid = valid_mask(candidate_counts[indices])
    flattened = selected[valid]
    return flattened.mean(0), flattened.std(0, unbiased=False).clamp_min(1e-4)


class FactorizedRanker:
    def __init__(self, dim, seed):
        import torch

        torch.manual_seed(seed)
        self.model = torch.nn.Linear(dim, 1)
        torch.nn.init.normal_(self.model.weight, std=.001)
        torch.nn.init.zeros_(self.model.bias)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def logits(self, values, cosine, candidate_counts):
        residual = self.model(values).squeeze(-1)
        return (cosine / COSINE_TEMPERATURE + residual).masked_fill(
            ~valid_mask(candidate_counts, values.shape[1]), float("-inf"))


def hard_negative_loss(groups, logits):
    import torch

    negative_mask = torch.tensor([
        [candidate["negative_type"] == "same_subject_different_chain"
         for candidate in group["candidates"]]
        + [False] * (logits.shape[1] - len(group["candidates"]))
        for group in groups
    ], dtype=torch.bool, device=logits.device)
    available = negative_mask.any(-1)
    if not available.any():
        return logits.sum() * 0
    negative = logits.masked_fill(~negative_mask, float("-inf")).max(-1).values
    positive = logits.gather(1, targets(groups).to(logits.device)[:, None]).squeeze(1)
    return torch.nn.functional.softplus(
        MARGIN + negative[available] - positive[available]).mean()


def rank_metrics(groups, logits):
    import torch

    label = targets(groups).to(logits.device)
    predictions = logits.argmax(-1); order = logits.argsort(-1, descending=True)
    ranks = order.eq(label[:, None]).nonzero()[:, 1].float() + 1
    same_results, different_results = [], []
    for index, group in enumerate(groups):
        target_score = logits[index, group["target"]]
        same = [candidate["index"] for candidate in group["candidates"]
                if candidate["negative_type"] == "same_subject_different_chain"]
        different = [candidate["index"] for candidate in group["candidates"]
                     if candidate["negative_type"] == "different_subject"]
        if same:
            same_results.append(bool(target_score > logits[index, same].max()))
        if different:
            different_results.append(bool(target_score > logits[index, different].max()))
    return {
        "events": len(groups),
        "top1_accuracy": float(predictions.eq(label).float().mean()),
        "mrr": float((1 / ranks).mean()),
        "nll": float(torch.nn.functional.cross_entropy(logits, label)),
        "target_beats_same_subject": statistics.fmean(same_results),
        "same_subject_events": len(same_results),
        "target_beats_all_different_subject": statistics.fmean(different_results),
        "different_subject_events": len(different_results),
    }


def train_for_epochs(groups, values, cosine, candidate_counts, indices, mean, std,
                     seed, epochs):
    import torch

    selected_groups = [groups[index] for index in indices]
    device = torch.device("cuda")
    x = ((values[indices] - mean) / std).to(device)
    c = cosine[indices].to(device); n = candidate_counts[indices].to(device)
    y = targets(selected_groups).to(device)
    head = FactorizedRanker(values.shape[-1], seed); head.model.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.005, weight_decay=.05)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(x, c, n)
        loss = torch.nn.functional.cross_entropy(logits, y) \
            + MARGIN_WEIGHT * hard_negative_loss(selected_groups, logits)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    head.model.cpu()
    return head


def choose_epoch(groups, values, cosine, candidate_counts, train_indices, tune_indices, seed):
    mean, std = fit_normalizer(values, candidate_counts, train_indices)
    curve, candidates = [], []
    for epoch in CHECKPOINT_EPOCHS:
        head = train_for_epochs(groups, values, cosine, candidate_counts,
                                train_indices, mean, std, seed, epoch)
        tune_groups = [groups[index] for index in tune_indices]
        with __import__("torch").no_grad():
            logits = head.logits((values[tune_indices] - mean) / std,
                                 cosine[tune_indices], candidate_counts[tune_indices])
        metric = rank_metrics(tune_groups, logits)
        curve.append({"epoch": epoch, "tune": metric})
        candidates.append((metric["top1_accuracy"], metric["mrr"],
                           metric["target_beats_same_subject"], -epoch))
    return -max(candidates)[-1], curve


def fit_outer(groups, values, cosine, candidate_counts, indices, seed, epoch):
    mean, std = fit_normalizer(values, candidate_counts, indices)
    head = train_for_epochs(groups, values, cosine, candidate_counts,
                            indices, mean, std, seed, epoch)
    return {"state": head.state_dict(), "mean": mean, "std": std,
            "dim": values.shape[-1], "seed": seed, "epoch": epoch}


def predict(groups, values, cosine, candidate_counts, indices, artifacts):
    import torch

    ensemble = []
    for artifact in artifacts:
        head = FactorizedRanker(artifact["dim"], artifact["seed"])
        head.load_state_dict(artifact["state"])
        with torch.no_grad():
            ensemble.append(head.logits((values[indices] - artifact["mean"]) / artifact["std"],
                                        cosine[indices], candidate_counts[indices]))
    return torch.stack(ensemble).mean(0)


def rows_from_logits(groups, logits, fold, arm):
    rows = []
    for group, scores in zip(groups, logits):
        target = group["target"]; prediction = int(scores.argmax())
        order = scores.argsort(descending=True).tolist()
        predicted_type = group["candidates"][prediction]["negative_type"]
        rows.append({
            "session_id": group["session_id"], "group_id": group["group_id"],
            "target": target, "prediction": prediction,
            "correct": int(prediction == target),
            "reciprocal_rank": 1 / (order.index(target) + 1),
            "predicted_type": predicted_type, "fold": fold, "arm": arm,
        })
    return rows


def load_v2_rows(root):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    path = root / "counterfact_mixed_oof_oracle_events.jsonl"
    if training["raw_rows_sha256"][path.name] != sha(path):
        raise ValueError("V2 oracle trace hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {(row["session_id"], row["group_id"]): row for row in rows
            if row["target"] < MAX_KEYS}


def compare(primary_rows, baseline, seed=13201):
    differences = []
    for row in primary_rows:
        old = baseline[(row["session_id"], row["group_id"])]
        differences.append(row["correct"] - int(old["top"] == old["target"]))
    return {"top1_gain": statistics.fmean(differences),
            "bootstrap_95ci": _bootstrap(differences, seed),
            "units": len(differences)}


def aggregate_rows(groups, rows, logits):
    metric = rank_metrics(groups, logits)
    metric["wrong_same_subject"] = sum(
        not row["correct"] and row["predicted_type"] == "same_subject_different_chain"
        for row in rows)
    metric["wrong_different_subject"] = sum(
        not row["correct"] and row["predicted_type"] == "different_subject"
        for row in rows)
    return metric


def development_gate(metrics, comparison):
    checks = {
        "top1_at_least_88": metrics["top1_accuracy"] >= .88,
        "mrr_at_least_93": metrics["mrr"] >= .93,
        "gain_at_least_5pp": comparison["top1_gain"] >= .05,
        "gain_ci_positive": comparison["bootstrap_95ci"][0] > 0,
        "same_subject_at_least_85": metrics["target_beats_same_subject"] >= .85,
        "different_subject_at_least_95": metrics["target_beats_all_different_subject"] >= .95,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    all_groups, corpus_manifest = load_corpus(args.corpus_root)
    groups = local_groups(all_groups)
    zsre_indices = [index for index, group in enumerate(groups) if group["source"] == "zsre"]
    counterfact_indices = [index for index, group in enumerate(groups)
                           if group["source"] == "counterfact"]
    counterfact_ids = {groups[index]["session_id"] for index in counterfact_indices}
    fold_map = deterministic_folds(counterfact_ids)

    source, scope = Path(args.source_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")

    names, pair_texts = all_visible_texts(groups)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, args.batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms["transforms"],
        source_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pair_texts, args.batch_size)
    full_pair_features = {key: value for key, value in pair_features.items()
                          if key.startswith("Memory update scope:")}
    scorer = FrozenScopeScorer(embeddings, full_pair_features, gate_artifact)
    values_by_arm, cosine, dimensions = feature_tables(
        groups, scorer, pair_features, embeddings)
    candidate_counts = counts(groups)
    del model; torch.cuda.empty_cache()

    v2_rows = load_v2_rows(args.v2_root)
    results, selections, checkpoints, raw_hashes = {}, {}, {}, {}
    primary_rows = None
    for arm in ARMS:
        values = values_by_arm[arm]; arm_rows = []; arm_logits = []
        selections[arm] = {}; checkpoints[arm] = {}
        for outer in range(FOLDS):
            inner = (outer + 1) % FOLDS
            inner_train = [*zsre_indices, *[index for index in counterfact_indices
                            if fold_map[groups[index]["session_id"]] not in (outer, inner)]]
            inner_tune = [index for index in counterfact_indices
                          if fold_map[groups[index]["session_id"]] == inner]
            outer_train = [*zsre_indices, *[index for index in counterfact_indices
                            if fold_map[groups[index]["session_id"]] != outer]]
            outer_test = [index for index in counterfact_indices
                          if fold_map[groups[index]["session_id"]] == outer]
            artifacts, curves, epochs = [], {}, {}
            for seed in SEEDS:
                epoch, curve = choose_epoch(groups, values, cosine, candidate_counts,
                                             inner_train, inner_tune, seed + outer * 100)
                artifact = fit_outer(groups, values, cosine, candidate_counts,
                                     outer_train, seed + outer * 100, epoch)
                artifacts.append(artifact); curves[str(seed)] = curve; epochs[str(seed)] = epoch
            logits = predict(groups, values, cosine, candidate_counts, outer_test, artifacts)
            selected_groups = [groups[index] for index in outer_test]
            arm_rows.extend(rows_from_logits(selected_groups, logits, outer, arm))
            arm_logits.append((outer_test, logits))
            selections[arm][str(outer)] = {"inner_fold": inner, "epochs": epochs,
                                            "curves": curves}
            checkpoints[arm][str(outer)] = artifacts
        order = sorted(range(len(arm_rows)), key=lambda i: (
            arm_rows[i]["session_id"], arm_rows[i]["group_id"]))
        arm_rows = [arm_rows[i] for i in order]
        # Reconstruct logits/groups in the same deterministic key order.
        logits_by_key = {}
        for indices, logits in arm_logits:
            for index, row_logits in zip(indices, logits):
                group = groups[index]; logits_by_key[(group["session_id"], group["group_id"])] = row_logits
        eval_groups = sorted((groups[index] for index in counterfact_indices),
                             key=lambda row: (row["session_id"], row["group_id"]))
        eval_logits = torch.stack([logits_by_key[(group["session_id"], group["group_id"])]
                                   for group in eval_groups])
        metrics = aggregate_rows(eval_groups, arm_rows, eval_logits)
        path = root / f"counterfact_oof_{arm}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in arm_rows))
        raw_hashes[path.name] = sha(path); results[arm] = metrics
        if arm == "structured_full": primary_rows = arm_rows

    comparison = compare(primary_rows, v2_rows)
    gate = development_gate(results["structured_full"], comparison)
    checkpoint_path = root / "rankers.pt"
    torch.save({"artifacts": checkpoints, "fold_map": fold_map,
                "arms": ARMS, "seeds": SEEDS, "dimensions": dimensions}, checkpoint_path)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    seconds = time.perf_counter() - started
    summary = {
        "protocol": PROTOCOL,
        "data_status": "observed_development_only",
        "counterfact_oof": results,
        "primary_vs_v2": comparison,
        "v2_candidate_top1": statistics.fmean(
            int(row["top"] == row["target"]) for row in v2_rows.values()),
        "development_gate": gate,
        "next_stage": "causal_delta_bank" if gate["passed"] else "stop_causal_gate_and_revise_scope_representation",
        "terminal_status": "completed_scoped_delta_ranker_v4",
    }
    training = {
        "seconds": seconds,
        "gpu_peak_allocated_gib": peak_allocated,
        "gpu_peak_reserved_gib": peak_reserved,
        "groups": {"zsre_local": len(zsre_indices),
                   "counterfact_local": len(counterfact_indices)},
        "unique_texts": len(names), "unique_pair_texts": len(pair_texts),
        "feature_dimensions": {arm: values.shape[-1] for arm, values in values_by_arm.items()},
        "component_dimensions": dimensions,
        "counterfact_folds": fold_map,
        "selection": selections,
        "embedding_hashes": embedding_hashes,
        "pair_features_sha256": table_hash(pair_features),
        "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": sha(checkpoint_path),
    }
    manifest = {
        "protocol": PROTOCOL,
        "corpus_manifest_sha256": sha(Path(args.corpus_root) / "manifest.json"),
        "v2_summary_sha256": sha(Path(args.v2_root) / "summary.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "arms": list(ARMS), "seeds": list(SEEDS), "folds": FOLDS,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "margin": MARGIN, "margin_weight": MARGIN_WEIGHT,
        "subject_metadata_role": "oracle_segmentation_diagnostic",
        "chain_identity_in_features": False,
        "relation_ids_used": False,
        "age_feature_used": False,
        "counterfact_status": "observed_development_only",
        "longmemeval_scored": False, "locomo_scored": False,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
