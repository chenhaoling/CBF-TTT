"""Causally replay a frozen V11 resolver with a nested retain/local gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import load_v5, relation_features
from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_relation_supervised_scope_v5 import ensemble_logits
from tasks.train_token_span_relation_scope_v11 import (
    extract_projected_features, jaccard, predict_spans,
)
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import (
    apply_controller, oracle_event_groups, paired_comparison, replay_controller,
    summarize_seed_metrics,
)
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "causal_token_span_forgetting_v12"
HEAD_SEEDS = (20101, 20102, 20103)
EPOCHS = (10, 25, 50)
THRESHOLDS = tuple(index / 20 for index in range(1, 20))
HARD_RETAIN_FLOOR = .95
PRIMARY_UPDATE_FLOOR = .90


def update_examples(sessions):
    examples = {}
    for session in sessions:
        for chain in session["chains"]:
            subject = chain["subject"]
            for version in chain["versions"]:
                text = version["prompt"]
                previous = examples.get(text)
                if previous is not None and previous["subject"] != subject:
                    raise ValueError("one update text maps to multiple subjects")
                examples[text] = {"text": text, "subject": subject,
                                  "sources": ["causal_update_audit"]}
    return [examples[key] for key in sorted(examples)]


def load_v11(root):
    import torch
    root = Path(root); training = json.loads((root / "training.json").read_text())
    checkpoint = root / "span_heads.pt"
    if training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("V11 span checkpoint mismatch")
    if any(row["selected_arm"] != "overlap8" for row in training["selection"].values()):
        raise ValueError("V12 requires the frozen V11 overlap8 arm")
    return torch.load(checkpoint, map_location="cpu", weights_only=False), training


def fold_predictions(feature_rows, checkpoints, training):
    output = {}
    for fold in range(FOLDS):
        artifacts = list(checkpoints[str(fold)].values())
        threshold = training["selection"][str(fold)]["threshold"]
        keyed = predict_spans(feature_rows, artifacts, threshold)
        table = {}
        for (text, _subject), value in keyed.items():
            if text in table and table[text] != value:
                raise ValueError("ambiguous text-only span prediction")
            table[text] = value
        output[fold] = table
    return output


class FrozenV11Resolver:
    def __init__(self, predictions, relation_table, relation_artifacts):
        self.predictions = predictions
        self.relation_table = relation_table
        self.relation_artifacts = relation_artifacts

    def score(self, query, keys):
        import torch
        if not keys:
            return None
        query_row = self.predictions[query]
        candidates = [self.predictions[key] for key in keys]
        texts = [query_row["relation_view"],
                 *[row["relation_view"] for row in candidates]]
        logits = ensemble_logits(texts, self.relation_table, self.relation_artifacts)
        query_posterior = logits[0].softmax(-1)
        candidate_posterior = logits[1:].softmax(-1)
        relation = candidate_posterior @ query_posterior
        overlap = torch.tensor([jaccard(query_row["span_tokens"], row["span_tokens"])
                                for row in candidates])
        exact = torch.tensor([float(bool(row["span_tokens"]) and
                                    row["span_tokens"] == query_row["span_tokens"])
                              for row in candidates])
        scores = relation.clamp_min(1e-12).log() + 8 * overlap
        top = int(scores.argmax())
        if len(keys) > 1:
            score_margin = float(torch.topk(scores, 2).values[0]
                                 - torch.topk(scores, 2).values[1])
            relation_margin = float(torch.topk(relation, 2).values[0]
                                    - torch.topk(relation, 2).values[1])
        else:
            score_margin = float(scores[0] + 1)
            relation_margin = float(relation[0])
        probability = torch.softmax(scores, -1)
        entropy = float(-(probability * probability.clamp_min(1e-12).log()).sum())
        feature = torch.tensor([
            float(scores[top]), score_margin, entropy,
            float(relation[top]), relation_margin,
            float(overlap[top]), float(exact[top]), math.log1p(len(keys)),
            float(relation.mean()), float(overlap.mean()),
        ])
        return {"top": top, "feature": feature, "scores": scores,
                "relation": relation, "overlap": overlap}


def gate_rows(groups, resolver):
    rows = []
    for group in groups:
        result = resolver.score(group["query"], group["keys"])
        rows.append({"group": group, "feature": result["feature"],
                     "top": result["top"],
                     "label": int(group["target"] < MAX_KEYS)})
    return rows


def candidate_metrics(rows):
    local = [row for row in rows if row["group"]["target"] < MAX_KEYS]
    return {"events": len(local), "top1_accuracy": statistics.fmean(
        row["top"] == row["group"]["target"] for row in local)}


def fit_gate_snapshots(rows, seed):
    import torch
    x = torch.stack([row["feature"] for row in rows])
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-4)
    counts = torch.bincount(y.long(), minlength=2).float().clamp_min(1)
    pos_weight = (counts[0] / counts[1]).to("cuda")
    train_x = ((x - mean) / std).to("cuda"); train_y = y.to("cuda")
    head = BinaryActionHead(x.shape[-1], seed); head.linear.to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    snapshots = {}
    for epoch in range(1, max(EPOCHS) + 1):
        optimizer.zero_grad(set_to_none=True); logits = head(train_x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, train_y, pos_weight=pos_weight)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        if epoch in EPOCHS:
            state = {key: value.detach().cpu().clone()
                     for key, value in head.state_dict().items()}
            snapshots[epoch] = {"state": state,
                                "mean": mean, "std": std, "dim": x.shape[-1],
                                "seed": seed, "epoch": epoch}
    head.linear.cpu()
    return snapshots


class CausalGateController:
    def __init__(self, resolver, artifacts, threshold):
        self.resolver, self.artifacts, self.threshold = resolver, artifacts, threshold
        self.heads = []
        for artifact in artifacts:
            head = BinaryActionHead(artifact["dim"], artifact["seed"])
            head.load_state_dict(artifact["state"]); self.heads.append(head)

    def predict(self, query, keys):
        import torch
        if not keys:
            return {"prediction": 0, "probability": 0.0}
        result = self.resolver.score(query, keys); logits = []
        for artifact, head in zip(self.artifacts, self.heads):
            with torch.no_grad():
                value = (result["feature"] - artifact["mean"]) / artifact["std"]
                logits.append(head(value.unsqueeze(0))[0])
        probability = float(torch.stack(logits).mean().sigmoid())
        prediction = result["top"] if probability >= self.threshold else len(keys)
        return {"prediction": prediction, "probability": probability}


def causal_action_metrics(sessions, controller):
    decisions = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}; memory = []
        for event in session["events"]:
            version = chains[event["chain_id"]]["versions"][event["version"]]
            memory, decision = apply_controller(memory, event, version, controller)
            decisions.append({**decision, "event_kind": event["kind"]})
    by_kind = {}
    for kind in ("primary_initial", "hard_retain", "primary_update"):
        values = [row["correct"] for row in decisions if row["event_kind"] == kind]
        by_kind[kind] = {"events": len(values), "accuracy": statistics.fmean(values)}
    return {"actions": len(decisions),
            "action_accuracy": statistics.fmean(row["correct"] for row in decisions),
            "by_event_kind": by_kind}


def select_threshold(grid):
    def floors(metric):
        return metric["by_event_kind"]["hard_retain"]["accuracy"] >= HARD_RETAIN_FLOOR \
            and metric["by_event_kind"]["primary_update"]["accuracy"] >= PRIMARY_UPDATE_FLOOR
    feasible = [value for value in THRESHOLDS if floors(grid[value])]
    if feasible:
        selected = max(feasible, key=lambda value: (
            grid[value]["action_accuracy"],
            min(grid[value]["by_event_kind"]["hard_retain"]["accuracy"],
                grid[value]["by_event_kind"]["primary_update"]["accuracy"]),
            -abs(value - .5)))
        return selected, "constraints_satisfied"
    def deficit(value):
        metric = grid[value]["by_event_kind"]
        return max(0, HARD_RETAIN_FLOOR - metric["hard_retain"]["accuracy"]) \
            + max(0, PRIMARY_UPDATE_FLOOR - metric["primary_update"]["accuracy"])
    selected = min(THRESHOLDS, key=lambda value: (
        deficit(value), -grid[value]["action_accuracy"], abs(value - .5)))
    return selected, "minimum_constraint_deficit"


def choose_inner(sessions, resolver, rows_by_fold, outer):
    inner = (outer + 1) % FOLDS
    train_rows = [row for fold, rows in rows_by_fold.items()
                  if fold not in (outer, inner) for row in rows]
    tune_sessions = sessions[inner]
    snapshots = {str(seed): fit_gate_snapshots(train_rows, seed + outer * 100)
                 for seed in HEAD_SEEDS}
    choices, curves = [], {}
    for epoch in EPOCHS:
        artifacts = [snapshots[str(seed)][epoch] for seed in HEAD_SEEDS]
        grid = {}
        for threshold in THRESHOLDS:
            controller = CausalGateController(resolver, artifacts, threshold)
            grid[threshold] = causal_action_metrics(tune_sessions, controller)
        threshold, status = select_threshold(grid); metric = grid[threshold]
        curves[str(epoch)] = {"threshold": threshold, "status": status,
                              "selected": metric,
                              "grid": {str(key): value for key, value in grid.items()}}
        choices.append((int(status == "constraints_satisfied"),
                        metric["action_accuracy"],
                        min(metric["by_event_kind"]["hard_retain"]["accuracy"],
                            metric["by_event_kind"]["primary_update"]["accuracy"]),
                        -epoch, epoch, threshold, status))
    selected = max(choices)
    return {"inner_fold": inner, "epoch": selected[4], "threshold": selected[5],
            "status": selected[6], "curves": curves}


def load_policy_records(root, policy):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    output = {}
    for seed in GATE_SEEDS:
        path = root / f"transfer_test_{policy}_seed{seed}.jsonl"
        if training["raw_rows_sha256"][path.name] != sha(path):
            raise ValueError(f"baseline trace mismatch: {path.name}")
        output[str(seed)] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return output


def development_gate(metric, candidate, versus_fixed, oracle_accuracy):
    checks = {"candidate_top1_at_least_96": candidate["top1_accuracy"] >= .96,
              "overall_action_at_least_95": metric["mean_action_accuracy"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "primary_update_at_least_90": metric["by_event_kind"]["primary_update"]["accuracy"] >= .90,
              "query_gain_fixed_retain_positive": versus_fixed["query_bootstrap_95ci"][0] > 0,
              "query_within_1pp_oracle": metric["mean_accuracy"] >= oracle_accuracy - .01}
    return {"checks": checks, "passed": all(checks.values())}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    sessions, data_manifest = load_transfer(args.counterfact_root)
    session_table = {session["id"]: session for session in sessions}
    fold_map = deterministic_folds(session_table)
    sessions_by_fold = {fold: [session for session in sessions
                               if fold_map[session["id"]] == fold]
                        for fold in range(FOLDS)}
    groups = oracle_event_groups(sessions)
    groups_by_fold = {fold: [group for group in groups
                             if fold_map[group["session_id"]] == fold]
                      for fold in range(FOLDS)}

    v11_heads, v11_training = load_v11(args.v11_root)
    v11_manifest = json.loads((Path(args.v11_root) / "training_manifest.json").read_text())
    v5_artifact, _ = load_v5(args.v5_root)
    context = Path(args.source_root)
    context_summary = json.loads((context / "summary.json").read_text())
    transforms = torch.load(context / "transforms.pt", map_location="cpu", weights_only=False)
    scope = Path(args.scope_root); scope_training = json.loads((scope / "training.json").read_text())
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("readout scope checkpoint mismatch")
    readout_gate = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    update_rows, projection_meta = extract_projected_features(
        model, tokenizer, update_examples(sessions), args.extract_batch_size,
        v11_manifest["max_length"], v11_manifest["projection_dim"])
    predictions = fold_predictions(update_rows, v11_heads, v11_training)
    relation_texts = sorted({row["relation_view"] for table in predictions.values()
                             for row in table.values()})

    readout_names, readout_pairs = all_names(sessions)
    combined_names = sorted(set(readout_names) | set(relation_texts))
    raw = extract(model, tokenizer, {"entity": combined_names, "value": ["unknown"]},
                  args.readout_batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": combined_names, "value": ["unknown"]}, transforms["transforms"],
        context_summary["selected_config"]["regularizer"])
    relation_table = relation_features(relation_texts, embeddings)
    readout_pairs_table = extract_pair_features(
        model, tokenizer, readout_pairs, args.pair_batch_size)
    readout_scorer = FrozenScopeScorer(embeddings, readout_pairs_table, readout_gate)
    resolvers = {fold: FrozenV11Resolver(predictions[fold], relation_table,
                                         v5_artifact["artifacts"][str(fold)])
                 for fold in range(FOLDS)}
    del raw, model; torch.cuda.empty_cache()

    gate_rows_by_outer, candidate_by_fold = {}, {}
    for outer in range(FOLDS):
        gate_rows_by_outer[outer] = {fold: gate_rows(groups_by_fold[fold], resolvers[outer])
                                     for fold in range(FOLDS)}
        candidate_by_fold[str(outer)] = candidate_metrics(gate_rows_by_outer[outer][outer])

    final_artifacts, selection, controllers = {}, {}, {}
    for outer in range(FOLDS):
        selected = choose_inner(sessions_by_fold, resolvers[outer],
                                gate_rows_by_outer[outer], outer)
        outer_train = [row for fold, rows in gate_rows_by_outer[outer].items()
                       if fold != outer for row in rows]
        snapshots = {str(seed): fit_gate_snapshots(outer_train, seed + outer * 1000)
                     for seed in HEAD_SEEDS}
        artifacts = [snapshots[str(seed)][selected["epoch"]] for seed in HEAD_SEEDS]
        controllers[outer] = CausalGateController(
            resolvers[outer], artifacts, selected["threshold"])
        final_artifacts[str(outer)] = artifacts; selection[str(outer)] = selected

    records_by_seed, metrics_by_seed, raw_hashes = {}, {}, {}
    for replicate, readout_seed in enumerate(GATE_SEEDS):
        records = []
        for session in sessions:
            fold = fold_map[session["id"]]
            records.append(replay_controller(
                session, readout_scorer, readout_seed, HEAD_SEEDS[replicate],
                controllers[fold], split="counterfact_observed_causal_v12"))
        records_by_seed[str(readout_seed)] = records
        metrics_by_seed[str(readout_seed)] = aggregate(records)
        path = root / f"counterfact_causal_v12_seed{readout_seed}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        raw_hashes[path.name] = sha(path)
    metric = summarize_seed_metrics(metrics_by_seed)

    fixed = load_policy_records(args.previous_root, "fixed_retain")
    oracle = load_policy_records(args.previous_root, "oracle_local")
    versus_fixed = paired_comparison(records_by_seed, fixed, 20201)
    versus_oracle = paired_comparison(records_by_seed, oracle, 20211)
    previous_summary = json.loads((Path(args.previous_root) / "summary.json").read_text())
    oracle_accuracy = previous_summary["policies"]["oracle_local"]["mean_accuracy"]
    candidate = {"events": sum(row["events"] for row in candidate_by_fold.values()),
                 "top1_accuracy": statistics.fmean(
                     row["top1_accuracy"] for row in candidate_by_fold.values())}
    # Folds are nearly balanced; retain the exact event-weighted value as primary.
    candidate["top1_accuracy"] = sum(
        row["events"] * row["top1_accuracy"] for row in candidate_by_fold.values()) \
        / candidate["events"]
    gate = development_gate(metric, candidate, versus_fixed, oracle_accuracy)
    checkpoint = root / "causal_gates.pt"; torch.save(final_artifacts, checkpoint)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "candidate_oof": candidate, "candidate_by_fold": candidate_by_fold,
               "causal": metric, "v12_vs_fixed_retain": versus_fixed,
               "v12_vs_oracle_local": versus_oracle,
               "oracle_local_accuracy": oracle_accuracy,
               "development_gate": gate,
               "next_stage": "external_domain_validation" if gate["passed"]
               else "audit_causal_token_span_failures",
               "terminal_status": "completed_causal_token_span_forgetting_v12"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "update_texts": len(update_rows), "projection": projection_meta,
                "unique_relation_texts": len(relation_texts),
                "unique_readout_texts": len(readout_names),
                "unique_readout_pairs": len(readout_pairs),
                "embedding_hashes": embedding_hashes,
                "readout_pair_features_sha256": table_hash(readout_pairs_table),
                "fold_map": fold_map, "selection": selection,
                "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
                "v11_training_sha256": sha(Path(args.v11_root) / "training.json"),
                "v11_checkpoint_sha256": sha(Path(args.v11_root) / "span_heads.pt"),
                "v5_training_sha256": sha(Path(args.v5_root) / "training.json"),
                "previous_summary_sha256": sha(Path(args.previous_root) / "summary.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"],
                "head_seeds": list(HEAD_SEEDS), "epochs": list(EPOCHS),
                "thresholds": list(THRESHOLDS), "current_write_gate": 1,
                "v11_frozen": True, "v5_frozen": True, "query_readout_frozen": True,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False,
                "units": data_manifest["eligible_subjects"]}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("counterfact-root", "v5-root", "v11-root", "previous-root",
                 "source-root", "scope-root", "output", "model", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--readout-batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
