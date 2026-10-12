"""Final nonlinear, one-round on-policy retain/local gate over frozen V11 scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.build_scoped_delta_corpus import MAX_KEYS
from tasks.evaluate_automatic_factorized_scope_v6 import load_v5, relation_features
from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate, all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_causal_token_span_forgetting_v12 import (
    EPOCHS, HEAD_SEEDS, THRESHOLDS, CausalGateController, FrozenV11Resolver,
    candidate_metrics, causal_action_metrics, fold_predictions, load_policy_records,
    load_v11, select_threshold, update_examples,
)
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import FOLDS, deterministic_folds
from tasks.train_token_span_relation_scope_v11 import extract_projected_features
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import (
    apply_controller, oracle_event_groups, paired_comparison, replay_controller,
    summarize_seed_metrics,
)
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "causal_onpolicy_scope_v13"
ARMS = ("oracle_mlp", "onpolicy_mlp")
RANK = 16


def augmented_feature(result):
    import torch
    top = result["top"]
    query = result["query_posterior"]
    candidate = result["candidate_posterior"][top]
    query_entropy = -(query * query.clamp_min(1e-12).log()).sum()
    candidate_entropy = -(candidate * candidate.clamp_min(1e-12).log()).sum()
    midpoint = .5 * (query + candidate)
    js = .5 * (query * (query.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum() \
        + .5 * (candidate * (candidate.clamp_min(1e-12)
                             / midpoint.clamp_min(1e-12)).log()).sum()
    cosine = torch.nn.functional.cosine_similarity(query, candidate, dim=0)
    additions = torch.stack((
        query.max(), candidate.max(), query_entropy, candidate_entropy,
        query.argmax().eq(candidate.argmax()).float(), cosine, js,
    )).float()
    return torch.cat((result["feature"], additions))


def augmented_rows(groups, resolver):
    output = []
    for group in groups:
        result = resolver.score(group["query"], group["keys"])
        output.append({"group": group, "feature": augmented_feature(result),
                       "top": result["top"],
                       "label": int(group["target"] < MAX_KEYS),
                       "state_source": "oracle"})
    return output


def collect_onpolicy_rows(sessions, resolver, base_controller):
    output = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}; memory = []
        for step, event in enumerate(session["events"]):
            chain = chains[event["chain_id"]]
            version = chain["versions"][event["version"]]
            if memory:
                matches = [index for index, row in enumerate(memory)
                           if row["chain_id"] == event["chain_id"]]
                target = matches[0] if event["action"] == "local" and len(matches) == 1 \
                    else MAX_KEYS
                group = {"session_id": session["id"],
                         "group_id": chain.get("parent_chain_id", chain["id"]),
                         "step": step, "event_kind": event["kind"],
                         "query": version["prompt"],
                         "keys": [row["prompt"] for row in memory], "target": target}
                result = resolver.score(group["query"], group["keys"])
                output.append({"group": group, "feature": augmented_feature(result),
                               "top": result["top"], "label": int(target < MAX_KEYS),
                               "state_source": "v12_onpolicy",
                               "desired_local_missing": int(
                                   event["action"] == "local" and not matches)})
            memory, _ = apply_controller(memory, event, version, base_controller)
    return output


class NonlinearGate:
    def __init__(self, dim, seed, rank=RANK):
        import torch
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(torch.nn.Linear(dim, rank), torch.nn.Tanh(),
                                         torch.nn.Linear(rank, 1))
        torch.nn.init.normal_(self.model[0].weight, std=.02)
        torch.nn.init.zeros_(self.model[0].bias)
        torch.nn.init.normal_(self.model[2].weight, std=.02)
        torch.nn.init.zeros_(self.model[2].bias)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def __call__(self, values):
        return self.model(values).squeeze(-1)


def fit_snapshots(rows, seed):
    import torch
    x = torch.stack([row["feature"] for row in rows])
    y = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-4)
    counts = torch.bincount(y.long(), minlength=2).float().clamp_min(1)
    pos_weight = (counts[0] / counts[1]).sqrt().to("cuda")
    train_x = ((x - mean) / std).to("cuda"); train_y = y.to("cuda")
    head = NonlinearGate(x.shape[-1], seed); head.model.to("cuda")
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
            snapshots[epoch] = {"state": state, "mean": mean, "std": std,
                                "dim": x.shape[-1], "rank": RANK,
                                "seed": seed, "epoch": epoch}
    head.model.cpu(); return snapshots


class NonlinearController:
    def __init__(self, resolver, artifacts, threshold):
        self.resolver, self.artifacts, self.threshold = resolver, artifacts, threshold
        self.heads = []
        for artifact in artifacts:
            head = NonlinearGate(artifact["dim"], artifact["seed"], artifact["rank"])
            head.load_state_dict(artifact["state"]); self.heads.append(head)

    def predict(self, query, keys):
        import torch
        if not keys:
            return {"prediction": 0, "probability": 0.0}
        result = self.resolver.score(query, keys); feature = augmented_feature(result)
        logits = []
        for artifact, head in zip(self.artifacts, self.heads):
            with torch.no_grad():
                value = (feature - artifact["mean"]) / artifact["std"]
                logits.append(head(value.unsqueeze(0))[0])
        probability = float(torch.stack(logits).mean().sigmoid())
        prediction = result["top"] if probability >= self.threshold else len(keys)
        return {"prediction": prediction, "probability": probability}


def select_inner(sessions_by_fold, resolver, train_rows, outer):
    inner = (outer + 1) % FOLDS
    snapshots = {str(seed): fit_snapshots(train_rows, seed + outer * 100)
                 for seed in HEAD_SEEDS}
    choices, curves = [], {}
    for epoch in EPOCHS:
        artifacts = [snapshots[str(seed)][epoch] for seed in HEAD_SEEDS]
        grid = {}
        for threshold in THRESHOLDS:
            controller = NonlinearController(resolver, artifacts, threshold)
            grid[threshold] = causal_action_metrics(sessions_by_fold[inner], controller)
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


def load_v12(root):
    import torch
    root = Path(root); training = json.loads((root / "training.json").read_text())
    checkpoint = root / "causal_gates.pt"
    if training["checkpoint_sha256"] != sha(checkpoint):
        raise ValueError("V12 checkpoint mismatch")
    return torch.load(checkpoint, map_location="cpu", weights_only=False), training


def load_v12_records(root):
    root = Path(root); training = json.loads((root / "training.json").read_text())
    output = {}
    for seed in GATE_SEEDS:
        path = root / f"counterfact_causal_v12_seed{seed}.jsonl"
        if training["raw_rows_sha256"][path.name] != sha(path):
            raise ValueError("V12 causal trace mismatch")
        output[str(seed)] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return output


def development_gate(metric, candidate, versus_v12, oracle_accuracy):
    checks = {"candidate_top1_at_least_96": candidate["top1_accuracy"] >= .96,
              "overall_action_at_least_95": metric["mean_action_accuracy"] >= .95,
              "hard_retain_at_least_95": metric["by_event_kind"]["hard_retain"]["accuracy"] >= .95,
              "primary_update_at_least_90": metric["by_event_kind"]["primary_update"]["accuracy"] >= .90,
              "query_within_half_pp_oracle": metric["mean_accuracy"] >= oracle_accuracy - .005,
              "query_not_materially_worse_v12": versus_v12["query_bootstrap_95ci"][0] > -.002}
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
    v12_artifacts, v12_training = load_v12(args.v12_root)
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
    readout_pairs_table = extract_pair_features(model, tokenizer, readout_pairs,
                                                 args.pair_batch_size)
    readout_scorer = FrozenScopeScorer(embeddings, readout_pairs_table, readout_gate)
    resolvers = {fold: FrozenV11Resolver(predictions[fold], relation_table,
                                         v5_artifact["artifacts"][str(fold)])
                 for fold in range(FOLDS)}
    del raw, model; torch.cuda.empty_cache()

    oracle_rows = {outer: {fold: augmented_rows(groups_by_fold[fold], resolvers[outer])
                           for fold in range(FOLDS)} for outer in range(FOLDS)}
    candidate_by_fold = {str(outer): candidate_metrics(oracle_rows[outer][outer])
                         for outer in range(FOLDS)}
    controllers, selections, checkpoints, state_counts = {}, {}, {}, {}
    for arm in ARMS:
        controllers[arm], selections[arm], checkpoints[arm], state_counts[arm] = {}, {}, {}, {}
        for outer in range(FOLDS):
            inner = (outer + 1) % FOLDS
            base = CausalGateController(resolvers[outer], v12_artifacts[str(outer)],
                                        v12_training["selection"][str(outer)]["threshold"])
            train_folds = [fold for fold in range(FOLDS) if fold not in (outer, inner)]
            inner_oracle = [row for fold in train_folds for row in oracle_rows[outer][fold]]
            inner_onpolicy = collect_onpolicy_rows(
                [session for fold in train_folds for session in sessions_by_fold[fold]],
                resolvers[outer], base) if arm == "onpolicy_mlp" else []
            inner_train = [*inner_oracle, *inner_onpolicy]
            selected = select_inner(sessions_by_fold, resolvers[outer], inner_train, outer)

            outer_folds = [fold for fold in range(FOLDS) if fold != outer]
            outer_oracle = [row for fold in outer_folds for row in oracle_rows[outer][fold]]
            outer_onpolicy = collect_onpolicy_rows(
                [session for fold in outer_folds for session in sessions_by_fold[fold]],
                resolvers[outer], base) if arm == "onpolicy_mlp" else []
            outer_train = [*outer_oracle, *outer_onpolicy]
            snapshots = {str(seed): fit_snapshots(outer_train, seed + outer * 1000)
                         for seed in HEAD_SEEDS}
            artifacts = [snapshots[str(seed)][selected["epoch"]] for seed in HEAD_SEEDS]
            controllers[arm][outer] = NonlinearController(
                resolvers[outer], artifacts, selected["threshold"])
            selections[arm][str(outer)] = selected; checkpoints[arm][str(outer)] = artifacts
            state_counts[arm][str(outer)] = {
                "inner_oracle": len(inner_oracle), "inner_onpolicy": len(inner_onpolicy),
                "outer_oracle": len(outer_oracle), "outer_onpolicy": len(outer_onpolicy),
                "outer_onpolicy_missing_local": sum(
                    row.get("desired_local_missing", 0) for row in outer_onpolicy)}

    arm_records, arm_metrics, raw_hashes = {}, {}, {}
    for arm in ARMS:
        records_by_seed, metrics_by_seed = {}, {}
        for replicate, readout_seed in enumerate(GATE_SEEDS):
            records = []
            for session in sessions:
                fold = fold_map[session["id"]]
                records.append(replay_controller(
                    session, readout_scorer, readout_seed, HEAD_SEEDS[replicate],
                    controllers[arm][fold], split=f"counterfact_observed_{arm}_v13"))
            records_by_seed[str(readout_seed)] = records
            metrics_by_seed[str(readout_seed)] = aggregate(records)
            path = root / f"counterfact_{arm}_v13_seed{readout_seed}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        arm_records[arm] = records_by_seed
        arm_metrics[arm] = summarize_seed_metrics(metrics_by_seed)

    fixed = load_policy_records(args.previous_root, "fixed_retain")
    oracle = load_policy_records(args.previous_root, "oracle_local")
    v12_records = load_v12_records(args.v12_root)
    primary = arm_records["onpolicy_mlp"]
    vs_fixed = paired_comparison(primary, fixed, 21201)
    vs_oracle = paired_comparison(primary, oracle, 21211)
    vs_v12 = paired_comparison(primary, v12_records, 21221)
    vs_ablation = paired_comparison(primary, arm_records["oracle_mlp"], 21231)
    previous_summary = json.loads((Path(args.previous_root) / "summary.json").read_text())
    oracle_accuracy = previous_summary["policies"]["oracle_local"]["mean_accuracy"]
    candidate = {"events": sum(row["events"] for row in candidate_by_fold.values())}
    candidate["top1_accuracy"] = sum(
        row["events"] * row["top1_accuracy"] for row in candidate_by_fold.values()) \
        / candidate["events"]
    gate = development_gate(arm_metrics["onpolicy_mlp"], candidate, vs_v12, oracle_accuracy)
    checkpoint = root / "nonlinear_gates.pt"; torch.save(checkpoints, checkpoint)
    seconds = time.perf_counter() - started
    summary = {"protocol": PROTOCOL, "data_status": "observed_development_only",
               "candidate_oof": candidate, "candidate_by_fold": candidate_by_fold,
               "arms": arm_metrics, "onpolicy_vs_fixed_retain": vs_fixed,
               "onpolicy_vs_oracle_local": vs_oracle, "onpolicy_vs_v12": vs_v12,
               "onpolicy_vs_oracle_mlp": vs_ablation,
               "oracle_local_accuracy": oracle_accuracy, "development_gate": gate,
               "next_stage": "external_domain_validation" if gate["passed"]
               else "stop_counterfact_controller_tuning",
               "terminal_status": "completed_causal_onpolicy_scope_v13"}
    training = {"seconds": seconds,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "update_texts": len(update_rows), "projection": projection_meta,
                "unique_relation_texts": len(relation_texts),
                "unique_readout_texts": len(readout_names),
                "unique_readout_pairs": len(readout_pairs),
                "embedding_hashes": embedding_hashes,
                "readout_pair_features_sha256": table_hash(readout_pairs_table),
                "fold_map": fold_map, "selection": selections,
                "state_counts": state_counts, "raw_rows_sha256": raw_hashes,
                "checkpoint_sha256": sha(checkpoint)}
    manifest = {"protocol": PROTOCOL,
                "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
                "v11_training_sha256": sha(Path(args.v11_root) / "training.json"),
                "v12_training_sha256": sha(Path(args.v12_root) / "training.json"),
                "v12_checkpoint_sha256": sha(Path(args.v12_root) / "causal_gates.pt"),
                "v5_training_sha256": sha(Path(args.v5_root) / "training.json"),
                "model": args.model, "tokenizer": args.tokenizer,
                "loading_info": model_meta["loading_info"], "arms": list(ARMS),
                "head_seeds": list(HEAD_SEEDS), "epochs": list(EPOCHS),
                "thresholds": list(THRESHOLDS), "rank": RANK, "onpolicy_rounds": 1,
                "current_write_gate": 1, "v11_frozen": True, "v5_frozen": True,
                "query_readout_frozen": True, "chain_identity_in_features": False,
                "inference_subject_metadata_used": False,
                "inference_relation_ids_used": False, "candidate_age_used": False,
                "counterfact_status": "observed_development_only",
                "longmemeval_scored": False, "locomo_scored": False,
                "units": data_manifest["eligible_subjects"]}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("counterfact-root", "v5-root", "v11-root", "v12-root",
                 "previous-root", "source-root", "scope-root", "output",
                 "model", "tokenizer"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--readout-batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
