"""One-round on-policy gate fitting with causal inner threshold selection."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from tasks.evaluate_counterfact_zero_shot_forgetting import load_transfer
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_counterfactual_forgetting_controller import load_split
from tasks.evaluate_zsre_hard_retain_forgetting import all_names
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_crossdomain_forgetting_reranker import (
    FOLDS,
    THRESHOLDS,
    DecomposedController,
    ResidualRanker,
    candidate_counts,
    cosine_values,
    deterministic_folds,
    development_gate,
    evaluate_records,
    gate_features,
    load_previous,
    tag_sessions,
)
from tasks.train_crossdomain_forgetting_reranker_v2 import load_v1_records
from tasks.evaluate_crossdomain_forgetting_reranker_v2_constraints import (
    select_constrained_threshold,
)
from tasks.train_zsre_forgetting_controller import (
    MAX_KEYS,
    apply_controller,
    pair_feature,
    paired_comparison,
)
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead
from tasks.train_zsre_scope_resolver import extract_pair_features


PROTOCOL = "onpolicy_forgetting_gate_v3"
GATE_EPOCHS = 25
RETAIN_LOSS_WEIGHT = 2.0


def collect_onpolicy_groups(sessions, scorer, artifact):
    controller = DecomposedController(scorer, artifact)
    output = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}; memory = []
        for step, event in enumerate(session["events"]):
            version = chains[event["chain_id"]]["versions"][event["version"]]
            if memory:
                matches = [index for index, row in enumerate(memory)
                           if row["chain_id"] == event["chain_id"]]
                target = matches[0] if event["action"] == "local" and len(matches) == 1 \
                    else len(memory)
                output.append({"session_id": session["id"], "group_id": chains[
                    event["chain_id"]].get("parent_chain_id", event["chain_id"]),
                    "step": step, "event_kind": event["kind"], "query": version["prompt"],
                    "keys": [row["prompt"] for row in memory], "target": target,
                    "desired": event["action"], "old_chain_present": bool(matches)})
            memory, _ = apply_controller(memory, event, version, controller)
    return output


def dynamic_feature_tensor(groups, scorer):
    import torch

    width = max(len(group["keys"]) for group in groups); rows = []
    for group in groups:
        values = [pair_feature(key, group["query"], scorer) for key in group["keys"]]
        values.extend(torch.zeros_like(values[0]) for _ in range(width - len(values)))
        rows.append(torch.stack(values))
    return torch.stack(rows)


def fit_onpolicy_gate(groups, values, cosines, counts, rank_artifact, seed):
    import torch

    ranker = ResidualRanker(rank_artifact["feature_dim"], rank_artifact["seed"])
    ranker.load_state_dict(rank_artifact["ranker"])
    normalized = (values - rank_artifact["rank_mean"]) / rank_artifact["rank_std"]
    with torch.no_grad(): logits = ranker.logits(normalized, cosines, counts)
    gate_raw, _ = gate_features(normalized, logits, counts)
    mean = gate_raw.mean(0); std = gate_raw.std(0, unbiased=False).clamp_min(1e-4)
    device = torch.device("cuda"); x = ((gate_raw - mean) / std).to(device)
    label = torch.tensor([group["desired"] == "local" and group["old_chain_present"]
                          for group in groups], dtype=torch.float32, device=device)
    weight = torch.where(label.bool(), torch.ones_like(label),
                         torch.full_like(label, RETAIN_LOSS_WEIGHT))
    gate = BinaryActionHead(x.shape[-1], seed); gate.linear.to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.01, weight_decay=.1)
    for _ in range(GATE_EPOCHS):
        optimizer.zero_grad(set_to_none=True); prediction = gate(x)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, label, reduction="none") * weight).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.); optimizer.step()
    gate.linear.cpu()
    artifact = copy.deepcopy(rank_artifact)
    artifact.update({"gate": gate.state_dict(), "gate_mean": mean, "gate_std": std,
                     "gate_epoch": GATE_EPOCHS, "gate_target": "onpolicy_available_conflict",
                     "retain_loss_weight": RETAIN_LOSS_WEIGHT, "threshold": .5})
    return artifact


def causal_action_metrics(sessions, scorer, artifact):
    controller = DecomposedController(scorer, artifact)
    rows = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}; memory = []
        for event in session["events"]:
            version = chains[event["chain_id"]]["versions"][event["version"]]
            memory, decision = apply_controller(memory, event, version, controller)
            rows.append({**decision, "event_kind": event["kind"]})
    kinds = ("primary_initial", "hard_retain", "primary_update")
    return {"action_accuracy": statistics.fmean(row["correct"] for row in rows),
            "by_event_kind": {kind: {"accuracy": statistics.fmean(
                row["correct"] for row in rows if row["event_kind"] == kind),
                "events": sum(row["event_kind"] == kind for row in rows)} for kind in kinds}}


def select_causal_threshold(sessions, scorer, artifact):
    grid = {}
    for threshold in THRESHOLDS:
        candidate = copy.deepcopy(artifact); candidate["threshold"] = threshold
        grid[threshold] = causal_action_metrics(sessions, scorer, candidate)
    serializable = {str(key): value for key, value in grid.items()}
    threshold, status = select_constrained_threshold(serializable)
    return threshold, status, serializable


def oracle_candidate_accuracy(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    local = [row for row in rows if row["target"] < MAX_KEYS]
    return statistics.fmean(row["top"] == row["target"] for row in local)


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    zsre = []
    for split in ("calibration", "dev", "test"):
        zsre.extend(tag_sessions(load_split(args.zsre_root, split), f"zsre_{split}"))
    counterfact, data_manifest = load_transfer(args.counterfact_root)
    all_sessions = [*zsre, *counterfact]; zsre_ids = {row["id"] for row in zsre}
    counterfact_ids = {row["id"] for row in counterfact}
    session_table = {row["id"]: row for row in all_sessions}

    v2 = Path(args.v2_root); v2_training = json.loads((v2 / "training.json").read_text())
    fold_map = {key: int(value) for key, value in v2_training["counterfact_folds"].items()}
    names, pairs = all_names(all_sessions)
    source, scope = Path(args.source_root), Path(args.scope_root)
    context_summary = json.loads((source / "summary.json").read_text())
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
        context_summary["selected_config"]["regularizer"])
    pair_features = extract_pair_features(model, tokenizer, pairs, args.batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    pair_dim = next(iter(pair_features.values())).numel()
    del model; torch.cuda.empty_cache()

    artifacts, selection, state_counts = {}, {}, {}
    for outer in range(FOLDS):
        tune_fold = (outer + 1) % FOLDS
        checkpoint = v2 / f"mixed_oof_fold{outer}.pt"
        if v2_training["checkpoint_sha256"][checkpoint.name] != sha(checkpoint):
            raise ValueError("V2 checkpoint hash mismatch")
        base = torch.load(checkpoint, map_location="cpu", weights_only=False)
        inner_grid = v2_training["selection"][str(outer)]["inner_grid"]
        base_threshold, _ = select_constrained_threshold(inner_grid)
        base["threshold"] = base_threshold
        train_ids = zsre_ids | {session_id for session_id, fold in fold_map.items()
                                if fold not in (outer, tune_fold)}
        tune_ids = {session_id for session_id, fold in fold_map.items() if fold == tune_fold}
        train_sessions = [session_table[value] for value in sorted(train_ids)]
        tune_sessions = [session_table[value] for value in sorted(tune_ids)]
        groups = collect_onpolicy_groups(train_sessions, scorer, base)
        values = dynamic_feature_tensor(groups, scorer); counts = candidate_counts(groups)
        cosines = cosine_values(values, pair_dim)
        artifact = fit_onpolicy_gate(groups, values, cosines, counts, base, 12101 + outer)
        threshold, status, grid = select_causal_threshold(tune_sessions, scorer, artifact)
        artifact["threshold"] = threshold; artifacts[outer] = artifact
        torch.save(artifact, root / f"onpolicy_oof_fold{outer}.pt")
        state_counts[str(outer)] = {
            "groups": len(groups), "local_available": sum(
                group["desired"] == "local" and group["old_chain_present"] for group in groups),
            "local_missing": sum(group["desired"] == "local" and not group["old_chain_present"]
                                 for group in groups),
            "retain": sum(group["desired"] == "retain" for group in groups),
        }
        selection[str(outer)] = {"threshold": threshold, "status": status,
                                 "causal_inner_grid": grid, "tune_fold": tune_fold}

    records, metrics, raw_hashes = evaluate_records(
        counterfact, scorer, lambda session_id: artifacts[fold_map[session_id]],
        "counterfact_observed_onpolicy_v3", root, "counterfact_onpolicy_v3")
    candidate_accuracy = oracle_candidate_accuracy(
        v2 / "counterfact_mixed_oof_oracle_events.jsonl")
    # Candidate top-1 is unchanged because V3 freezes every V2 ranker.
    oracle_proxy = {"candidate_top1_accuracy": candidate_accuracy}
    v1_records, v1_summary = load_v1_records(args.v1_root)
    old_records, old_summary = load_previous(args.previous_root)
    fixed_retain_query = old_summary["policies"]["fixed_retain"]["mean_accuracy"]
    gate_proxy = {**oracle_proxy}
    gate = development_gate(metrics, gate_proxy, fixed_retain_query)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30

    summary = {
        "protocol": PROTOCOL, "data_status": "observed_development_only",
        "metrics": metrics, "candidate_top1_accuracy": candidate_accuracy,
        "v1_mixed_oof": v1_summary["mixed_counterfact_oof"]["metrics"],
        "onpolicy_vs_v1": paired_comparison(records, v1_records, 12201),
        "onpolicy_vs_previous": paired_comparison(records, old_records, 12211),
        "development_gate": gate,
        "terminal_status": "completed_onpolicy_forgetting_gate_v3",
    }
    training = {
        "seconds": time.perf_counter() - started,
        "gpu_peak_allocated_gib": peak_allocated, "gpu_peak_reserved_gib": peak_reserved,
        "features": {"unique_texts": len(names), "unique_pairs": len(pairs),
                     "embedding_hashes": embedding_hashes,
                     "pair_features_sha256": table_hash(pair_features)},
        "state_counts": state_counts, "selection": selection,
        "counterfact_folds": fold_map, "raw_rows_sha256": raw_hashes,
        "checkpoint_sha256": {f"onpolicy_oof_fold{fold}.pt": sha(
            root / f"onpolicy_oof_fold{fold}.pt") for fold in range(FOLDS)},
    }
    manifest = {
        "protocol": PROTOCOL, "v2_training_sha256": sha(v2 / "training.json"),
        "counterfact_manifest_sha256": sha(Path(args.counterfact_root) / "manifest.json"),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "folds": FOLDS,
        "gate_epochs": GATE_EPOCHS, "retain_loss_weight": RETAIN_LOSS_WEIGHT,
        "thresholds": list(THRESHOLDS), "current_write_gate": 1,
        "ranker_frozen": True, "chain_identity_in_features": False,
        "counterfact_status": "observed_development_only",
        "longmemeval_scored": False, "locomo_scored": False,
        "units": data_manifest["eligible_subjects"],
    }
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    write_json(root / "training_manifest.json", manifest); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-root", required=True)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--v1-root", required=True)
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
