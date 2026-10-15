"""Calibration-only OOF repair of the natural-memory retrieve/defer gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load, normalize
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    BOOTSTRAP_DRAWS,
    read_stage,
)
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    candidate_scores,
    generate_fixed,
    prepare,
    value_rows,
)
from tasks.evaluate_zsre_contextual_readout import apply, extract
from tasks.evaluate_zsre_natural_forgetting import FrozenScopeScorer, table_hash
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_decomposed_scope_gate import BinaryGate
from tasks.train_zsre_scope_resolver import PAIR_TEMPLATE, extract_pair_features, tensor_hash


PROTOCOL = "qwen_natural_scope_transfer_v2"
PLAN = "QWEN_NATURAL_SCOPE_TRANSFER_V2_PLAN.md"
GATE_SEED = 7301
FOLDS = 5
ANCHORS = (.01, .1, 1., 10.)
LEARNING_RATES = (.001, .005)
EPOCHS = (25, 50, 100)
THRESHOLDS = tuple(value / 100 for value in range(10, 91, 5))
STRATA = ("natural_correct_top", "natural_wrong_top", "no_match")
BOOTSTRAP_SEED = 202610181


def fold_for(session_id):
    digest = hashlib.sha256(f"natural-scope-transfer-v2|{session_id}".encode()).hexdigest()
    return int(digest, 16) % FOLDS


def snapshot_map(states):
    return {(row["session_id"], snapshot["phase"]): snapshot
            for row in states for snapshot in row["snapshots"]}


def subject_map(session):
    return {chain["id"]: chain["subject"] for chain in session["chains"]}


def donor_for(session, phase, memory, sessions_by_fold):
    memory_targets = {normalize(row["target"]) for row in memory}
    memory_subjects = set(subject_map(session).values())
    candidates = []
    for donor in sessions_by_fold[fold_for(session["id"])]:
        if donor["id"] == session["id"]:
            continue
        for chain in donor["chains"]:
            version = chain["versions"][0]
            if chain["subject"] in memory_subjects or normalize(version["target"]) in memory_targets:
                continue
            rank = hashlib.sha256(
                f"natural-scope-transfer-v2|{session['id']}|{phase}|{chain['id']}".encode()
            ).hexdigest()
            candidates.append((rank, donor["id"], chain["id"], chain["subject"], version))
    if not candidates:
        raise ValueError("no subject/target-disjoint in-fold donor")
    return min(candidates)


def build_examples(sessions, rows, states):
    by_id = {session["id"]: session for session in sessions}
    snapshots = snapshot_map(states)
    examples = []
    for index, row in enumerate(rows):
        snapshot = snapshots[(row["session_id"], row["phase"])]
        memory = snapshot["states"]["oracle"]
        examples.append({
            "id": f"natural|{index}", "kind": "natural", "row_index": index,
            "session_id": row["session_id"], "fold": fold_for(row["session_id"]),
            "phase": row["phase"], "surface": row["surface"],
            "query": row["query"], "keys": [record["prompt"] for record in memory],
            "memory": memory, "expected_chain_id": row["chain_id"],
            "expected_version": row["active_version"],
        })
    sessions_by_fold = {fold: sorted(
        [session for session in sessions if fold_for(session["id"]) == fold],
        key=lambda row: row["id"],
    ) for fold in range(FOLDS)}
    if any(len(part) < 2 for part in sessions_by_fold.values()):
        raise ValueError("five-fold donor construction needs two sessions per fold")
    for session in sessions:
        state = next(row for row in states if row["session_id"] == session["id"])
        for snapshot in state["snapshots"]:
            memory = snapshot["states"]["oracle"]
            _, donor_session, donor_chain, donor_subject, version = donor_for(
                session, snapshot["phase"], memory, sessions_by_fold
            )
            for surface, query in (("exact", version["prompt"]),
                                   ("rephrase", version["rephrase_prompt"])):
                examples.append({
                    "id": f"negative|{session['id']}|{snapshot['phase']}|{surface}",
                    "kind": "no_match", "row_index": None,
                    "session_id": session["id"], "fold": fold_for(session["id"]),
                    "phase": snapshot["phase"], "surface": surface,
                    "query": query, "keys": [record["prompt"] for record in memory],
                    "memory": memory, "donor_session_id": donor_session,
                    "donor_chain_id": donor_chain, "donor_subject": donor_subject,
                })
    if len({row["id"] for row in examples}) != len(examples):
        raise ValueError("scope-transfer example ids are not unique")
    return examples


def build_scorer(examples, model, tokenizer, transforms, regularizer,
                 gate_artifact, batch_size):
    names = sorted({text for row in examples for text in (row["query"], *row["keys"])})
    pairs = sorted({PAIR_TEMPLATE.format(key=key, query=row["query"])
                    for row in examples for key in row["keys"]})
    raw = extract(model, tokenizer, {"entity": names, "value": ["unknown"]}, batch_size)
    embeddings, embedding_hashes = apply(
        raw, {"entity": names, "value": ["unknown"]}, transforms, regularizer
    )
    pair_features = extract_pair_features(model, tokenizer, pairs, batch_size)
    scorer = FrozenScopeScorer(embeddings, pair_features, gate_artifact)
    return scorer, {"unique_texts": len(names), "unique_pairs": len(pairs),
                    "embedding_hashes": embedding_hashes,
                    "pair_features_sha256": table_hash(pair_features)}


def encode_examples(examples, scorer):
    import torch

    features = []; labels = []; metadata = []
    for example_index, row in enumerate(examples):
        result, feature = scorer.encode(row["query"], row["keys"], GATE_SEED)
        top = result["top"]; record = row["memory"][top]
        top_correct = int(row["kind"] == "natural"
                          and record["chain_id"] == row["expected_chain_id"]
                          and record["version"] == row["expected_version"])
        stratum = ("no_match" if row["kind"] == "no_match" else
                   "natural_correct_top" if top_correct else "natural_wrong_top")
        features.append(feature); labels.append(float(top_correct))
        metadata.append({key: value for key, value in row.items() if key not in ("keys", "memory")})
        metadata[-1].update({
            "example_index": example_index,
            "stratum": stratum, "label": top_correct, "top": top,
            "top_chain_id": record["chain_id"], "top_version": record["version"],
            "top_target": record["target"],
            "frozen_probability": result["retrieve_probability"],
        })
    return torch.stack(features), torch.tensor(labels), metadata


def stratum_weights(strata, indices):
    import torch

    counts = Counter(strata[index] for index in indices)
    if set(counts) != set(STRATA):
        raise ValueError("each training fold must contain all registered strata")
    return torch.tensor([1. / (len(STRATA) * counts[strata[index]]) for index in indices])


def fit_head(features, labels, strata, indices, initial_state, anchor, lr, epochs):
    import torch

    device = features.device
    head = BinaryGate(features.shape[1], GATE_SEED)
    head.load_state_dict(initial_state); head.linear.to(device)
    initial = {key: value.detach().to(device).clone() for key, value in initial_state.items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
    index = torch.tensor(indices, dtype=torch.long, device=device)
    weights = stratum_weights(strata, indices).to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features[index])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[index], reduction="none"
        )
        data_loss = (losses * weights).sum() / weights.sum()
        penalty = ((head.linear.weight - initial["weight"]).square().mean()
                   + (head.linear.bias - initial["bias"]).square().mean())
        loss = data_loss + anchor * penalty
        loss.backward(); optimizer.step()
    return head


def decision_metrics(logits, metadata, threshold):
    probability = logits.sigmoid(); prediction = probability >= threshold
    output = {}
    for stratum in STRATA:
        indices = [index for index, row in enumerate(metadata) if row["stratum"] == stratum]
        if stratum == "natural_correct_top":
            value = float(prediction[indices].float().mean())
            output["correct_top_recall"] = value
        elif stratum == "natural_wrong_top":
            value = float((~prediction[indices]).float().mean())
            output["natural_wrong_top_specificity"] = value
        else:
            value = float((~prediction[indices]).float().mean())
            output["no_match_specificity"] = value
        output[f"{stratum}_examples"] = len(indices)
    output["balanced_accuracy"] = statistics.fmean((
        output["correct_top_recall"], output["natural_wrong_top_specificity"],
        output["no_match_specificity"],
    ))
    return output


def search(features, labels, metadata, initial_state):
    import torch

    device = torch.device("cuda")
    values = features.to(device); targets = labels.to(device)
    strata = [row["stratum"] for row in metadata]
    folds = [row["fold"] for row in metadata]
    records, oof_logits = [], {}
    for anchor in ANCHORS:
        for lr in LEARNING_RATES:
            for epochs in EPOCHS:
                key = (anchor, lr, epochs)
                logits = torch.full((len(metadata),), float("nan"), device=device)
                for fold in range(FOLDS):
                    train = [index for index, value in enumerate(folds) if value != fold]
                    valid = [index for index, value in enumerate(folds) if value == fold]
                    head = fit_head(values, targets, strata, train, initial_state,
                                    anchor, lr, epochs)
                    with torch.no_grad():
                        index = torch.tensor(valid, dtype=torch.long, device=device)
                        logits[index] = head(values[index])
                if torch.isnan(logits).any():
                    raise ValueError("OOF logits are incomplete")
                oof_logits[key] = logits.detach().cpu()
                for threshold in THRESHOLDS:
                    metric = decision_metrics(oof_logits[key], metadata, threshold)
                    records.append({"anchor": anchor, "learning_rate": lr,
                                    "epochs": epochs, "threshold": threshold, **metric})
    eligible = [row for row in records
                if row["no_match_specificity"] >= .92
                and row["natural_wrong_top_specificity"] >= .85]
    if not eligible:
        return records, None, None
    selected = max(eligible, key=lambda row: (
        row["correct_top_recall"], row["no_match_specificity"],
        row["natural_wrong_top_specificity"], row["balanced_accuracy"],
        row["anchor"], -row["learning_rate"], -row["epochs"],
        -abs(row["threshold"] - .5),
    ))
    key = (selected["anchor"], selected["learning_rate"], selected["epochs"])
    return records, selected, oof_logits[key]


def bootstrap_group(values, seed=BOOTSTRAP_SEED):
    groups = sorted(values); rng = random.Random(seed); samples = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        samples.append(statistics.fmean(values[group] for group in chosen))
    samples.sort()
    return [samples[max(0, math.ceil(.025 * len(samples)) - 1)],
            samples[min(len(samples) - 1, math.ceil(.975 * len(samples)) - 1)]]


def exact_oof_score(model, tokenizer, rows, metadata, decisions):
    natural = sorted((row for row in metadata if row["kind"] == "natural"),
                     key=lambda row: row["row_index"])
    if len(natural) != len(rows):
        raise ValueError("natural OOF rows do not cover V1 rows")
    output = []
    for meta, source in zip(natural, rows):
        retrieve = bool(decisions[meta["example_index"]])
        if retrieve and meta["stratum"] == "natural_correct_top":
            result = source["arms"]["oracle_oracle"]; score_source = "oracle_oracle"
        elif not retrieve:
            result = source["arms"]["zero"]; score_source = "zero"
        else:
            prepared = prepare(model, tokenizer, source["query"], source["candidates"])
            residual = value_rows(model, prepared["encoded"][meta["top_target"]])
            scores = candidate_scores(model, prepared, residual, 8.0)
            prediction = min(source["candidates"], key=lambda value: (scores[value], value))
            result = {"scores": scores, "prediction": prediction,
                      "candidate_correct": int(prediction == source["answer"]),
                      "answer_nll": scores[source["answer"]]}
            if source["final"]:
                generation = generate_fixed(
                    model, tokenizer, prepared, residual, 8.0,
                    len(prepared["encoded"][source["answer"]]),
                )
                generation["token_exact"] = int(
                    generation["ids"] == prepared["encoded"][source["answer"]]
                )
                result["generation"] = generation
            score_source = "new_wrong_top"
        output.append({
            "session_id": source["session_id"], "group_id": source["group_id"],
            "phase": source["phase"], "surface": source["surface"],
            "chain_kind": source["chain_kind"], "final": source["final"],
            "retrieve": int(retrieve), "top_correct": int(
                meta["stratum"] == "natural_correct_top"),
            "score_source": score_source,
            "candidate_correct": result["candidate_correct"],
            "answer_nll": result["answer_nll"],
            "generation_exact": (result.get("generation", {}).get("token_exact")
                                 if source["final"] else None),
            "v1_candidate_correct": source["arms"]["oracle_auto"]["candidate_correct"],
            "oracle_candidate_correct": source["arms"]["oracle_oracle"]["candidate_correct"],
        })
    return output


def end_to_end_metrics(rows):
    final = [row for row in rows if row["final"]]
    gains = {}
    for group in sorted({row["group_id"] for row in rows}):
        part = [row for row in rows if row["group_id"] == group]
        gains[group] = statistics.fmean(
            row["candidate_correct"] - row["v1_candidate_correct"] for row in part
        )
    accuracy = statistics.fmean(row["candidate_correct"] for row in rows)
    oracle = statistics.fmean(row["oracle_candidate_correct"] for row in rows)
    return {
        "candidate_accuracy": accuracy,
        "answer_nll": statistics.fmean(row["answer_nll"] for row in rows),
        "oracle_oracle_candidate_accuracy": oracle,
        "oracle_gap": oracle - accuracy,
        "final_candidate_accuracy": statistics.fmean(
            row["candidate_correct"] for row in final),
        "final_generation_exact": statistics.fmean(
            row["generation_exact"] for row in final),
        "gain_over_v1_oracle_auto": statistics.fmean(gains.values()),
        "gain_bootstrap_95ci": bootstrap_group(gains),
        "subject_groups": len(gains),
        "score_sources": dict(sorted(Counter(row["score_source"] for row in rows).items())),
        "by_surface": {surface: statistics.fmean(
            row["candidate_correct"] for row in rows if row["surface"] == surface
        ) for surface in ("exact", "rephrase")},
    }


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _, sessions = load(args.data_root, "calibration")
    rows, states, resources = read_stage(args.v1_root, "calibration")
    examples = build_examples(sessions, rows, states)
    source = Path(args.context_root); scope = Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms_artifact = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    torch.manual_seed(202610181); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    scorer, feature_meta = build_scorer(
        examples, model, tokenizer, transforms_artifact["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size,
    )
    features, labels, metadata = encode_examples(examples, scorer)
    torch.save({"features": features, "labels": labels, "metadata": metadata},
               root / "calibration_features.pt")
    feature_hash = sha(root / "calibration_features.pt")
    initial_state = gate_artifact["heads"][str(GATE_SEED)]
    search_records, selected, logits = search(features, labels, metadata, initial_state)
    write_json(root / "search.json", {"records": search_records, "selected": selected})
    if selected is None:
        summary = {"protocol": PROTOCOL, "stage": "calibration",
                   "selected": None, "gate": {"passed": False,
                   "checks": {"specificity_feasible": False}},
                   "development_loaded": False, "test_loaded": False}
        write_json(root / "summary.json", summary)
        return summary
    decisions = logits.sigmoid() >= selected["threshold"]
    oof_rows = exact_oof_score(model, tokenizer, rows, metadata, decisions)
    with (root / "oof_rows.jsonl").open("x") as sink:
        for row in oof_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = end_to_end_metrics(oof_rows)
    decision = decision_metrics(logits, metadata, selected["threshold"])
    all_indices = list(range(len(metadata))); strata = [row["stratum"] for row in metadata]
    final_head = fit_head(features.to("cuda"), labels.to("cuda"), strata, all_indices,
                          initial_state, selected["anchor"], selected["learning_rate"],
                          selected["epochs"])
    checkpoint = {
        "protocol": PROTOCOL, "head": {key: value.detach().cpu()
                                         for key, value in final_head.state_dict().items()},
        "feature_dim": features.shape[1], "seed": GATE_SEED,
        "selected": selected, "threshold": selected["threshold"],
        "folds": FOLDS, "strata": STRATA,
    }
    torch.save(checkpoint, root / "scope_transfer_gate.pt")
    checks = {
        "oof_correct_top_recall_at_least_point_90": decision["correct_top_recall"] >= .90,
        "oof_no_match_specificity_at_least_point_92": decision["no_match_specificity"] >= .92,
        "oof_wrong_top_specificity_at_least_point_85":
            decision["natural_wrong_top_specificity"] >= .85,
        "oof_balanced_accuracy_at_least_point_91": decision["balanced_accuracy"] >= .91,
        "candidate_accuracy_at_least_point_90": metric["candidate_accuracy"] >= .90,
        "oracle_gap_at_most_point_06": metric["oracle_gap"] <= .06,
        "final_candidate_accuracy_at_least_point_88": metric["final_candidate_accuracy"] >= .88,
        "final_generation_exact_at_least_point_70": metric["final_generation_exact"] >= .70,
        "gain_ci_lower_above_zero": metric["gain_bootstrap_95ci"][0] > 0,
    }
    summary = {
        "protocol": PROTOCOL, "stage": "calibration", "selected": selected,
        "decision_metrics": decision, "end_to_end": metric,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "examples": dict(sorted(Counter(row["stratum"] for row in metadata).items())),
        "feature_meta": feature_meta,
        "resources": {
            "seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
        "calibration_features_sha256": feature_hash,
        "oof_rows_sha256": sha(root / "oof_rows.jsonl"),
        "checkpoint_sha256": sha(root / "scope_transfer_gate.pt"),
        "plan_sha256": sha(PLAN),
        "v1_row_hashes": [row["rows_sha256"] for row in resources],
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "v1_root", "model", "tokenizer", "context_root",
                 "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
