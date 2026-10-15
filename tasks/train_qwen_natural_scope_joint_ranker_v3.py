"""Calibration-only OOF joint key/defer ranker for natural Qwen memory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_qwen_natural_causal_memory_v1 import read_stage
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    candidate_scores,
    generate_fixed,
    prepare,
    value_rows,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import (
    GATE_SEED,
    FOLDS,
    bootstrap_group,
    build_examples,
    build_scorer,
)
from tasks.train_zsre_scope_resolver import MAX_KEYS, PAIR_TEMPLATE


PROTOCOL = "qwen_natural_scope_joint_ranker_v3"
PLAN = "QWEN_NATURAL_SCOPE_JOINT_RANKER_V3_PLAN.md"
ANCHORS = (.001, .01, .1, 1.)
LEARNING_RATES = (.001, .005)
EPOCHS = (25, 50, 100)
DEFER_OFFSETS = tuple(value / 4 for value in range(-4, 5))
STRATA = (
    "primary_exact", "primary_rephrase", "auxiliary_exact",
    "auxiliary_rephrase", "no_match_exact", "no_match_rephrase",
)


class JointRanker:
    def __init__(self, dim, initial_state, device):
        import torch

        self.linear = torch.nn.Linear(dim, 1).to(device)
        self.linear.load_state_dict(initial_state)
        self.defer = torch.nn.Parameter(torch.tensor(math.log(.4 / .6), device=device))

    def parameters(self):
        return [*self.linear.parameters(), self.defer]

    def logits(self, features, mask, defer_offset=0.):
        import torch

        key = self.linear(features).squeeze(-1)
        key = key.masked_fill(~mask, float("-inf"))
        defer = (self.defer + defer_offset).expand(len(features), 1)
        return torch.cat((key, defer), dim=1)

    def state_dict(self):
        return {"linear": self.linear.state_dict(), "defer": self.defer.detach().clone()}


def encode_candidates(examples, scorer):
    import torch

    features, masks, labels, metadata = [], [], [], []
    for example_index, row in enumerate(examples):
        result, top_feature = scorer.encode(row["query"], row["keys"], GATE_SEED)
        geometry = top_feature[-4:]
        values = [torch.cat((scorer.pair_features[
            PAIR_TEMPLATE.format(key=key, query=row["query"])], geometry))
                  for key in row["keys"]]
        dim = values[0].numel(); count = len(values)
        values.extend(torch.zeros(dim) for _ in range(MAX_KEYS - count))
        mask = torch.tensor([True] * count + [False] * (MAX_KEYS - count))
        if row["kind"] == "natural":
            matches = [index for index, record in enumerate(row["memory"])
                       if record["chain_id"] == row["expected_chain_id"]
                       and record["version"] == row["expected_version"]]
            if len(matches) != 1:
                raise ValueError("natural query has no unique active record")
            label = matches[0]
            source = row["memory"][label]
            stratum = f"{source['chain_kind']}_{row['surface']}"
        else:
            label = MAX_KEYS; stratum = f"no_match_{row['surface']}"
        if stratum not in STRATA:
            raise ValueError(f"unregistered ranker stratum {stratum}")
        features.append(torch.stack(values)); masks.append(mask); labels.append(label)
        meta = {key: value for key, value in row.items() if key not in ("keys", "memory")}
        meta.update({"example_index": example_index, "stratum": stratum,
                     "target": label, "memory": row["memory"],
                     "frozen_top": result["top"]})
        metadata.append(meta)
    return torch.stack(features), torch.stack(masks), torch.tensor(labels), metadata


def sample_weights(metadata, indices):
    import torch

    counts = Counter(metadata[index]["stratum"] for index in indices)
    if set(counts) != set(STRATA):
        raise ValueError("each ranker training fold must contain all six strata")
    return torch.tensor([1. / (len(STRATA) * counts[metadata[index]["stratum"]])
                         for index in indices])


def fit(features, masks, labels, metadata, indices, initial_state, anchor, lr, epochs):
    import torch

    device = features.device
    ranker = JointRanker(features.shape[-1], initial_state, device)
    initial_weight = initial_state["weight"].to(device)
    initial_bias = initial_state["bias"].to(device)
    initial_defer = torch.tensor(math.log(.4 / .6), device=device)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr, weight_decay=0.)
    index = torch.tensor(indices, dtype=torch.long, device=device)
    weights = sample_weights(metadata, indices).to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = ranker.logits(features[index], masks[index])
        losses = torch.nn.functional.cross_entropy(logits, labels[index], reduction="none")
        data_loss = (losses * weights).sum() / weights.sum()
        penalty = ((ranker.linear.weight - initial_weight).square().mean()
                   + (ranker.linear.bias - initial_bias).square().mean()
                   + (ranker.defer - initial_defer).square())
        loss = data_loss + anchor * penalty
        loss.backward(); optimizer.step()
    return ranker


def ranker_metrics(logits, metadata, defer_offset):
    adjusted = logits.clone(); adjusted[:, MAX_KEYS] += defer_offset
    prediction = adjusted.argmax(-1)
    output = {}
    per_stratum = {}
    for stratum in STRATA:
        indices = [index for index, row in enumerate(metadata) if row["stratum"] == stratum]
        value = float((prediction[indices] == __import__("torch").tensor(
            [metadata[index]["target"] for index in indices])).float().mean())
        per_stratum[stratum] = {"examples": len(indices), "accuracy": value}
    natural = [index for index, row in enumerate(metadata) if row["kind"] == "natural"]
    no_match = [index for index, row in enumerate(metadata) if row["kind"] == "no_match"]
    exact = [index for index in natural if metadata[index]["surface"] == "exact"]
    rephrase = [index for index in natural if metadata[index]["surface"] == "rephrase"]
    targets = __import__("torch").tensor([row["target"] for row in metadata])
    def acc(indices):
        return float((prediction[indices] == targets[indices]).float().mean())
    wrong_key = [index for index in natural
                 if prediction[index] != MAX_KEYS and prediction[index] != targets[index]]
    deferred = [index for index in natural if prediction[index] == MAX_KEYS]
    output.update({
        "natural_selection_accuracy": acc(natural),
        "exact_selection_accuracy": acc(exact),
        "rephrase_selection_accuracy": acc(rephrase),
        "auxiliary_rephrase_selection_accuracy": per_stratum[
            "auxiliary_rephrase"]["accuracy"],
        "no_match_specificity": acc(no_match),
        "macro_accuracy": statistics.fmean(row["accuracy"] for row in per_stratum.values()),
        "natural_wrong_key_rate": len(wrong_key) / len(natural),
        "natural_defer_rate": len(deferred) / len(natural),
        "per_stratum": per_stratum,
    })
    return output, prediction


def search(features, masks, labels, metadata, initial_state):
    import torch

    device = torch.device("cuda")
    values, valid, targets = features.to(device), masks.to(device), labels.to(device)
    folds = [row["fold"] for row in metadata]
    records, logits_by_config = [], {}
    for anchor in ANCHORS:
        for lr in LEARNING_RATES:
            for epochs in EPOCHS:
                key = (anchor, lr, epochs)
                oof = torch.full((len(metadata), MAX_KEYS + 1), float("nan"), device=device)
                for fold in range(FOLDS):
                    train = [index for index, value in enumerate(folds) if value != fold]
                    holdout = [index for index, value in enumerate(folds) if value == fold]
                    ranker = fit(values, valid, targets, metadata, train, initial_state,
                                 anchor, lr, epochs)
                    with torch.no_grad():
                        index = torch.tensor(holdout, dtype=torch.long, device=device)
                        oof[index] = ranker.logits(values[index], valid[index])
                if torch.isnan(oof).any():
                    raise ValueError("joint-ranker OOF logits are incomplete")
                logits_by_config[key] = oof.detach().cpu()
                for offset in DEFER_OFFSETS:
                    metric, _ = ranker_metrics(logits_by_config[key], metadata, offset)
                    records.append({"anchor": anchor, "learning_rate": lr,
                                    "epochs": epochs, "defer_offset": offset, **metric})
    eligible = [row for row in records if row["no_match_specificity"] >= .92
                and row["exact_selection_accuracy"] >= .98]
    if not eligible:
        return records, None, None
    selected = max(eligible, key=lambda row: (
        row["rephrase_selection_accuracy"], row["natural_selection_accuracy"],
        row["no_match_specificity"], row["macro_accuracy"], row["anchor"],
        -row["learning_rate"], -row["epochs"], -abs(row["defer_offset"]),
    ))
    key = (selected["anchor"], selected["learning_rate"], selected["epochs"])
    return records, selected, logits_by_config[key]


def exact_oof_score(model, tokenizer, source_rows, v2_rows, metadata, prediction):
    natural = sorted((row for row in metadata if row["kind"] == "natural"),
                     key=lambda row: row["row_index"])
    if len(natural) != len(source_rows) or len(v2_rows) != len(source_rows):
        raise ValueError("V3 source rows do not align")
    output = []
    for meta, source, v2 in zip(natural, source_rows, v2_rows):
        selected = int(prediction[meta["example_index"]])
        if selected == meta["target"]:
            result = source["arms"]["oracle_oracle"]; score_source = "oracle_oracle"
        elif selected == MAX_KEYS:
            result = source["arms"]["zero"]; score_source = "zero"
        else:
            record = meta["memory"][selected]
            prepared = prepare(model, tokenizer, source["query"], source["candidates"])
            residual = value_rows(model, prepared["encoded"][record["target"]])
            scores = candidate_scores(model, prepared, residual, 8.0)
            answer = min(source["candidates"], key=lambda value: (scores[value], value))
            result = {"candidate_correct": int(answer == source["answer"]),
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
            score_source = "new_wrong_key"
        output.append({
            "session_id": source["session_id"], "group_id": source["group_id"],
            "phase": source["phase"], "surface": source["surface"],
            "chain_kind": source["chain_kind"], "final": source["final"],
            "selection_correct": int(selected == meta["target"]),
            "defer": int(selected == MAX_KEYS), "score_source": score_source,
            "candidate_correct": result["candidate_correct"],
            "answer_nll": result["answer_nll"],
            "generation_exact": (result.get("generation", {}).get("token_exact")
                                 if source["final"] else None),
            "v2_candidate_correct": v2["candidate_correct"],
            "oracle_candidate_correct": source["arms"]["oracle_oracle"]["candidate_correct"],
        })
    return output


def end_to_end(rows):
    final = [row for row in rows if row["final"]]
    gains = {}
    for group in sorted({row["group_id"] for row in rows}):
        part = [row for row in rows if row["group_id"] == group]
        gains[group] = statistics.fmean(
            row["candidate_correct"] - row["v2_candidate_correct"] for row in part
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
        "gain_over_v2": statistics.fmean(gains.values()),
        "gain_bootstrap_95ci": bootstrap_group(gains, 202610191),
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
    source_rows, states, resources = read_stage(args.v1_root, "calibration")
    v2_path = Path(args.v2_root) / "result" / "oof_rows.jsonl"
    v2_rows = [json.loads(line) for line in v2_path.read_text().splitlines() if line]
    examples = build_examples(sessions, source_rows, states)
    source, scope = Path(args.context_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_training = json.loads((scope / "training.json").read_text())
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    if scope_training["checkpoint_sha256"] != sha(scope / "scope_gates.pt"):
        raise ValueError("scope checkpoint mismatch")
    torch.manual_seed(202610191); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    scorer, feature_meta = build_scorer(
        examples, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], gate_artifact, args.batch_size,
    )
    features, masks, labels, metadata = encode_candidates(examples, scorer)
    torch.save({"features": features.half(), "masks": masks, "labels": labels,
                "metadata": metadata}, root / "calibration_candidate_features.pt")
    initial_state = gate_artifact["heads"][str(GATE_SEED)]
    records, selected, logits = search(features, masks, labels, metadata, initial_state)
    write_json(root / "search.json", {"records": records, "selected": selected})
    if selected is None:
        summary = {"protocol": PROTOCOL, "stage": "calibration", "selected": None,
                   "gate": {"checks": {"constraints_feasible": False}, "passed": False},
                   "development_loaded": False, "test_loaded": False}
        write_json(root / "summary.json", summary); return summary
    decision, prediction = ranker_metrics(logits, metadata, selected["defer_offset"])
    oof_rows = exact_oof_score(model, tokenizer, source_rows, v2_rows, metadata, prediction)
    with (root / "oof_rows.jsonl").open("x") as sink:
        for row in oof_rows: sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = end_to_end(oof_rows)
    all_indices = list(range(len(metadata)))
    final = fit(features.to("cuda"), masks.to("cuda"), labels.to("cuda"), metadata,
                all_indices, initial_state, selected["anchor"], selected["learning_rate"],
                selected["epochs"])
    checkpoint = {"protocol": PROTOCOL,
                  "linear": {key: value.detach().cpu()
                             for key, value in final.linear.state_dict().items()},
                  "defer": final.defer.detach().cpu(), "feature_dim": features.shape[-1],
                  "selected": selected, "defer_offset": selected["defer_offset"],
                  "folds": FOLDS, "strata": STRATA, "seed": GATE_SEED}
    torch.save(checkpoint, root / "joint_scope_ranker.pt")
    checks = {
        "oof_natural_selection_at_least_point_93":
            decision["natural_selection_accuracy"] >= .93,
        "oof_exact_selection_at_least_point_98":
            decision["exact_selection_accuracy"] >= .98,
        "oof_rephrase_selection_at_least_point_88":
            decision["rephrase_selection_accuracy"] >= .88,
        "oof_aux_rephrase_selection_at_least_point_85":
            decision["auxiliary_rephrase_selection_accuracy"] >= .85,
        "oof_no_match_specificity_at_least_point_92": decision["no_match_specificity"] >= .92,
        "oof_macro_accuracy_at_least_point_92": decision["macro_accuracy"] >= .92,
        "candidate_accuracy_at_least_point_93": metric["candidate_accuracy"] >= .93,
        "oracle_gap_at_most_point_055": metric["oracle_gap"] <= .055,
        "final_candidate_accuracy_at_least_point_91": metric["final_candidate_accuracy"] >= .91,
        "final_generation_exact_at_least_point_74": metric["final_generation_exact"] >= .74,
        "gain_over_v2_ci_lower_above_zero": metric["gain_bootstrap_95ci"][0] > 0,
    }
    summary = {
        "protocol": PROTOCOL, "stage": "calibration", "selected": selected,
        "decision_metrics": decision, "end_to_end": metric,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "examples": dict(sorted(Counter(row["stratum"] for row in metadata).items())),
        "feature_meta": feature_meta,
        "resources": {"seconds": time.perf_counter() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "candidate_features_sha256": sha(root / "calibration_candidate_features.pt"),
        "oof_rows_sha256": sha(root / "oof_rows.jsonl"),
        "checkpoint_sha256": sha(root / "joint_scope_ranker.pt"),
        "plan_sha256": sha(PLAN), "v2_rows_sha256": sha(v2_path),
        "v1_row_hashes": [row["rows_sha256"] for row in resources],
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(root / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "v1_root", "v2_root", "model", "tokenizer",
                 "context_root", "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
