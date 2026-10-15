"""Calibration-only nonlinear residual gate over the frozen natural top key."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from tasks.evaluate_qwen_natural_causal_memory_v1 import read_stage
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import (
    FOLDS,
    STRATA,
    bootstrap_group,
    decision_metrics,
    exact_oof_score,
    stratum_weights,
)


PROTOCOL = "qwen_natural_scope_residual_gate_v4"
PLAN = "QWEN_NATURAL_SCOPE_RESIDUAL_GATE_V4_PLAN.md"
EXPECTED_FEATURE_SHA256 = "1d3f89ee7db51208b1d1934d7229a6b1c5d3e302cc2ebc6a67b8c6c09535a140"
HIDDEN = (16, 32)
PENALTIES = (.01, .1, 1.)
LEARNING_RATES = (.001, .005)
EPOCHS = (25, 50)
THRESHOLDS = tuple(value / 100 for value in range(10, 91, 5))
SEED = 202610201


class ResidualGate:
    def __init__(self, dim, hidden, base_state, device):
        import torch

        torch.manual_seed(SEED)
        self.first = torch.nn.Linear(dim, hidden).to(device)
        torch.nn.init.kaiming_uniform_(self.first.weight, a=5 ** .5)
        torch.nn.init.zeros_(self.first.bias)
        self.second = torch.nn.Linear(hidden, 1).to(device)
        torch.nn.init.zeros_(self.second.weight); torch.nn.init.zeros_(self.second.bias)
        self.base_weight = base_state["weight"].detach().to(device)
        self.base_bias = base_state["bias"].detach().to(device)

    def parameters(self):
        return [*self.first.parameters(), *self.second.parameters()]

    def __call__(self, features):
        import torch

        base = features @ self.base_weight.t() + self.base_bias
        residual = self.second(torch.nn.functional.gelu(self.first(features)))
        return (base + residual).squeeze(-1)

    def state_dict(self):
        return {"first": self.first.state_dict(), "second": self.second.state_dict()}


def fit(features, labels, metadata, indices, base_state, hidden, penalty, lr, epochs):
    import torch

    device = features.device; model = ResidualGate(features.shape[1], hidden, base_state, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    index = torch.tensor(indices, dtype=torch.long, device=device)
    weights = stratum_weights([row["stratum"] for row in metadata], indices).to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True); logits = model(features[index])
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels[index], reduction="none"
        )
        data_loss = (losses * weights).sum() / weights.sum()
        parameters = model.parameters()
        regularizer = sum(parameter.square().mean() for parameter in parameters) / len(parameters)
        loss = data_loss + penalty * regularizer
        loss.backward(); optimizer.step()
    return model


def search(features, labels, metadata, base_state):
    import torch

    device = torch.device("cuda"); values = features.to(device); targets = labels.to(device)
    folds = [row["fold"] for row in metadata]
    records, logits_by_config = [], {}
    for hidden in HIDDEN:
        for penalty in PENALTIES:
            for lr in LEARNING_RATES:
                for epochs in EPOCHS:
                    key = (hidden, penalty, lr, epochs)
                    logits = torch.full((len(metadata),), float("nan"), device=device)
                    for fold in range(FOLDS):
                        train = [index for index, value in enumerate(folds) if value != fold]
                        holdout = [index for index, value in enumerate(folds) if value == fold]
                        model = fit(values, targets, metadata, train, base_state,
                                    hidden, penalty, lr, epochs)
                        with torch.no_grad():
                            index = torch.tensor(holdout, dtype=torch.long, device=device)
                            logits[index] = model(values[index])
                    if torch.isnan(logits).any():
                        raise ValueError("residual-gate OOF logits are incomplete")
                    logits_by_config[key] = logits.detach().cpu()
                    for threshold in THRESHOLDS:
                        metric = decision_metrics(logits_by_config[key], metadata, threshold)
                        records.append({"hidden": hidden, "penalty": penalty,
                                        "learning_rate": lr, "epochs": epochs,
                                        "threshold": threshold, **metric})
    eligible = [row for row in records if row["no_match_specificity"] >= .92
                and row["natural_wrong_top_specificity"] >= .85]
    if not eligible:
        return records, None, None
    selected = max(eligible, key=lambda row: (
        row["correct_top_recall"], row["no_match_specificity"],
        row["natural_wrong_top_specificity"], row["balanced_accuracy"],
        -row["hidden"], row["penalty"], -row["learning_rate"], -row["epochs"],
        -abs(row["threshold"] - .5),
    ))
    key = (selected["hidden"], selected["penalty"],
           selected["learning_rate"], selected["epochs"])
    return records, selected, logits_by_config[key]


def end_to_end(rows, v2_rows):
    if len(rows) != len(v2_rows):
        raise ValueError("V4 and V2 OOF row counts differ")
    final = [row for row in rows if row["final"]]
    gains = {}
    for group in sorted({row["group_id"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["group_id"] == group]
        gains[group] = statistics.fmean(
            rows[index]["candidate_correct"] - v2_rows[index]["candidate_correct"]
            for index in indices
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
        "gain_bootstrap_95ci": bootstrap_group(gains, 202610202),
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
    started = time.perf_counter(); v2_root = Path(args.v2_root)
    feature_path = v2_root / "result" / "calibration_features.pt"
    if sha(feature_path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("unexpected V2 feature artifact")
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    features, labels, metadata = artifact["features"], artifact["labels"], artifact["metadata"]
    scope = Path(args.scope_root)
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    base_state = gate_artifact["heads"]["7301"]
    torch.manual_seed(SEED); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    records, selected, logits = search(features, labels, metadata, base_state)
    write_json(root / "search.json", {"records": records, "selected": selected})
    if selected is None:
        summary = {"protocol": PROTOCOL, "stage": "calibration", "selected": None,
                   "gate": {"checks": {"constraints_feasible": False}, "passed": False},
                   "development_loaded": False, "test_loaded": False}
        write_json(root / "summary.json", summary); return summary
    decision = decision_metrics(logits, metadata, selected["threshold"])
    decisions = logits.sigmoid() >= selected["threshold"]
    source_rows, _, resources = read_stage(args.v1_root, "calibration")
    v2_rows = [json.loads(line) for line in
               (v2_root / "result" / "oof_rows.jsonl").read_text().splitlines() if line]
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    oof_rows = exact_oof_score(model, tokenizer, source_rows, metadata, decisions)
    with (root / "oof_rows.jsonl").open("x") as sink:
        for row in oof_rows: sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = end_to_end(oof_rows, v2_rows)
    all_indices = list(range(len(metadata)))
    final = fit(features.to("cuda"), labels.to("cuda"), metadata, all_indices, base_state,
                selected["hidden"], selected["penalty"], selected["learning_rate"],
                selected["epochs"])
    checkpoint = {"protocol": PROTOCOL, "state": {
        part: {key: value.detach().cpu() for key, value in values.items()}
        for part, values in final.state_dict().items()},
        "feature_dim": features.shape[1], "selected": selected, "folds": FOLDS,
        "strata": STRATA, "seed": SEED}
    torch.save(checkpoint, root / "residual_scope_gate.pt")
    checks = {
        "oof_correct_top_recall_at_least_point_92": decision["correct_top_recall"] >= .92,
        "oof_wrong_top_specificity_at_least_point_85":
            decision["natural_wrong_top_specificity"] >= .85,
        "oof_no_match_specificity_at_least_point_92": decision["no_match_specificity"] >= .92,
        "oof_balanced_accuracy_at_least_point_92": decision["balanced_accuracy"] >= .92,
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
        "resources": {"seconds": time.perf_counter() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "v2_features_sha256": sha(feature_path),
        "v2_rows_sha256": sha(v2_root / "result" / "oof_rows.jsonl"),
        "oof_rows_sha256": sha(root / "oof_rows.jsonl"),
        "checkpoint_sha256": sha(root / "residual_scope_gate.pt"),
        "plan_sha256": sha(PLAN), "scope_checkpoint_sha256": sha(scope / "scope_gates.pt"),
        "v1_row_hashes": [row["rows_sha256"] for row in resources],
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(root / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1_root", "v2_root", "scope_root", "model", "tokenizer", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
