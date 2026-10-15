"""Constrained V5 selection from the immutable residual-gate V4 search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tasks.evaluate_qwen_natural_causal_memory_v1 import read_stage
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_residual_gate_v4 import (
    PROTOCOL as V4_PROTOCOL,
    end_to_end,
    fit,
)
from tasks.train_qwen_natural_scope_transfer_v2 import (
    FOLDS,
    STRATA,
    decision_metrics,
    exact_oof_score,
)


PROTOCOL = "qwen_natural_scope_constrained_v5"
PLAN = "QWEN_NATURAL_SCOPE_CONSTRAINED_V5_PLAN.md"
EXPECTED_V2_FEATURE_SHA256 = "1d3f89ee7db51208b1d1934d7229a6b1c5d3e302cc2ebc6a67b8c6c09535a140"
EXPECTED_V4_SEARCH_SHA256 = "bc75b7bcfec1117113734d53ac744edbe3aeb2e2e0fdd1ddb8ecdea9a8296f33"
EXPECTED_V4_SUMMARY_SHA256 = "2b7217a295bfb56b029120760b7bd4cbf47d63aa0f8330bd626ae94bfe24167e"


def select_record(records):
    eligible = [row for row in records
                if row["correct_top_recall"] >= .92
                and row["natural_wrong_top_specificity"] >= .85
                and row["no_match_specificity"] >= .92
                and row["balanced_accuracy"] >= .92]
    if not eligible:
        raise ValueError("immutable V4 search has no fully eligible record")
    return max(eligible, key=lambda row: (
        row["correct_top_recall"], row["no_match_specificity"],
        row["natural_wrong_top_specificity"], row["balanced_accuracy"],
        -row["hidden"], row["penalty"], -row["learning_rate"], -row["epochs"],
        -abs(row["threshold"] - .5),
    ))


def reproduce_oof(features, labels, metadata, base_state, selected):
    import torch

    device = torch.device("cuda"); values = features.to(device); targets = labels.to(device)
    folds = [row["fold"] for row in metadata]
    logits = torch.full((len(metadata),), float("nan"), device=device)
    for fold in range(FOLDS):
        train = [index for index, value in enumerate(folds) if value != fold]
        holdout = [index for index, value in enumerate(folds) if value == fold]
        model = fit(values, targets, metadata, train, base_state,
                    selected["hidden"], selected["penalty"],
                    selected["learning_rate"], selected["epochs"])
        with torch.no_grad():
            index = torch.tensor(holdout, dtype=torch.long, device=device)
            logits[index] = model(values[index])
    if torch.isnan(logits).any():
        raise ValueError("V5 OOF reproduction is incomplete")
    return logits.detach().cpu()


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); v2_root, v4_root = Path(args.v2_root), Path(args.v4_root)
    feature_path = v2_root / "result" / "calibration_features.pt"
    search_path, v4_summary_path = (v4_root / "result" / "search.json",
                                    v4_root / "result" / "summary.json")
    if sha(feature_path) != EXPECTED_V2_FEATURE_SHA256 \
            or sha(search_path) != EXPECTED_V4_SEARCH_SHA256 \
            or sha(v4_summary_path) != EXPECTED_V4_SUMMARY_SHA256:
        raise ValueError("V5 immutable input hash mismatch")
    v4_summary = json.loads(v4_summary_path.read_text())
    if v4_summary["protocol"] != V4_PROTOCOL or v4_summary["gate"]["passed"]:
        raise ValueError("V5 expects the stopped V4 calibration")
    search = json.loads(search_path.read_text()); selected = select_record(search["records"])
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    features, labels, metadata = artifact["features"], artifact["labels"], artifact["metadata"]
    scope = Path(args.scope_root)
    gate_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    base_state = gate_artifact["heads"]["7301"]
    torch.manual_seed(202610201); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    logits = reproduce_oof(features, labels, metadata, base_state, selected)
    decision = decision_metrics(logits, metadata, selected["threshold"])
    for key in ("correct_top_recall", "natural_wrong_top_specificity",
                "no_match_specificity", "balanced_accuracy"):
        if abs(decision[key] - selected[key]) > 1e-12:
            raise ValueError(f"V5 failed to reproduce selected V4 metric {key}")
    decisions = logits.sigmoid() >= selected["threshold"]
    source_rows, _, resources = read_stage(args.v1_root, "calibration")
    v2_rows_path = v2_root / "result" / "oof_rows.jsonl"
    v2_rows = [json.loads(line) for line in v2_rows_path.read_text().splitlines() if line]
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    oof_rows = exact_oof_score(model, tokenizer, source_rows, metadata, decisions)
    with (root / "oof_rows.jsonl").open("x") as sink:
        for row in oof_rows: sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = end_to_end(oof_rows, v2_rows)
    all_indices = list(range(len(metadata)))
    final = fit(features.to("cuda"), labels.to("cuda"), metadata, all_indices,
                base_state, selected["hidden"], selected["penalty"],
                selected["learning_rate"], selected["epochs"])
    checkpoint = {"protocol": PROTOCOL, "state": {
        part: {key: value.detach().cpu() for key, value in values.items()}
        for part, values in final.state_dict().items()},
        "feature_dim": features.shape[1], "selected": selected,
        "threshold": selected["threshold"], "folds": FOLDS,
        "strata": STRATA, "seed": 202610201}
    torch.save(checkpoint, root / "constrained_scope_gate.pt")
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
        "resources": {"seconds": time.perf_counter() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "v2_features_sha256": sha(feature_path), "v2_rows_sha256": sha(v2_rows_path),
        "v4_search_sha256": sha(search_path), "v4_summary_sha256": sha(v4_summary_path),
        "oof_rows_sha256": sha(root / "oof_rows.jsonl"),
        "checkpoint_sha256": sha(root / "constrained_scope_gate.pt"),
        "scope_checkpoint_sha256": sha(scope / "scope_gates.pt"),
        "plan_sha256": sha(PLAN),
        "v1_row_hashes": [row["rows_sha256"] for row in resources],
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(root / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v1_root", "v2_root", "v4_root", "scope_root", "model",
                 "tokenizer", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
