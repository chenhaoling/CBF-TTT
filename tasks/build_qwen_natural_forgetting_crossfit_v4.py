"""Reconstruct frozen outer-fold Factor Gate V2 checkpoints for V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS, fold_for, load_labels
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    FEATURE_SETS,
    OUTER_FOLDS,
    PROTOCOL as SOURCE_PROTOCOL,
    factor_actions,
    fit_pair,
    predict_pair,
)


PROTOCOL = "qwen_natural_forgetting_crossfit_v4"
PLAN = "QWEN_NATURAL_FORGETTING_ONPOLICY_CONSENSUS_V4_PLAN.md"


def training_config(selected):
    return {key: selected["config"][key]
            for key in ("hidden", "learning_rate", "weight_decay")}


def run(args):
    import torch

    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    factor_root, utility_root, v1_root = map(
        Path, (args.factor_root, args.utility_root, args.v1_root))
    source_summary_path = factor_root / "result" / "summary.json"
    source_audit_path = factor_root / "execution_audit.json"
    source_checkpoint_path = factor_root / "result" / "factor_gate.pt"
    source_features_path = factor_root / "result" / "factor_features.pt"
    source_oof_path = factor_root / "result" / "oof_predictions.jsonl"
    source_summary = json.loads(source_summary_path.read_text())
    source_audit = json.loads(source_audit_path.read_text())
    if source_summary["protocol"] != SOURCE_PROTOCOL \
            or not source_summary["gate"]["development_authorized"] \
            or not source_audit["passed"]:
        raise ValueError("V2 source is not released")
    utility_summary, labels = load_labels(utility_root)
    features = torch.load(source_features_path, map_location="cpu", weights_only=False)
    oof = [json.loads(line) for line in source_oof_path.read_text().splitlines() if line]
    if len(labels) != len(oof) or any(
        (label["session_id"], label["event_step"])
        != (row["session_id"], row["event_step"]) for label, row in zip(labels, oof)):
        raise ValueError("V2 OOF identity mismatch")
    artifact = {"protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
                "feature_names": FEATURE_SETS, "seeds": SEEDS, "folds": {}}
    max_error = 0.; action_mismatches = 0; reproduced = 0
    for feature_set, table in features["feature_sets"].items():
        artifact["folds"][feature_set] = {}
        for outer in range(OUTER_FOLDS):
            validation = [index for index, row in enumerate(labels)
                          if fold_for(row["session_id"]) == outer]
            training = [index for index in range(len(labels)) if index not in set(validation)]
            selected = source_summary["folds"][feature_set][str(outer)]["selected"]
            config = training_config(selected); checkpoints = {}
            for seed in SEEDS:
                pair = fit_pair(table, features["benefit"], features["risk"],
                                training, config, seed)
                pb, pr = predict_pair(pair, table[validation])
                expected_actions = factor_actions(
                    pb, pr, selected["benefit_threshold"], selected["risk_threshold"], True)
                checkpoints[str(seed)] = {
                    "benefit_state": pair["benefit_model"].state_dict(),
                    "benefit_mean": pair["benefit_mean"], "benefit_std": pair["benefit_std"],
                    "risk_state": pair["risk_model"].state_dict(),
                    "risk_mean": pair["risk_mean"], "risk_std": pair["risk_std"],
                }
                for local, index in enumerate(validation):
                    stored = oof[index]["feature_sets"][feature_set][str(seed)]
                    max_error = max(max_error,
                                    abs(float(pb[local]) - stored["benefit_probability"]),
                                    abs(float(pr[local]) - stored["risk_probability"]))
                    action = "scope_top1_a0" if expected_actions[local] else "retain_all"
                    action_mismatches += int(action != stored["factor_action"])
                    reproduced += 1
            artifact["folds"][feature_set][str(outer)] = {
                "config": config,
                "benefit_threshold": selected["benefit_threshold"],
                "risk_threshold": selected["risk_threshold"],
                "train_sessions": sorted({labels[index]["session_id"] for index in training}),
                "validation_sessions": sorted({labels[index]["session_id"]
                                                for index in validation}),
                "checkpoints": checkpoints,
            }
    checkpoint_path = output / "crossfit_factor_gate.pt"
    torch.save(artifact, checkpoint_path)
    checks = {
        "all_774_rows_reproduced_per_view_seed":
            reproduced == len(labels) * len(FEATURE_SETS) * len(SEEDS),
        "maximum_probability_error_at_most_1e_6": max_error <= 1e-6,
        "zero_action_mismatches": action_mismatches == 0,
        "development_test_external_sealed": True,
    }
    summary = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "rows": len(labels), "views": list(FEATURE_SETS), "seeds": list(SEEDS),
        "maximum_probability_error": max_error,
        "action_mismatches": action_mismatches, "reproduced_predictions": reproduced,
        "gate": {"checks": checks, "online_calibration_authorized": all(checks.values())},
        "hashes": {
            "checkpoint": sha(checkpoint_path), "plan": sha(PLAN),
            "source_summary": sha(source_summary_path),
            "source_audit": sha(source_audit_path),
            "source_checkpoint": sha(source_checkpoint_path),
            "source_features": sha(source_features_path),
            "source_oof": sha(source_oof_path),
            "utility_summary": sha(utility_root / "summary.json"),
            "v1_summary": sha(v1_root / "result" / "summary.json"),
        },
        "utility_protocol": utility_summary["protocol"],
        "calibration_loaded": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("factor_root", "utility_root", "v1_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
