"""Independent audit of V4 cross-fit checkpoint reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_natural_forgetting_crossfit_v4 import PLAN, PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import SEEDS, fold_for, load_labels
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    BinaryHead,
    FEATURE_SETS,
    OUTER_FOLDS,
    factor_actions,
)


def probabilities(row, features, hidden, seed):
    import torch
    benefit = BinaryHead.create(features.shape[1], hidden, seed)
    risk = BinaryHead.create(features.shape[1], hidden, seed + 100_000)
    benefit.load_state_dict(row["benefit_state"]); risk.load_state_dict(row["risk_state"])
    with torch.no_grad():
        pb = benefit((features - row["benefit_mean"]) / row["benefit_std"] \
                     ).squeeze(-1).sigmoid()
        pr = risk((features - row["risk_mean"]) / row["risk_std"] \
                  ).squeeze(-1).sigmoid()
    return pb, pr


def audit(args):
    import torch

    root, repo = Path(args.root), Path(args.repo)
    factor_root, utility_root, v1_root = map(
        Path, (args.factor_root, args.utility_root, args.v1_root))
    result = root / "result"; summary = json.loads((result / "summary.json").read_text())
    checkpoint_path = result / "crossfit_factor_gate.pt"
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_path = factor_root / "result" / "factor_features.pt"
    source_features = torch.load(feature_path, map_location="cpu", weights_only=False)
    source_summary_path = factor_root / "result" / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    oof_path = factor_root / "result" / "oof_predictions.jsonl"
    oof = [json.loads(line) for line in oof_path.read_text().splitlines() if line]
    _, labels = load_labels(utility_root)
    fold_checks = artifact["protocol"] == PROTOCOL \
        and artifact["feature_names"] == FEATURE_SETS \
        and tuple(artifact["seeds"]) == SEEDS
    probability_checks = True; max_error = 0.; action_mismatches = 0; reproduced = 0
    all_sessions = {row["session_id"] for row in labels}
    for feature_set, table in source_features["feature_sets"].items():
        for outer in range(OUTER_FOLDS):
            fold = artifact["folds"][feature_set][str(outer)]
            train = set(fold["train_sessions"]); validation = set(fold["validation_sessions"])
            expected_validation = {value for value in all_sessions if fold_for(value) == outer}
            fold_checks &= not train & validation and train | validation == all_sessions \
                and validation == expected_validation
            selected = source_summary["folds"][feature_set][str(outer)]["selected"]
            fold_checks &= fold["config"] == {key: selected["config"][key]
                for key in ("hidden", "learning_rate", "weight_decay")} \
                and fold["benefit_threshold"] == selected["benefit_threshold"] \
                and fold["risk_threshold"] == selected["risk_threshold"]
            indices = [index for index, row in enumerate(labels)
                       if row["session_id"] in validation]
            for seed in SEEDS:
                pb, pr = probabilities(fold["checkpoints"][str(seed)], table[indices],
                                       fold["config"]["hidden"], seed)
                actions = factor_actions(pb, pr, fold["benefit_threshold"],
                                         fold["risk_threshold"], True)
                for local, index in enumerate(indices):
                    stored = oof[index]["feature_sets"][feature_set][str(seed)]
                    max_error = max(max_error,
                        abs(float(pb[local]) - stored["benefit_probability"]),
                        abs(float(pr[local]) - stored["risk_probability"]))
                    action = "scope_top1_a0" if actions[local] else "retain_all"
                    action_mismatches += int(action != stored["factor_action"])
                    reproduced += 1
    checks_expected = {
        "all_774_rows_reproduced_per_view_seed":
            reproduced == len(labels) * len(FEATURE_SETS) * len(SEEDS),
        "maximum_probability_error_at_most_1e_6": max_error <= 1e-6,
        "zero_action_mismatches": action_mismatches == 0,
        "development_test_external_sealed": True,
    }
    hashes = summary["hashes"]
    hash_checks = all((
        hashes["checkpoint"] == sha(checkpoint_path),
        hashes["plan"] == sha(repo / PLAN),
        hashes["source_summary"] == sha(source_summary_path),
        hashes["source_audit"] == sha(factor_root / "execution_audit.json"),
        hashes["source_checkpoint"] == sha(factor_root / "result" / "factor_gate.pt"),
        hashes["source_features"] == sha(feature_path),
        hashes["source_oof"] == sha(oof_path),
        hashes["utility_summary"] == sha(utility_root / "summary.json"),
        hashes["v1_summary"] == sha(v1_root / "result" / "summary.json"),
    ))
    terminal = ("completed_qwen_natural_forgetting_crossfit_v4"
                if all(checks_expected.values()) else
                "failed_qwen_natural_forgetting_crossfit_v4")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip()
            == terminal and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "protocol_features_and_folds": fold_checks,
        "checkpoint_probability_and_action_reproduction": probability_checks
            and summary["maximum_probability_error"] == max_error
            and summary["action_mismatches"] == action_mismatches
            and summary["reproduced_predictions"] == reproduced,
        "gate_reconstruction": summary["gate"]["checks"] == checks_expected
            and summary["gate"]["online_calibration_authorized"]
            == all(checks_expected.values()),
        "artifact_hashes": hash_checks,
        "heldout_seals": summary["calibration_loaded"]
            and not summary["development_loaded"] and not summary["test_loaded"]
            and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": {
                "maximum_probability_error": max_error,
                "action_mismatches": action_mismatches,
                "reproduced_predictions": reproduced,
                "gate": checks_expected}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "factor_root", "utility_root", "v1_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(); result = audit(args)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
