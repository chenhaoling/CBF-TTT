"""Independent audit of the ZsRE development threshold diagnostic."""

import argparse
import json
from pathlib import Path

from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import (
    CONTROLLER_SEEDS,
    paired_comparison,
    summarize_seed_metrics,
)
from tasks.diagnose_zsre_forgetting_dev_threshold import (
    PROTOCOL,
    THRESHOLDS,
    rank_threshold,
    read_rows,
    threshold_key,
)


def audit(root, formal_root, controller_root):
    root, formal, controller = Path(root), Path(formal_root), Path(controller_root)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    formal_training = json.loads((formal / "training.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "thresholds": manifest["thresholds"] == list(THRESHOLDS),
        "test_sealed": not summary["test_loaded"] and not summary["test_scored"]
                       and not manifest["test_loaded"] and not manifest["test_scored"]
                       and not list(root.glob("test*")),
        "formal_hashes": training["formal_summary_sha256"] == sha(formal / "summary.json")
                         and training["formal_training_sha256"] == sha(formal / "training.json")
                         and training["formal_manifest_sha256"] == sha(
                             formal / "training_manifest.json"),
        "controller_hash": training["controller_checkpoint_sha256"] == sha(
            controller / "counterfactual_forgetting_action_oof_gate.pt"),
    }
    baseline_records, baseline_metrics = {}, {}
    for seed in GATE_SEEDS:
        path = formal / f"dev_cosine_local_seed{seed}.jsonl"
        checks[f"baseline_hash_{seed}"] = formal_training["raw_rows_sha256"][path.name] == sha(path)
        rows = read_rows(path); baseline_records[str(seed)] = rows
        baseline_metrics[str(seed)] = aggregate(rows)
    checks["baseline_recomputed"] = summarize_seed_metrics(baseline_metrics) == summary["baseline"]

    rebuilt = {}
    for threshold in THRESHOLDS:
        records_by_seed, metrics_by_seed = {}, {}
        for seed in CONTROLLER_SEEDS:
            path = root / f"dev_threshold_{threshold_key(threshold)}_controller_seed{seed}.jsonl"
            checks[f"raw_hash_{threshold_key(threshold)}_{seed}"] = \
                training["raw_rows_sha256"][path.name] == sha(path)
            rows = read_rows(path)
            checks[f"raw_identity_{threshold_key(threshold)}_{seed}"] = all(
                row["split"] == "development_threshold_diagnostic" and row["seed"] == seed
                for row in rows)
            records_by_seed[str(seed)] = rows; metrics_by_seed[str(seed)] = aggregate(rows)
        controller_summary = summarize_seed_metrics(metrics_by_seed)
        comparison = paired_comparison(
            records_by_seed, baseline_records, 9501 + int(round(threshold * 10)))
        from tasks.evaluate_zsre_counterfactual_forgetting_controller import development_gate
        rebuilt[threshold_key(threshold)] = {
            "threshold": threshold,
            "controller": controller_summary,
            "comparison": comparison,
            "development_gate": development_gate(controller_summary, comparison),
        }
    checks["grid_recomputed"] = rebuilt == summary["grid"]
    selected = max(rebuilt.values(), key=rank_threshold)
    checks["selection_recomputed"] = selected == summary["selected_by_reliability"]
    passing = [row["threshold"] for row in rebuilt.values() if row["development_gate"]["passed"]]
    checks["sufficiency_recomputed"] = passing == summary["thresholds_passing_original_gate"] \
        and bool(passing) == summary["threshold_only_sufficient"]
    return {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "manifest_sha256": sha(root / "training_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.formal_root, args.controller_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
