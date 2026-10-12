"""Independent audit for CounterFact zero-shot forgetting transfer."""

import argparse
import json
from pathlib import Path

from scripts.audit_zsre_counterfactual_forgetting_formal import recompute
from tasks.build_counterfact_zero_shot_forgetting import PROTOCOL as DATA_PROTOCOL
from tasks.evaluate_counterfact_zero_shot_forgetting import (
    CONTROL_POLICIES,
    PROTOCOL,
    transfer_gate,
)
from tasks.evaluate_zsre_hard_retain_forgetting import aggregate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_zsre_decomposed_scope_gate import GATE_SEEDS
from tasks.train_zsre_forgetting_controller import summarize_seed_metrics


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root, data_root, boundary_root, boundary_audit, controller_root, source_root, scope_root):
    root, data, boundary = Path(root), Path(data_root), Path(boundary_root)
    controller, source, scope = map(Path, (controller_root, source_root, scope_root))
    boundary_audit_path = Path(boundary_audit)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data / "manifest.json").read_text())
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "data_protocol": data_manifest["protocol"] == manifest["data_protocol"] == DATA_PROTOCOL,
        "zero_shot": summary["counterfact_training_rows"] == 0
                     and summary["counterfact_tuning_rows"] == 0
                     and manifest["counterfact_training_rows"] == 0
                     and manifest["counterfact_tuning_rows"] == 0,
        "units": summary["units"] == data_manifest["eligible_subjects"],
        "data_hash": training["data_manifest_sha256"] == sha(data / "manifest.json"),
        "boundary_hashes": training["boundary_summary_sha256"] == sha(boundary / "summary.json")
                           and training["boundary_manifest_sha256"] == sha(
                               boundary / "training_manifest.json")
                           and training["boundary_audit_sha256"] == sha(boundary_audit_path),
        "controller_hash": training["controller_checkpoint_sha256"] == sha(
            controller / "counterfactual_forgetting_action_oof_gate.pt"),
        "source_hashes": training["source_summary_sha256"] == sha(source / "summary.json")
                         and training["source_transforms_sha256"] == sha(source / "transforms.pt"),
        "scope_hashes": training["scope_training_sha256"] == sha(scope / "training.json")
                        and training["scope_checkpoint_sha256"] == sha(scope / "scope_gates.pt"),
    }
    primary, raw_checks = recompute(root, "transfer_test", training, 9701)
    checks.update(raw_checks)
    checks["learned_recomputed"] = primary["controller"] == summary["policies"]["learned"]
    checks["cosine_recomputed"] = primary["cosine_local"] == summary["policies"]["cosine_local"]
    checks["comparison_recomputed"] = primary["comparison"] == summary["primary_comparison"]
    checks["paired_units"] = primary["comparison"]["paired_subject_units"] == summary["units"]
    for policy in CONTROL_POLICIES:
        metrics = {}
        for seed in GATE_SEEDS:
            path = root / f"transfer_test_{policy}_seed{seed}.jsonl"
            checks[f"{policy}_hash_{seed}"] = training["raw_rows_sha256"][path.name] == sha(path)
            rows = read_rows(path)
            checks[f"{policy}_identity_{seed}"] = all(
                row["split"] == "transfer_test" and row["seed"] == seed
                and row["policy"] == policy for row in rows)
            metrics[str(seed)] = aggregate(rows)
        checks[f"{policy}_recomputed"] = summarize_seed_metrics(metrics) == summary["policies"][policy]
    checks["gate_recomputed"] = transfer_gate(
        primary["controller"], primary["comparison"]) == summary["transfer_gate"]
    return {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "units": summary["units"],
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "manifest_sha256": sha(root / "training_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--boundary-root", required=True)
    parser.add_argument("--boundary-audit", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.boundary_root, args.boundary_audit,
                   args.controller_root, args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
