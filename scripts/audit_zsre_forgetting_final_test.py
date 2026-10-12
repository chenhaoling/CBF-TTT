"""Independent audit of the one-time held-out ZsRE forgetting test."""

import argparse
import json
from pathlib import Path

from scripts.audit_zsre_counterfactual_forgetting_formal import recompute
from tasks.evaluate_zsre_counterfactual_forgetting_controller import development_gate
from tasks.evaluate_zsre_forgetting_final_test import PROTOCOL, release_threshold
from tasks.pack_cbf_event_warmup import sha


def audit(root, boundary_root, boundary_audit, controller_root, source_root, scope_root):
    root, boundary = Path(root), Path(boundary_root)
    controller, source, scope = map(Path, (controller_root, source_root, scope_root))
    boundary_audit_path = Path(boundary_audit)
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    boundary_summary = json.loads((boundary / "summary.json").read_text())
    boundary_manifest = json.loads((boundary / "training_manifest.json").read_text())
    boundary_audit_data = json.loads(boundary_audit_path.read_text())
    threshold = release_threshold(boundary_summary, boundary_manifest, boundary_audit_data)
    checks = {
        "protocol": summary["protocol"] == manifest["protocol"] == PROTOCOL,
        "threshold": summary["selected_threshold"] == manifest["selected_threshold"] == threshold,
        "test_scored": summary["test_loaded"] and summary["test_scored"]
                       and manifest["test_loaded"] and manifest["test_scored"],
        "dev_absent": not list(root.glob("dev_*.jsonl")),
        "boundary_hashes": training["boundary_summary_sha256"] == sha(boundary / "summary.json")
                           and training["boundary_training_sha256"] == sha(boundary / "training.json")
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
    rebuilt, raw_checks = recompute(root, "test", training, 9601); checks.update(raw_checks)
    checks["test_recomputed"] = rebuilt == summary["test"]
    checks["test_units"] = rebuilt["comparison"]["paired_subject_units"] == 516
    checks["gate_recomputed"] = development_gate(
        rebuilt["controller"], rebuilt["comparison"]) == summary["test_gate"]
    return {
        "protocol": PROTOCOL,
        "audit_passed": all(checks.values()),
        "checks": checks,
        "selected_threshold": threshold,
        "test_units": rebuilt["comparison"]["paired_subject_units"],
        "summary_sha256": sha(root / "summary.json"),
        "training_sha256": sha(root / "training.json"),
        "manifest_sha256": sha(root / "training_manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--boundary-root", required=True)
    parser.add_argument("--boundary-audit", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.boundary_root, args.boundary_audit, args.controller_root,
                   args.source_root, args.scope_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
