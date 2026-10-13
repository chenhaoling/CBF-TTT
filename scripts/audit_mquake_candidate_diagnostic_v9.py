"""Independently audit MQuAKE candidate diagnostic V9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.diagnose_mquake_candidate_scope_v9 import WIDTHS, record_metrics, select_arm
from tasks.pack_cbf_event_warmup import sha


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, data_root, source_root, v8_root):
    root, data_root, source_root, v8_root = map(
        Path, (root, data_root, source_root, v8_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    records = read_jsonl(root / "tune_candidate_diagnostic.jsonl")
    metrics = {"learned_full": record_metrics(records, "rank_learned_full")}
    controls = {"max_mean_raw": record_metrics(records, "rank_max_mean_raw"),
                "anchor_raw": record_metrics(records, "rank_anchor_raw")}
    for kind in ("max", "anchor"):
        for width in WIDTHS:
            metrics[f"{kind}_{width}"] = record_metrics(
                records, f"rank_{kind}_{width}", f"recall_{kind}_{width}")
    gate = select_arm(metrics)
    recompute_ok = (metrics == summary["metrics"] and controls == summary["controls"]
                    and gate == summary["release_gate"])
    identities_ok = (len(records) == len({row["episode_id"] for row in records})
                     == data_manifest["counts"]["tune"]["episodes"]
                     and {row["length"] for row in records} == {16, 32, 48, 64})
    checkpoint = v8_root / "mquake_adapted_forgetting.pt"
    source_ok = (manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["v8_hashes"] == {
                     "summary": sha(v8_root / "summary.json"),
                     "training": sha(v8_root / "training.json"),
                     "checkpoint": sha(checkpoint)})
    locks_ok = (summary["protocol"] == manifest["protocol"]
                == "mquake_candidate_diagnostic_v9"
                and summary["split"] == "tune" and manifest["widths"] == list(WIDTHS)
                and manifest["explicit_scope_anchor"]
                and not manifest["development_loaded"] and not summary["development_loaded"]
                and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
                and not manifest["locomo_used"] and not summary["locomo_scored"]
                and len(manifest["git_commit"]) == 40)
    checks = {"metrics_and_release_gate": recompute_ok,
              "tune_episode_identities": identities_ok,
              "raw_output_hash": training["raw_rows_sha256"] == {
                  "tune_candidate_diagnostic.jsonl": sha(root / "tune_candidate_diagnostic.jsonl")},
              "source_and_v8_hashes": source_ok, "protocol_and_access_locks": locks_ok}
    return {"protocol": "mquake_candidate_diagnostic_v9_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_release_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v8-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(
        args.root, args.data_root, args.source_root, args.v8_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
