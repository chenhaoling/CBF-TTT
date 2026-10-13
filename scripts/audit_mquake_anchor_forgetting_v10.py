"""Independently audit MQuAKE anchor forgetting V10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_crossfit_calibrated_forgetting_v5 import selected_threshold
from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_causal_forgetting_v4 import aggregate, paired_bootstrap
from tasks.train_mquake_anchor_forgetting_v10 import candidate_metrics, development_gate, tune_gate


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit(root, data_root, source_root, v8_root):
    root, data_root, source_root, v8_root = map(Path, (root, data_root, source_root, v8_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    tune = read_jsonl(root / "tune_controller.jsonl")
    tune_candidates = read_jsonl(root / "tune_candidate_rows.jsonl")
    tune_metric = aggregate(tune); tune_candidate = candidate_metrics(tune_candidates)
    prerequisite = tune_gate(tune_metric, tune_candidate)
    tune_ok = (tune_metric == summary["tune"] and tune_candidate == summary["tune_candidate"]
               and prerequisite == summary["tune_prerequisite"])
    calibration_ok = all(selected_threshold(row["grid"]) == row["threshold"]
                         for row in training["calibration"].values())
    dev_ok = True; random_ok = True
    if prerequisite["passed"]:
        dev = read_jsonl(root / "dev_controller.jsonl")
        candidates = read_jsonl(root / "dev_candidate_rows.jsonl")
        controls = {name: read_jsonl(root / f"dev_{name}.jsonl")
                    for name in ("retain_all", "random_matched", "oracle")}
        metric = aggregate(dev); candidate = candidate_metrics(candidates)
        comparison = paired_bootstrap(dev, controls["random_matched"], 35105)
        gate = development_gate(metric, candidate, comparison)
        dev_ok = (summary["development_scored"] and metric == summary["development"]
                  and candidate == summary["development_candidate"]
                  and {name: aggregate(rows) for name, rows in controls.items()} == summary["controls"]
                  and comparison == summary["primary_vs_random"] and gate == summary["development_gate"]
                  and len(dev) == data_manifest["counts"]["dev"]["episodes"])
        primary_deletions = sum(x["deleted"] is not None for row in dev for x in row["decisions"])
        random_deletions = sum(x["deleted"] is not None for row in controls["random_matched"] for x in row["decisions"])
        random_ok = primary_deletions == random_deletions == training["matched_random_deletions"]
        expected_next = "preregister_locomo_confirmation" if gate["passed"] else "stop_anchor_dev_no_retuning"
    else:
        absent = ("dev_controller.jsonl", "dev_candidate_rows.jsonl", "dev_retain_all.jsonl",
                  "dev_random_matched.jsonl", "dev_oracle.jsonl")
        dev_ok = not summary["development_scored"] and all(not (root / name).exists() for name in absent)
        expected_next = "stop_anchor_tune"
    v8_checkpoint = v8_root / "mquake_adapted_forgetting.pt"
    source_ok = (manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
                 and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
                 and manifest["source_training_sha256"] == sha(source_root / "training.json")
                 and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
                 and training["checkpoint_sha256"] == sha(root / "mquake_anchor_forgetting.pt")
                 and training["v8_hashes"] == {"summary": sha(v8_root / "summary.json"),
                                               "training": sha(v8_root / "training.json"),
                                               "checkpoint": sha(v8_checkpoint)})
    locks_ok = (summary["protocol"] == manifest["protocol"] == "mquake_anchor_forgetting_v10"
                and summary["candidate_policy"] == manifest["candidate_policy"]
                == "anchor_4_then_v8_ranker" and manifest["shortlist"] == 4
                and manifest["gate_epoch"] == 50 and not manifest["refit_after_calibration"]
                and manifest["development_labels_loaded_conditionally"]
                and not manifest["mquake_cf_used"] and not manifest["longmemeval_used"]
                and not manifest["locomo_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == expected_next and len(manifest["git_commit"]) == 40)
    checks = {"tune_metrics_and_gate": tune_ok, "threshold_selection": calibration_ok,
              "conditional_development": dev_ok, "matched_random": random_ok,
              "raw_hashes": training["raw_rows_sha256"]
              == {path.name: sha(path) for path in root.glob("*.jsonl")},
              "source_v8_checkpoint_hashes": source_ok, "protocol_and_access_locks": locks_ok}
    return {"protocol": "mquake_anchor_forgetting_v10_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "recomputed_tune_prerequisite": prerequisite}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v8-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); result = audit(args.root, args.data_root, args.source_root, args.v8_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
