"""Independently recompute factorized scope candidate V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts.audit_long_session_candidate_v2 import aggregate, mean_metric, read_jsonl, selected_epoch
from tasks.pack_cbf_event_warmup import sha
from tasks.train_factorized_scope_candidate_v3 import SEEDS, load_v2, release_gate
from tasks.train_long_session_candidate_v2 import CONTROLS, bootstrap


def audit(root, data_root, source_root, v2_root):
    root, data_root, source_root, v2_root = map(Path, (root, data_root, source_root, v2_root))
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text())
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    learned_records = {str(seed): read_jsonl(root / f"dev_scope_learned_seed{seed}.jsonl")
                       for seed in SEEDS}
    control_records = {name: read_jsonl(root / f"dev_scope_{name}.jsonl") for name in CONTROLS}
    v2_records, v2_hashes = load_v2(v2_root)
    expected = data_manifest["counts"]["dev"]["episodes"]
    all_records = (*learned_records.values(), *control_records.values(), *v2_records.values())
    identity_sets = [{row["episode_id"] for row in records} for records in all_records]
    identity_ok = (all(len(records) == expected for records in all_records)
                   and all(len(ids) == expected for ids in identity_sets)
                   and all(ids == identity_sets[0] for ids in identity_sets[1:]))
    row_ok = all(row["rank"] >= 1 and row["rank"] <= row["candidates"]
                 and row["correct"] == int(row["rank"] == 1)
                 and row["reciprocal_rank"] == 1 / row["rank"]
                 for records in all_records for row in records)
    learned = {seed: aggregate(records) for seed, records in learned_records.items()}
    controls = {f"scope_{name}": aggregate(records)
                for name, records in control_records.items()}
    mean = mean_metric(learned)
    aggregate_ok = (learned == summary["learned_per_seed"]
                    and controls == summary["controls"] and mean == summary["learned"])
    comparison = bootstrap(learned_records[str(SEEDS[0])], v2_records["learned"], 30201)
    scope_max = controls["scope_max_mean"]
    full_max = {"top1": statistics.fmean(row["correct"] for row in v2_records["full_max"])}
    comparison_ok = (comparison == summary["primary_vs_v2_learned"]
                     and full_max["top1"] == summary["full_text_v1_max_top1"])
    gate = release_gate(mean, learned, comparison, scope_max, full_max)
    selection_ok = all(selected_epoch(training["selections"][str(seed)]["tune_grid"])
                       == training["selections"][str(seed)]["epoch"] for seed in SEEDS)
    source_ok = (
        manifest["data_manifest_sha256"] == sha(data_root / "manifest.json")
        and manifest["source_summary_sha256"] == sha(source_root / "summary.json")
        and manifest["source_training_sha256"] == sha(source_root / "training.json")
        and manifest["source_transforms_sha256"] == sha(source_root / "transforms.pt")
        and training["checkpoint_sha256"] == sha(root / "scope_candidate_rankers.pt")
        and training["v2_hashes"] == v2_hashes)
    locks_ok = (manifest["payload_excluded_from_candidate_key"]
                and not manifest["candidate_age_feature"] and not manifest["unit_position_feature"]
                and not manifest["longmemeval_used"] and not manifest["locomo_used"]
                and not summary["longmemeval_used"] and not summary["locomo_scored"]
                and summary["next_stage"] == (
                    "preregister_factorized_sparse_gate" if gate["passed"] else "stop_candidate_v3")
                and len(manifest["git_commit"]) == 40)
    raw_hashes = {path.name: sha(path) for path in root.glob("*.jsonl")}
    checks = {
        "raw_hashes": raw_hashes == training["raw_rows_sha256"],
        "record_identities": identity_ok,
        "row_invariants": row_ok,
        "aggregates": aggregate_ok,
        "fit_tune_selection": selection_ok,
        "v2_comparison": comparison_ok,
        "release_gate": gate == summary["release_gate"],
        "source_checkpoint_v2_hashes": source_ok,
        "factorization_and_claim_locks": locks_ok,
    }
    return {"protocol": "factorized_scope_candidate_v3_audit",
            "audit_passed": all(checks.values()), "checks": checks,
            "summary_sha256": sha(root / "summary.json"),
            "training_sha256": sha(root / "training.json"),
            "training_manifest_sha256": sha(root / "training_manifest.json"),
            "episodes": expected, "recomputed_gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v2-root", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.v2_root)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
