"""Audit V10 FewRel pilot labels, resources, metrics, and release gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    AUDIT_PROTOCOL as DATA_AUDIT_PROTOCOL,
)
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.pilot_qwen_utility_record_controller_v10_fewrel import (
    ALPHAS, AUDIT_PROTOCOL, EXPECTED_EPISODES, EXPECTED_GROUPS,
    EXPECTED_LABELS, PROTOCOL, action_grid, load_jsonl, pilot_gate,
    single_candidate_benefit, summarize,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import choose_safe_oracle


def audit(root: Path, data_root: Path, data_audit_path: Path,
          summary_path: Path, num_shards: int) -> dict:
    data_audit = json.loads(data_audit_path.read_text())
    summary = json.loads(summary_path.read_text())
    rows, resources = [], []
    hashes = {}
    for shard in range(num_shards):
        label_path = root / f"labels_shard{shard}.jsonl"
        resource_path = root / f"resource_shard{shard}.json"
        part = load_jsonl(label_path)
        resource = json.loads(resource_path.read_text())
        rows.extend(part); resources.append(resource)
        hashes[label_path.name] = sha256(label_path)
        hashes[resource_path.name] = sha256(resource_path)
    episodes = load_jsonl(data_root / "pilot_episodes.jsonl")
    expected = {(row["episode_id"], boundary["boundary_index"])
                for row in episodes for boundary in row["boundaries"]}
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}

    action_integrity = True
    oracle_integrity = True
    finite_outcomes = True
    for row in rows:
        actions = action_grid(row["candidate_count"], ALPHAS)
        action_integrity &= set(row["actions"]) == {
            action["name"] for action in actions
        }
        oracle_integrity &= row["safe_oracle_action"] == choose_safe_oracle(
            {name: value for name, value in row["actions"].items()},
            actions, row["retain_action"],
        )
        for outcome in row["actions"].values():
            finite_outcomes &= all(
                math.isfinite(float(outcome[key]))
                for key in ("correct", "total_answer_nll",
                            "unrelated_correct", "mean_memory_mass")
            )
    reconstructed = summarize(SimpleNamespace(
        root=str(root), data_root=str(data_root), num_shards=num_shards,
    ))
    candidate_reconstructed = single_candidate_benefit(rows)
    gate_reconstructed = pilot_gate(
        finite_complete=summary["gate"]["checks"][
            "finite_complete_auditable_labels"
        ],
        peak_reserved_gib=summary["resource"]["peak_reserved_gib"],
        answer_coverage=summary["answer_token_coverage"]["coverage"],
        positive_fraction=candidate_reconstructed[
            "positive_safe_benefit_fraction"
        ],
        oracle_headroom=summary["safe_oracle_headroom"],
        heldouts_sealed=(
            not summary["development_labels_loaded"]
            and not summary["development_model_scores_loaded"]
            and not summary["heldout_benchmarks_loaded"]
        ),
    )
    checks = {
        "audited_data_dependency_locked": (
            data_audit.get("protocol") == DATA_AUDIT_PROTOCOL
            and data_audit.get("passed")
            and all(resource["data_audit_sha256"] == sha256(data_audit_path)
                    and resource["data_manifest_sha256"]
                    == sha256(data_root / "manifest.json")
                    for resource in resources)
        ),
        "complete_unique_expected_labels": (
            len(rows) == EXPECTED_LABELS and identities == expected
            and len(episodes) == EXPECTED_EPISODES
            and len({row["source_group"] for row in rows}) == EXPECTED_GROUPS
        ),
        "shard_hashes_and_counts": all(
            resource["rows_sha256"] == hashes[f"labels_shard{shard}.jsonl"]
            and resource["labels"] == len(load_jsonl(
                root / f"labels_shard{shard}.jsonl"
            ))
            for shard, resource in enumerate(resources)
        ),
        "ternary_action_grid_reconstructed": bool(action_integrity),
        "safe_oracle_reconstructed": bool(oracle_integrity),
        "finite_outcomes": bool(finite_outcomes),
        "summary_exactly_reconstructed": summary == reconstructed,
        "candidate_benefit_reconstructed": (
            summary["single_candidate_benefit"] == candidate_reconstructed
        ),
        "gate_reconstructed": summary["gate"] == gate_reconstructed,
        "plain_final_qwen_and_fixed_grid": all(
            resource["protocol"] == PROTOCOL
            and resource["model"]["config"].get("ttt_mode") is False
            and len(resource["model"]["excluded_adapters"]) == 14
            and resource["action_alphas"] == [0.0, 0.5, 1.0]
            for resource in resources
        ),
        "two_rtx_5090_shards": (
            len(resources) == 2
            and all("5090" in resource["gpu"] for resource in resources)
        ),
        "heldout_sources_sealed": all(
            not resource["development_labels_loaded"]
            and not resource["development_model_scores_loaded"]
            and not resource["heldout_benchmarks_loaded"]
            for resource in resources
        ),
    }
    integrity_passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": integrity_passed,
        "release_passed": integrity_passed and summary["gate"]["passed"],
        "checks": checks,
        "pilot_gate": summary["gate"],
        "summary_sha256": sha256(summary_path),
        "artifact_sha256": hashes,
        "next_stage": (
            "v10_formal_fewrel_label_construction"
            if integrity_passed and summary["gate"]["passed"]
            else "stop_v10_before_formal_labeling"
        ),
        "terminal_status": (
            "v10_fewrel_pilot_audit_passed_gate_passed"
            if integrity_passed and summary["gate"]["passed"]
            else "v10_fewrel_pilot_audit_passed_gate_failed"
            if integrity_passed else "v10_fewrel_pilot_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.data_audit,
                   args.summary, args.num_shards)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
