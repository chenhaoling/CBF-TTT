"""Audit all V11 FewRel development labels and the scale-consistency gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json
from tasks.label_qwen_utility_record_controller_v11_fewrel_development import (
    ALPHAS, AUDIT_PROTOCOL, EXPECTED_EPISODES, EXPECTED_GROUPS,
    EXPECTED_LABELS, PROTOCOL, _verified_inputs, action_grid, formal_gate,
    load_jsonl, single_candidate_benefit, summarize,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import choose_safe_oracle


def audit(root: Path, data_root: Path, data_audit: Path,
          summary_path: Path, num_shards: int) -> dict:
    _, _, locks = _verified_inputs(data_root, data_audit)
    summary = json.loads(summary_path.read_text())
    episodes = load_jsonl(data_root / "development_episodes.jsonl")
    expected = {(row["episode_id"], boundary["boundary_index"])
                for row in episodes for boundary in row["boundaries"]}
    rows, resources, hashes = [], [], {}
    for shard in range(num_shards):
        label_path = root / f"labels_shard{shard}.jsonl"
        resource_path = root / f"resource_shard{shard}.json"
        part = load_jsonl(label_path)
        resource = json.loads(resource_path.read_text())
        rows.extend(part); resources.append(resource)
        hashes[label_path.name] = sha256(label_path)
        hashes[resource_path.name] = sha256(resource_path)
    identities = {(row["episode_id"], row["boundary_index"]) for row in rows}
    action_ok = oracle_ok = finite_ok = True
    for row in rows:
        actions = action_grid(row["candidate_count"], ALPHAS)
        action_ok &= set(row["actions"]) == {value["name"] for value in actions}
        oracle_ok &= row["safe_oracle_action"] == choose_safe_oracle(
            row["actions"], actions, row["retain_action"]
        )
        finite_ok &= math.isfinite(row["label_seconds"]) and row["label_seconds"] > 0
        finite_ok &= all(math.isfinite(float(outcome[key]))
                         for outcome in row["actions"].values()
                         for key in ("correct", "total_answer_nll",
                                     "unrelated_correct", "mean_memory_mass"))
    reconstructed = summarize(SimpleNamespace(
        root=str(root), data_root=str(data_root), num_shards=num_shards,
    ))
    candidates = single_candidate_benefit(rows)
    gate = formal_gate(
        finite_complete=summary["gate"]["checks"][
            "finite_complete_auditable_labels"
        ],
        peak_reserved_gib=summary["resource"]["peak_reserved_gib"],
        answer_coverage=summary["answer_token_coverage"]["coverage"],
        positive_fraction=candidates["positive_safe_benefit_fraction"],
        oracle_headroom=summary["safe_oracle_headroom"],
        relation_coverage=summary["gate"]["checks"][
            "complete_balanced_relation_coverage"
        ],
        zero_oracle_collateral=summary["gate"]["checks"][
            "safe_oracle_zero_added_unrelated_errors"
        ],
        heldouts_sealed=(not summary["development_labels_loaded"]
                         and not summary["development_model_scores_loaded"]
                         and not summary["heldout_benchmarks_loaded"]
                         and not summary["internal_test_loaded"]
                         and not summary["external_test_loaded"]),
    )
    checks = {
        "audited_data_dependency_locked": all(
            resource[key] == value for resource in resources
            for key, value in locks.items()
        ),
        "complete_unique_expected_labels": (
            len(episodes) == EXPECTED_EPISODES
            and len(rows) == EXPECTED_LABELS and identities == expected
            and len({row["source_group"] for row in rows}) == EXPECTED_GROUPS
        ),
        "shard_hashes_and_counts": all(
            resource["rows_sha256"] == hashes[f"labels_shard{shard}.jsonl"]
            and resource["labels"] == len(load_jsonl(
                root / f"labels_shard{shard}.jsonl"
            )) for shard, resource in enumerate(resources)
        ),
        "ternary_action_grid_reconstructed": bool(action_ok),
        "safe_oracle_reconstructed": bool(oracle_ok),
        "finite_outcomes": bool(finite_ok),
        "summary_exactly_reconstructed": summary == reconstructed,
        "candidate_benefit_reconstructed": (
            summary["single_candidate_benefit"] == candidates
        ),
        "gate_reconstructed": summary["gate"] == gate,
        "plain_final_qwen_and_two_5090_shards": (
            len(resources) == 2
            and all(resource["protocol"] == PROTOCOL
                    and resource["model"]["config"].get("ttt_mode") is False
                    and len(resource["model"]["excluded_adapters"]) == 14
                    and resource["action_alphas"] == [0.0, 0.5, 1.0]
                    and "5090" in resource["gpu"] for resource in resources)
        ),
        "heldout_sources_sealed": all(
            not resource["development_labels_loaded"]
            and not resource["development_model_scores_loaded"]
            and not resource["heldout_benchmarks_loaded"]
            and not resource["internal_test_loaded"]
            and not resource["external_test_loaded"]
            for resource in resources
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL, "passed": passed,
        "release_passed": passed and summary["gate"]["passed"],
        "checks": checks, "formal_gate": summary["gate"],
        "summary_sha256": sha256(summary_path),
        "artifact_sha256": hashes,
        "next_stage": (
            "v11_development_feature_extraction"
            if passed and summary["gate"]["passed"]
            else "stop_v11_before_feature_extraction"
        ),
        "terminal_status": (
            "v11_development_label_audit_passed_gate_passed"
            if passed and summary["gate"]["passed"]
            else "v11_development_label_audit_passed_gate_failed"
            if passed else "v11_development_label_audit_failed"
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
