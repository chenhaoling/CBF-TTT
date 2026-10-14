"""Independent audit of Qwen rescoring for the V18 learned deletion policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_v18_physical_conflict_rules import _query_ok
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.evaluate_v18_causal_deletion_policy import (
    PROTOCOL, aggregate, read_json, verify_policy_release,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_v18_causal_deletion_policy import SEEDS


def audit(data_root, formal_root, bridge_root, physical_root, policy_root,
          model, tokenizer, inputs, summary_path, output):
    data_root, policy_root = Path(data_root), Path(policy_root)
    model, tokenizer, summary_path = Path(model), Path(tokenizer), Path(summary_path)
    dependencies = verify_policy_release(
        data_root, formal_root, bridge_root, physical_root, policy_root)
    summary = read_json(summary_path)
    episodes = load_jsonl(data_root / "formal.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    policy_rows = load_jsonl(policy_root / "result/oof_policy_states.jsonl")
    policy_index = {row["episode_id"]: row for row in policy_rows}
    rows, resources, hashes, observed = [], [], {}, {0: [], 1: []}
    for value in inputs:
        path = Path(value); part = load_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]
    expected_order = {shard: [row["episode_id"] for index, row in enumerate(episodes)
                               if index % 2 == shard] for shard in (0, 1)}
    identities_ok = states_ok = queries_ok = True
    expected_unique = {0: 0, 1: 0}; expected_reused = {0: 0, 1: 0}
    for row in rows:
        episode = episode_index.get(row.get("episode_id"), {})
        policy = policy_index.get(row.get("episode_id"), {})
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "target_key_id",
            "current_session_id")) and row.get("selection") == policy.get("selection")
        arms = row.get("arms", []); arm_index = {arm.get("seed"): arm for arm in arms}
        states_ok &= set(arm_index) == set(SEEDS) and len(arms) == len(SEEDS)
        first_by_state = {}; valid_ids = {value["session_id"] for value in episode["sessions"]}
        for seed in SEEDS:
            expected = policy.get("per_seed", {}).get(str(seed), {})
            identifiers = expected.get("retained_session_ids", [])
            state = tuple(identifiers); first_by_state.setdefault(state, seed)
            arm = arm_index.get(seed, {})
            states_ok &= (
                arm.get("memory_session_ids") == identifiers
                and arm.get("deleted_session_ids") == expected.get("deleted_session_ids")
                and arm.get("component_scores") == expected.get("components")
                and arm.get("score_source_seed") == first_by_state[state]
                and set(identifiers) | set(arm.get("deleted_session_ids", [])) == valid_ids
                and not set(identifiers) & set(arm.get("deleted_session_ids", []))
                and episode.get("current_session_id") in identifiers)
            qindex = {query.get("kind"): query for query in arm.get("queries", [])}
            for spec in episode.get("queries", []):
                queries_ok &= _query_ok(qindex.get(spec["kind"], {}), spec, identifiers)
        shard = episode.get("row_index", 0) % 2
        expected_unique[shard] += len(set(tuple(
            policy["per_seed"][str(seed)]["retained_session_ids"]) for seed in SEEDS))
        expected_reused[shard] += len(SEEDS) - len(set(tuple(
            policy["per_seed"][str(seed)]["retained_session_ids"]) for seed in SEEDS))
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 320
                   and observed == expected_order)
    shard_ok = (len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
                and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                        and row.get("rows") == 160
                        and row.get("rows_sha256") == hashes[Path(path).name]
                        and row.get("dependencies") == dependencies
                        and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                        and row.get("reused_arm_states") == expected_reused[row["shard"]]
                        and row.get("query_measurements") == 5 * expected_unique[row["shard"]]
                        and row.get("seeds") == list(SEEDS)
                        and row.get("fresh_kv") is True and row.get("backbone_frozen") is True
                        and row.get("query_features_used") is False
                        and row.get("current_update_always_retained") is True
                        and row.get("external_sets_loaded") is False
                        for row, path in zip(resources, inputs)))
    tokenizer_hashes = {path.name: sha(path) for path in sorted(tokenizer.iterdir())
                        if path.is_file() and path.suffix in {
                            ".json", ".model", ".txt", ".tiktoken"}}
    model_hashes = {path.name: sha(path) for path in sorted(model.glob("*.safetensors"))}
    provenance_ok = bool(model_hashes) and bool(tokenizer_hashes) and all(
        row.get("model") == str(model) and row.get("tokenizer") == str(tokenizer)
        and row.get("model_provenance", {}).get("path") == str(model)
        and row.get("model_provenance", {}).get("config_sha256") == sha(model / "config.json")
        and row.get("model_provenance", {}).get("source_files_sha256") == model_hashes
        and row.get("model_provenance", {}).get("ttt_mode") is True
        and row.get("model_provenance", {}).get("excluded_adapters") == []
        and row.get("tokenizer_files_sha256") == tokenizer_hashes
        and all(not row.get("loading_info", {}).get(key) for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        for row in resources)
    table = aggregate(rows, physical_root)
    checks = {"audited_oof_policy_release": True,
              "complete_ordered_two_shard_coverage": coverage_ok,
              "episode_split_and_selection_identities": identities_ok,
              "three_seed_component_state_reconstruction": states_ok,
              "fresh_kv_scores_generation_and_resources": queries_ok,
              "shard_counts_dependencies_and_hashes": shard_ok,
              "model_tokenizer_and_native_adapter_provenance": provenance_ok,
              "independent_controlled_policy_gate_recomputation": all(
                  summary.get(key) == value for key, value in table.items()),
              "terminal_summary_locks": (
                  summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
                  and summary.get("raw_rows_sha256") == hashes
                  and summary.get("shard_resources") == resources
                  and summary.get("dependencies") == dependencies
                  and summary.get("model_scored") is True
                  and summary.get("formal_scored") is True
                  and summary.get("external_sets_loaded") is False
                  and summary.get("terminal_status") in {
                      "controlled_policy_supported", "controlled_policy_unsupported"})}
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes,
              "recomputed_controlled_policy_gate": table["controlled_policy_gate"],
              "model_scored": True, "formal_scored": True,
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 learned-policy Qwen audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--formal-root", required=True)
    parser.add_argument("--bridge-root", required=True); parser.add_argument("--physical-root", required=True)
    parser.add_argument("--policy-root", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True); parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
