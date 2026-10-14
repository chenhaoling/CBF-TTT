"""Independent reconstruction audit for V18 physical conflict rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl, normalize_generation
from tasks.evaluate_v18_lowrank_fastmemory_bridge import SCALES, render_query
from tasks.evaluate_v18_physical_conflict_rules import (
    COSINE_THRESHOLDS, PRIMARY_SCALE, PROTOCOL, RANDOM_SEEDS, aggregate,
    bridge_references, bridge_rows, physical_states, read_json, verify_bridge_release,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def _expected_arms(episode, row):
    similarities = {}
    for name, value in row.get("key_similarities", {}).items():
        pair = tuple(name.split("||"))
        if len(pair) != 2 or not math.isfinite(value):
            raise ValueError("invalid serialized key similarity")
        similarities[pair] = value
    return physical_states(episode, similarities), similarities


def _query_ok(query, spec, identifiers):
    if (query.get("kind") != spec["kind"] or query.get("statement") != spec["statement"]
            or query.get("answer") != spec["answer"]
            or query.get("candidates") != spec["candidates"]
            or query.get("memory_session_ids") != identifiers
            or set(query.get("answer_token_lengths", {}).values()) != {5}):
        return False
    prompt = render_query(spec["statement"])
    if (query.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
            or query.get("prompt_tokens", 0) <= 0
            or query.get("total_tokens_max") != query.get("prompt_tokens") + 5):
        return False
    contribution_ids = [value["session_id"] for value in query.get("contributions", [])]
    if spec["kind"] == "locality":
        if contribution_ids or any(query.get("residual_norms", [])):
            return False
    elif contribution_ids != identifiers:
        return False
    for scale in SCALES:
        result = query.get("by_scale", {}).get(str(float(scale)), {})
        scores = result.get("scores", {})
        if set(scores) != set(spec["candidates"]) or not all(
                math.isfinite(value) for value in scores.values()):
            return False
        predicted = min(spec["candidates"], key=lambda value: (scores[value], value))
        if (result.get("prediction") != predicted
                or result.get("correct") != int(predicted == spec["answer"])
                or result.get("answer_nll") != scores[spec["answer"]]):
            return False
    generation = query.get("generation_primary", {})
    normalized = normalize_generation(generation.get("text", ""))
    if (generation.get("exact") != int(generation.get("text", "") == spec["answer"])
            or generation.get("normalized_exact") != int(
                normalized == normalize_generation(spec["answer"]))
            or generation.get("max_new_tokens") != 13
            or not 0 <= generation.get("tokens", -1) <= 13):
        return False
    resources = query.get("resources", {})
    return (all(math.isfinite(resources.get(key, math.nan)) for key in (
        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
        and resources.get("seconds", 0) > 0
        and resources.get("candidate_scores") == len(spec["candidates"]) * len(SCALES))


def audit(data_root, formal_root, bridge_root, model, tokenizer, inputs,
          summary_path, output):
    data_root, bridge_root = Path(data_root), Path(bridge_root)
    model, tokenizer, summary_path = Path(model), Path(tokenizer), Path(summary_path)
    dependencies = verify_bridge_release(data_root, formal_root, bridge_root)
    summary = read_json(summary_path)
    episodes = load_jsonl(data_root / "formal.jsonl")
    source_index = bridge_rows(bridge_root)
    bridge_summary = read_json(bridge_root / "summary.json")
    rows, resources, hashes, observed = [], [], {}, {0: [], 1: []}
    for value in inputs:
        path = Path(value); part = load_jsonl(path)
        resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]
    episode_index = {row["episode_id"]: row for row in episodes}
    expected_order = {shard: [row["episode_id"] for index, row in enumerate(episodes)
                               if index % 2 == shard] for shard in (0, 1)}
    identities_ok = arms_ok = queries_ok = similarities_ok = references_ok = True
    expected_unique = {0: 0, 1: 0}; expected_reused = {0: 0, 1: 0}
    for row in rows:
        episode = episode_index.get(row.get("episode_id"))
        if episode is None:
            identities_ok = False; continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "target_key_id",
            "current_session_id"))
        expected, similarities = _expected_arms(episode, row)
        keys = sorted({item["key_id"] for item in episode["sessions"]})
        similarities_ok &= (set(similarities) == {
            (left, right) for left in keys for right in keys if left <= right}
            and all(abs(similarities[(key, key)] - 1.0) < 1e-4 for key in keys))
        arms = row.get("arms", []); arm_index = {arm.get("arm_name"): arm for arm in arms}
        arms_ok &= list(arm_index) == list(expected) and len(arm_index) == len(arms)
        first_by_state = {}
        for name, identifiers in expected.items():
            arm = arm_index.get(name, {})
            state = tuple(identifiers); first_by_state.setdefault(state, name)
            arms_ok &= (arm.get("memory_session_ids") == identifiers
                        and arm.get("score_source_arm_name") == first_by_state[state]
                        and episode["current_session_id"] in identifiers)
            qindex = {query.get("kind"): query for query in arm.get("queries", [])}
            queries_ok &= set(qindex) == set(KINDS)
            for spec in episode["queries"]:
                queries_ok &= _query_ok(qindex.get(spec["kind"], {}), spec, identifiers)
        shard = episode["row_index"] % 2
        expected_unique[shard] += len(set(tuple(value) for value in expected.values()))
        expected_reused[shard] += len(expected) - len(set(tuple(value) for value in expected.values()))
        references_ok &= row.get("bridge_reference") == bridge_references(
            source_index[row["episode_id"]], bridge_summary)
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
                        and row.get("cosine_thresholds") == list(COSINE_THRESHOLDS)
                        and row.get("random_seeds") == list(RANDOM_SEEDS)
                        and row.get("primary_scale") == PRIMARY_SCALE
                        and row.get("fresh_kv") is True and row.get("backbone_frozen") is True
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
    table = aggregate(rows)
    checks = {"positive_fastmemory_bridge_release": True,
              "complete_ordered_two_shard_coverage": coverage_ok,
              "episode_and_outer_fold_identities": identities_ok,
              "frozen_pairwise_qwen_similarities": similarities_ok,
              "physical_and_matched_random_state_reconstruction": arms_ok,
              "fresh_kv_scores_generation_and_resources": queries_ok,
              "bridge_fixed_and_oracle_references": references_ok,
              "shard_counts_dependencies_and_hashes": shard_ok,
              "model_tokenizer_and_native_adapter_provenance": provenance_ok,
              "independent_outer_fold_summary_recomputation": all(
                  summary.get(key) == value for key, value in table.items()),
              "terminal_summary_locks": (
                  summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
                  and summary.get("raw_rows_sha256") == hashes
                  and summary.get("shard_resources") == resources
                  and summary.get("dependencies") == dependencies
                  and summary.get("model_scored") is True
                  and summary.get("formal_scored") is True
                  and summary.get("external_sets_loaded") is False
                  and summary.get("terminal_status") == "physical_conflict_rules_evaluated")}
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes, "model_scored": True,
              "formal_scored": True, "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 physical-rule audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
