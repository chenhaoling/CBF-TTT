"""Independent reconstruction audit for the V18 low-rank fast-memory bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "v18_qwen_lowrank_fastmemory_bridge_v1"
FORMAL_PROTOCOL = "controlled_memory_interference_v18_qwen_formal"
DATA_PROTOCOL = "controlled_memory_interference_v18_formal"
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")
UPDATE = frozenset(KINDS[:2]); RETAIN = frozenset(KINDS[2:4]); JOINT = UPDATE | RETAIN
SCALES = (4.0, 8.0, 16.0); PRIMARY = 8.0
HEADER = "Use the session fast memory and answer with only the requested code."
BOOTSTRAP_SEED = 1831; DRAWS = 20_000


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render(statement):
    return f"{HEADER}\n\nQuery: {statement}\nAnswer:\n"


def expected_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]} for policy in FIXED]
    arms.extend({"arm_id": f"delete_one:{row['deleted_session_id']}",
                 "kind": "delete_one", "deletion_index": index,
                 "deleted_session_id": row["deleted_session_id"],
                 "memory_session_ids": row["memory_session_ids"]}
                for index, row in enumerate(state["delete_one"]))
    arms.extend({"arm_id": f"delete_key:{row['deleted_key_id']}",
                 "kind": "delete_key", "deletion_index": index,
                 "deleted_key_id": row["deleted_key_id"],
                 "deleted_session_ids": row["deleted_session_ids"],
                 "memory_session_ids": row["memory_session_ids"]}
                for index, row in enumerate(state["delete_key"]))
    return arms


def arm_metrics(arm, scale):
    queries = {query["kind"]: query for query in arm["queries"]}; key = str(float(scale))
    return {"joint_success": int(all(queries[kind]["by_scale"][key]["correct"] for kind in JOINT)),
            "update_accuracy": statistics.fmean(
                queries[kind]["by_scale"][key]["correct"] for kind in UPDATE),
            "retained_key_accuracy": statistics.fmean(
                queries[kind]["by_scale"][key]["correct"] for kind in RETAIN),
            "joint_answer_nll": statistics.fmean(
                queries[kind]["by_scale"][key]["answer_nll"] for kind in JOINT),
            "locality_accuracy": queries["locality"]["by_scale"][key]["correct"],
            "generation_joint_success": int(all(
                queries[kind]["generation_primary"]["normalized_exact"] for kind in JOINT))
                if scale == PRIMARY else None}


def arm_index(row):
    result = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(result) != len(row["arms"]):
        raise ValueError("duplicate audited bridge arm")
    return result


def select_fixed(rows, scale):
    table = {}
    for policy in FIXED:
        values = [arm_metrics(arm_index(row)[f"fixed:{policy}"], scale) for row in rows]
        table[policy] = {key: statistics.fmean(value[key] for value in values)
                         for key in ("joint_success", "update_accuracy",
                                     "retained_key_accuracy", "joint_answer_nll",
                                     "locality_accuracy")}
    winner = max(FIXED, key=lambda policy: (
        table[policy]["joint_success"], table[policy]["update_accuracy"],
        table[policy]["retained_key_accuracy"], -table[policy]["joint_answer_nll"],
        -FIXED.index(policy)))
    return winner, table


def select_oracle(row, scale):
    arms = arm_index(row)
    candidates = [arms["fixed:retain_all"]] + [
        arm for arm in row["arms"] if arm["kind"] in {"delete_one", "delete_key"}]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        value = arm_metrics(arm, scale)
        ranked.append((value["joint_success"], value["update_accuracy"],
                       value["retained_key_accuracy"], -value["joint_answer_nll"],
                       -tie_index, arm, value))
    winner = max(ranked, key=lambda value: value[:5])
    return winner[5], winner[6]


def interval(records, cluster, seed):
    grouped = {}
    for row in records:
        grouped.setdefault(row[cluster], []).append(row["joint_gain"])
    values = [statistics.fmean(value) for _, value in sorted(grouped.items())]
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(DRAWS))
    return {"mean": statistics.fmean(values), "lower_95": samples[math.floor(.025 * DRAWS)],
            "upper_95": samples[min(DRAWS - 1, math.ceil(.975 * DRAWS) - 1)],
            "clusters": len(values), "draws": DRAWS, "seed": seed}


def aggregate_scale(rows, scale):
    selected, evidence = {}, {}
    for fold in range(5):
        selected[str(fold)], evidence[str(fold)] = select_fixed(
            [row for row in rows if row["fold"] != fold], scale)
    records, selections = [], []
    action_counts = {"retain_all": 0, "delete_one": 0, "delete_key": 0}
    for row in rows:
        oracle_arm, oracle = select_oracle(row, scale)
        policy = selected[str(row["fold"])]
        fixed = arm_metrics(arm_index(row)[f"fixed:{policy}"], scale)
        action = "retain_all" if oracle_arm["arm_id"] == "fixed:retain_all" \
            else oracle_arm["kind"]
        action_counts[action] += 1
        records.append({"subject": row["subject"], "template_group": row["template_group"],
                        "fold": row["fold"], "length": row["length"],
                        "mechanism": row["mechanism"], "joint_gain": oracle["joint_success"]
                        - fixed["joint_success"], "oracle": oracle, "fixed": fixed})
        selections.append({"episode_id": row["episode_id"], "oracle_arm_id": oracle_arm["arm_id"],
                           "fixed_policy": policy})
    mean = lambda key, side: statistics.fmean(row[side][key] for row in records)
    oracle_table = {key: mean(key, "oracle") for key in (
        "joint_success", "update_accuracy", "retained_key_accuracy",
        "joint_answer_nll", "locality_accuracy")}
    fixed_table = {key: mean(key, "fixed") for key in (
        "joint_success", "update_accuracy", "retained_key_accuracy",
        "joint_answer_nll", "locality_accuracy")}
    if scale == PRIMARY:
        oracle_table["generation_joint_success"] = mean("generation_joint_success", "oracle")
        fixed_table["generation_joint_success"] = mean("generation_joint_success", "fixed")
    by_mechanism = {name: {"rows": sum(row["mechanism"] == name for row in records),
                           "joint_gain": statistics.fmean(
                               row["joint_gain"] for row in records
                               if row["mechanism"] == name)}
                    for name in sorted({row["mechanism"] for row in records})}
    by_length = {str(length): {"rows": sum(row["length"] == length for row in records),
                               "joint_gain": statistics.fmean(
                                   row["joint_gain"] for row in records
                                   if row["length"] == length)}
                 for length in sorted({row["length"] for row in records})}
    return {"scale": scale, "fixed_selected_by_outer_fold": selected,
            "fixed_selection_evidence": evidence, "oracle_selection": selections,
            "oracle_action_counts": action_counts,
            "oracle": oracle_table, "outer_fold_fixed": fixed_table,
            "joint_gain_micro": statistics.fmean(row["joint_gain"] for row in records),
            "joint_gain_subject_macro": interval(records, "subject", BOOTSTRAP_SEED + int(scale)),
            "joint_gain_template_macro": interval(
                records, "template_group", BOOTSTRAP_SEED + 100 + int(scale)),
            "by_fold_gain": {str(fold): statistics.fmean(
                row["joint_gain"] for row in records if row["fold"] == fold)
                             for fold in range(5)},
            "by_mechanism": by_mechanism, "by_length": by_length}


def recompute_table(rows):
    scales = {str(scale): aggregate_scale(rows, scale) for scale in SCALES}
    content = {}
    for policy in ("correct", "wrong", "zero"):
        queries = [control[policy] for row in rows for control in row["content_controls"]]
        content[policy] = {"accuracy": statistics.fmean(
            query["by_scale"][str(PRIMARY)]["correct"] for query in queries),
                           "answer_nll": statistics.fmean(
            query["by_scale"][str(PRIMARY)]["answer_nll"] for query in queries)}
    primary = scales[str(PRIMARY)]
    checks = {"correct_component_accuracy_at_least_90": content["correct"]["accuracy"] >= .90,
              "wrong_component_accuracy_at_most_20": content["wrong"]["accuracy"] <= .20,
              "oracle_update_accuracy_at_least_90": primary["oracle"]["update_accuracy"] >= .90,
              "oracle_retained_key_accuracy_at_least_90":
                  primary["oracle"]["retained_key_accuracy"] >= .90,
              "oracle_locality_accuracy_at_least_97": primary["oracle"]["locality_accuracy"] >= .97,
              "oracle_joint_gain_at_least_10_points": primary["joint_gain_micro"] >= .10,
              "subject_cluster_ci_lower_above_zero":
                  primary["joint_gain_subject_macro"]["lower_95"] > 0,
              "template_cluster_ci_lower_above_zero":
                  primary["joint_gain_template_macro"]["lower_95"] > 0,
              "positive_joint_gain_at_scales_4_8_16":
                  all(scales[str(scale)]["joint_gain_micro"] > 0 for scale in SCALES)}
    supported = all(checks.values())
    return {"scales": scales, "primary_scale": PRIMARY, "content_binding": content,
            "bridge_gate": {"checks": checks, "passed": supported},
            "fastmemory_bridge_supported": supported,
            "next_stage": "evaluate_physical_conflict_rules" if supported
                else "stop_fastmemory_controller_claim"}


def recompute_resources(rows, resources):
    measurements = [query for row in rows for arm in row["arms"]
                    if arm["score_source_arm_id"] == arm["arm_id"]
                    for query in arm["queries"]]
    measurements.extend(query for row in rows for control in row["content_controls"]
                        for name in ("correct", "wrong", "zero")
                        for query in [control[name]])
    latencies = sorted(query["resources"]["seconds"] for query in measurements)
    return {"unique_memory_states": sum(row["unique_memory_states_scored"]
                                        for row in resources),
            "query_measurements": len(measurements),
            "wall_seconds_slower_shard": max(row["seconds"] for row in resources),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in measurements),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in measurements)}


def audit(data_root, formal_root, model, tokenizer, inputs, summary_path, output):
    data_root, formal_root, model, tokenizer = map(
        Path, (data_root, formal_root, model, tokenizer))
    summary, manifest = read_json(summary_path), read_json(data_root / "manifest.json")
    data_audit_path = data_root.parent / "audit.json"
    if not data_audit_path.exists():
        data_audit_path = data_root / "audit.json"
    data_audit = read_json(data_audit_path)
    formal_summary, formal_audit = (read_json(formal_root / "summary.json"),
                                    read_json(formal_root / "audit.json"))
    episodes, states = read_jsonl(data_root / "formal.jsonl"), read_jsonl(
        data_root / "formal_states.jsonl")
    episode_index = {row["episode_id"]: row for row in episodes}
    state_index = {row["episode_id"]: row for row in states}
    rows, resources, hashes, observed = [], [], {}, {}
    for value in inputs:
        path = Path(value); part = read_jsonl(path); resource = read_json(str(path) + ".audit.json")
        rows.extend(part); resources.append(resource); hashes[path.name] = sha(path)
        observed[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = arms_ok = queries_ok = controls_ok = measurements_ok = True
    for row in rows:
        episode, state = episode_index.get(row.get("episode_id")), state_index.get(
            row.get("episode_id"))
        if episode is None or state is None:
            identities_ok = False; continue
        identities_ok &= all(row.get(key) == episode.get(key) for key in (
            "row_index", "subject", "subject_index", "template_group",
            "template_group_index", "fold", "length", "mechanism", "target_key_id",
            "current_session_id"))
        expected = expected_arms(state); arms_ok &= len(row.get("arms", [])) == len(expected)
        session_index = {session["session_id"]: session for session in episode["sessions"]}
        first_source, first_queries = {}, {}
        for actual, wanted in zip(row.get("arms", []), expected):
            arms_ok &= all(actual.get(key) == value for key, value in wanted.items())
            memory_key = tuple(wanted["memory_session_ids"])
            source = first_source.setdefault(memory_key, wanted["arm_id"])
            arms_ok &= actual.get("score_source_arm_id") == source
            if memory_key in first_queries:
                arms_ok &= actual.get("queries") == first_queries[memory_key]
            else:
                first_queries[memory_key] = actual.get("queries")
            for query, spec in zip(actual.get("queries", []), episode["queries"]):
                prompt = render(spec["statement"]); candidates = spec["candidates"]
                queries_ok &= (
                    query.get("kind") == spec["kind"] and query.get("statement") == spec["statement"]
                    and query.get("answer") == spec["answer"]
                    and query.get("candidates") == candidates
                    and query.get("memory_session_ids") == wanted["memory_session_ids"]
                    and query.get("prompt_sha256") == hashlib.sha256(prompt.encode()).hexdigest()
                    and query.get("answer_token_lengths") == {value: 5 for value in candidates}
                    and query.get("total_tokens_max") == query.get("prompt_tokens") + 5
                    and len(query.get("residual_norms", [])) == 5
                    and all(math.isfinite(value) and value >= 0
                            for value in query.get("residual_norms", [])))
                expected_contributions = [] if spec["kind"] == "locality" else [
                    (identifier, session_index[identifier]["key_id"],
                     session_index[identifier]["value"])
                    for identifier in wanted["memory_session_ids"]]
                observed_contributions = [(value.get("session_id"), value.get("key_id"),
                                           value.get("value"))
                                          for value in query.get("contributions", [])]
                queries_ok &= observed_contributions == expected_contributions and all(
                    math.isfinite(value.get("key_similarity", math.nan))
                    for value in query.get("contributions", []))
                for scale in SCALES:
                    table = query.get("by_scale", {}).get(str(scale), {})
                    scores = table.get("scores", {})
                    queries_ok &= (
                        set(scores) == set(candidates) and all(math.isfinite(value)
                                                               for value in scores.values())
                        and table.get("prediction") == min(
                            candidates, key=lambda value: (scores[value], value))
                        and table.get("correct") == int(table.get("prediction") == spec["answer"])
                        and table.get("answer_nll") == scores.get(spec["answer"]))
                generation = query.get("generation_primary", {})
                normalized = " ".join(generation.get("text", "").strip().casefold().split())
                truth = " ".join(spec["answer"].strip().casefold().split())
                queries_ok &= (generation.get("exact") == int(generation.get("text") == spec["answer"])
                               and generation.get("normalized_exact") == int(normalized == truth)
                               and generation.get("max_new_tokens") == 13
                               and 0 <= generation.get("tokens", -1) <= 13)
                measurement = query.get("resources", {})
                measurements_ok &= (
                    all(math.isfinite(measurement.get(key, math.nan)) for key in (
                        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
                    and measurement.get("seconds", 0) > 0
                    and measurement.get("candidate_scores") == len(candidates) * len(SCALES))
        controls = row.get("content_controls", [])
        controls_ok &= [control.get("kind") for control in controls] == list(KINDS[:4])
        for control in controls:
            spec = next(spec for spec in episode["queries"] if spec["kind"] == control["kind"])
            key_index = {"update_old_surface": 0, "update_current_surface": 0,
                         "retain_1": 1, "retain_2": 2}[control["kind"]]
            key_id = episode["keys"][key_index]
            correct_session = [session for session in episode["sessions"]
                               if session["key_id"] == key_id][-1]
            wrong_value = {0: episode["primary_old_value"],
                           1: episode["current_values"][3],
                           2: episode["current_values"][4]}[key_index]
            for name in ("correct", "wrong", "zero"):
                query = control.get(name, {})
                expected_ids = ({"correct": [correct_session["session_id"]],
                                 "wrong": [f"wrong:{control['kind']}"], "zero": []}[name])
                expected_contributions = ({
                    "correct": [(correct_session["session_id"], key_id,
                                 correct_session["value"])],
                    "wrong": [(f"wrong:{control['kind']}", key_id, wrong_value)],
                    "zero": [],
                }[name])
                observed_contributions = [(value.get("session_id"), value.get("key_id"),
                                           value.get("value"))
                                          for value in query.get("contributions", [])]
                controls_ok &= (
                    query.get("kind") == control["kind"]
                    and query.get("statement") == spec["statement"]
                    and query.get("answer") == spec["answer"]
                    and query.get("candidates") == spec["candidates"]
                    and query.get("memory_session_ids") == expected_ids
                    and observed_contributions == expected_contributions
                    and all(math.isfinite(value.get("key_similarity", math.nan))
                            for value in query.get("contributions", []))
                    and query.get("prompt_sha256")
                        == hashlib.sha256(render(spec["statement"]).encode()).hexdigest()
                    and query.get("answer_token_lengths")
                        == {value: 5 for value in spec["candidates"]}
                    and len(query.get("residual_norms", [])) == 5
                    and all(math.isfinite(value) and value >= 0
                            for value in query.get("residual_norms", [])))
                for scale in SCALES:
                    table = query.get("by_scale", {}).get(str(scale), {})
                    scores = table.get("scores", {})
                    controls_ok &= (
                        set(scores) == set(spec["candidates"])
                        and all(math.isfinite(value) for value in scores.values())
                        and table.get("prediction") == min(
                            spec["candidates"], key=lambda value: (scores[value], value))
                        and table.get("correct") == int(
                            table.get("prediction") == spec["answer"])
                        and table.get("answer_nll") == scores.get(spec["answer"]))
                generation = query.get("generation_primary", {})
                normalized = " ".join(generation.get("text", "").strip().casefold().split())
                truth = " ".join(spec["answer"].strip().casefold().split())
                controls_ok &= (
                    generation.get("exact") == int(generation.get("text") == spec["answer"])
                    and generation.get("normalized_exact") == int(normalized == truth)
                    and generation.get("max_new_tokens") == 13
                    and 0 <= generation.get("tokens", -1) <= 13)
                measurement = query.get("resources", {})
                measurements_ok &= (
                    all(math.isfinite(measurement.get(key, math.nan)) for key in (
                        "seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                        "gpu_peak_allocated_gib", "gpu_peak_reserved_gib"))
                    and measurement.get("seconds", 0) > 0
                    and measurement.get("candidate_scores")
                        == len(spec["candidates"]) * len(SCALES))

    expected_shards = {shard: [row["episode_id"] for index, row in enumerate(episodes)
                                if index % 2 == shard] for shard in (0, 1)}
    expected_unique, expected_reused = {}, {}
    for shard in (0, 1):
        unique = reused = 0
        for index, state in enumerate(states):
            if index % 2 != shard:
                continue
            memories = [tuple(arm["memory_session_ids"]) for arm in expected_arms(state)]
            unique += len(set(memories)); reused += len(memories) - len(set(memories))
        expected_unique[shard], expected_reused[shard] = unique, reused
    dependencies = {"formal_data_manifest_sha256": sha(data_root / "manifest.json"),
                    "formal_data_audit_sha256": sha(data_audit_path),
                    "formal_data_sha256": sha(data_root / "formal.jsonl"),
                    "formal_states_sha256": sha(data_root / "formal_states.jsonl"),
                    "formal_summary_sha256": sha(formal_root / "summary.json"),
                    "formal_outcome_audit_sha256": sha(formal_root / "audit.json")}
    source_ok = (manifest.get("protocol") == DATA_PROTOCOL
                 and data_audit.get("audit_passed") is True
                 and formal_summary.get("protocol") == FORMAL_PROTOCOL
                 and formal_summary.get("causal_headroom_supported") is True
                 and formal_audit.get("audit_passed") is True
                 and formal_audit.get("summary_sha256") == dependencies["formal_summary_sha256"]
                 and (formal_root / "runner_exit_code.txt").read_text().strip() == "0")
    coverage_ok = len(rows) == len({row["episode_id"] for row in rows}) == 320 \
        and observed == expected_shards
    shard_ok = (len(resources) == 2 and {row.get("shard") for row in resources} == {0, 1}
                and all(row.get("protocol") == PROTOCOL and row.get("num_shards") == 2
                        and row.get("rows") == 160
                        and row.get("unique_memory_states_scored") == expected_unique[row["shard"]]
                        and row.get("reused_arm_states") == expected_reused[row["shard"]]
                        and row.get("content_control_measurements") == 160 * 12
                        and row.get("query_measurements")
                            == 5 * expected_unique[row["shard"]] + 160 * 12
                        and row.get("rows_sha256") == hashes[Path(path).name]
                        and row.get("dependencies") == dependencies
                        and row.get("scales") == list(SCALES)
                        and row.get("primary_scale") == PRIMARY
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
        and all(not row.get("loading_info", {}).get(key)
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
        for row in resources)
    table, resource_table = recompute_table(rows), recompute_resources(rows, resources)
    checks = {"positive_formal_release_and_frozen_data": source_ok,
              "complete_ordered_two_shard_coverage": coverage_ok,
              "episode_state_and_split_identities": identities_ok,
              "fixed_single_key_arms_and_dedup_sources": arms_ok,
              "prompt_scores_contributions_and_generation": queries_ok,
              "correct_wrong_zero_content_controls": controls_ok,
              "per_query_resource_records": measurements_ok,
              "shard_counts_dependencies_and_hashes": shard_ok,
              "model_tokenizer_and_native_adapter_provenance": provenance_ok,
              "independent_bridge_gate_recomputation": all(
                  summary.get(key) == table[key] for key in table),
              "independent_resource_recomputation":
                  summary.get("resources") == resource_table,
              "terminal_and_access_locks": (
                  summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
                  and summary.get("raw_rows_sha256") == hashes
                  and summary.get("shard_resources") == resources
                  and summary.get("model_scored") is True
                  and summary.get("formal_scored") is True
                  and summary.get("external_sets_loaded") is False
                  and summary.get("terminal_status") in {
                      "fastmemory_bridge_supported", "fastmemory_bridge_unsupported"})}
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes, "recomputed_bridge_gate": table["bridge_gate"],
              "model_scored": True, "formal_scored": True,
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 low-rank fast-memory bridge audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
