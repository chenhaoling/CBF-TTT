"""Independently audit the frozen nine-arm V16 Qwen outcome experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_mquake_qwen_outcome_v16_formal_states import POLICIES
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_qwen_outcome_v16_formal"
KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")
PRIMARY = frozenset(("primary_old_phrase", "primary_current_phrase"))
FIXED = ("no_memory", "retain_all", "global_reset", "last_1", "last_2")
HEADER = (
    "The following are chronological session memories. When statements conflict, "
    "use the most recent statement. Answer the query with only the requested value."
)
DRAWS = 20_000
SEED = 1611


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def render(memory, texts, statement):
    chunks = [HEADER, "", "Session memories:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.append(f"[Session {index + 1}]")
            chunks.extend(texts[identifier] for identifier in session["units"])
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def specs(sample):
    primary = [sample["old"]["value"], sample["current"]["value"]]
    safety = []
    for name in ("old", "current", "hard_retain", "unrelated"):
        if sample[name]["value"] not in safety:
            safety.append(sample[name]["value"])
    return (
        ("primary_old_phrase", sample["old"]["statement"], sample["current"]["value"], primary),
        ("primary_current_phrase", sample["current"]["statement"], sample["current"]["value"], primary),
        ("hard_retain", sample["hard_retain"]["statement"], sample["hard_retain"]["value"], safety),
        ("unrelated", sample["unrelated"]["statement"], sample["unrelated"]["value"], safety),
    )


def normalize(value):
    return " ".join(value.strip().casefold().split())


def reconstruct_memory(episode, sample, identifiers):
    visible = episode["sessions"][:sample["query_step"] + 1]
    index = {row["session_id"]: row for row in visible}
    if len(index) != len(visible) or any(identifier not in index for identifier in identifiers):
        raise ValueError("noncausal formal memory")
    return [index[identifier] for identifier in identifiers]


def pairs(rows, policy, kinds=None):
    kinds = set(kinds) if kinds is not None else None
    return [(row, query) for row in rows for query in row["policies"][policy]["queries"]
            if kinds is None or query["kind"] in kinds]


def table(rows, policy):
    values = pairs(rows, policy)
    subjects = sorted({row["subject"] for row in rows})
    def metrics(items):
        return {"queries": len(items),
                "accuracy": statistics.fmean(query["correct"] for _, query in items),
                "answer_nll": statistics.fmean(query["answer_nll"] for _, query in items),
                "generation_exact": statistics.fmean(query["generation"]["exact"]
                                                       for _, query in items),
                "generation_normalized_exact": statistics.fmean(
                    query["generation"]["normalized_exact"] for _, query in items)}
    primary = pairs(rows, policy, PRIMARY)
    pair_accuracy = statistics.fmean(int(all(query["correct"] for query in
        row["policies"][policy]["queries"] if query["kind"] in PRIMARY)) for row in rows)
    queries = [query for _, query in values]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    sizes = [len(row["policies"][policy]["memory_session_ids"]) for row in rows]
    return {
        **metrics(values), "primary": metrics(primary), "primary_pair_accuracy": pair_accuracy,
        "by_kind": {kind: metrics(pairs(rows, policy, (kind,))) for kind in KINDS},
        "by_fold": {str(fold): metrics(pairs(
            [row for row in rows if row["fold"] == fold], policy)) for fold in range(5)},
        "subject_macro": statistics.fmean(statistics.fmean(
            query["correct"] for row, query in values if row["subject"] == subject)
            for subject in subjects),
        "memory_sessions": {"mean": statistics.fmean(sizes), "maximum": max(sizes)},
        "max_prompt_tokens": max(query["prompt_tokens"] for query in queries),
        "max_total_tokens": max(query["total_tokens_max"] for query in queries),
        "resources": {
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                query["resources"]["gpu_peak_allocated_gib"] for query in queries),
            "max_gpu_peak_reserved_gib": max(
                query["resources"]["gpu_peak_reserved_gib"] for query in queries),
        },
    }


def fixed_selection(rows):
    selected, evidence = {}, {}
    for fold in range(5):
        development = [row for row in rows if row["fold"] != fold]
        scores = {}
        for policy in FIXED:
            values = pairs(development, policy, PRIMARY)
            scores[policy] = {
                "primary_accuracy": statistics.fmean(query["correct"] for _, query in values),
                "primary_answer_nll": statistics.fmean(query["answer_nll"] for _, query in values),
                "queries": len(values),
            }
        selected[str(fold)] = max(FIXED, key=lambda policy: (
            scores[policy]["primary_accuracy"], -scores[policy]["primary_answer_nll"],
            -FIXED.index(policy)))
        evidence[str(fold)] = scores
    return selected, evidence


def comparison_rows(rows, selected):
    output = []
    for row in rows:
        fixed = selected[str(row["fold"])]
        baseline = {query["kind"]: query for query in row["policies"][fixed]["queries"]}
        learned = {query["kind"]: query for query in row["policies"]["v15"]["queries"]}
        for kind in PRIMARY:
            output.append({"subject": row["subject"], "fold": row["fold"],
                "fixed_policy": fixed, "kind": kind,
                "accuracy_gain": learned[kind]["correct"] - baseline[kind]["correct"],
                "nll_gain": baseline[kind]["answer_nll"] - learned[kind]["answer_nll"]})
    return output


def interval(records, field, seed):
    subjects = sorted({row["subject"] for row in records})
    means = {subject: statistics.fmean(row[field] for row in records
                                       if row["subject"] == subject)
             for subject in subjects}
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(means[rng.choice(subjects)] for _ in subjects)
                     for _ in range(DRAWS))
    return {"mean": statistics.fmean(means.values()),
            "lower_95": samples[int(.025 * DRAWS)],
            "upper_95": samples[math.ceil(.975 * DRAWS) - 1],
            "subjects": len(subjects), "draws": DRAWS, "seed": seed}


def recompute(rows):
    policies = {policy: table(rows, policy) for policy in POLICIES}
    selected, evidence = fixed_selection(rows)
    records = comparison_rows(rows, selected)
    accuracy, nll = interval(records, "accuracy_gain", SEED), interval(records, "nll_gain", SEED + 1)
    folds = {}
    for fold in range(5):
        subset = [row for row in records if row["fold"] == fold]
        folds[str(fold)] = {"fixed_policy": selected[str(fold)], "queries": len(subset),
            "v15_minus_fixed_accuracy": statistics.fmean(row["accuracy_gain"] for row in subset),
            "fixed_minus_v15_nll": statistics.fmean(row["nll_gain"] for row in subset)}
    v15 = policies["v15"]
    checks = {
        "v15_primary_accuracy_gain_ci_lower_above_zero": accuracy["mean"] > 0 and accuracy["lower_95"] > 0,
        "v15_primary_nll_gain_ci_lower_above_zero": nll["mean"] > 0 and nll["lower_95"] > 0,
        "v15_primary_current_accuracy_at_least_97": v15["by_kind"]["primary_current_phrase"]["accuracy"] >= .97,
        "v15_unrelated_accuracy_at_least_97": v15["by_kind"]["unrelated"]["accuracy"] >= .97,
        "v15_hard_retain_accuracy_at_least_97": v15["by_kind"]["hard_retain"]["accuracy"] >= .97,
        "no_outer_fold_has_negative_accuracy_gain": all(
            value["v15_minus_fixed_accuracy"] >= 0 for value in folds.values()),
        "all_scores_finite": all(math.isfinite(query["answer_nll"])
            for row in rows for policy in POLICIES for query in row["policies"][policy]["queries"]),
        "all_prompts_within_checkpoint_limit": max(
            value["max_total_tokens"] for value in policies.values()) <= 40960,
    }
    return {"policies": policies,
        "fixed_selection": {"rule": "leave-one-outer-fold-out primary accuracy, then NLL, then registered order",
                            "candidates": list(FIXED), "selected": selected,
                            "development_evidence": evidence},
        "primary_comparison": {"v15_minus_crossfit_fixed_accuracy": accuracy,
            "crossfit_fixed_minus_v15_nll": nll, "by_fold": folds},
        "outcome_gate": {"checks": checks, "passed": all(checks.values())}}


def audit(data_root, sample_root, state_root, model, tokenizer, inputs,
          summary_path, generation_bridge, output):
    from transformers import AutoTokenizer

    data_root, sample_root, state_root = map(Path, (data_root, sample_root, state_root))
    summary = read_json(summary_path)
    bridge = read_json(generation_bridge)
    samples = read_jsonl(sample_root / "formal.jsonl")
    states = read_jsonl(state_root / "formal_states.jsonl")
    episodes = read_jsonl(data_root / "train.jsonl") + read_jsonl(data_root / "tune.jsonl")
    sample_index = {row["episode_id"]: row for row in samples}
    state_index = {row["episode_id"]: row for row in states}
    episode_index = {row["episode_id"]: row for row in episodes}
    texts = read_json(data_root / "texts.json")
    tokenizer_object = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    rows, audits, hashes = [], [], {}
    ordered_shards = {}
    for value in inputs:
        path = Path(value)
        part, resource = read_jsonl(path), read_json(str(path) + ".audit.json")
        rows.extend(part); audits.append(resource); hashes[path.name] = sha(path)
        ordered_shards[resource["shard"]] = [row["episode_id"] for row in part]

    identities_ok = prompt_score_ok = generation_ok = resource_ok = dedup_ok = True
    expected_unique = 0
    for row in rows:
        sample, state, episode = (mapping.get(row.get("episode_id")) for mapping in
                                  (sample_index, state_index, episode_index))
        if sample is None or state is None or episode is None:
            identities_ok = False
            continue
        identities_ok &= all(row[key] == sample[key] == state[key] for key in
            ("episode_id", "subject", "fold", "length", "hard_relation_matched", "query_step"))
        identities_ok &= tuple(row["policies"]) == POLICIES
        first_by_state = {}
        for policy in POLICIES:
            identifiers = state["policies"][policy]
            arm = row["policies"][policy]
            key = tuple(identifiers)
            source = first_by_state.setdefault(key, policy)
            expected_unique += int(source == policy)
            dedup_ok &= (arm["memory_session_ids"] == identifiers
                         and arm["score_source_policy"] == source)
            memory = reconstruct_memory(episode, sample, identifiers)
            if source != policy:
                dedup_ok &= arm["queries"] == row["policies"][source]["queries"]
            if len(arm["queries"]) != 4:
                prompt_score_ok = False
                continue
            for query, (kind, statement, answer, candidates) in zip(arm["queries"], specs(sample)):
                prompt = render(memory, texts, statement)
                prompt_ids = tokenizer_object.encode(prompt, add_special_tokens=False)
                encoded = {}
                for candidate in candidates:
                    combined = tokenizer_object.encode(prompt + candidate, add_special_tokens=False)
                    if combined[:len(prompt_ids)] != prompt_ids or len(combined) == len(prompt_ids):
                        prompt_score_ok = False
                    encoded[candidate] = combined[len(prompt_ids):]
                prompt_score_ok &= (
                    query["kind"] == kind and query["statement"] == statement
                    and query["answer"] == answer and query["candidates"] == candidates
                    and query["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
                    and query["prompt_tokens"] == len(prompt_ids)
                    and query["answer_token_lengths"] == {key: len(value) for key, value in encoded.items()}
                    and query["total_tokens_max"] == len(prompt_ids) + max(map(len, encoded.values()))
                    and query["total_tokens_max"] <= 40960
                    and set(query["scores"]) == set(candidates)
                    and all(math.isfinite(value) for value in query["scores"].values())
                    and query["prediction"] == min(candidates,
                        key=lambda candidate: (query["scores"][candidate], candidate))
                    and query["correct"] == int(query["prediction"] == answer)
                    and query["answer_nll"] == query["scores"][answer])
                generation = query.get("generation", {})
                token_ids = generation.get("token_ids", [])
                raw = tokenizer_object.decode(token_ids, skip_special_tokens=True)
                text = raw.replace("\r", "\n").split("\n", 1)[0].strip()
                generation_ok &= (
                    generation.get("raw_text") == raw and generation.get("text") == text
                    and generation.get("tokens") == len(token_ids)
                    and generation.get("max_new_tokens") == min(64, max(map(len, encoded.values())) + 8)
                    and len(token_ids) <= generation.get("max_new_tokens", -1)
                    and generation.get("stop_reason") in {"eos", "newline", "max_new_tokens"}
                    and generation.get("exact") == int(text == answer)
                    and generation.get("normalized_exact") == int(normalize(text) == normalize(answer)))
                profile = query.get("resources", {})
                resource_ok &= (profile.get("seconds", 0) > 0
                    and profile.get("candidate_scores") == len(candidates)
                    and profile.get("gpu_peak_allocated_gib", -1) >= profile.get("gpu_start_allocated_gib", 0) >= 0
                    and profile.get("gpu_peak_reserved_gib", -1) >= profile.get("gpu_start_reserved_gib", 0) >= 0
                    and all(math.isfinite(profile.get(name, math.nan)) for name in
                        ("seconds", "gpu_start_allocated_gib", "gpu_start_reserved_gib",
                         "gpu_peak_allocated_gib", "gpu_peak_reserved_gib")))

    expected_order = {shard: [sample["episode_id"] for index, sample in enumerate(samples)
                              if index % 2 == shard] for shard in (0, 1)}
    coverage_ok = (len(rows) == len({row["episode_id"] for row in rows}) == 272
                   and {row["episode_id"] for row in rows} == set(sample_index)
                   and ordered_shards == expected_order)
    state_manifest, state_audit = read_json(state_root / "manifest.json"), read_json(state_root.parent / "audit.json")
    dependencies = {"state_manifest_sha256": sha(state_root / "manifest.json"),
        "state_audit_sha256": sha(state_root.parent / "audit.json"),
        "formal_states_sha256": sha(state_root / "formal_states.jsonl"),
        "formal_sample_sha256": sha(sample_root / "formal.jsonl")}
    shard_ok = (len(audits) == 2 and {value["shard"] for value in audits} == {0, 1}
        and sum(value["unique_memory_states_scored"] for value in audits) == expected_unique
        and all(value.get("protocol") == PROTOCOL and value.get("num_shards") == 2
            and value.get("rows") == 136 and value.get("rows_sha256") == hashes[Path(path).name]
            and value.get("dependencies") == dependencies
            and value.get("query_measurements") == value.get("unique_memory_states_scored") * 4
            and value.get("generation") == "greedy_from_independent_shared_prompt_cache"
            and not value.get("external_sets_loaded") for value, path in zip(audits, inputs)))
    model_root, tokenizer_root = Path(model), Path(tokenizer)
    model_hashes = {path.name: sha(path) for path in sorted(model_root.glob("*.safetensors"))}
    tokenizer_hashes = {path.name: sha(path) for path in sorted(tokenizer_root.iterdir())
        if path.is_file() and path.suffix in {".json", ".model", ".txt", ".tiktoken"}}
    provenance_ok = bool(model_hashes) and bool(tokenizer_hashes) and all(
        value.get("model") == str(model_root) and value.get("tokenizer") == str(tokenizer_root)
        and value.get("model_provenance", {}).get("path") == str(model_root)
        and value.get("model_provenance", {}).get("config_sha256") == sha(model_root / "config.json")
        and value.get("model_provenance", {}).get("source_files_sha256") == model_hashes
        and value.get("tokenizer_files_sha256") == tokenizer_hashes
        and value.get("readout_mode") == "plain_ttt_disabled_fresh_kv"
        and value.get("candidate_scoring") == "one_prompt_forward_with_independent_answer_cache_branches"
        and value.get("model_provenance", {}).get("ttt_mode") is False
        and bool(value.get("model_provenance", {}).get("excluded_adapters"))
        and all(not value.get("loading_info", {}).get(name)
                for name in ("missing_keys", "mismatched_keys", "error_msgs"))
        and set(value.get("loading_info", {}).get("unexpected_keys", ()))
            == set(value.get("model_provenance", {}).get("excluded_adapters", ()))
        for value in audits)
    independent = recompute(rows)
    checks = {
        "synthetic_generation_bridge_passed_and_hashed":
            bridge.get("protocol") == "mquake_qwen_outcome_v16_generation_bridge"
            and bridge.get("passed") is True and bridge.get("synthetic_only") is True
            and bridge.get("experiment_rows_loaded") == 0
            and bridge.get("external_sets_loaded") is False
            and summary.get("generation_bridge_sha256") == sha(generation_bridge),
        "frozen_state_terminal_audit_and_hashes": state_manifest.get("terminal_status")
            == "frozen_after_passed_v16_pilot_before_formal_scoring"
            and state_audit.get("audit_passed") is True
            and summary.get("formal_states_dependencies") == dependencies,
        "complete_ordered_disjoint_shards": coverage_ok,
        "sample_state_episode_identities": identities_ok,
        "causal_memory_and_first_identical_state_deduplication": dedup_ok,
        "prompt_token_candidate_score_and_prediction_reconstruction": prompt_score_ok,
        "greedy_generation_token_decode_and_exact_match_reconstruction": generation_ok,
        "per_query_latency_and_peak_memory": resource_ok,
        "shard_hash_resource_and_unique_score_accounting": shard_ok,
        "exact_model_tokenizer_and_loading_provenance": provenance_ok,
        "independent_policy_fold_bootstrap_and_gate_recomputation": all(
            summary.get(key) == independent[key] for key in
            ("policies", "fixed_selection", "primary_comparison", "outcome_gate")),
        "terminal_and_external_locks": summary.get("protocol") == PROTOCOL
            and summary.get("rows") == 272 and summary.get("raw_rows_sha256") == hashes
            and summary.get("resources") == audits and summary.get("bootstrap_draws") == DRAWS
            and summary.get("terminal_status") in {"formal_outcome_passed", "formal_outcome_failed"}
            and summary.get("external_sets_loaded") is False,
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
        "checks": checks, "summary_sha256": sha(summary_path), "raw_rows_sha256": hashes,
        "recomputed_outcome_gate": independent["outcome_gate"],
        "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V16 formal outcome audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "sample-root", "state-root", "model", "tokenizer",
                 "summary-path", "generation-bridge", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
