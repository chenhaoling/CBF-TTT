"""Score the post-V19 natural semantic-conflict pilot or confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_post_v19_natural_semantic_headroom import (
    FIXED_POLICIES,
    FORMAL_PER_FOLD,
    FOLDS,
    PILOT_PER_FOLD,
    PROTOCOL as DATA_PROTOCOL,
    QUERY_KINDS,
)
from tasks.evaluate_controlled_interference_v18_formal import clustered_interval
from tasks.evaluate_controlled_interference_v18_pilot import (
    arm_index,
    arm_metrics,
    memory_from_ids,
    select_best_fixed,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    encode_candidates,
    greedy_from_prefix,
    load_jsonl,
    normalize_generation,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "post_v19_natural_semantic_headroom_qwen_v1"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2117
MAX_ANSWER_TOKENS = 10
MAX_NEW_TOKENS = 16
HEADER = (
    "The following chronological session notes are available memory. Some notes may "
    "conflict, repeat, or use different wording for the same relation. Answer the "
    "query with only the requested value."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def data_audit_path(data_root):
    candidates = (Path(data_root) / "audit.json", Path(data_root).parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("natural data audit location is missing or ambiguous")
    return paths[0]


def verify_data(data_root):
    data_root = Path(data_root)
    manifest_path, audit_path = data_root / "manifest.json", data_audit_path(data_root)
    manifest, audit = read_json(manifest_path), read_json(audit_path)
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("terminal_status")
            == "frozen_post_v19_natural_data_before_model_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("query_kinds") == list(QUERY_KINDS),
        manifest.get("splits", {}).get("pilot", {}).get("rows")
            == PILOT_PER_FOLD * FOLDS,
        manifest.get("splits", {}).get("formal", {}).get("rows")
            == FORMAL_PER_FOLD * FOLDS,
        manifest.get("model_scored") is False,
        manifest.get("controller_trained") is False,
        manifest.get("sealed_benchmarks_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(manifest_path),
    )
    for split in ("pilot", "formal"):
        checks += (
            manifest["files_sha256"][f"{split}.jsonl"]
                == sha(data_root / f"{split}.jsonl"),
            manifest["files_sha256"][f"{split}_states.jsonl"]
                == sha(data_root / f"{split}_states.jsonl"),
        )
    if not all(checks):
        raise ValueError("natural scoring requires frozen independently audited data")
    return {
        "data_manifest_sha256": sha(manifest_path),
        "data_audit_sha256": sha(audit_path),
        "pilot_sha256": sha(data_root / "pilot.jsonl"),
        "pilot_states_sha256": sha(data_root / "pilot_states.jsonl"),
        "formal_sha256": sha(data_root / "formal.jsonl"),
        "formal_states_sha256": sha(data_root / "formal_states.jsonl"),
    }


def verify_formal_release(pilot_release_root):
    if not pilot_release_root:
        raise ValueError("formal scoring requires an audited passing pilot release")
    root = Path(pilot_release_root)
    summary, audit = read_json(root / "summary.json"), read_json(root / "audit.json")
    checks = (
        summary.get("protocol") == PROTOCOL,
        summary.get("split") == "pilot",
        summary.get("pilot_gate", {}).get("passed") is True,
        summary.get("next_stage") == "score_sealed_natural_confirmation",
        summary.get("terminal_status") == "pilot_semantic_headroom_supported",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == sha(root / "summary.json"),
        (root / "runner_exit_code.txt").read_text().strip() == "0",
        (root / "terminal_status.txt").read_text().strip()
            == "completed_post_v19_natural_semantic_pilot",
    )
    if not all(checks):
        raise ValueError("formal natural scoring is locked by the pilot gate")
    return {
        "pilot_summary_sha256": sha(root / "summary.json"),
        "pilot_audit_sha256": sha(root / "audit.json"),
    }


def render_prompt(memory, statement):
    chunks = [HEADER, "", "Session notes:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.extend((f"[Session {index + 1}]", session["statement"]))
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def score_query(model, tokenizer, memory, spec):
    import torch
    from cbf_ttt.runtime import _nll_from_hidden
    from tasks.cbf_checkpoint_readout import PlainSession

    prompt = render_prompt(memory, spec["statement"])
    prompt_ids, encoded = encode_candidates(tokenizer, prompt, spec["candidates"])
    lengths = {answer: len(ids) for answer, ids in encoded.items()}
    if not lengths or min(lengths.values()) < 1 or max(lengths.values()) > MAX_ANSWER_TOKENS:
        raise ValueError("natural answer token length is outside the frozen 1-10 range")
    if any(len(prompt_ids) + len(answer_ids) > model.config.max_position_embeddings
           for answer_ids in encoded.values()):
        raise ValueError("natural prompt exceeds checkpoint context")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated, start_reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    started = time.perf_counter()
    prefix_session = PlainSession(model)
    prefix_hidden = prefix_session._forward(prompt_ids)
    prefix_cache_length = prefix_session.cache.get_seq_length()
    scores = {}
    for answer, answer_ids in encoded.items():
        branch = prefix_session.clone()
        tail = branch._forward(answer_ids[:-1])
        hidden = torch.cat((prefix_hidden[:, -1:], tail), dim=1)
        labels = torch.tensor(answer_ids, dtype=torch.long, device=prefix_session.device)
        value = _nll_from_hidden(model, hidden, labels)
        if not math.isfinite(value):
            raise RuntimeError("non-finite natural answer NLL")
        scores[answer] = value
        del branch, tail, hidden, labels
    generation = greedy_from_prefix(
        model, tokenizer, prefix_session, prefix_hidden, MAX_NEW_TOKENS)
    generation.update({
        "exact": int(generation["text"] == spec["answer"]),
        "normalized_exact": int(normalize_generation(generation["text"])
                                == normalize_generation(spec["answer"])),
        "max_new_tokens": MAX_NEW_TOKENS,
    })
    if prefix_session.cache.get_seq_length() != prefix_cache_length:
        raise RuntimeError("natural candidate branch mutated shared prompt cache")
    torch.cuda.synchronize()
    resources = {
        "seconds": time.perf_counter() - started,
        "candidate_scores": len(encoded),
        "gpu_start_allocated_gib": start_allocated / 2**30,
        "gpu_start_reserved_gib": start_reserved / 2**30,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    prediction = min(spec["candidates"], key=lambda value: (scores[value], value))
    return {
        "kind": spec["kind"],
        "statement": spec["statement"],
        "answer": spec["answer"],
        "candidates": spec["candidates"],
        "scores": scores,
        "answer_token_lengths": lengths,
        "prediction": prediction,
        "correct": int(prediction == spec["answer"]),
        "answer_nll": scores[spec["answer"]],
        "generation": generation,
        "prompt_tokens": len(prompt_ids),
        "total_tokens_max": len(prompt_ids) + max(lengths.values()),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "resources": resources,
    }


def state_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]}
            for policy in FIXED_POLICIES]
    for policy, removed in (("exact_conflict", "exact_removed"),
                            ("semantic_oracle", "semantic_removed")):
        arms.append({
            "arm_id": f"special:{policy}",
            "kind": "semantic_rule",
            "policy": policy,
            "removed_session_ids": state[removed],
            "memory_session_ids": state["special"][policy],
        })
    if len({row["arm_id"] for row in arms}) != len(arms):
        raise ValueError("duplicate natural arm identity")
    return arms


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_data(data_root)
    if args.split == "formal":
        dependencies.update(verify_formal_release(args.pilot_release_root))
    episodes = load_jsonl(data_root / f"{args.split}.jsonl")
    states = load_jsonl(data_root / f"{args.split}_states.jsonl")
    if [row["episode_id"] for row in episodes] != [row["episode_id"] for row in states]:
        raise ValueError("natural episode/state order mismatch")
    selected = [(episode, state) for index, (episode, state) in enumerate(
        zip(episodes, states)) if index % args.num_shards == args.shard]
    torch.manual_seed(2129)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("natural headroom requires final slow weights with TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    observed_lengths = {len(tokenizer.encode(value, add_special_tokens=False))
                        for episode in episodes for query in episode["queries"]
                        for value in query["candidates"]}
    if not observed_lengths or min(observed_lengths) < 1 or max(observed_lengths) > 10:
        raise ValueError("frozen natural data answer-length invariant failed")

    records, unique_states, reused = [], 0, 0
    started = time.perf_counter()
    with output.open("x") as sink:
        for index, (episode, state) in enumerate(selected):
            scored, arms = {}, []
            for arm in state_arms(state):
                key = tuple(arm["memory_session_ids"])
                if key not in scored:
                    memory = memory_from_ids(episode, arm["memory_session_ids"])
                    scored[key] = {
                        "source_arm_id": arm["arm_id"],
                        "queries": [score_query(model, tokenizer, memory, spec)
                                    for spec in episode["queries"]],
                    }
                    unique_states += 1
                else:
                    reused += 1
                result = copy.deepcopy(arm)
                result["score_source_arm_id"] = scored[key]["source_arm_id"]
                result["queries"] = copy.deepcopy(scored[key]["queries"])
                arms.append(result)
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "source", "source_split",
                "source_row_sha256", "subject", "relation_id", "fold", "length",
                "mechanism", "current_session_id")}
            row["arms"] = arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    queries = [query for row in records for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"]
               for query in arm["queries"]]
    audit = {
        "protocol": PROTOCOL,
        "split": args.split,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "rows": len(records),
        "rows_sha256": sha(output),
        "dependencies": dependencies,
        "seconds": time.perf_counter() - started,
        "unique_memory_states_scored": unique_states,
        "reused_arm_states": reused,
        "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"]
                                      for q in queries),
        "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"]
                                     for q in queries),
        "model": args.model,
        "tokenizer": args.tokenizer,
        "gpu": torch.cuda.get_device_name(),
        "loading_info": model_meta["loading_info"],
        "readout_mode": "plain_ttt_disabled_fresh_kv_contextual_evidence_memory",
        "candidate_scoring": "variable_1_to_10_token_normalized_answer_nll",
        "generation": f"greedy_from_independent_shared_prompt_cache_{MAX_NEW_TOKENS}_tokens",
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files,
        "answer_token_lengths": sorted(observed_lengths),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "model_scored": True,
        "controller_trained": False,
        "sealed_benchmarks_loaded": False,
    }
    write_json(str(output) + ".audit.json", audit)
    return audit


def aggregate(rows, split):
    expected = (PILOT_PER_FOLD if split == "pilot" else FORMAL_PER_FOLD) * FOLDS
    if (len(rows) != expected or len({row["episode_id"] for row in rows}) != expected
            or {row["fold"] for row in rows} != set(range(FOLDS))):
        raise ValueError("natural aggregation requires complete unique split coverage")
    fixed_by_fold = {str(fold): select_best_fixed(
        [row for row in rows if row["fold"] != fold])[0] for fold in range(FOLDS)}
    metric_keys = ("joint_success", "update_accuracy", "retained_key_accuracy",
                   "locality_accuracy", "generation_joint_success", "joint_answer_nll")
    records, exact_identity = [], True
    for row in rows:
        arms = arm_index(row)
        semantic = arm_metrics(arms["special:semantic_oracle"])
        exact = arm_metrics(arms["special:exact_conflict"])
        retain = arm_metrics(arms["fixed:retain_all"])
        fixed = arm_metrics(arms[f"fixed:{fixed_by_fold[str(row['fold'])]}"])
        exact_identity &= (arms["special:exact_conflict"]["queries"]
                           == arms["fixed:retain_all"]["queries"])
        records.append({
            "episode_id": row["episode_id"],
            "subject": row["subject"],
            "relation_id": row["relation_id"],
            "source": row["source"],
            "fold": row["fold"],
            "mechanism": row["mechanism"],
            "length": row["length"],
            "joint_gain": semantic["joint_success"] - exact["joint_success"],
            "generation_joint_gain": (semantic["generation_joint_success"]
                                      - exact["generation_joint_success"]),
            "semantic": semantic,
            "exact": exact,
            "retain_all": retain,
            "fixed": fixed,
        })
    mean = lambda arm, key: statistics.fmean(row[arm][key] for row in records)
    tables = {arm: {key: mean(arm, key) for key in metric_keys}
              for arm in ("semantic", "exact", "retain_all", "fixed")}
    subject = clustered_interval(records, "joint_gain", "subject",
                                 BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    relation = clustered_interval(records, "joint_gain", "relation_id",
                                  BOOTSTRAP_SEED + 1, BOOTSTRAP_DRAWS)
    by_fold = {str(fold): statistics.fmean(
        row["joint_gain"] for row in records if row["fold"] == fold)
               for fold in range(FOLDS)}
    by_source = {source: statistics.fmean(
        row["joint_gain"] for row in records if row["source"] == source)
                 for source in sorted({row["source"] for row in records})}
    semantic = tables["semantic"]
    if split == "pilot":
        checks = {
            "subject_macro_joint_gain_over_exact_at_least_5_points":
                subject["mean"] >= .05,
            "semantic_update_accuracy_at_least_85": semantic["update_accuracy"] >= .85,
            "semantic_retained_fact_accuracy_at_least_90":
                semantic["retained_key_accuracy"] >= .90,
            "semantic_locality_accuracy_at_least_97":
                semantic["locality_accuracy"] >= .97,
            "no_negative_fold": all(value >= 0 for value in by_fold.values()),
            "exact_surface_identical_to_retain_all": (
                exact_identity and tables["exact"] == tables["retain_all"]),
        }
    else:
        checks = {
            "subject_macro_joint_gain_over_exact_at_least_10_points":
                subject["mean"] >= .10,
            "subject_cluster_ci_lower_above_zero": subject["lower_95"] > 0,
            "relation_cluster_ci_lower_above_zero": relation["lower_95"] > 0,
            "semantic_update_accuracy_at_least_90": semantic["update_accuracy"] >= .90,
            "semantic_retained_fact_accuracy_at_least_90":
                semantic["retained_key_accuracy"] >= .90,
            "semantic_locality_accuracy_at_least_97":
                semantic["locality_accuracy"] >= .97,
            "no_negative_fold": all(value >= 0 for value in by_fold.values()),
            "no_negative_source": all(value >= 0 for value in by_source.values()),
            "exact_surface_identical_to_retain_all": (
                exact_identity and tables["exact"] == tables["retain_all"]),
        }
    return {
        "fixed_selected_by_outer_fold": fixed_by_fold,
        "arm_metrics": {
            "semantic_oracle": tables["semantic"],
            "exact_surface": tables["exact"],
            "retain_all": tables["retain_all"],
            "outer_fold_fixed": tables["fixed"],
        },
        "semantic_oracle_vs_exact_surface": {
            "joint_gain_micro": (tables["semantic"]["joint_success"]
                                 - tables["exact"]["joint_success"]),
            "joint_gain_subject_macro": subject,
            "joint_gain_relation_macro": relation,
            "by_fold_gain": by_fold,
            "by_source_gain": by_source,
        },
        f"{split}_gate": {"checks": checks, "passed": all(checks.values())},
    }


def summarize(args):
    data_root = Path(args.data_root)
    dependencies = verify_data(data_root)
    if args.split == "formal":
        dependencies.update(verify_formal_release(args.pilot_release_root))
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value)
        part, resource = load_jsonl(path), read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL
                or resource.get("split") != args.split
                or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("natural shard audit mismatch")
        rows.extend(part)
        shard_audits.append(resource)
        hashes[path.name] = sha(path)
    expected = {row["episode_id"] for row in load_jsonl(data_root / f"{args.split}.jsonl")}
    if len(rows) != len(expected) or {row["episode_id"] for row in rows} != expected:
        raise ValueError("natural scored split coverage mismatch")
    table = aggregate(rows, args.split)
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    passed = table[f"{args.split}_gate"]["passed"]
    if args.split == "pilot":
        terminal = "pilot_semantic_headroom_supported" if passed \
            else "pilot_semantic_headroom_not_supported"
        next_stage = "score_sealed_natural_confirmation" if passed \
            else "stop_post_v19_semantic_controller_branch"
    else:
        terminal = "formal_semantic_headroom_supported" if passed \
            else "formal_semantic_headroom_not_supported"
        next_stage = "train_natural_semantic_residual_controller" if passed \
            else "stop_post_v19_semantic_controller_branch"
    result = {
        "protocol": PROTOCOL,
        "split": args.split,
        **table,
        "rows": len(rows),
        "raw_rows_sha256": hashes,
        "shard_resources": shard_audits,
        "dependencies": dependencies,
        "resources": {
            "unique_memory_states": sum(a["unique_memory_states_scored"]
                                         for a in shard_audits),
            "query_measurements": len(queries),
            "wall_seconds_slower_shard": max(a["seconds"] for a in shard_audits),
            "mean_seconds_per_query": statistics.fmean(latencies),
            "median_seconds_per_query": statistics.median(latencies),
            "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "max_seconds_per_query": max(latencies),
            "max_gpu_peak_allocated_gib": max(
                q["resources"]["gpu_peak_allocated_gib"] for q in queries),
            "max_gpu_peak_reserved_gib": max(
                q["resources"]["gpu_peak_reserved_gib"] for q in queries),
        },
        "model_scored": True,
        "controller_trained": False,
        "sealed_benchmarks_loaded": False,
        "memory_substrate": "contextual_evidence_not_native_dense_ttt_writer",
        "next_stage": next_stage,
        "terminal_status": terminal,
    }
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--split", choices=("pilot", "formal"), required=True)
    collect_parser.add_argument("--pilot-release-root")
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--split", choices=("pilot", "formal"), required=True)
    summary_parser.add_argument("--pilot-release-root")
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else summarize(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
