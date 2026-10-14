"""Score the frozen V18 controlled-interference pilot with Qwen3-4B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_controlled_interference_v18 import FIXED_POLICIES, QUERY_KINDS
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    encode_candidates, greedy_from_prefix, load_jsonl, normalize_generation,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "controlled_memory_interference_v18_qwen_pilot"
DATA_PROTOCOL = "controlled_memory_interference_v18_pilot"
UPDATE_KINDS = frozenset(("update_old_surface", "update_current_surface"))
RETAIN_KINDS = frozenset(("retain_1", "retain_2"))
JOINT_KINDS = UPDATE_KINDS | RETAIN_KINDS
HEADER = (
    "The following chronological session notes are available memory. Some notes may "
    "conflict or repeat. Answer the query with only the requested code."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def data_audit_path(data_root):
    data_root = Path(data_root)
    candidates = (data_root / "audit.json", data_root.parent / "audit.json")
    paths = [path for path in candidates if path.exists()]
    if len(paths) != 1:
        raise ValueError("V18 data audit location is missing or ambiguous")
    return paths[0]


def verify_data(data_root):
    data_root = Path(data_root)
    manifest = read_json(data_root / "manifest.json")
    audit_path = data_audit_path(data_root)
    audit = read_json(audit_path)
    checks = (
        manifest.get("protocol") == DATA_PROTOCOL,
        manifest.get("terminal_status") == "frozen_v18_pilot_before_model_scoring",
        manifest.get("fixed_policies") == list(FIXED_POLICIES),
        manifest.get("query_kinds") == list(QUERY_KINDS),
        manifest.get("counts", {}).get("rows") == 40,
        manifest.get("pilot_sha256") == sha(data_root / "pilot.jsonl"),
        manifest.get("pilot_states_sha256") == sha(data_root / "pilot_states.jsonl"),
        manifest.get("model_scored") is False,
        manifest.get("formal_generated") is False,
        manifest.get("external_sets_loaded") is False,
        audit.get("audit_passed") is True,
        audit.get("manifest_sha256") == sha(data_root / "manifest.json"),
        audit.get("pilot_sha256") == sha(data_root / "pilot.jsonl"),
        audit.get("pilot_states_sha256") == sha(data_root / "pilot_states.jsonl"),
    )
    if not all(checks):
        raise ValueError("V18 Qwen pilot requires frozen and independently audited data")
    return {"manifest_sha256": sha(data_root / "manifest.json"),
            "data_audit_sha256": sha(audit_path),
            "pilot_sha256": sha(data_root / "pilot.jsonl"),
            "pilot_states_sha256": sha(data_root / "pilot_states.jsonl")}


def render_prompt(memory, statement):
    chunks = [HEADER, "", "Session notes:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.extend((f"[Session {index + 1}]", session["statement"]))
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def memory_from_ids(episode, identifiers):
    index = {session["session_id"]: session for session in episode["sessions"]}
    if (len(index) != len(episode["sessions"]) or len(identifiers) != len(set(identifiers))
            or any(identifier not in index for identifier in identifiers)):
        raise ValueError("V18 state contains duplicate or unknown sessions")
    result = [index[identifier] for identifier in identifiers]
    if [session["session_index"] for session in result] != sorted(
            session["session_index"] for session in result):
        raise ValueError("V18 memory state is not chronological")
    return result


def state_arms(state):
    arms = [{"arm_id": f"fixed:{policy}", "kind": "fixed", "policy": policy,
             "memory_session_ids": state["fixed"][policy]} for policy in FIXED_POLICIES]
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
    if len({arm["arm_id"] for arm in arms}) != len(arms):
        raise ValueError("duplicate V18 arm identity")
    return arms


def score_query(model, tokenizer, memory, spec):
    import torch
    from cbf_ttt.runtime import _nll_from_hidden
    from tasks.cbf_checkpoint_readout import PlainSession

    prompt = render_prompt(memory, spec["statement"])
    prompt_ids, encoded = encode_candidates(tokenizer, prompt, spec["candidates"])
    if any(len(answer_ids) != 5 for answer_ids in encoded.values()):
        raise ValueError("V18 token-matched answer invariant failed in prompt context")
    if any(len(prompt_ids) + len(answer_ids) > model.config.max_position_embeddings
           for answer_ids in encoded.values()):
        raise ValueError("V18 prompt exceeds checkpoint context")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
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
            raise RuntimeError("non-finite V18 answer NLL")
        scores[answer] = value
        del branch, tail, hidden, labels
    generation = greedy_from_prefix(model, tokenizer, prefix_session, prefix_hidden, 13)
    generation.update({
        "exact": int(generation["text"] == spec["answer"]),
        "normalized_exact": int(normalize_generation(generation["text"])
                                == normalize_generation(spec["answer"])),
        "max_new_tokens": 13,
    })
    if prefix_session.cache.get_seq_length() != prefix_cache_length:
        raise RuntimeError("V18 candidate/generation branch mutated shared prompt cache")
    torch.cuda.synchronize()
    resources = {
        "seconds": time.perf_counter() - started, "candidate_scores": len(encoded),
        "gpu_start_allocated_gib": start_allocated / 2**30,
        "gpu_start_reserved_gib": start_reserved / 2**30,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    prediction = min(spec["candidates"], key=lambda value: (scores[value], value))
    return {
        "kind": spec["kind"], "statement": spec["statement"], "answer": spec["answer"],
        "candidates": spec["candidates"], "scores": scores,
        "answer_token_lengths": {answer: len(ids) for answer, ids in encoded.items()},
        "prediction": prediction, "correct": int(prediction == spec["answer"]),
        "answer_nll": scores[spec["answer"]], "generation": generation,
        "prompt_tokens": len(prompt_ids), "total_tokens_max": len(prompt_ids) + 5,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "resources": resources,
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    data_root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    dependencies = verify_data(data_root)
    episodes, states = load_jsonl(data_root / "pilot.jsonl"), load_jsonl(
        data_root / "pilot_states.jsonl")
    if [row["episode_id"] for row in episodes] != [row["episode_id"] for row in states]:
        raise ValueError("V18 data/state order mismatch")
    selected = [(episode, state) for index, (episode, state) in enumerate(zip(episodes, states))
                if index % args.num_shards == args.shard]
    torch.manual_seed(1803); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V18 requires final slow weights with native TTT disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
                       if path.is_file() and path.suffix in {
                           ".json", ".model", ".txt", ".tiktoken"}}
    if not tokenizer_files:
        raise ValueError("no tokenizer provenance files")
    values = {value for episode in episodes for value in
              episode["initial_values"] + [episode["primary_current_value"],
                                             episode["locality_value"]]}
    if len(values) != 320 or {len(tokenizer.encode(value, add_special_tokens=False))
                              for value in values} != {5}:
        raise ValueError("V18 tokenizer-wide five-token invariant failed")

    started = time.perf_counter(); records = []; unique_states = reused = 0
    with output.open("x") as sink:
        for index, (episode, state) in enumerate(selected):
            scored, arms = {}, []
            for arm in state_arms(state):
                key = tuple(arm["memory_session_ids"])
                if key not in scored:
                    memory = memory_from_ids(episode, arm["memory_session_ids"])
                    queries = [score_query(model, tokenizer, memory, spec)
                               for spec in episode["queries"]]
                    scored[key] = {"source_arm_id": arm["arm_id"], "queries": queries}
                    unique_states += 1
                else:
                    reused += 1
                result = copy.deepcopy(arm)
                result["score_source_arm_id"] = scored[key]["source_arm_id"]
                result["queries"] = copy.deepcopy(scored[key]["queries"])
                arms.append(result)
            row = {key: episode[key] for key in (
                "episode_id", "row_index", "subject", "length", "mechanism",
                "target_key_id", "current_session_id")}
            row["arms"] = arms
            sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(selected),
                              "episode_id": episode["episode_id"],
                              "unique_states": len(scored)}), flush=True)
    queries = [query for row in records for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    audit = {
        "protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(records), "rows_sha256": sha(output), "dependencies": dependencies,
        "seconds": time.perf_counter() - started, "unique_memory_states_scored": unique_states,
        "reused_arm_states": reused, "query_measurements": len(queries),
        "gpu_peak_allocated_gib": max(q["resources"]["gpu_peak_allocated_gib"] for q in queries),
        "gpu_peak_reserved_gib": max(q["resources"]["gpu_peak_reserved_gib"] for q in queries),
        "model": args.model, "tokenizer": args.tokenizer, "gpu": torch.cuda.get_device_name(),
        "loading_info": model_meta["loading_info"], "readout_mode": "plain_ttt_disabled_fresh_kv",
        "candidate_scoring": "one_prompt_forward_with_independent_answer_cache_branches",
        "generation": "greedy_from_independent_shared_prompt_cache_13_tokens",
        "model_provenance": {key: model_meta[key] for key in (
            "path", "class", "config_sha256", "attention", "parameter_count",
            "source_files_sha256", "excluded_adapters")}
            | {"ttt_mode": model_meta["config"]["ttt_mode"]},
        "tokenizer_files_sha256": tokenizer_files, "opaque_answer_tokens": 5,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "external_sets_loaded": False, "formal_generated": False,
    }
    write_json(str(output) + ".audit.json", audit); return audit


def arm_index(row):
    result = {arm["arm_id"]: arm for arm in row["arms"]}
    if len(result) != len(row["arms"]):
        raise ValueError("duplicate scored V18 arm")
    return result


def qmap(arm):
    result = {query["kind"]: query for query in arm["queries"]}
    if set(result) != set(QUERY_KINDS):
        raise ValueError("incomplete V18 query kinds")
    return result


def arm_metrics(arm):
    queries = qmap(arm)
    update = statistics.fmean(queries[kind]["correct"] for kind in UPDATE_KINDS)
    retain = statistics.fmean(queries[kind]["correct"] for kind in RETAIN_KINDS)
    joint = int(all(queries[kind]["correct"] for kind in JOINT_KINDS))
    nll = statistics.fmean(queries[kind]["answer_nll"] for kind in JOINT_KINDS)
    generation_joint = int(all(queries[kind]["generation"]["normalized_exact"]
                               for kind in JOINT_KINDS))
    return {"joint_success": joint, "update_accuracy": update,
            "retained_key_accuracy": retain, "joint_answer_nll": nll,
            "locality_accuracy": queries["locality"]["correct"],
            "generation_joint_success": generation_joint}


def select_oracle(row):
    candidates = [arm_index(row)["fixed:retain_all"]] + [
        arm for arm in row["arms"] if arm["kind"] in {"delete_one", "delete_key"}]
    ranked = []
    for tie_index, arm in enumerate(candidates):
        metric = arm_metrics(arm)
        ranked.append((metric["joint_success"], metric["update_accuracy"],
                       metric["retained_key_accuracy"], -metric["joint_answer_nll"],
                       -tie_index, arm, metric))
    winner = max(ranked, key=lambda value: value[:5])
    return winner[5], winner[6]


def select_best_fixed(rows):
    table = {}
    for policy in FIXED_POLICIES:
        metrics = [arm_metrics(arm_index(row)[f"fixed:{policy}"]) for row in rows]
        table[policy] = {key: statistics.fmean(metric[key] for metric in metrics)
                         for key in metrics[0]}
    winner = max(FIXED_POLICIES, key=lambda policy: (
        table[policy]["joint_success"], table[policy]["update_accuracy"],
        table[policy]["retained_key_accuracy"], -table[policy]["joint_answer_nll"],
        -FIXED_POLICIES.index(policy)))
    return winner, table


def aggregate(rows):
    fixed_policy, fixed_table = select_best_fixed(rows)
    oracle_metrics, fixed_metrics, selections = [], [], []
    by_mechanism, by_length = {}, {}
    for row in rows:
        oracle, metric = select_oracle(row)
        fixed_metric = arm_metrics(arm_index(row)[f"fixed:{fixed_policy}"])
        oracle_metrics.append(metric); fixed_metrics.append(fixed_metric)
        selections.append({"episode_id": row["episode_id"], "oracle_arm_id": oracle["arm_id"],
                           "oracle_kind": oracle["kind"], "metrics": metric})
    for name, field in ((name, "mechanism") for name in sorted({r["mechanism"] for r in rows})):
        indices = [index for index, row in enumerate(rows) if row[field] == name]
        by_mechanism[name] = {
            "rows": len(indices), "oracle_joint_success": statistics.fmean(
                oracle_metrics[index]["joint_success"] for index in indices),
            "fixed_joint_success": statistics.fmean(
                fixed_metrics[index]["joint_success"] for index in indices)}
    for length in sorted({row["length"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["length"] == length]
        by_length[str(length)] = {
            "rows": len(indices), "oracle_joint_success": statistics.fmean(
                oracle_metrics[index]["joint_success"] for index in indices),
            "fixed_joint_success": statistics.fmean(
                fixed_metrics[index]["joint_success"] for index in indices)}
    oracle = {key: statistics.fmean(metric[key] for metric in oracle_metrics)
              for key in oracle_metrics[0]}
    fixed = {key: statistics.fmean(metric[key] for metric in fixed_metrics)
             for key in fixed_metrics[0]}
    checks = {
        "no_memory_joint_success_at_most_70": fixed_table["no_memory"]["joint_success"] <= .70,
        "retain_all_joint_success_between_20_and_90":
            .20 <= fixed_table["retain_all"]["joint_success"] <= .90,
        "global_reset_retained_key_accuracy_below_80":
            fixed_table["global_reset"]["retained_key_accuracy"] < .80,
        "oracle_update_accuracy_at_least_90": oracle["update_accuracy"] >= .90,
        "oracle_retained_key_accuracy_at_least_90": oracle["retained_key_accuracy"] >= .90,
        "oracle_joint_gain_over_best_fixed_at_least_10":
            oracle["joint_success"] - fixed["joint_success"] >= .10,
    }
    return {"fixed_arms": fixed_table, "best_fixed_policy": fixed_policy,
            "oracle": oracle, "best_fixed": fixed,
            "oracle_minus_fixed_joint_success": oracle["joint_success"] - fixed["joint_success"],
            "oracle_selection": selections, "by_mechanism": by_mechanism,
            "by_length": by_length,
            "corpus_admission_gate": {"checks": checks, "passed": all(checks.values())}}


def summarize(args):
    data_root = Path(args.data_root); dependencies = verify_data(data_root)
    rows, shard_audits, hashes = [], [], {}
    for value in args.inputs:
        path = Path(value); part = load_jsonl(path); resource = read_json(str(path) + ".audit.json")
        if (resource.get("protocol") != PROTOCOL or resource.get("rows") != len(part)
                or resource.get("rows_sha256") != sha(path)
                or resource.get("dependencies") != dependencies):
            raise ValueError("V18 shard audit mismatch")
        rows.extend(part); shard_audits.append(resource); hashes[path.name] = sha(path)
    expected = {row["episode_id"] for row in load_jsonl(data_root / "pilot.jsonl")}
    if len(rows) != 40 or len({row["episode_id"] for row in rows}) != 40 \
            or {row["episode_id"] for row in rows} != expected:
        raise ValueError("V18 pilot coverage mismatch")
    table = aggregate(rows)
    queries = [query for row in rows for arm in row["arms"]
               if arm["score_source_arm_id"] == arm["arm_id"] for query in arm["queries"]]
    latencies = sorted(query["resources"]["seconds"] for query in queries)
    result = {
        "protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
        "shard_resources": shard_audits, "dependencies": dependencies,
        "resources": {"unique_memory_states": sum(a["unique_memory_states_scored"]
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
                          q["resources"]["gpu_peak_reserved_gib"] for q in queries)},
        "model_scored": True, "formal_generated": False, "external_sets_loaded": False,
        "terminal_status": "pilot_corpus_admitted" if table["corpus_admission_gate"]["passed"]
            else "pilot_corpus_rejected",
    }
    write_json(args.output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
