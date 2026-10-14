"""Pilot frozen-Qwen scoring of causally retained MQuAKE textual memories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_mquake_temporal_adaptation_v8 import load_split, load_texts


PROTOCOL = "mquake_qwen_outcome_v16_pilot"
POLICIES = ("no_memory", "oracle")
QUERY_KINDS = ("primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated")
HEADER = (
    "The following are chronological session memories. When statements conflict, "
    "use the most recent statement. Answer the query with only the requested value."
)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def source_index(data_root):
    train, manifest = load_split(data_root, "train")
    tune, _ = load_split(data_root, "tune")
    return {row["episode_id"]: row for row in train + tune}, load_texts(data_root, manifest)


def visible_sessions(episode, sample):
    query_step = sample["query_step"]
    if not 0 <= query_step < len(episode["sessions"]):
        raise ValueError("invalid query step")
    visible = episode["sessions"][:query_step + 1]
    if visible[-1]["session_id"] != sample["primary_current_session_id"]:
        raise ValueError("query step is not the primary-current event")
    return visible


def policy_memory(episode, sample, policy):
    visible = visible_sessions(episode, sample)
    if policy == "no_memory":
        return []
    if policy == "oracle":
        old = sample["primary_old_session_id"]
        result = [row for row in visible if row["session_id"] != old]
        if any(row["session_id"] == old for row in result) \
                or not any(row["session_id"] == sample["primary_current_session_id"]
                           for row in result):
            raise RuntimeError("oracle memory state mismatch")
        return result
    raise ValueError("pilot exposes only no_memory and oracle")


def query_specs(sample):
    primary = [sample["old"]["value"], sample["current"]["value"]]
    safety = []
    for name in ("old", "current", "hard_retain", "unrelated"):
        value = sample[name]["value"]
        if value not in safety:
            safety.append(value)
    return (
        {"kind": "primary_old_phrase", "statement": sample["old"]["statement"],
         "answer": sample["current"]["value"], "candidates": primary},
        {"kind": "primary_current_phrase", "statement": sample["current"]["statement"],
         "answer": sample["current"]["value"], "candidates": primary},
        {"kind": "hard_retain", "statement": sample["hard_retain"]["statement"],
         "answer": sample["hard_retain"]["value"], "candidates": safety},
        {"kind": "unrelated", "statement": sample["unrelated"]["statement"],
         "answer": sample["unrelated"]["value"], "candidates": safety},
    )


def render_prompt(memory, texts, statement):
    chunks = [HEADER, "", "Session memories:"]
    if memory:
        for index, session in enumerate(memory):
            chunks.append(f"[Session {index + 1}]")
            chunks.extend(texts[identifier] for identifier in session["units"])
    else:
        chunks.append("(none)")
    chunks.extend(("", f"Query: {statement}", "Answer:", ""))
    return "\n".join(chunks)


def encode_pair(tokenizer, prompt, answer):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    combined = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if combined[:len(prefix)] != prefix or len(combined) == len(prefix):
        raise ValueError("answer tokenization does not preserve the prompt prefix")
    return prefix, combined[len(prefix):]


def score_query(model, tokenizer, memory, texts, spec):
    import torch
    from cbf_ttt.runtime import _nll_from_hidden
    from tasks.cbf_checkpoint_readout import PlainSession

    prompt = render_prompt(memory, texts, spec["statement"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    scores, answer_lengths = {}, {}
    for answer in spec["candidates"]:
        prefix, answer_ids = encode_pair(tokenizer, prompt, answer)
        if prefix != prompt_ids:
            raise RuntimeError("inconsistent prompt tokenization")
        total = len(prefix) + len(answer_ids)
        if total > model.config.max_position_embeddings:
            raise ValueError("prompt and answer exceed the checkpoint context limit")
        session = PlainSession(model)
        hidden = session._forward(prefix + answer_ids[:-1])
        labels = torch.tensor(answer_ids, dtype=torch.long, device=session.device)
        value = _nll_from_hidden(
            model, hidden[:, len(prefix) - 1:len(prefix) - 1 + len(answer_ids)], labels)
        if not math.isfinite(value):
            raise RuntimeError("non-finite Qwen answer NLL")
        scores[answer] = value; answer_lengths[answer] = len(answer_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    resources = {
        "seconds": elapsed,
        "candidate_scores": len(spec["candidates"]),
        "gpu_start_allocated_gib": start_allocated / 2**30,
        "gpu_start_reserved_gib": start_reserved / 2**30,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    prediction = min(spec["candidates"], key=lambda value: (scores[value], value))
    return {"kind": spec["kind"], "statement": spec["statement"],
            "answer": spec["answer"], "candidates": spec["candidates"],
            "scores": scores, "answer_token_lengths": answer_lengths,
            "prediction": prediction, "correct": int(prediction == spec["answer"]),
            "answer_nll": scores[spec["answer"]], "prompt_tokens": len(prompt_ids),
            "total_tokens_max": len(prompt_ids) + max(answer_lengths.values()),
            "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "resources": resources}


def verify_v15(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    audit = json.loads((root.parent / "audit.json").read_text()) \
        if (root.parent / "audit.json").exists() else json.loads((root / "audit.json").read_text())
    if summary.get("protocol") != "mquake_jackknife_consensus_v15" \
            or not summary.get("jackknife_consensus_supported") \
            or summary.get("terminal_status") != "completed_mquake_jackknife_consensus_v15" \
            or not audit.get("audit_passed"):
        raise ValueError("V16 pilot requires a completed, audited, supported V15")
    return {"summary_sha256": sha(root / "summary.json"),
            "audit_sha256": sha(root.parent / "audit.json") if (root.parent / "audit.json").exists()
                            else sha(root / "audit.json")}


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_readout import load_plain

    sample_root, output = Path(args.sample_root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    sample_manifest = json.loads((sample_root / "manifest.json").read_text())
    if sample_manifest["protocol"] != "mquake_qwen_outcome_v16" \
            or sample_manifest["model_scored"] or sample_manifest["external_sets_loaded"] \
            or sha(sample_root / "pilot.jsonl") != sample_manifest["pilot_sha256"]:
        raise ValueError("frozen V16 sample mismatch")
    v15_hashes = verify_v15(args.v15_root)
    samples = [row for index, row in enumerate(load_jsonl(sample_root / "pilot.jsonl"))
               if index % args.num_shards == args.shard]
    episodes, texts = source_index(args.data_root)
    torch.manual_seed(1601); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    model, model_meta = load_plain(args.model)
    if model.config.ttt_mode or not model_meta["excluded_adapters"]:
        raise ValueError("V16 requires the final slow weights with native TTT adapters disabled")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer_files = {
        path.name: sha(path) for path in sorted(Path(args.tokenizer).iterdir())
        if path.is_file() and path.suffix in {".json", ".model", ".txt", ".tiktoken"}
    }
    if not tokenizer_files:
        raise ValueError("no tokenizer provenance files")
    started = time.perf_counter(); records = []
    with output.open("x") as sink:
        for index, sample in enumerate(samples):
            episode = episodes[sample["episode_id"]]
            policies = {}
            for policy in POLICIES:
                memory = policy_memory(episode, sample, policy)
                policies[policy] = {
                    "memory_session_ids": [row["session_id"] for row in memory],
                    "queries": [score_query(model, tokenizer, memory, texts, spec)
                                for spec in query_specs(sample)],
                }
            row = {"episode_id": sample["episode_id"], "subject": sample["subject"],
                   "fold": sample["fold"], "length": sample["length"],
                   "hard_relation_matched": sample["hard_relation_matched"],
                   "query_step": sample["query_step"], "policies": policies}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n"); sink.flush(); records.append(row)
            print(json.dumps({"completed": index + 1, "total": len(samples),
                              "episode_id": sample["episode_id"]}), flush=True)
    queries = [query for row in records for policy in POLICIES
               for query in row["policies"][policy]["queries"]]
    audit = {"protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
             "rows": len(records), "rows_sha256": sha(output),
             "sample_manifest_sha256": sha(sample_root / "manifest.json"),
             "pilot_sha256": sample_manifest["pilot_sha256"], "v15_hashes": v15_hashes,
             "seconds": time.perf_counter() - started,
             "query_measurements": len(queries),
             "gpu_peak_allocated_gib": max(query["resources"]["gpu_peak_allocated_gib"]
                                            for query in queries),
             "gpu_peak_reserved_gib": max(query["resources"]["gpu_peak_reserved_gib"]
                                           for query in queries),
             "model": args.model, "tokenizer": args.tokenizer,
             "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
             "readout_mode": "plain_ttt_disabled_fresh_kv",
             "model_provenance": {key: model_meta[key] for key in (
                 "path", "class", "config_sha256", "attention", "parameter_count",
                 "source_files_sha256", "excluded_adapters")}
                 | {"ttt_mode": model_meta["config"]["ttt_mode"]},
             "tokenizer_files_sha256": tokenizer_files,
             "torch": torch.__version__, "transformers": __import__("transformers").__version__,
             "policies": list(POLICIES), "external_sets_loaded": False}
    write_json(str(output) + ".audit.json", audit); return audit


def aggregate(rows):
    output = {}
    for policy in POLICIES:
        queries = [query for row in rows for query in row["policies"][policy]["queries"]]
        latencies = sorted(row["resources"]["seconds"] for row in queries)
        output[policy] = {
            "queries": len(queries), "accuracy": statistics.fmean(row["correct"] for row in queries),
            "answer_nll": statistics.fmean(row["answer_nll"] for row in queries),
            "max_prompt_tokens": max(row["prompt_tokens"] for row in queries),
            "max_total_tokens": max(row["total_tokens_max"] for row in queries),
            "resources": {
                "mean_seconds_per_query": statistics.fmean(latencies),
                "median_seconds_per_query": statistics.median(latencies),
                "p95_seconds_per_query": latencies[math.ceil(.95 * len(latencies)) - 1],
                "max_seconds_per_query": max(latencies),
                "max_gpu_peak_allocated_gib": max(
                    row["resources"]["gpu_peak_allocated_gib"] for row in queries),
                "max_gpu_peak_reserved_gib": max(
                    row["resources"]["gpu_peak_reserved_gib"] for row in queries),
            },
            "by_kind": {kind: {"queries": sum(row["kind"] == kind for row in queries),
                                "accuracy": statistics.fmean(row["correct"] for row in queries
                                                             if row["kind"] == kind),
                                "answer_nll": statistics.fmean(row["answer_nll"] for row in queries
                                                               if row["kind"] == kind)}
                        for kind in QUERY_KINDS},
        }
    primary = ("primary_old_phrase", "primary_current_phrase")
    oracle_primary = statistics.fmean(
        query["correct"] for row in rows for query in row["policies"]["oracle"]["queries"]
        if query["kind"] in primary)
    checks = {"oracle_primary_at_least_80": oracle_primary >= .80,
              "all_rows_present": len(rows) == 40,
              "all_scores_finite": all(math.isfinite(query["answer_nll"])
                  for row in rows for policy in POLICIES for query in row["policies"][policy]["queries"]),
              "all_within_context": max(value["max_total_tokens"] for value in output.values()) <= 40960,
              "per_query_resources_complete": all(
                  query["resources"]["seconds"] > 0
                  and query["resources"]["candidate_scores"] == len(query["candidates"])
                  and query["resources"]["gpu_peak_allocated_gib"]
                      >= query["resources"]["gpu_start_allocated_gib"]
                  and query["resources"]["gpu_peak_reserved_gib"]
                      >= query["resources"]["gpu_start_reserved_gib"]
                  for row in rows for policy in POLICIES
                  for query in row["policies"][policy]["queries"])}
    return {"policies": output, "oracle_primary_accuracy": oracle_primary,
            "pilot_gate": {"checks": checks, "passed": all(checks.values())}}


def summarize(args):
    sample_root = Path(args.sample_root); rows = []
    hashes, resources = {}, []
    for path in map(Path, args.inputs):
        audit = json.loads(Path(str(path) + ".audit.json").read_text()); part = load_jsonl(path)
        if audit["rows"] != len(part) or audit["rows_sha256"] != sha(path) \
                or audit["pilot_sha256"] != json.loads((sample_root / "manifest.json").read_text())["pilot_sha256"]:
            raise ValueError("pilot shard audit mismatch")
        rows.extend(part); hashes[path.name] = sha(path); resources.append(audit)
    expected = [row["episode_id"] for row in load_jsonl(sample_root / "pilot.jsonl")]
    if sorted(row["episode_id"] for row in rows) != sorted(expected) \
            or len({row["episode_id"] for row in rows}) != 40:
        raise ValueError("pilot episode coverage mismatch")
    table = aggregate(rows)
    result = {"protocol": PROTOCOL, **table, "rows": len(rows), "raw_rows_sha256": hashes,
              "resources": resources, "formal_scored": False, "external_sets_loaded": False,
              "terminal_status": "pilot_passed" if table["pilot_gate"]["passed"] else "stopped_by_pilot"}
    write_json(args.output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--sample-root", required=True)
    collect_parser.add_argument("--data-root", required=True)
    collect_parser.add_argument("--v15-root", required=True)
    collect_parser.add_argument("--model", required=True); collect_parser.add_argument("--tokenizer", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--sample-root", required=True)
    summary_parser.add_argument("--inputs", nargs="+", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args) if args.command == "collect" else summarize(args), indent=2))


if __name__ == "__main__":
    main()
