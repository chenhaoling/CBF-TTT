"""One-time zero-shot LongMemEval-S knowledge-update forgetting transfer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_zsre_forgetting_controller import lexical_features
from tasks.train_zsre_forgetting_controller_v2 import BinaryActionHead, relation_tokens, set_similarity


PROTOCOL = "longmemeval_forgetting_transfer_v1"
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SEEDS = (5101, 5102, 5103)
WINDOW_TOKENS = 1024
POLICIES = ("retain_all", "window8", "fixed_top", "random_matched",
            "frozen_controller", "oracle_pair")
METRICS = ("pair_state_correct", "latest_retained", "stale_deleted",
           "nonanswer_retention", "memory_fraction", "current_hit1", "current_hit5",
           "stale_hit1", "stale_hit5", "latest_over_stale")


def load_rows(path) -> list[dict]:
    path = Path(path)
    if sha(path) != DATA_SHA256:
        raise ValueError("LongMemEval-S hash mismatch")
    rows = [row for row in json.loads(path.read_text())
            if row.get("question_type") == "knowledge-update"]
    if len(rows) != 78 or len({row["question_id"] for row in rows}) != 78:
        raise ValueError("expected 78 unique knowledge-update questions")
    for row in rows:
        if len(row["answer_session_ids"]) != 2:
            raise ValueError("knowledge-update example must have two answer sessions")
        if len(row["haystack_sessions"]) != len(row["haystack_session_ids"]) \
                or len(row["haystack_sessions"]) != len(row["haystack_dates"]):
            raise ValueError("session fields are not aligned")
        positions = [row["haystack_session_ids"].index(value)
                     for value in row["answer_session_ids"]]
        if positions[0] == positions[1]:
            raise ValueError("answer sessions must be distinct")
    return sorted(rows, key=lambda row: row["question_id"])


def render_session(session: list[dict], date: str) -> str:
    lines = [f"Session date: {date}"]
    for turn in session:
        role, content = str(turn.get("role", "")).strip(), str(turn.get("content", "")).strip()
        # The cleaned source contains three role-labelled placeholder turns
        # with empty content. They carry no model-visible information.
        if not content:
            continue
        if not role:
            raise ValueError("nonempty LongMemEval turn has no role")
        lines.append(f"{role}: {content}")
    if len(lines) == 1:
        raise ValueError("LongMemEval session has no nonempty turns")
    return "\n".join(lines)


def token_windows(tokenizer, text: str, prefix: str = "Memory key: ") -> list[list[int]]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("empty tokenized text")
    width = WINDOW_TOKENS - len(prefix_ids)
    if width < 1:
        raise ValueError("prefix exceeds window")
    return [prefix_ids + ids[start:start + width] for start in range(0, len(ids), width)]


def embed_sequences(model, sequences: list[list[int]], batch_size: int):
    import torch

    pad = model.config.pad_token_id
    if pad is None:
        pad = model.config.eos_token_id
    output = []
    for offset in range(0, len(sequences), batch_size):
        rows = sequences[offset:offset + batch_size]; width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, device="cuda"); mask[index, :len(row)] = 1
        with torch.inference_mode():
            hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 return_dict=True).last_hidden_state.float()
            selected = hidden[torch.arange(len(rows), device="cuda"),
                              torch.tensor([len(row) - 1 for row in rows], device="cuda")]
        output.append(selected.cpu())
    return torch.cat(output)


def transform_embeddings(raw, transforms, regularizer):
    import torch

    result = {}
    for seed in SEEDS:
        item = transforms["seeds"][seed]["entity"]
        centered = raw - transforms["means"]["entity"]
        projected = centered @ item["projection"]
        inverse = torch.rsqrt(item["eigenvalues"] + regularizer * item["regularizer_scale"])
        values = (projected @ item["eigenvectors"] * inverse) @ item["eigenvectors"].T
        result[str(seed)] = values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return result


def cosine_triplet(tables, left: list[int], right: list[int]) -> tuple[float, float, float]:
    values = []
    for seed in SEEDS:
        table = tables[str(seed)]
        values.append(float((table[left] @ table[right].T).max()))
    return tuple(values)


def feature(cosines, key_text, query_text):
    import torch

    relation = set_similarity(relation_tokens(key_text), relation_tokens(query_text))
    return torch.tensor([*cosines, statistics.fmean(cosines), statistics.pstdev(cosines),
                         *lexical_features(key_text, query_text), *relation])


def top_candidate(memory, incoming, session_chunks, session_texts, tables):
    if not memory:
        return None, None, None
    triples = [cosine_triplet(tables, session_chunks[index], session_chunks[incoming])
               for index in memory]
    top_offset = max(range(len(memory)), key=lambda index: statistics.fmean(triples[index]))
    top = memory[top_offset]
    return top_offset, top, feature(triples[top_offset], session_texts[top], session_texts[incoming])


def controller_rollout(question_id, session_ids, session_chunks, session_texts, tables,
                       head, mean, std, threshold):
    import torch

    memory, decisions = [], []
    for step in range(len(session_ids)):
        delete_offset = None; probability = 0.0
        if memory:
            offset, _, value = top_candidate(memory, step, session_chunks, session_texts, tables)
            with torch.no_grad():
                probability = float(head(((value - mean) / std).unsqueeze(0)).sigmoid()[0])
            if probability >= threshold:
                delete_offset = offset; memory.pop(offset)
        decisions.append({"step": step, "deleted": delete_offset is not None,
                          "probability": probability})
        memory.append(step)
    return memory, decisions


def fixed_top_rollout(session_ids, session_chunks, session_texts, tables):
    memory = []
    for step in range(len(session_ids)):
        if memory:
            offset, _, _ = top_candidate(memory, step, session_chunks, session_texts, tables)
            memory.pop(offset)
        memory.append(step)
    return memory


def random_matched_rollout(question_id, count, decisions):
    memory = []
    for step in range(count):
        if memory and decisions[step]["deleted"]:
            digest = hashlib.sha256(f"{PROTOCOL}|{question_id}|{step}".encode()).digest()
            offset = int.from_bytes(digest[:8], "big") % len(memory); memory.pop(offset)
        memory.append(step)
    return memory


def oracle_rollout(session_ids, answer_ids):
    positions = sorted(session_ids.index(value) for value in answer_ids)
    stale, current = positions
    memory = []
    for step in range(len(session_ids)):
        if step == current and stale in memory:
            memory.remove(stale)
        memory.append(step)
    return memory


def score_memory(memory, session_ids, answer_ids, question_chunks, session_chunks, tables):
    positions = sorted(session_ids.index(value) for value in answer_ids)
    stale, current = positions
    retained = set(memory); nonanswer = set(range(len(session_ids))) - {stale, current}
    scores = {index: statistics.fmean(cosine_triplet(
        tables, question_chunks, session_chunks[index])) for index in memory}
    ranking = sorted(memory, key=lambda index: (-scores[index], index))
    current_rank = ranking.index(current) + 1 if current in retained else math.inf
    stale_rank = ranking.index(stale) + 1 if stale in retained else math.inf
    if current not in retained:
        latest_over_stale = 0.0
    elif stale not in retained:
        latest_over_stale = 1.0
    else:
        latest_over_stale = float(scores[current] > scores[stale])
    return {
        "pair_state_correct": float(current in retained and stale not in retained),
        "latest_retained": float(current in retained),
        "stale_deleted": float(stale not in retained),
        "nonanswer_retention": len(nonanswer & retained) / len(nonanswer),
        "memory_fraction": len(memory) / len(session_ids),
        "current_hit1": float(current_rank <= 1), "current_hit5": float(current_rank <= 5),
        "stale_hit1": float(stale_rank <= 1), "stale_hit5": float(stale_rank <= 5),
        "latest_over_stale": latest_over_stale,
        "memory_size": len(memory), "sessions": len(session_ids),
        "stale_position": stale, "current_position": current,
    }


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = [row for index, row in enumerate(load_rows(args.data))
            if index % args.num_shards == args.shard]
    source = Path(args.source_root); controller = Path(args.controller_root)
    final = json.loads(Path(args.final_summary).read_text())
    if final["protocol"] != "zsre_counterfactual_forgetting_final_test_v1" \
            or final["selected_threshold"] != .675 or not final["test_scored"]:
        raise ValueError("frozen final ZsRE threshold source mismatch")
    source_summary = json.loads((source / "summary.json").read_text())
    source_training = json.loads((source / "training.json").read_text())
    if source_training["transforms_sha256"] != sha(source / "transforms.pt"):
        raise ValueError("ZsRE contextual transform mismatch")
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)["transforms"]
    artifact_path = controller / "counterfactual_forgetting_action_oof_gate.pt"
    controller_training = json.loads((controller / "training.json").read_text())
    if controller_training["checkpoint_sha256"] != sha(artifact_path):
        raise ValueError("ZsRE controller checkpoint mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    head = BinaryActionHead(artifact["feature_dim"], artifact["training_seed"])
    head.load_state_dict(artifact["head"]); head.linear.eval()

    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    sequences, layouts, texts = [], [], []
    for row in rows:
        session_chunks, session_texts = [], []
        for session, date in zip(row["haystack_sessions"], row["haystack_dates"]):
            text = render_session(session, date); ids = token_windows(tokenizer, text)
            session_chunks.append(list(range(len(sequences), len(sequences) + len(ids))))
            session_texts.append(text); sequences.extend(ids)
        question_ids = token_windows(tokenizer, row["question"])
        question_chunks = list(range(len(sequences), len(sequences) + len(question_ids)))
        sequences.extend(question_ids); layouts.append((session_chunks, question_chunks)); texts.append(session_texts)
    raw = embed_sequences(model, sequences, args.batch_size)
    tables = transform_embeddings(raw, transforms, source_summary["selected_config"]["regularizer"])
    records = []
    for row, (session_chunks, question_chunks), session_texts in zip(rows, layouts, texts):
        ids = row["haystack_session_ids"]
        learned, decisions = controller_rollout(
            row["question_id"], ids, session_chunks, session_texts, tables,
            head, artifact["mean"], artifact["std"], .675)
        memories = {
            "retain_all": list(range(len(ids))),
            "window8": list(range(max(0, len(ids) - 8), len(ids))),
            "fixed_top": fixed_top_rollout(ids, session_chunks, session_texts, tables),
            "random_matched": random_matched_rollout(row["question_id"], len(ids), decisions),
            "frozen_controller": learned,
            "oracle_pair": oracle_rollout(ids, row["answer_session_ids"]),
        }
        policies = {name: score_memory(memory, ids, row["answer_session_ids"],
                                       question_chunks, session_chunks, tables)
                    for name, memory in memories.items()}
        records.append({"question_id": row["question_id"], "shard": args.shard,
                        "policy_metrics": policies,
                        "controller_deletions": sum(item["deleted"] for item in decisions),
                        "encoded_windows": sum(map(len, session_chunks)) + len(question_chunks)})
        print(json.dumps({"shard": args.shard, "completed": len(records), "total": len(rows)}), flush=True)
    with output.open("x") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")
    audit = {"protocol": PROTOCOL, "shard": args.shard, "num_shards": args.num_shards,
             "questions": len(records), "encoded_windows": len(sequences),
             "seconds": time.perf_counter() - started,
             "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
             "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
             "rows_sha256": sha(output), "data_sha256": DATA_SHA256,
             "source_transforms_sha256": sha(source / "transforms.pt"),
             "controller_checkpoint_sha256": sha(artifact_path),
             "model_loading_info": model_meta["loading_info"]}
    write_json(Path(str(output) + ".audit.json"), audit)


def mean_policy(records, policy):
    return {metric: statistics.fmean(row["policy_metrics"][policy][metric] for row in records)
            for metric in METRICS}


def bootstrap_difference(records, left, right, metric, seed, direction=1.0, draws=10000):
    rng = random.Random(seed); values = []
    n = len(records)
    for _ in range(draws):
        chosen = [records[rng.randrange(n)] for _ in range(n)]
        values.append(direction * statistics.fmean(
            row["policy_metrics"][left][metric] - row["policy_metrics"][right][metric]
            for row in chosen))
    values.sort()
    return [values[int(.025 * draws)], values[int(.975 * draws) - 1]]


def summarize(args):
    root = Path(args.root); all_rows = load_rows(args.data); records = []
    audits = []
    for shard in range(args.num_shards):
        path = root / f"shard_{shard}.jsonl"; audit_path = Path(str(path) + ".audit.json")
        audit = json.loads(audit_path.read_text())
        if audit["protocol"] != PROTOCOL or audit["rows_sha256"] != sha(path):
            raise ValueError("shard audit mismatch")
        audits.append(audit); records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    expected = {row["question_id"] for row in all_rows}
    if len(records) != 78 or {row["question_id"] for row in records} != expected:
        raise ValueError("incomplete or duplicate LongMemEval question set")
    records.sort(key=lambda row: row["question_id"])
    policies = {policy: mean_policy(records, policy) for policy in POLICIES}
    comparisons = {
        "pair_state_vs_random": {
            "gain": policies["frozen_controller"]["pair_state_correct"]
                    - policies["random_matched"]["pair_state_correct"],
            "bootstrap_95ci": bootstrap_difference(
                records, "frozen_controller", "random_matched", "pair_state_correct", 27101),
        },
        "current_hit5_vs_retain": {
            "gain": policies["frozen_controller"]["current_hit5"]
                    - policies["retain_all"]["current_hit5"],
            "bootstrap_95ci": bootstrap_difference(
                records, "frozen_controller", "retain_all", "current_hit5", 27102),
        },
        "stale_hit5_reduction_vs_retain": {
            "gain": policies["retain_all"]["stale_hit5"]
                    - policies["frozen_controller"]["stale_hit5"],
            "bootstrap_95ci": bootstrap_difference(
                records, "frozen_controller", "retain_all", "stale_hit5", 27103, direction=-1),
        },
    }
    checks = {
        "pair_state_beats_random": comparisons["pair_state_vs_random"]["bootstrap_95ci"][0] > 0,
        "current_hit5_noninferior": comparisons["current_hit5_vs_retain"]["bootstrap_95ci"][0] > -.02,
        "stale_hit5_reduced": comparisons["stale_hit5_reduction_vs_retain"]["bootstrap_95ci"][0] > 0,
        "latest_retention_at_least_95": policies["frozen_controller"]["latest_retained"] >= .95,
        "nonanswer_retention_at_least_80": policies["frozen_controller"]["nonanswer_retention"] >= .80,
    }
    summary = {"protocol": PROTOCOL, "data_status": "one_time_knowledge_update_transfer",
               "questions": len(records), "policies": policies, "comparisons": comparisons,
               "transfer_gate": {"checks": checks, "passed": all(checks.values())},
               "official_longmemeval_score": False,
               "locomo_scored": False,
               "next_stage": "preregister_official_generation" if all(checks.values())
                             else "stop_longmemeval_zero_shot_transfer",
               "terminal_status": "completed_longmemeval_forgetting_transfer_v1"}
    training = {"seconds_sum": sum(row["seconds"] for row in audits),
                "gpu_peak_allocated_gib": max(row["gpu_peak_allocated_gib"] for row in audits),
                "gpu_peak_reserved_gib": max(row["gpu_peak_reserved_gib"] for row in audits),
                "encoded_windows": sum(row["encoded_windows"] for row in audits),
                "shards": audits, "raw_rows_sha256": {f"shard_{i}.jsonl": audits[i]["rows_sha256"]
                                                       for i in range(args.num_shards)}}
    write_json(root / "summary.json", summary); write_json(root / "training.json", training)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("data", "source-root", "controller-root", "final-summary",
                 "model", "tokenizer", "output"):
        collect_parser.add_argument(f"--{name}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=4)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--data", required=True); summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.command == "collect": collect(args)
    else: summarize(args)


if __name__ == "__main__":
    main()
