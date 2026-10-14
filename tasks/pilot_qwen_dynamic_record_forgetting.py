"""Resource pilot and counterfactual labels for dynamic record forgetting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_qwen_dynamic_record_forgetting import (
    DATA_SEED,
    FAMILIES,
    PROTOCOL as DATA_PROTOCOL,
    load_corpus,
    parse_visible_block,
)
from tasks.diagnose_qwen_sparse_reader import clone_state, query_metrics
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import episode_candidates
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_cbf_structured_forgetting import boundary_truth
from tasks.train_qwen_cohort_forgetting import (
    GEOMETRY_HASHES,
    batch_delta,
    extract_embeddings,
    move_embeddings,
)


PROTOCOL = "qwen_dynamic_record_resource_pilot_v1"
GRID = tuple(itertools.product((0.0, 0.5, 1.0), repeat=3))
ACTIONS = tuple((clear, retention) for clear in (0, 1) for retention in GRID)
READER = "component_top1"
FORMAL_CHOICES = (128, 256, 512)


def _assignment(parsed):
    return {
        "batch": parsed["record"],
        "facts": parsed["facts"],
        "old_refs": parsed["old_refs"],
        "reset": parsed["reset"],
        "prelude_lines": parsed["prelude_lines"],
        "text": parsed["text"],
    }


def candidate_indices(state, assignment, embeddings):
    """Return three stable record slots: reference, overlap, Qwen conflict."""
    slots = []
    for reference in assignment["old_refs"]:
        index = next(
            (offset for offset, row in enumerate(state) if row["batch"] == reference),
            None,
        )
        if index is not None and index not in slots:
            slots.append(index)
        if len(slots) == 3:
            return tuple(slots)

    current_keys = set(assignment["facts"])
    overlap = [
        index
        for index, row in enumerate(state)
        if index not in slots and current_keys.intersection(row["facts"])
    ]
    if overlap:
        slots.append(max(overlap))
    if len(slots) == 3:
        return tuple(slots)

    current_vectors = [embeddings.get("entity", key) for key in current_keys]
    scored = []
    for index, row in enumerate(state):
        if index in slots or row["weight"] <= 1e-12:
            continue
        score = max(
            float(__import__("torch").dot(embeddings.get("entity", key), current))
            for key in row["facts"]
            for current in current_vectors
        )
        scored.append((score, index))
    if scored:
        slots.append(max(scored, key=lambda item: (item[0], item[1]))[1])
    return tuple(slots)


def valid_actions(assignment, slots):
    output = []
    for action in ACTIONS:
        clear, retention = action
        if bool(clear) != bool(assignment["reset"]):
            continue
        if any(retention[index] != 1.0 for index in range(len(slots), 3)):
            continue
        output.append(action)
    if not output:
        raise ValueError("physical masks removed every dynamic action")
    return tuple(output)


def apply_dynamic_action(state, assignment, delta, block_index, slots, action):
    clear, retention = action
    output = clone_state(state)
    if clear:
        for row in output:
            row["weight"] = 0.0
    else:
        for slot, state_index in enumerate(slots):
            output[state_index]["weight"] *= float(retention[slot])
    output.append(
        {
            "batch": assignment["batch"],
            "facts": dict(assignment["facts"]),
            "delta": delta,
            "weight": 1.0,
            "block_index": int(block_index),
        }
    )
    return output


def score_action(state, assignment, delta, block_index, slots, action, truth,
                 candidates, embeddings):
    candidate = apply_dynamic_action(
        state, assignment, delta, block_index, slots, action
    )
    metrics = [
        query_metrics(candidate, row, candidates, embeddings, READER)
        for row in truth
    ]
    removed = 0.0
    if action[0]:
        removed = sum(row["weight"] for row in state)
    else:
        for slot, state_index in enumerate(slots):
            removed += (1.0 - action[1][slot]) * state[state_index]["weight"]
    return {
        "accuracy": statistics.fmean(row["correct"] for row in metrics),
        "nll": statistics.fmean(row["nll"] for row in metrics),
        "removed_mass": removed,
        "query_metrics": metrics,
    }


def lexicographic_label(outcomes, actions):
    best_accuracy = max(outcomes[action]["accuracy"] for action in actions)
    accurate = [
        action
        for action in actions
        if outcomes[action]["accuracy"] >= best_accuracy - 1e-12
    ]
    best_nll = min(outcomes[action]["nll"] for action in accurate)
    near = [
        action for action in accurate if outcomes[action]["nll"] <= best_nll + 1e-6
    ]
    return min(
        near,
        key=lambda action: (outcomes[action]["removed_mass"], ACTIONS.index(action)),
    )


def _token_stats(inp, block_index, truth, tokenizer):
    prefix = "\n".join(
        block["text"] for block in inp["blocks"] if block["index"] <= block_index
    )
    prompt_lengths = [
        len(tokenizer.encode(prefix + "\n" + query["text"], add_special_tokens=False))
        for query in inp["queries"]
        if query["after_block"] == block_index
    ]
    answer_lengths = [
        len(tokenizer.encode(" " + row["answer"], add_special_tokens=False))
        for row in truth
    ]
    return {
        "prompt_token_lengths": prompt_lengths,
        "answer_token_lengths": answer_lengths,
        "max_prompt_tokens": max(prompt_lengths),
        "max_answer_tokens": max(answer_lengths),
    }


def construct_labels(pairs, embeddings, tokenizer, device):
    import torch

    rows = []
    for inp, target in pairs:
        state = []
        candidates = episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            parsed = parse_visible_block(block["text"], inp["split"])
            assignment = _assignment(parsed)
            delta = batch_delta(assignment, block["index"], embeddings)
            slots = candidate_indices(state, assignment, embeddings)
            if block["index"] not in boundaries:
                state = apply_dynamic_action(
                    state,
                    assignment,
                    delta,
                    block["index"],
                    slots,
                    (0, (1.0, 1.0, 1.0)),
                )
                continue
            truth = boundary_truth(inp, target, block["index"])
            actions = valid_actions(assignment, slots)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            outcomes = {
                action: score_action(
                    state,
                    assignment,
                    delta,
                    block["index"],
                    slots,
                    action,
                    truth,
                    candidates,
                    embeddings,
                )
                for action in actions
            }
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            label = lexicographic_label(outcomes, actions)
            ordered = sorted(
                actions,
                key=lambda action: (
                    -outcomes[action]["accuracy"],
                    outcomes[action]["nll"],
                    outcomes[action]["removed_mass"],
                    ACTIONS.index(action),
                ),
            )
            runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
            token_stats = _token_stats(inp, block["index"], truth, tokenizer)
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "projection_seed": embeddings.seed,
                    "episode_id": inp["id"],
                    "group_id": inp["group_id"],
                    "family": target["family"],
                    "length": target["length"],
                    "block_index": block["index"],
                    "active_records": sum(row["weight"] > 1e-12 for row in state),
                    "addressable_records": len(state),
                    "candidate_slots": [
                        state[index]["batch"] for index in slots
                    ],
                    "candidate_slot_count": len(slots),
                    "visible_reference_count": len(assignment["old_refs"]),
                    "visible_reset": assignment["reset"],
                    "valid_action_count": len(actions),
                    "label": [label[0], list(label[1])],
                    "label_index": ACTIONS.index(label),
                    "label_accuracy": outcomes[label]["accuracy"],
                    "label_nll": outcomes[label]["nll"],
                    "label_removed_mass": outcomes[label]["removed_mass"],
                    "runner_up_accuracy_gap": (
                        outcomes[label]["accuracy"] - outcomes[runner_up]["accuracy"]
                    ),
                    "runner_up_nll_gap": outcomes[runner_up]["nll"] - outcomes[label]["nll"],
                    "label_seconds": seconds,
                    "seconds_per_action": seconds / len(actions),
                    "peak_allocated_gib": (
                        torch.cuda.max_memory_allocated() / 2**30
                        if device.startswith("cuda")
                        else 0.0
                    ),
                    "peak_reserved_gib": (
                        torch.cuda.max_memory_reserved() / 2**30
                        if device.startswith("cuda")
                        else 0.0
                    ),
                    "candidate_value_count": len(candidates),
                    **token_stats,
                }
            )
            state = apply_dynamic_action(
                state, assignment, delta, block["index"], slots, label
            )
    return rows


def summarize(rows, wall_seconds, extraction_peak_allocated_gib=0.0,
              extraction_peak_reserved_gib=0.0):
    if not rows:
        raise ValueError("empty dynamic-record resource rows")
    groups = len({row["group_id"] for row in rows})
    seeds = sorted({row["projection_seed"] for row in rows})
    estimates = {}
    for choice in FORMAL_CHOICES:
        # ``wall_seconds`` is the two-GPU critical path after shard merging.
        # Formal choices are total source groups and retain the same three
        # projections, so group-linear extrapolation is the registered rule.
        estimates[str(choice)] = wall_seconds * choice / groups
    label_peak_allocated = max(row["peak_allocated_gib"] for row in rows)
    label_peak_reserved = max(row["peak_reserved_gib"] for row in rows)
    overall_peak_allocated = max(label_peak_allocated, extraction_peak_allocated_gib)
    overall_peak_reserved = max(label_peak_reserved, extraction_peak_reserved_gib)
    eligible = [
        choice
        for choice in FORMAL_CHOICES
        if overall_peak_allocated < 24.0
        and estimates[str(choice)] <= 6 * 3600
    ]
    selected = max(eligible) if eligible else None
    return {
        "protocol": PROTOCOL,
        "rows": len(rows),
        "groups": groups,
        "projection_seeds": seeds,
        "families": {
            family: sum(row["family"] == family for row in rows)
            for family in FAMILIES
        },
        "wall_seconds": wall_seconds,
        "mean_label_seconds": statistics.fmean(row["label_seconds"] for row in rows),
        "p95_label_seconds": sorted(row["label_seconds"] for row in rows)[
            max(0, math.ceil(0.95 * len(rows)) - 1)
        ],
        "mean_seconds_per_action": statistics.fmean(
            row["seconds_per_action"] for row in rows
        ),
        "label_peak_allocated_gib": label_peak_allocated,
        "label_peak_reserved_gib": label_peak_reserved,
        "extraction_peak_allocated_gib": extraction_peak_allocated_gib,
        "extraction_peak_reserved_gib": extraction_peak_reserved_gib,
        "overall_peak_allocated_gib": overall_peak_allocated,
        "overall_peak_reserved_gib": overall_peak_reserved,
        "max_prompt_tokens": max(row["max_prompt_tokens"] for row in rows),
        "max_answer_tokens": max(row["max_answer_tokens"] for row in rows),
        "max_active_records": max(row["active_records"] for row in rows),
        "valid_action_counts": dict(Counter(row["valid_action_count"] for row in rows)),
        "label_counts": dict(Counter(str(row["label"]) for row in rows)),
        "estimated_two_gpu_seconds": estimates,
        "selected_formal_train_groups": selected,
        "resource_gate_passed": selected is not None,
    }


def verify_data(data_root):
    manifest, pairs = load_corpus(data_root)
    if (
        manifest.get("phase") != "pilot"
        or manifest.get("seed") != DATA_SEED
        or manifest.get("groups_per_family") != 8
        or manifest.get("families") != list(FAMILIES)
        or manifest.get("audit", {}).get("causal_answers_verified") is not True
        or manifest.get("audit", {}).get("oracle_actions_generated") is not False
    ):
        raise ValueError("unexpected dynamic-record pilot corpus")
    return manifest, pairs


def run(args):
    import torch
    from transformers import AutoTokenizer

    data_root, geometry_root, output = map(
        Path, (args.data_root, args.geometry_root, args.output)
    )
    if output.exists():
        raise FileExistsError(output)
    manifest, pairs = verify_data(data_root)
    observed_geometry = {name: sha(geometry_root / name) for name in GEOMETRY_HASHES}
    if observed_geometry != GEOMETRY_HASHES:
        raise ValueError("frozen contextual geometry dependency changed")
    requested = tuple(int(value) for value in args.projection_seeds.split(",") if value)
    if not requested or any(seed not in SEEDS for seed in requested):
        raise ValueError("projection seeds must be a nonempty subset of 2301,2302,2303")
    output.mkdir(parents=True)
    write_json(output / "progress.json", {"stage": "embedding_extraction"})
    transform_artifact = torch.load(
        geometry_root / "contextual_transforms.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings, hashes, extraction, model_meta = extract_embeddings(
        pairs,
        transform_artifact["transforms"],
        args.model,
        args.tokenizer,
        args.batch_size,
    )
    extraction_peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    all_rows, row_hashes = [], {}
    for seed in requested:
        write_json(output / "progress.json", {"stage": "labels", "seed": seed})
        memory = move_embeddings(embeddings[str(seed)], args.device)
        rows = construct_labels(pairs, memory, tokenizer, args.device)
        path = output / f"labels_seed{seed}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        row_hashes[path.name] = sha(path)
        all_rows.extend(rows)
        del memory
        torch.cuda.empty_cache()
    wall_seconds = time.perf_counter() - started
    summary = summarize(
        all_rows,
        wall_seconds,
        extraction_peak_allocated,
        extraction_peak_reserved,
    )
    write_json(output / "summary.json", summary)
    run_manifest = {
        "protocol": PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "geometry_hashes": observed_geometry,
        "projection_seeds": list(requested),
        "grid": [list(row) for row in GRID],
        "actions": [[row[0], list(row[1])] for row in ACTIONS],
        "reader": READER,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "device": args.device,
        "batch_size": args.batch_size,
        "embedding_hashes": {str(seed): hashes[str(seed)] for seed in requested},
        "embedding_extraction": extraction,
        "loading_info": model_meta["loading_info"],
        "raw_rows_sha256": row_hashes,
        "data_audit": manifest["audit"],
    }
    write_json(output / "manifest.json", run_manifest)
    write_json(output / "progress.json", {"stage": "completed", "seeds": list(requested)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-seeds", default="2301,2302,2303")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
