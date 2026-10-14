"""Diagnose dense-reader interference with causal sparse component routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import episode_candidates
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_cbf_structured_forgetting import boundary_truth, load_split, parse_block
from tasks.train_qwen_cohort_forgetting import (
    GEOMETRY_HASHES,
    GRID,
    batch_delta,
    extract_embeddings,
    physics_retention,
    verify_inputs,
)


PROTOCOL = "qwen_sparse_reader_diagnostic_v1"
READERS = (
    "dense_all",
    "component_top1",
    "component_top2",
    "component_top4",
    "component_top8",
    "component_top16",
    "exact_key_upper",
)
ELIGIBLE = READERS[1:-1]
POLICIES = ("fixed_retain", "physics_cohort", "exact_reference", "oracle_grid")


def clone_state(state):
    return [
        {
            "batch": row["batch"],
            "facts": dict(row["facts"]),
            "delta": row["delta"],
            "weight": float(row["weight"]),
            "block_index": int(row["block_index"]),
        }
        for row in state
    ]


def apply_retention(state, retention, assignment, delta, block_index):
    if len(retention) != 3 or any(not 0 <= value <= 1 for value in retention):
        raise ValueError("cohort retention requires three values in [0,1]")
    output, size = clone_state(state), len(state)
    for index, row in enumerate(output):
        cohort = 2 if index == size - 1 else 1 if index == size - 2 else 0
        row["weight"] *= float(retention[cohort])
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


def apply_exact_reference(state, assignment, delta, block_index):
    output = clone_state(state)
    if assignment["prelude_lines"] and not assignment["old_refs"]:
        output = []
    elif assignment["old_refs"]:
        target = assignment["old_refs"][0]
        output = [row for row in output if row["batch"] != target]
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


def dense_output(state, query_key, dim):
    import torch

    output = torch.zeros(dim, dtype=torch.float32, device=query_key.device)
    for row in state:
        output += row["weight"] * (row["delta"] @ query_key)
    return output, sum(len(row["facts"]) for row in state)


def sparse_output(state, query, embeddings, reader):
    import torch

    query_key = embeddings.get("entity", query["key"])
    if reader == "dense_all":
        return dense_output(state, query_key, embeddings.dim)

    components = []
    for row_index, row in enumerate(state):
        if row["weight"] <= 1e-12:
            continue
        strength = 1.0 + 0.15 * row["block_index"]
        for fact_index, (entity, value) in enumerate(row["facts"].items()):
            key = embeddings.get("entity", entity)
            cosine = float(torch.dot(key, query_key))
            components.append(
                {
                    "cosine": cosine,
                    "row_index": row_index,
                    "fact_index": fact_index,
                    "entity": entity,
                    "value": value,
                    "coefficient": row["weight"] * strength * cosine,
                }
            )

    if reader == "exact_key_upper":
        selected = [row for row in components if row["entity"] == query["key"]]
    else:
        if not reader.startswith("component_top"):
            raise ValueError(f"unknown reader: {reader}")
        k = int(reader.removeprefix("component_top"))
        selected = sorted(
            components,
            key=lambda row: (-row["cosine"], -row["row_index"], row["fact_index"]),
        )[:k]

    output = torch.zeros(embeddings.dim, dtype=torch.float32, device=query_key.device)
    for row in selected:
        output += row["coefficient"] * embeddings.get("value", row["value"])
    return output, len(selected)


def query_metrics(state, query, candidates, embeddings, reader, unknown_raw=0.75,
                  temperature=0.2):
    import torch

    output, used = sparse_output(state, query, embeddings, reader)
    logits = torch.stack(
        [
            output.new_tensor(unknown_raw)
            if value == "unknown"
            else torch.dot(embeddings.get("value", value), output)
            for value in candidates
        ]
    ) / temperature
    target = candidates.index(query["answer"])
    prediction = int(logits.argmax())
    return {
        "nll": float(-torch.log_softmax(logits, -1)[target]),
        "correct": int(prediction == target),
        "prediction": candidates[prediction],
        "components_used": used,
    }


def outcome(state, retention, assignment, delta, block_index, truth, candidates,
            embeddings, reader):
    candidate = apply_retention(state, retention, assignment, delta, block_index)
    metrics = [
        query_metrics(candidate, row, candidates, embeddings, reader) for row in truth
    ]
    removed = 0.0
    size = len(state)
    for index, row in enumerate(state):
        cohort = 2 if index == size - 1 else 1 if index == size - 2 else 0
        removed += (1 - retention[cohort]) * row["weight"]
    return {
        "nll": statistics.fmean(row["nll"] for row in metrics),
        "accuracy": statistics.fmean(row["correct"] for row in metrics),
        "removed_mass": removed,
    }


def counterfactual_label(state, assignment, delta, block_index, truth, candidates,
                         embeddings, reader):
    values = {
        retention: outcome(
            state,
            retention,
            assignment,
            delta,
            block_index,
            truth,
            candidates,
            embeddings,
            reader,
        )
        for retention in GRID
    }
    best = min(row["nll"] for row in values.values())
    near = [retention for retention in GRID if values[retention]["nll"] <= best + 1e-6]
    return min(near, key=lambda value: (values[value]["removed_mass"], GRID.index(value)))


def evaluate_policy(pairs, policy, embeddings, reader):
    records = []
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates = episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                if policy == "fixed_retain":
                    retention, direct = (1.0, 1.0, 1.0), None
                elif policy == "physics_cohort":
                    retention, direct = physics_retention(state, assignment), None
                elif policy == "exact_reference":
                    retention = None
                    direct = apply_exact_reference(
                        state, assignment, delta, block["index"]
                    )
                elif policy == "oracle_grid":
                    retention = counterfactual_label(
                        state,
                        assignment,
                        delta,
                        block["index"],
                        truth,
                        candidates,
                        embeddings,
                        reader,
                    )
                    direct = None
                else:
                    raise ValueError(f"unknown policy: {policy}")
                state = (
                    direct
                    if direct is not None
                    else apply_retention(
                        state, retention, assignment, delta, block["index"]
                    )
                )
                decisions.append(
                    {
                        "block": block["index"],
                        "retention": None if retention is None else list(retention),
                        "retained_mass": sum(row["weight"] for row in state),
                    }
                )
                for row in truth:
                    queries.append(
                        {
                            **row,
                            **query_metrics(
                                state, row, candidates, embeddings, reader
                            ),
                        }
                    )
            else:
                state = apply_retention(
                    state, (1.0, 1.0, 1.0), assignment, delta, block["index"]
                )
        records.append(
            {
                "episode_id": inp["id"],
                "group_id": inp["group_id"],
                "family": target["family"],
                "reader": reader,
                "policy": policy,
                "queries": queries,
                "decisions": decisions,
            }
        )
    return records


def aggregate(records):
    queries = [row for episode in records for row in episode["queries"]]
    decisions = [row for episode in records for row in episode["decisions"]]
    groups, families, roles = {}, {}, {}
    for group in sorted({row["group_id"] for row in records}):
        values = [
            q
            for episode in records
            if episode["group_id"] == group
            for q in episode["queries"]
        ]
        groups[group] = {
            "accuracy": statistics.fmean(q["correct"] for q in values),
            "nll": statistics.fmean(q["nll"] for q in values),
        }
    for family in sorted({row["family"] for row in records}):
        values = [
            q
            for episode in records
            if episode["family"] == family
            for q in episode["queries"]
        ]
        families[family] = statistics.fmean(q["correct"] for q in values)
    for role in ("current", "historical", "anchor"):
        values = [row for row in queries if row["role"] == role]
        roles[role] = statistics.fmean(row["correct"] for row in values)
    return {
        "accuracy": statistics.fmean(row["correct"] for row in queries),
        "nll": statistics.fmean(row["nll"] for row in queries),
        "roles": roles,
        "families": families,
        "groups": groups,
        "mean_components_used": statistics.fmean(
            row["components_used"] for row in queries
        ),
        "mean_retained_mass": statistics.fmean(
            row["retained_mass"] for row in decisions
        ),
        "queries": len(queries),
        "decisions": len(decisions),
    }


def multi_seed_summary(per_seed):
    output = {}
    for reader in READERS:
        output[reader] = {}
        for policy in POLICIES:
            rows = [per_seed[str(seed)][reader][policy] for seed in SEEDS]
            output[reader][policy] = {
                "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
                "mean_nll": statistics.fmean(row["nll"] for row in rows),
                "min_seed_accuracy": min(row["accuracy"] for row in rows),
                "mean_components_used": statistics.fmean(
                    row["mean_components_used"] for row in rows
                ),
                "mean_retained_mass": statistics.fmean(
                    row["mean_retained_mass"] for row in rows
                ),
                "roles": {
                    role: statistics.fmean(row["roles"][role] for row in rows)
                    for role in rows[0]["roles"]
                },
                "families": {
                    family: statistics.fmean(
                        row["families"][family] for row in rows
                    )
                    for family in rows[0]["families"]
                },
                "per_seed": {
                    str(seed): per_seed[str(seed)][reader][policy] for seed in SEEDS
                },
            }
    return output


def select_reader(summary):
    dense = summary["dense_all"]["oracle_grid"]
    checks = {}
    passing = []
    for reader in ELIGIBLE:
        row = summary[reader]["oracle_grid"]
        reader_checks = {
            "accuracy_at_least_point_95": row["mean_accuracy"] >= 0.95,
            "current_at_least_point_95": row["roles"]["current"] >= 0.95,
            "anchor_at_least_point_95": row["roles"]["anchor"] >= 0.95,
            "historical_at_least_point_90": row["roles"]["historical"] >= 0.90,
            "min_family_at_least_point_85": min(row["families"].values()) >= 0.85,
            "every_seed_improves_dense": all(
                row["per_seed"][str(seed)]["accuracy"]
                > dense["per_seed"][str(seed)]["accuracy"]
                for seed in SEEDS
            ),
        }
        checks[reader] = reader_checks
        if all(reader_checks.values()):
            passing.append(reader)
    if passing:
        best_accuracy = max(
            summary[reader]["oracle_grid"]["mean_accuracy"] for reader in passing
        )
        near_best = [
            reader
            for reader in passing
            if best_accuracy
            - summary[reader]["oracle_grid"]["mean_accuracy"]
            < 0.005
        ]
        selected = near_best[0]
    else:
        selected = None
    upper = summary["exact_key_upper"]["oracle_grid"]
    upper_checks = {
        "accuracy_at_least_point_95": upper["mean_accuracy"] >= 0.95,
        "current_at_least_point_95": upper["roles"]["current"] >= 0.95,
        "anchor_at_least_point_95": upper["roles"]["anchor"] >= 0.95,
        "historical_at_least_point_90": upper["roles"]["historical"] >= 0.90,
        "min_family_at_least_point_85": min(upper["families"].values()) >= 0.85,
    }
    return {
        "checks": checks,
        "exact_key_upper_checks": upper_checks,
        "passed": selected is not None,
        "selected": selected,
    }


def run(args):
    import torch

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    data_manifest, config = verify_inputs(args.data_root, args.geometry_root)
    transform_artifact = torch.load(
        Path(args.geometry_root) / "contextual_transforms.pt",
        map_location="cpu",
        weights_only=False,
    )
    dev_pairs = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings, embedding_hashes, extraction, model_meta = extract_embeddings(
        dev_pairs,
        transform_artifact["transforms"],
        args.model,
        args.tokenizer,
        args.batch_size,
    )
    extraction_seconds = time.perf_counter() - started

    per_seed, row_hashes, timings = {}, {}, {}
    for seed in SEEDS:
        memory = embeddings[str(seed)]
        memory.values = {key: value.to("cuda") for key, value in memory.values.items()}
        per_seed[str(seed)] = {}
        for reader in READERS:
            per_seed[str(seed)][reader] = {}
            for policy in POLICIES:
                policy_started = time.perf_counter()
                records = evaluate_policy(dev_pairs, policy, memory, reader)
                timings[f"{seed}:{reader}:{policy}"] = (
                    time.perf_counter() - policy_started
                )
                per_seed[str(seed)][reader][policy] = aggregate(records)
                path = root / f"dev_seed{seed}_{reader}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
                row_hashes[path.name] = sha(path)
        del memory
        torch.cuda.empty_cache()

    dev = multi_seed_summary(per_seed)
    decision = select_reader(dev)
    training = {
        "seconds": time.perf_counter() - started,
        "extraction_seconds": extraction_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "embedding_hashes": embedding_hashes,
        "extraction": extraction,
        "timings": timings,
        "raw_rows_sha256": row_hashes,
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "geometry_hashes": GEOMETRY_HASHES,
        "contextual_config": config,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "readers": READERS,
        "eligible_readers": ELIGIBLE,
        "policies": POLICIES,
        "projection_seeds": SEEDS,
        "split_loaded": "dev",
        "test_loaded": False,
        "test_scored": False,
        "source_group_count": data_manifest["group_counts"]["dev"],
    }
    summary = {
        "protocol": PROTOCOL,
        "dev": dev,
        "decision": decision,
        "test_scored": False,
        "stage_passed": decision["passed"],
        "next_stage": (
            "freeze_fresh_sparse_reader_experiment"
            if decision["passed"]
            else "stop_sparse_reader_branch"
        ),
        "terminal_status": "completed_qwen_sparse_reader_diagnostic",
    }
    write_json(root / "training.json", training)
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "test_scored": result["test_scored"],
                "next_stage": result["next_stage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
