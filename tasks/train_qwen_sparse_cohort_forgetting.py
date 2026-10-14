"""Train cohort forgetting with the audited top-1 contextual Qwen reader."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
    CONTROLLER_SEEDS,
    FEATURE_NAMES,
    FIXED,
    GEOMETRY_HASHES,
    GRID,
    POLICIES,
    aggregate,
    batch_delta,
    comparison,
    controller_retention,
    extract_embeddings,
    features,
    multi_seed_summary,
    physics_retention,
    restore_controllers,
    serialize_controller,
    train_controllers,
)
from tasks.diagnose_qwen_sparse_reader import (
    apply_exact_reference,
    apply_retention,
    counterfactual_label,
    query_metrics,
)


PROTOCOL = "qwen_sparse_cohort_forgetting_v1"
DATA_SEED = 202610152
GROUP_COUNTS = {"train": 256, "dev": 64, "test": 128}
READER = "component_top1"


def verify_inputs(data_root, geometry_root, reader_audit):
    data_root, geometry_root = Path(data_root), Path(geometry_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if (
        manifest.get("protocol") != "cbf_event_curriculum_v1"
        or manifest.get("seed") != DATA_SEED
        or manifest.get("group_counts") != GROUP_COUNTS
        or manifest.get("audit", {}).get("causal_answers_verified") is not True
        or manifest.get("audit", {}).get("entities_disjoint") is not True
        or manifest.get("audit", {}).get("templates_disjoint") is not True
    ):
        raise ValueError("unexpected sparse-cohort corpus release")
    observed = {name: sha(geometry_root / name) for name in GEOMETRY_HASHES}
    if observed != GEOMETRY_HASHES:
        raise ValueError("frozen contextual geometry dependency changed")
    geometry_summary = json.loads((geometry_root / "summary.json").read_text())
    config = geometry_summary["selected_configs"]["context_last"]
    if (
        geometry_summary.get("stage_passed") is not True
        or geometry_summary.get("selected_variant") != "context_last"
        or geometry_summary.get("test_scored") is not False
        or config
        != {
            "internal": "context_last_r0.01",
            "regularizer": 0.01,
            "unknown_raw": 0.75,
            "train_accuracy": 0.9988787615740741,
        }
    ):
        raise ValueError("unexpected contextual geometry configuration")
    audit = json.loads(Path(reader_audit).read_text())
    if not all(
        (
            audit.get("protocol") == "audit_qwen_sparse_reader_diagnostic_v1",
            audit.get("audit_passed") is True,
            audit.get("decision", {}).get("passed") is True,
            audit.get("decision", {}).get("selected") == READER,
            audit.get("test_scored") is False,
        )
    ):
        raise ValueError("sparse-reader diagnostic did not release component_top1")
    return manifest, config, sha(reader_audit)


def build_examples(pairs, embeddings, history=True):
    rows = []
    for inp, target in pairs:
        state, candidates = [], episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                label = counterfactual_label(
                    state,
                    assignment,
                    delta,
                    block["index"],
                    truth,
                    candidates,
                    embeddings,
                    READER,
                )
                rows.append(
                    {
                        "group_id": inp["group_id"],
                        "episode_id": inp["id"],
                        "block_index": block["index"],
                        "features": features(
                            state, assignment, delta, block["index"], history
                        ),
                        "retention": list(label),
                        "label": GRID.index(label),
                    }
                )
                state = apply_retention(
                    state, label, assignment, delta, block["index"]
                )
            else:
                state = apply_retention(
                    state, (1.0, 1.0, 1.0), assignment, delta, block["index"]
                )
    interior = sum(any(value == 0.5 for value in row["retention"]) for row in rows)
    return rows, {
        "examples": len(rows),
        "interior_examples": interior,
        "interior_fraction": interior / len(rows),
        "label_counts": dict(Counter(str(tuple(row["retention"])) for row in rows)),
    }


def select(policy, state, assignment, delta, block_index, truth, candidates,
           embeddings, controllers, episode_id):
    if policy in FIXED:
        return FIXED[policy], None
    if policy == "deterministic_random":
        digest = hashlib.sha256(
            f"{embeddings.seed}|{episode_id}|{block_index}".encode()
        ).hexdigest()
        return GRID[int(digest[:12], 16) % len(GRID)], None
    if policy == "physics_cohort":
        return physics_retention(state, assignment), None
    if policy == "oracle_grid":
        return (
            counterfactual_label(
                state,
                assignment,
                delta,
                block_index,
                truth,
                candidates,
                embeddings,
                READER,
            ),
            None,
        )
    if policy == "exact_reference":
        return None, apply_exact_reference(state, assignment, delta, block_index)
    visible_history = policy != "reference_blind"
    values = features(state, assignment, delta, block_index, visible_history)
    return controller_retention(controllers[policy], values), None


def evaluate_policy(pairs, policy, embeddings, controllers):
    records = []
    score_oracle = policy in {
        "reference_blind",
        "grid_classifier",
        "learned",
        "oracle_grid",
    }
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates = episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                oracle = (
                    counterfactual_label(
                        state,
                        assignment,
                        delta,
                        block["index"],
                        truth,
                        candidates,
                        embeddings,
                        READER,
                    )
                    if score_oracle
                    else None
                )
                if policy == "oracle_grid":
                    retention, direct_state = oracle, None
                else:
                    retention, direct_state = select(
                        policy,
                        state,
                        assignment,
                        delta,
                        block["index"],
                        truth,
                        candidates,
                        embeddings,
                        controllers,
                        inp["id"],
                    )
                state = (
                    direct_state
                    if direct_state is not None
                    else apply_retention(
                        state, retention, assignment, delta, block["index"]
                    )
                )
                decisions.append(
                    {
                        "block": block["index"],
                        "retention": None if retention is None else list(retention),
                        "oracle_retention": None if oracle is None else list(oracle),
                        "retention_mae": (
                            None
                            if retention is None or oracle is None
                            else statistics.fmean(
                                abs(a - b) for a, b in zip(retention, oracle)
                            )
                        ),
                        "grid_action_match": (
                            None
                            if retention is None or oracle is None
                            else int(
                                tuple(
                                    min(
                                        (0.0, 0.5, 1.0),
                                        key=lambda point: abs(point - value),
                                    )
                                    for value in retention
                                )
                                == oracle
                            )
                        ),
                        "retained_mass": sum(row["weight"] for row in state),
                    }
                )
                for row in truth:
                    queries.append(
                        {
                            **row,
                            **query_metrics(
                                state, row, candidates, embeddings, READER
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
                "policy": policy,
                "reader": READER,
                "queries": queries,
                "decisions": decisions,
            }
        )
    return records


def development_gate(summary, training_meta):
    learned = summary["learned"]
    physics = summary["physics_cohort"]
    oracle = summary["oracle_grid"]
    gain = comparison(learned, physics, 2711)
    checks = {
        "accuracy_at_least_point_97": learned["mean_accuracy"] >= 0.97,
        "accuracy_gain_over_physics_at_least_point_03": gain["accuracy_gain"] >= 0.03,
        "accuracy_ci_lower_above_point_015": gain["accuracy_bootstrap_95ci"][0] > 0.015,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_beats_physics": all(
            learned["per_seed"][str(seed)]["accuracy"]
            > physics["per_seed"][str(seed)]["accuracy"]
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_02": (
            oracle["mean_accuracy"] - learned["mean_accuracy"] <= 0.02
        ),
        "current_and_anchor_at_least_point_98": min(
            learned["roles"]["current"], learned["roles"]["anchor"]
        )
        >= 0.98,
        "historical_at_least_point_93": learned["roles"]["historical"] >= 0.93,
        "all_families_at_least_point_90": min(learned["families"].values()) >= 0.90,
        "interior_label_fraction_at_least_point_10": min(
            value["interior_fraction"] for value in training_meta.values()
        )
        >= 0.10,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - learned["mean_accuracy"],
    }


def test_gate(summary):
    learned = summary["learned"]
    physics = summary["physics_cohort"]
    oracle = summary["oracle_grid"]
    gain = comparison(learned, physics, 2731)
    checks = {
        "accuracy_at_least_point_97": learned["mean_accuracy"] >= 0.97,
        "accuracy_gain_over_physics_at_least_point_03": gain["accuracy_gain"] >= 0.03,
        "accuracy_ci_lower_above_point_015": gain["accuracy_bootstrap_95ci"][0] > 0.015,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_beats_physics": all(
            learned["per_seed"][str(seed)]["accuracy"]
            > physics["per_seed"][str(seed)]["accuracy"]
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_02": (
            oracle["mean_accuracy"] - learned["mean_accuracy"] <= 0.02
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - learned["mean_accuracy"],
    }


def move_embeddings(embeddings, device):
    from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings

    return QwenEmbeddings(
        embeddings.seed,
        embeddings.dim,
        {key: value.to(device) for key, value in embeddings.values.items()},
    )


def run(args):
    import torch

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    data_manifest, config, reader_audit_sha = verify_inputs(
        args.data_root, args.geometry_root, args.reader_audit
    )
    transform_artifact = torch.load(
        Path(args.geometry_root) / "contextual_transforms.pt",
        map_location="cpu",
        weights_only=False,
    )
    transforms = transform_artifact["transforms"]
    train_pairs = load_split(args.data_root, "train")
    dev_pairs = load_split(args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings, embedding_hashes, extraction, model_meta = extract_embeddings(
        train_pairs + dev_pairs,
        transforms,
        args.model,
        args.tokenizer,
        args.batch_size,
    )
    extraction_seconds = time.perf_counter() - started

    controllers_saved, training_meta, dev_seed, raw_hashes = {}, {}, {}, {}
    for offset, seed in enumerate(SEEDS):
        memory = move_embeddings(embeddings[str(seed)], "cuda")
        train_rows, meta = build_examples(train_pairs, memory, True)
        blind_rows, blind_meta = build_examples(train_pairs, memory, False)
        controllers, curves = train_controllers(
            train_rows, blind_rows, CONTROLLER_SEEDS[offset], args.epochs
        )
        controllers_saved[str(seed)] = {
            name: serialize_controller(value) for name, value in controllers.items()
        }
        training_meta[str(seed)] = {**meta, "blind": blind_meta, "curves": curves}
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            records = evaluate_policy(dev_pairs, policy, memory, controllers)
            dev_seed[str(seed)][policy] = aggregate(records)
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        del memory
        torch.cuda.empty_cache()
    dev = multi_seed_summary(dev_seed)
    dev_gate = development_gate(dev, training_meta)

    test_scored, test, test_release, test_embedding_meta = False, None, None, None
    if dev_gate["passed"]:
        test_pairs = load_split(args.data_root, "test")
        test_embeddings, test_hashes, test_extraction, _ = extract_embeddings(
            test_pairs, transforms, args.model, args.tokenizer, args.batch_size
        )
        test_seed = {}
        for seed in SEEDS:
            memory = move_embeddings(test_embeddings[str(seed)], "cuda")
            controllers = restore_controllers(
                controllers_saved[str(seed)], len(FEATURE_NAMES)
            )
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                records = evaluate_policy(test_pairs, policy, memory, controllers)
                test_seed[str(seed)][policy] = aggregate(records)
                path = root / f"test_seed{seed}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
                raw_hashes[path.name] = sha(path)
            del memory
            torch.cuda.empty_cache()
        test = multi_seed_summary(test_seed)
        test_release = test_gate(test)
        test_scored = True
        test_embedding_meta = {"hashes": test_hashes, "extraction": test_extraction}

    checkpoint = root / "controllers.pt"
    torch.save(
        {
            "protocol": PROTOCOL,
            "controllers": controllers_saved,
            "features": FEATURE_NAMES,
            "grid": GRID,
            "reader": READER,
            "projection_seeds": SEEDS,
            "controller_seeds": CONTROLLER_SEEDS,
            "epochs": args.epochs,
        },
        checkpoint,
    )
    training = {
        "seconds": time.perf_counter() - started,
        "train_dev_extraction_seconds": extraction_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "embedding_hashes": embedding_hashes,
        "extraction": extraction,
        "test_embedding_meta": test_embedding_meta,
        "seeds": training_meta,
        "checkpoint_sha256": sha(checkpoint),
        "raw_rows_sha256": raw_hashes,
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "reader_audit_sha256": reader_audit_sha,
        "geometry_hashes": GEOMETRY_HASHES,
        "contextual_config": config,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "reader": READER,
        "grid": GRID,
        "fixed": FIXED,
        "policies": POLICIES,
        "features": FEATURE_NAMES,
        "projection_seeds": SEEDS,
        "controller_seeds": CONTROLLER_SEEDS,
        "epochs": args.epochs,
        "test_loaded": test_scored,
        "test_scored": test_scored,
        "group_counts": data_manifest["group_counts"],
    }
    result = {
        "protocol": PROTOCOL,
        "dev": dev,
        "dev_gate": dev_gate,
        "test": test,
        "test_gate": test_release,
        "test_scored": test_scored,
        "stage_passed": bool(test_release and test_release["passed"]),
        "next_stage": (
            "integrate_segmented_native_fast_weights"
            if test_release and test_release["passed"]
            else "report_failed_one_time_test"
            if test_scored
            else "stop_sparse_cohort_controller"
        ),
        "terminal_status": "completed_qwen_sparse_cohort_forgetting",
    }
    write_json(root / "training.json", training)
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--reader-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "dev_gate": result["dev_gate"],
                "test_scored": result["test_scored"],
                "test_gate": result["test_gate"],
                "next_stage": result["next_stage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
