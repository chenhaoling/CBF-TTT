"""Fresh confirmation of exact discrete cohort-state control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_cbf_structured_forgetting import boundary_truth, load_split, parse_block
from tasks.train_qwen_cohort_forgetting import (
    CONTROLLER_SEEDS,
    FEATURE_NAMES,
    FIXED,
    GEOMETRY_HASHES,
    GRID,
    POLICIES as BASE_POLICIES,
    REFERENCE_INDICES,
    aggregate,
    batch_delta,
    comparison,
    controller_retention,
    extract_embeddings,
    features,
    physics_retention,
    restore_controllers,
    serialize_controller,
    train_controllers,
)
from tasks.train_qwen_sparse_cohort_forgetting import build_examples
from tasks.diagnose_qwen_sparse_reader import (
    apply_exact_reference,
    apply_retention,
    counterfactual_label,
    query_metrics,
)


PROTOCOL = "qwen_discrete_cohort_confirmation_v1"
DATA_SEED = 202610153
GROUP_COUNTS = {"train": 256, "dev": 64, "test": 128}
READER = "component_top1"
POLICIES = BASE_POLICIES[:7] + ("learned_snapped",) + BASE_POLICIES[7:]


def verify_inputs(data_root, geometry_root, reader_audit, predecessor_audit):
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
        raise ValueError("unexpected discrete-cohort corpus release")
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

    reader = json.loads(Path(reader_audit).read_text())
    if not all(
        (
            reader.get("protocol") == "audit_qwen_sparse_reader_diagnostic_v1",
            reader.get("audit_passed") is True,
            reader.get("decision", {}).get("selected") == READER,
            reader.get("test_scored") is False,
        )
    ):
        raise ValueError("top-1 reader dependency did not pass")

    predecessor = json.loads(Path(predecessor_audit).read_text())
    if not all(
        (
            predecessor.get("protocol")
            == "audit_qwen_sparse_cohort_forgetting_v1",
            predecessor.get("audit_passed") is True,
            predecessor.get("test_scored") is False,
        )
    ):
        raise ValueError("predecessor audit did not preserve the sealed test")
    dev = predecessor["development"]
    primary = dev["grid_classifier"]
    physics = dev["physics_cohort"]
    oracle = dev["oracle_grid"]
    prerequisites = {
        "grid_accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "every_seed_at_least_point_98": min(
            row["accuracy"] for row in primary["per_seed"].values()
        )
        >= 0.98,
        "gain_over_physics_at_least_point_05": (
            primary["mean_accuracy"] - physics["mean_accuracy"] >= 0.05
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
    }
    if not all(prerequisites.values()):
        raise ValueError(f"predecessor discrete prerequisite failed: {prerequisites}")
    return manifest, config, sha(reader_audit), sha(predecessor_audit), prerequisites


def blind_examples(rows):
    output = []
    for row in rows:
        values = list(row["features"])
        for index in REFERENCE_INDICES:
            values[index] = 0.0
        output.append({**row, "features": values})
    return output


def snap_retention(retention):
    return tuple(
        min((0.0, 0.5, 1.0), key=lambda point: abs(point - value))
        for value in retention
    )


def select(policy, state, assignment, delta, block_index, truth, candidates,
           embeddings, controllers, episode_id):
    from tasks.train_qwen_sparse_cohort_forgetting import select as base_select

    if policy == "learned_snapped":
        values = features(state, assignment, delta, block_index, True)
        retention = controller_retention(controllers["learned"], values)
        return snap_retention(retention), None
    return base_select(
        policy,
        state,
        assignment,
        delta,
        block_index,
        truth,
        candidates,
        embeddings,
        controllers,
        episode_id,
    )


def evaluate_policy(pairs, policy, embeddings, controllers):
    records = []
    score_oracle = policy in {
        "grid_classifier",
        "learned",
        "learned_snapped",
        "oracle_grid",
    }
    for inp, target in pairs:
        from tasks.train_cbf_lowrank_forgetting import episode_candidates

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
                            else int(snap_retention(retention) == oracle)
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


def multi_seed_summary(per_seed):
    output = {}
    for policy in POLICIES:
        rows = [per_seed[str(seed)][policy] for seed in SEEDS]
        output[policy] = {
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "mean_nll": statistics.fmean(row["nll"] for row in rows),
            "min_seed_accuracy": min(row["accuracy"] for row in rows),
            "mean_retention_mae": (
                None
                if rows[0]["retention_mae"] is None
                else statistics.fmean(row["retention_mae"] for row in rows)
            ),
            "mean_grid_action_accuracy": (
                None
                if rows[0]["grid_action_accuracy"] is None
                else statistics.fmean(row["grid_action_accuracy"] for row in rows)
            ),
            "mean_retained_mass": statistics.fmean(
                row["mean_retained_mass"] for row in rows
            ),
            "roles": {
                role: statistics.fmean(row["roles"][role] for row in rows)
                for role in rows[0]["roles"]
            },
            "families": {
                family: statistics.fmean(row["families"][family] for row in rows)
                for family in rows[0]["families"]
            },
            "per_seed": {
                str(seed): per_seed[str(seed)][policy] for seed in SEEDS
            },
        }
    return output


def development_gate(summary, training_meta):
    primary = summary["grid_classifier"]
    physics = summary["physics_cohort"]
    oracle = summary["oracle_grid"]
    blind = summary["reference_blind"]
    gain = comparison(primary, physics, 2811)
    checks = {
        "accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "accuracy_gain_over_physics_at_least_point_05": gain["accuracy_gain"] >= 0.05,
        "accuracy_ci_lower_above_point_035": gain["accuracy_bootstrap_95ci"][0] > 0.035,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_gain_at_least_point_04": all(
            primary["per_seed"][str(seed)]["accuracy"]
            - physics["per_seed"][str(seed)]["accuracy"]
            >= 0.04
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
        "current_and_anchor_at_least_point_99": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        )
        >= 0.99,
        "historical_at_least_point_96": primary["roles"]["historical"] >= 0.96,
        "all_families_at_least_point_95": min(primary["families"].values()) >= 0.95,
        "grid_action_agreement_at_least_point_88": (
            primary["mean_grid_action_accuracy"] >= 0.88
        ),
        "gain_over_reference_blind_at_least_point_20": (
            primary["mean_accuracy"] - blind["mean_accuracy"] >= 0.20
        ),
        "interior_label_fraction_at_least_point_10": min(
            value["interior_fraction"] for value in training_meta.values()
        )
        >= 0.10,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def test_gate(summary):
    primary, physics, oracle = (
        summary["grid_classifier"],
        summary["physics_cohort"],
        summary["oracle_grid"],
    )
    gain = comparison(primary, physics, 2831)
    checks = {
        "accuracy_at_least_point_985": primary["mean_accuracy"] >= 0.985,
        "accuracy_gain_over_physics_at_least_point_05": gain["accuracy_gain"] >= 0.05,
        "accuracy_ci_lower_above_point_035": gain["accuracy_bootstrap_95ci"][0] > 0.035,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "every_seed_gain_at_least_point_04": all(
            primary["per_seed"][str(seed)]["accuracy"]
            - physics["per_seed"][str(seed)]["accuracy"]
            >= 0.04
            for seed in SEEDS
        ),
        "oracle_gap_at_most_point_01": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.01
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
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
    progress_path = root / "progress.json"
    write_json(
        progress_path,
        {"stage": "verify_inputs", "completed_seeds": [], "test_loaded": False},
    )
    (
        data_manifest,
        config,
        reader_audit_sha,
        predecessor_audit_sha,
        prerequisites,
    ) = verify_inputs(
        args.data_root,
        args.geometry_root,
        args.reader_audit,
        args.predecessor_audit,
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
    write_json(
        progress_path,
        {"stage": "development", "completed_seeds": [], "test_loaded": False},
    )

    controllers_saved, training_meta, dev_seed, raw_hashes = {}, {}, {}, {}
    for offset, seed in enumerate(SEEDS):
        memory = move_embeddings(embeddings[str(seed)], "cuda")
        train_rows, meta = build_examples(train_pairs, memory, True)
        blind_rows = blind_examples(train_rows)
        controllers, curves = train_controllers(
            train_rows, blind_rows, CONTROLLER_SEEDS[offset], args.epochs
        )
        controllers_saved[str(seed)] = {
            name: serialize_controller(value) for name, value in controllers.items()
        }
        training_meta[str(seed)] = {
            **meta,
            "blind_derived_from_primary_rows": True,
            "curves": curves,
        }
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            write_json(
                progress_path,
                {
                    "stage": "development",
                    "completed_seeds": list(SEEDS[:offset]),
                    "current_seed": seed,
                    "current_policy": policy,
                    "test_loaded": False,
                },
            )
            records = evaluate_policy(dev_pairs, policy, memory, controllers)
            dev_seed[str(seed)][policy] = aggregate(records)
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        del memory
        torch.cuda.empty_cache()
        write_json(
            progress_path,
            {
                "stage": "development",
                "completed_seeds": list(SEEDS[: offset + 1]),
                "test_loaded": False,
            },
        )
    dev = multi_seed_summary(dev_seed)
    dev_gate = development_gate(dev, training_meta)

    test_scored, test, test_release, test_embedding_meta = False, None, None, None
    if dev_gate["passed"]:
        write_json(
            progress_path,
            {
                "stage": "test_extraction",
                "completed_seeds": list(SEEDS),
                "test_loaded": True,
            },
        )
        test_pairs = load_split(args.data_root, "test")
        test_embeddings, test_hashes, test_extraction, _ = extract_embeddings(
            test_pairs, transforms, args.model, args.tokenizer, args.batch_size
        )
        test_seed = {}
        for test_offset, seed in enumerate(SEEDS):
            memory = move_embeddings(test_embeddings[str(seed)], "cuda")
            controllers = restore_controllers(
                controllers_saved[str(seed)], len(FEATURE_NAMES)
            )
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                write_json(
                    progress_path,
                    {
                        "stage": "test",
                        "completed_seeds": list(SEEDS),
                        "completed_test_seeds": list(SEEDS[:test_offset]),
                        "current_seed": seed,
                        "current_policy": policy,
                        "test_loaded": True,
                    },
                )
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
        "predecessor_audit_sha256": predecessor_audit_sha,
        "predecessor_prerequisites": prerequisites,
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
            else "stop_discrete_cohort_controller"
        ),
        "terminal_status": "completed_qwen_discrete_cohort_confirmation",
    }
    write_json(root / "training.json", training)
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", result)
    write_json(
        progress_path,
        {
            "stage": "complete",
            "completed_seeds": list(SEEDS),
            "test_loaded": test_scored,
            "development_gate_passed": dev_gate["passed"],
            "test_gate_passed": bool(test_release and test_release["passed"]),
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--reader-audit", required=True)
    parser.add_argument("--predecessor-audit", required=True)
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
