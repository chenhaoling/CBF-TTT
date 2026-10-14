"""Evaluate a frozen exact-state controller on registered lifecycle shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.build_qwen_discrete_cohort_ood import (
    BLOCKS,
    BOUNDARIES,
    PROTOCOL as DATA_PROTOCOL,
    SUITES,
    load_corpus,
    parse_visible_block,
)
from tasks.diagnose_qwen_sparse_reader import (
    apply_retention,
    counterfactual_label,
    query_metrics,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import episode_candidates
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_cbf_structured_forgetting import boundary_truth
from tasks.train_qwen_cohort_forgetting import (
    FEATURE_NAMES,
    GEOMETRY_HASHES,
    aggregate,
    batch_delta,
    comparison,
    extract_embeddings,
    restore_controllers,
)
from tasks.train_qwen_discrete_cohort_confirmation import (
    POLICIES,
    move_embeddings,
    multi_seed_summary,
    select,
    snap_retention,
)
from scripts.summarize_qwen_sparse_action_diagnostic import (
    combine as combine_action_diagnostics,
    summarize_rows as summarize_action_rows,
)


PROTOCOL = "qwen_discrete_cohort_ood_v1"
DATA_SEED = 202610154
GROUPS_PER_SUITE = 64
READER = "component_top1"
CONFIRMATION_AUDIT_PROTOCOL = "audit_qwen_discrete_cohort_confirmation_v1"
CONFIRMATION_PROTOCOL = "qwen_discrete_cohort_confirmation_v1"


def verify_confirmation(root, audit_path):
    root, audit_path = Path(root), Path(audit_path)
    audit = json.loads(audit_path.read_text())
    summary_path = root / "result" / "summary.json"
    checkpoint_path = root / "result" / "controllers.pt"
    summary = json.loads(summary_path.read_text())
    if not all(
        (
            audit.get("protocol") == CONFIRMATION_AUDIT_PROTOCOL,
            audit.get("audit_passed") is True,
            audit.get("test_scored") is True,
            audit.get("test_gate", {}).get("passed") is True,
            summary.get("protocol") == CONFIRMATION_PROTOCOL,
            summary.get("test_scored") is True,
            summary.get("stage_passed") is True,
            summary.get("test_gate", {}).get("passed") is True,
            audit.get("summary_sha256") == sha(summary_path),
            audit.get("checkpoint_sha256") == sha(checkpoint_path),
        )
    ):
        raise ValueError("discrete confirmation did not release OOD stress")
    return {
        "confirmation_audit_sha256": sha(audit_path),
        "confirmation_summary_sha256": sha(summary_path),
        "confirmation_checkpoint_sha256": sha(checkpoint_path),
    }


def verify_data(data_root):
    manifest, suites = load_corpus(data_root)
    audit = manifest.get("audit", {})
    if not all(
        (
            manifest.get("protocol") == DATA_PROTOCOL,
            manifest.get("seed") == DATA_SEED,
            manifest.get("groups_per_suite") == GROUPS_PER_SUITE,
            tuple(manifest.get("suites", ())) == SUITES,
            manifest.get("blocks") == BLOCKS,
            manifest.get("boundaries")
            == {key: list(value) for key, value in BOUNDARIES.items()},
            audit.get("causal_answers_verified") is True,
            audit.get("suite_identifiers_and_values_disjoint") is True,
            audit.get("complete_twins") is True,
            audit.get("oracle_actions_generated") is False,
            all(len(suites[suite]) == 2 * GROUPS_PER_SUITE for suite in SUITES),
        )
    ):
        raise ValueError("unexpected lifecycle stress corpus release")
    return manifest, suites


def evaluate_policy(pairs, policy, embeddings, controllers):
    records = []
    score_oracle = policy in {
        "learned_snapped",
        "grid_classifier",
        "learned",
        "oracle_grid",
    }
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates = episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_visible_block(block["text"], state)
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
                                abs(left - right)
                                for left, right in zip(retention, oracle)
                            )
                        ),
                        "grid_action_match": (
                            None
                            if retention is None or oracle is None
                            else int(snap_retention(retention) == tuple(oracle))
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
                    state,
                    (1.0, 1.0, 1.0),
                    assignment,
                    delta,
                    block["index"],
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


def registered_gate(suites, macro, action):
    primary, physics, oracle = (
        macro["grid_classifier"],
        macro["physics_cohort"],
        macro["oracle_grid"],
    )
    gain = comparison(primary, physics, 3011)
    checks = {
        "macro_accuracy_at_least_point_97": primary["mean_accuracy"] >= 0.97,
        "every_suite_accuracy_at_least_point_95": all(
            suites[suite]["grid_classifier"]["mean_accuracy"] >= 0.95
            for suite in SUITES
        ),
        "minimum_projection_seed_at_least_point_95": (
            primary["min_seed_accuracy"] >= 0.95
        ),
        "physics_gain_at_least_point_03": gain["accuracy_gain"] >= 0.03,
        "physics_gain_ci_lower_above_point_01": (
            gain["accuracy_bootstrap_95ci"][0] > 0.01
        ),
        "nll_gain_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.0,
        "no_suite_negative_physics_gain": all(
            suites[suite]["grid_classifier"]["mean_accuracy"]
            >= suites[suite]["physics_cohort"]["mean_accuracy"]
            for suite in SUITES
        ),
        "oracle_gap_at_most_point_02": (
            oracle["mean_accuracy"] - primary["mean_accuracy"] <= 0.02
        ),
        "current_and_anchor_at_least_point_98": min(
            primary["roles"]["current"], primary["roles"]["anchor"]
        )
        >= 0.98,
        "historical_at_least_point_93": primary["roles"]["historical"] >= 0.93,
        "endpoint_action_at_least_point_97": (
            action["buckets"]["endpoint"]["action_accuracy"] >= 0.97
        ),
        "interior_action_at_least_point_65": (
            action["buckets"]["interior"]["action_accuracy"] >= 0.65
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "comparison_to_physics": gain,
        "oracle_gap": oracle["mean_accuracy"] - primary["mean_accuracy"],
    }


def run(args):
    import torch

    output = Path(args.output)
    dependency = verify_confirmation(args.confirmation_root, args.confirmation_audit)
    data_manifest, suite_pairs = verify_data(args.data_root)
    geometry_root = Path(args.geometry_root)
    observed_geometry = {name: sha(geometry_root / name) for name in GEOMETRY_HASHES}
    if observed_geometry != GEOMETRY_HASHES:
        raise ValueError("frozen contextual geometry dependency changed")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "progress.json",
        {"stage": "embedding_extraction", "completed_seeds": []},
    )
    transform_artifact = torch.load(
        geometry_root / "contextual_transforms.pt",
        map_location="cpu",
        weights_only=False,
    )
    all_pairs = [pair for suite in SUITES for pair in suite_pairs[suite]]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embeddings, embedding_hashes, extraction, model_meta = extract_embeddings(
        all_pairs,
        transform_artifact["transforms"],
        args.model,
        args.tokenizer,
        args.batch_size,
    )
    extraction_seconds = time.perf_counter() - started
    checkpoint_path = Path(args.confirmation_root) / "result" / "controllers.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("protocol") != CONFIRMATION_PROTOCOL
        or tuple(checkpoint.get("projection_seeds", ())) != SEEDS
        or checkpoint.get("reader") != READER
    ):
        raise ValueError("unexpected confirmation controller checkpoint")

    suite_summaries, macro_records, raw_hashes = {}, {
        str(seed): {policy: [] for policy in POLICIES} for seed in SEEDS
    }, {}
    action_per_seed = {}
    for seed in SEEDS:
        memory = move_embeddings(embeddings[str(seed)], "cuda")
        controllers = restore_controllers(
            checkpoint["controllers"][str(seed)], len(FEATURE_NAMES)
        )
        action_rows = []
        for suite in SUITES:
            for policy in POLICIES:
                write_json(
                    output / "progress.json",
                    {
                        "stage": "evaluation",
                        "completed_seeds": [
                            completed for completed in SEEDS if completed < seed
                        ],
                        "current_seed": seed,
                        "current_suite": suite,
                        "current_policy": policy,
                    },
                )
                records = evaluate_policy(
                    suite_pairs[suite], policy, memory, controllers
                )
                macro_records[str(seed)][policy].extend(records)
                if policy == "grid_classifier":
                    action_rows.extend(records)
                path = output / f"{suite}_seed{seed}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
                raw_hashes[path.name] = sha(path)
        action_per_seed[str(seed)] = summarize_action_rows(action_rows)
        del memory
        torch.cuda.empty_cache()

    for suite in SUITES:
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {}
            for policy in POLICIES:
                path = output / f"{suite}_seed{seed}_{policy}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
                per_seed[str(seed)][policy] = aggregate(rows)
        suite_summaries[suite] = multi_seed_summary(per_seed)
    macro_per_seed = {
        str(seed): {
            policy: aggregate(macro_records[str(seed)][policy])
            for policy in POLICIES
        }
        for seed in SEEDS
    }
    macro = multi_seed_summary(macro_per_seed)
    action = combine_action_diagnostics(action_per_seed)
    gate = registered_gate(suite_summaries, macro, action)
    training = {
        "seconds": time.perf_counter() - started,
        "embedding_extraction_seconds": extraction_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "embedding_hashes": embedding_hashes,
        "extraction": extraction,
        "raw_rows_sha256": raw_hashes,
    }
    manifest = {
        "protocol": PROTOCOL,
        **dependency,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "geometry_hashes": observed_geometry,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "reader": READER,
        "projection_seeds": list(SEEDS),
        "suites": list(SUITES),
        "groups_per_suite": data_manifest["groups_per_suite"],
        "blocks": BLOCKS,
        "policies": list(POLICIES),
        "controller_retrained": False,
        "calibration_performed": False,
    }
    result = {
        "protocol": PROTOCOL,
        "suites": suite_summaries,
        "macro": macro,
        "action_diagnostic": action,
        "gate": gate,
        "stage_passed": gate["passed"],
        "next_stage": (
            "release_native_and_public_benchmark_prerequisites"
            if gate["passed"]
            else "stop_exact_state_controller_branch"
        ),
        "terminal_status": "completed_qwen_discrete_cohort_ood",
    }
    write_json(output / "training.json", training)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", result)
    write_json(
        output / "progress.json",
        {"stage": "completed", "completed_seeds": list(SEEDS)},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--confirmation-root", required=True)
    verify.add_argument("--confirmation-audit", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--confirmation-root", required=True)
    evaluate.add_argument("--confirmation-audit", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--geometry-root", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_confirmation(args.confirmation_root, args.confirmation_audit)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
