"""Independent replay audit for on-policy natural forgetting V3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import scope_examples
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    active_queries,
    chain_map,
    checkpoint_label,
    primary_count,
    record_for,
)
from tasks.evaluate_qwen_natural_forgetting_online_v3 import (
    ARMS,
    EXPECTED_V5,
    FEATURE_SETS,
    FactorController,
    GENERATIVE_ARMS,
    PLAN,
    PROTOCOL,
    SEEDS,
    aggregate,
    engineered_features,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.evaluate_qwen_natural_value_bridge_v1 import render_query
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


def close(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def audit(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root, repo = Path(args.root), Path(args.repo)
    factor_root, data_root = Path(args.factor_root), Path(args.data_root)
    v5_root, context_root, scope_root = map(
        Path, (args.v5_root, args.context_root, args.scope_root))
    summary = json.loads((root / "summary.json").read_text())
    release = json.loads((root / "release_manifest.json").read_text())
    rows = []
    decisions = []
    resources = []
    for shard in range(args.num_shards):
        row_path = root / f"rows_shard{shard}.jsonl"
        decision_path = root / f"decisions_shard{shard}.jsonl"
        resource = json.loads((root / f"resource_shard{shard}.json").read_text())
        rows.extend(json.loads(line) for line in row_path.read_text().splitlines() if line)
        decisions.extend(json.loads(line) for line in decision_path.read_text().splitlines() if line)
        resources.append(resource)
    rows.sort(key=lambda row: (row["session_id"], row["phase"],
                               row["chain_id"], row["surface"]))
    decisions.sort(key=lambda row: row["session_id"])
    _, sessions = load(data_root, "dev")
    sessions_by_id = {row["id"]: row for row in sessions}
    decisions_by_id = {row["session_id"]: row for row in decisions}

    factor_path = factor_root / "result" / "factor_gate.pt"
    factor = FactorController(torch.load(factor_path, map_location="cpu", weights_only=False))
    model, _ = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    source_summary = json.loads((context_root / "summary.json").read_text())
    transforms = torch.load(context_root / "transforms.pt", map_location="cpu", weights_only=False)
    scope_artifact = torch.load(scope_root / "scope_gates.pt", map_location="cpu", weights_only=False)
    scorer, _ = build_scorer(
        scope_examples(sessions), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact, args.batch_size)
    v5_path = v5_root / "result" / "constrained_scope_gate.pt"
    v5_gate, v5_threshold, _ = load_v5_gate(v5_path, scope_artifact["heads"]["7301"])

    transition_checks = True
    feature_checks = True
    probability_checks = True
    expected_identity = []
    for session in sessions:
        trace = decisions_by_id[session["id"]]
        chains = chain_map(session); states = {arm: [] for arm in ARMS}; active = {}
        if set(trace["decisions"]) != set(ARMS):
            transition_checks = False; continue
        for step, event in enumerate(session["events"]):
            chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
            for arm in ARMS:
                observed = trace["decisions"][arm][step]; memory = states[arm]
                features, top, probability = engineered_features(
                    version["prompt"], memory, step, len(session["events"]), scorer, v5_gate)
                # The collector embeds 32 sessions per GPU while the auditor
                # embeds all 64 together.  Keep representation replay
                # independent, with tolerance only for batch-dependent GPU
                # floating point accumulation.  Checkpoint replay below uses
                # the stored feature exactly and remains at 1e-6.
                feature_checks &= len(features) == len(observed["features"]) and all(
                    close(left, right, 1e-4)
                    for left, right in zip(features, observed["features"]))
                feature_checks &= observed["top"] == top and close(
                    observed["v5_probability"], probability, 1e-4)
                predicted = {}; expected_delete = False
                if arm == "always_scope_top1":
                    expected_delete = bool(memory)
                elif arm == "source_action_rule":
                    expected_delete = bool(memory) and event["action"] == "local"
                elif arm.startswith("factor_seed_"):
                    predicted = factor.probabilities(observed["features"])
                    seed = int(arm.rsplit("_", 1)[1]); config = factor.configs["engineered"]
                    expected_delete = bool(memory) \
                        and predicted[seed][0] >= config["benefit_threshold"] \
                        and predicted[seed][1] <= config["risk_threshold"]
                elif arm in ("factor_ensemble", "benefit_only_ensemble",
                             "no_schedule_ensemble"):
                    feature_set = ("no_schedule" if arm == "no_schedule_ensemble"
                                   else "engineered")
                    predicted = factor.probabilities(observed["features"], feature_set)
                    benefit = sum(value[0] for value in predicted.values()) / len(predicted)
                    risk = sum(value[1] for value in predicted.values()) / len(predicted)
                    config = factor.configs[feature_set]
                    expected_delete = bool(memory) \
                        and benefit >= config["benefit_threshold"] \
                        and (arm == "benefit_only_ensemble" or risk <= config["risk_threshold"])
                elif arm != "retain_all":
                    transition_checks = False
                stored = {int(seed): tuple(value)
                          for seed, value in observed["probabilities"].items()}
                probability_checks &= set(stored) == set(predicted) and all(
                    close(stored[seed][index], predicted[seed][index])
                    for seed in stored for index in (0, 1))
                action = "scope_top1_a0" if expected_delete else "retain_all"
                transition_checks &= all((
                    observed["arm"] == arm, observed["step"] == step,
                    observed["before_size"] == len(memory), observed["top"] == top,
                    observed["action"] == action,
                    observed["after_delete_size"] == len(memory) - int(expected_delete),
                    observed["after_write_size"] == len(memory) + 1 - int(expected_delete),
                ))
                if expected_delete:
                    transition_checks &= observed["deleted"] == [memory[top]]
                    memory = [row for index, row in enumerate(memory) if index != top]
                else:
                    transition_checks &= observed["deleted"] == []
                states[arm] = [*memory, {**record_for(chain, version), "strength": 1.0}]
            active[event["chain_id"]] = event["version"]
            phase = checkpoint_label(step, session)
            if phase is not None:
                final = phase == f"update_{primary_count(session)}"
                expected_identity.extend((session["id"], phase, query["chain_id"],
                                          query["active_version"], query["surface"],
                                          query["query"], query["answer"], final,
                                          event["chain_id"])
                                         for query in active_queries(session, active))

    observed_identity = [(row["session_id"], row["phase"], row["chain_id"],
                          row["active_version"], row["surface"], row["query"],
                          row["answer"], row["final"], row["recent_chain_id"])
                         for row in rows]
    expected_identity.sort(); observed_identity.sort()
    row_checks = len(rows) == len(expected_identity) == 5120 \
        and observed_identity == expected_identity
    score_checks = True
    for row in rows:
        for arm in ARMS:
            value = row["arms"][arm]
            score_checks &= value["correct"] == int(value["prediction"] == row["answer"])
            score_checks &= value["memory_size"] >= 1
            if row["final"] and arm in GENERATIVE_ARMS:
                _, encoded = encode_candidates(
                    tokenizer, render_query(row["query"]),
                    sessions_by_id[row["session_id"]]["candidates"])
                score_checks &= value["generation"]["token_exact"] == int(
                    value["generation"]["ids"] == encoded[row["answer"]])
            else:
                score_checks &= "generation" not in value

    recomputed = aggregate(rows, decisions)
    hash_checks = all(
        summary["rows_sha256"][path.name] == sha(path)
        for path in sorted(root.glob("rows_shard*.jsonl"))) and all(
        summary["decisions_sha256"][path.name] == sha(path)
        for path in sorted(root.glob("decisions_shard*.jsonl"))) and all(
        resource["rows_sha256"] == sha(root / f"rows_shard{index}.jsonl")
        and resource["decisions_sha256"] == sha(root / f"decisions_shard{index}.jsonl")
        for index, resource in enumerate(resources))
    release_hashes = {
        "plan": sha(repo / PLAN),
        "factor_summary": sha(factor_root / "result" / "summary.json"),
        "factor_audit": sha(factor_root / "execution_audit.json"),
        "factor_checkpoint": sha(factor_path),
        "data_manifest": sha(data_root / "manifest.json"),
        "v5_checkpoint": sha(v5_path),
        "context_transforms": sha(context_root / "transforms.pt"),
        "scope_artifact": sha(scope_root / "scope_gates.pt"),
    }
    terminal = ("completed_qwen_natural_forgetting_online_v3_development_pass"
                if recomputed["gate"]["passed"] else
                "completed_qwen_natural_forgetting_online_v3_development_stop")
    checks = {
        "terminal_status_and_exit": (root / "terminal_status.txt").read_text().strip()
            == terminal and (root / "runner_exit_code.txt").read_text().strip() == "0",
        "release_before_development_and_hashes": release["protocol"] == PROTOCOL
            and not release["development_loaded"] and release["hashes"] == release_hashes
            and release_hashes["v5_checkpoint"] == EXPECTED_V5,
        "complete_subject_disjoint_development_rows": row_checks,
        "state_transition_reconstruction": transition_checks,
        "feature_reconstruction": feature_checks,
        "checkpoint_probability_and_action_replay": probability_checks,
        "score_and_generation_consistency": score_checks,
        "artifact_hashes": hash_checks,
        "independent_aggregate_bootstrap_and_gate": all(
            recomputed[key] == summary[key] for key in recomputed),
        "heldout_seals": summary["development_scored"]
            and not summary["test_loaded"] and not summary["external_benchmarks_loaded"],
    }
    return {"protocol": PROTOCOL + "_audit", "checks": checks,
            "passed": all(checks.values()), "recomputed": recomputed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "factor_root", "data_root", "v5_root", "model",
                 "tokenizer", "context_root", "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); result = audit(args)
    write_json(Path(args.output), result); print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
