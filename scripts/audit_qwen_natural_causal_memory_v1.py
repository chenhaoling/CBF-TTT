"""Independently audit natural causal-memory construction, replay and gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from tasks.build_qwen_natural_causal_memory_v1 import (
    EXPECTED,
    PROTOCOL as DATA_PROTOCOL,
    SPLITS,
    load,
    materialize_split,
)
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    ARMS,
    AUTO_POLICIES,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    FIXED_CONTROLS,
    FORGETTING_THRESHOLD,
    PROTOCOL,
    SCALE,
    SCOPE_SEED,
    STATE_POLICIES,
    active_queries,
    action_correct,
    checkpoint_label,
    record_for,
)
from tasks.evaluate_qwen_natural_value_bridge_v1 import render_query
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


AUDIT_PROTOCOL = "qwen_natural_causal_memory_v1_audit"
EXPECTED_HASHES = {
    "manifest.json": "c5056824efa867532a33ff460976aaaf6d709ff1900acc562fc26c4b5ae1026f",
    "calibration.jsonl": "04bcc19bfe0524b16294eb529fd1c168148743cf58987039abfab99bf6b3243d",
    "dev.jsonl": "79f6c29b49981dabd1f90aee5fe0dce748d87fc6351efd29f8fface5dbb4301d",
    "context_summary.json": "843a0c23b15125e2298f71fcf5d99ef3f96d997bc5a6616ee102d6d949e8cafe",
    "transforms.pt": "6f6a696005185d80d6b8f468fa6442146dc8d9683e232e7a591676ba9515c378",
    "scope_training.json": "ff9186cf16df7cbab91536e77c588a29aaadc5f0ed06618eb2054054d6799297",
    "scope_gates.pt": "0ed4dab8b77fd0fb91fdd49720c783261ab2fd5c64cd705ebe36346d99615619",
    "controller_training.json": "63aa061bfd512aa7541bb0bf0a65fbbd4427a8cf760b979c00bdc20ae6788580",
    "controller.pt": "fc766618db0e6847da893c4db0866eaafa98bb94889f58d6ed0b9fc6b77538e1",
    "boundary_summary.json": "7d24223491b2fd2f73fafbce4e3dece1411c70633e01631d6c6f0b11f5902a2c",
    "boundary_audit.json": "b88776f1907f874d6171d8df95ed9329795174fd7a6cacac6644aaba4fc2eeec",
    "value_summary.json": "f13dd631d9378a27006e6289218270ee5b952f3d213984a83ae24fdf8da8208a",
    "value_audit.json": "c27933a071b1b2c1d719fae76e4b63d06f848e8087d101b3596ca8c30d4b49ee",
}


def close(left, right, tolerance=1e-12):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def dependency_paths(args):
    source, context, scope, controller, boundary, value = map(Path, (
        args.source_root, args.context_root, args.scope_root, args.controller_root,
        args.boundary_root, args.value_root,
    ))
    return {
        "manifest.json": source / "manifest.json",
        "calibration.jsonl": source / "calibration.jsonl",
        "dev.jsonl": source / "dev.jsonl",
        "context_summary.json": context / "summary.json",
        "transforms.pt": context / "transforms.pt",
        "scope_training.json": scope / "training.json",
        "scope_gates.pt": scope / "scope_gates.pt",
        "controller_training.json": controller / "training.json",
        "controller.pt": controller / "counterfactual_forgetting_action_oof_gate.pt",
        "boundary_summary.json": boundary / "diagnostic/summary.json",
        "boundary_audit.json": boundary / "execution_audit.json",
        "value_summary.json": value / "development_summary.json",
        "value_audit.json": value / "execution_audit.json",
    }


def audit_dependencies(root, args):
    paths = dependency_paths(args)
    observed = {name: sha(path) for name, path in paths.items()}
    registered = json.loads((Path(root) / "dependencies.json").read_text())
    expected_registered = {str(path): observed[name] for name, path in paths.items()}
    value_summary = json.loads(paths["value_summary.json"].read_text())
    value_audit = json.loads(paths["value_audit.json"].read_text())
    boundary_summary = json.loads(paths["boundary_summary.json"].read_text())
    boundary_audit = json.loads(paths["boundary_audit.json"].read_text())
    return all((
        observed == EXPECTED_HASHES,
        registered == expected_registered,
        value_summary["gate"]["passed"], value_summary["selected_scale"] == SCALE,
        value_audit["passed"], boundary_audit["audit_passed"],
        boundary_summary["selected_passing_original_gate"]["development_gate"]["passed"],
        float(boundary_summary["selected_passing_original_gate"]["threshold"])
            == FORGETTING_THRESHOLD,
    )), expected_registered


def audit_data(root, source_root):
    root = Path(root); manifest = json.loads((root / "data/manifest.json").read_text())
    checks = [manifest["protocol"] == DATA_PROTOCOL, not manifest["test_loaded"],
              not manifest["test_scored"], not manifest["external_benchmarks_loaded"]]
    for split in SPLITS:
        expected = materialize_split(source_root, split)
        path = root / f"data/{split}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected)
        checks.extend((path.read_text() == content,
                       manifest["files_sha256"][path.name] == sha(path)))
    return all(checks), manifest


def expected_rows(session):
    chains = {chain["id"]: chain for chain in session["chains"]}; active = {}; output = []
    for step, event in enumerate(session["events"]):
        active[event["chain_id"]] = event["version"]
        phase = checkpoint_label(step, session)
        if phase is None:
            continue
        for query in active_queries(session, active):
            output.append((session["id"], phase, query["chain_id"], query["surface"]))
    return output


def audit_state(state, session):
    chains = {chain["id"]: chain for chain in session["chains"]}
    memories = {policy: [] for policy in STATE_POLICIES}
    recorded_snapshots = {row["phase"]: row for row in state["snapshots"]}
    seen_phases = set()
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        for policy in STATE_POLICIES:
            decision = state["decisions"][policy][step]
            before = [row["chain_id"] for row in memories[policy]]
            if decision["before"] != before or decision["chain_id"] != event["chain_id"] \
                    or decision["version"] != event["version"] \
                    or decision["event_kind"] != event["kind"]:
                return False
            work = list(memories[policy])
            for identifier in decision["deleted"]:
                matches = [index for index, row in enumerate(work)
                           if row["chain_id"] == identifier]
                if not matches:
                    return False
                work.pop(matches[0])
            work.append(record_for(chain, version)); memories[policy] = work
            if decision["after"] != [row["chain_id"] for row in work] \
                    or decision["correct"] != int(action_correct(event, decision["deleted"])):
                return False
        phase = checkpoint_label(step, session)
        if phase is not None:
            if phase not in recorded_snapshots:
                return False
            snapshot = recorded_snapshots[phase]; seen_phases.add(phase)
            for policy in STATE_POLICIES:
                recorded = snapshot["states"][policy]
                # A retain error can leave two versions with the same chain ID.
                # Decisions intentionally record only public chain IDs, so the
                # exact deleted version is ambiguous. Validate the ID-level
                # transition, every full record and the newly appended record,
                # then carry the recorded full state into the next step.
                if [row["chain_id"] for row in recorded] != decision_after(
                        state["decisions"][policy][step]):
                    return False
                valid = all(
                    any(row == record_for(chain, version)
                        for chain in session["chains"]
                        for version in chain["versions"])
                    for row in recorded
                )
                if not valid or not recorded or recorded[-1] != record_for(chain, version):
                    return False
                memories[policy] = list(recorded)
    return seen_phases == set(recorded_snapshots)


def decision_after(decision):
    return decision["after"]


def audit_row(row, episode, snapshot, tokenizer):
    query_version = next(
        version for chain in episode["chains"] if chain["id"] == row["chain_id"]
        for version in chain["versions"] if version["version"] == row["active_version"]
    )
    expected_query = query_version["prompt"] if row["surface"] == "exact" \
        else query_version["rephrase_prompt"]
    prompt_ids, encoded = encode_candidates(
        tokenizer, render_query(expected_query), episode["candidates"]
    )
    if any((row["protocol"] != PROTOCOL, row["split"] != episode["split"],
            row["query"] != expected_query, row["answer"] != query_version["target"],
            row["candidates"] != episode["candidates"], row["candidate_token_ids"] != encoded,
            row["prompt_tokens"] != len(prompt_ids),
            row["answer_tokens"] != len(encoded[row["answer"]]),
            set(row["arms"]) != set(ARMS), not math.isfinite(row["seconds"]))):
        return False
    policy_memory = {arm: snapshot["states"][policy]
                     for arm, policy in AUTO_POLICIES.items()}
    policy_memory.update({"oracle_oracle": snapshot["states"]["oracle"],
                          "oracle_dense": snapshot["states"]["oracle"], "zero": []})
    for arm, value in row["arms"].items():
        prediction = min(episode["candidates"], key=lambda answer: (
            value["scores"][answer], answer
        ))
        selected = None
        if value["selected_chain_id"] is not None:
            matches = [record for record in policy_memory[arm]
                       if record["chain_id"] == value["selected_chain_id"]
                       and record["version"] == value["selected_version"]
                       and record["target"] == value["selected_target"]]
            if len(matches) != 1:
                return False
            selected = matches[0]
        expected_size = len(policy_memory[arm])
        if any((set(value["scores"]) != set(episode["candidates"]),
                not all(math.isfinite(score) for score in value["scores"].values()),
                value["prediction"] != prediction,
                value["candidate_correct"] != int(prediction == row["answer"]),
                not close(value["answer_nll"], value["scores"][row["answer"]]),
                value["selection_correct"] != int(
                    selected is not None and selected["target"] == row["answer"]),
                value["stale_selection"] != int(selected is not None
                    and selected["chain_id"] == row["chain_id"]
                    and selected["version"] != row["active_version"]),
                value["wrong_chain_selection"] != int(selected is not None
                    and selected["chain_id"] != row["chain_id"]),
                value["defer"] != int(selected is None),
                value["memory_size"] != expected_size)):
            return False
        if row["final"]:
            generation = value.get("generation")
            if not generation or generation["tokens"] != generation["conditioned_length"] \
                    or generation["conditioned_length"] != len(encoded[row["answer"]]) \
                    or generation["token_exact"] != int(
                        generation["ids"] == encoded[row["answer"]]):
                return False
        elif "generation" in value:
            return False
    return True


def read_stage(root, data_root, tokenizer, stage):
    root = Path(root); _, sessions = load(data_root, stage)
    by_id = {session["id"]: session for session in sessions}
    order = {session["id"]: index for index, session in enumerate(sessions)}
    expected = {item for session in sessions for item in expected_rows(session)}
    rows, states, resources = [], [], []; shard_ok = True
    for shard in (0, 1):
        row_path = root / f"{stage}_shard{shard}.jsonl"
        state_path = root / f"{stage}_states_shard{shard}.jsonl"
        resource_path = root / f"{stage}_resource_shard{shard}.json"
        part = [json.loads(line) for line in row_path.read_text().splitlines() if line]
        state_part = [json.loads(line) for line in state_path.read_text().splitlines() if line]
        resource = json.loads(resource_path.read_text())
        expected_sessions = sum(index % 2 == shard for index in range(EXPECTED[stage]["sessions"]))
        shard_ok &= all((resource["protocol"] == PROTOCOL, resource["stage"] == stage,
                         resource["shard"] == shard, resource["num_shards"] == 2,
                         resource["sessions"] == expected_sessions,
                         resource["rows"] == len(part), resource["states"] == len(state_part),
                         resource["rows_sha256"] == sha(row_path),
                         resource["states_sha256"] == sha(state_path),
                         resource["data_manifest_sha256"] == sha(Path(data_root) / "manifest.json"),
                         resource["scale"] == SCALE, resource["scope_seed"] == SCOPE_SEED,
                         resource["forgetting_threshold"] == FORGETTING_THRESHOLD,
                         not resource["loading_info"]["missing_keys"],
                         not resource["loading_info"]["unexpected_keys"],
                         not resource["loading_info"]["mismatched_keys"],
                         resource["calibration_scored"],
                         resource["development_scored"] == (stage == "dev"),
                         not resource["test_loaded"], not resource["test_scored"],
                         not resource["external_benchmarks_loaded"]))
        shard_ok &= all(order[row["session_id"]] % 2 == shard and row["shard"] == shard
                        for row in part)
        shard_ok &= all(order[state["session_id"]] % 2 == shard and state["shard"] == shard
                        for state in state_part)
        rows.extend(part); states.extend(state_part); resources.append(resource)
    coverage = len(rows) == len(expected) and {
        (row["session_id"], row["phase"], row["chain_id"], row["surface"])
        for row in rows
    } == expected and len(states) == len(sessions)
    state_by_session = {state["session_id"]: state for state in states}
    state_ok = len(state_by_session) == len(states) and all(
        audit_state(state_by_session[identifier], episode)
        for identifier, episode in by_id.items()
    )
    snapshot_map = {
        (state["session_id"], snapshot["phase"]): snapshot
        for state in states for snapshot in state["snapshots"]
    }
    row_ok = all(audit_row(
        row, by_id[row["session_id"]], snapshot_map[row["session_id"], row["phase"]], tokenizer
    ) for row in rows)
    return rows, states, resources, shard_ok, coverage, state_ok, row_ok


def interval(values, seed):
    groups = sorted(values); rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    draws.sort()
    return [draws[max(0, math.ceil(.025 * len(draws)) - 1)],
            draws[min(len(draws) - 1, math.ceil(.975 * len(draws)) - 1)]]


def metric(rows, arm):
    final = [row for row in rows if row["final"]]
    acc = lambda part: statistics.fmean(row["arms"][arm]["candidate_correct"] for row in part)
    return {
        "candidate_accuracy": acc(rows),
        "answer_nll": statistics.fmean(row["arms"][arm]["answer_nll"] for row in rows),
        "final_candidate_accuracy": acc(final),
        "final_generation_exact": statistics.fmean(
            row["arms"][arm]["generation"]["token_exact"] for row in final),
        "selection_accuracy": statistics.fmean(row["arms"][arm]["selection_correct"] for row in rows),
        "stale_selection_rate": statistics.fmean(row["arms"][arm]["stale_selection"] for row in rows),
        "wrong_chain_selection_rate": statistics.fmean(
            row["arms"][arm]["wrong_chain_selection"] for row in rows),
        "defer_rate": statistics.fmean(row["arms"][arm]["defer"] for row in rows),
        "mean_memory_size": statistics.fmean(row["arms"][arm]["memory_size"] for row in rows),
        "by_surface": {surface: acc([row for row in rows if row["surface"] == surface])
                       for surface in ("exact", "rephrase")},
        "by_chain_kind": {kind: acc([row for row in rows if row["chain_kind"] == kind])
                          for kind in ("primary", "auxiliary")},
        "by_phase": {phase: acc([row for row in rows if row["phase"] == phase])
                     for phase in sorted({row["phase"] for row in rows})},
    }


def compare(rows, left, right, seed):
    groups = sorted({row["group_id"] for row in rows})
    values = {group: statistics.fmean(
        row["arms"][left]["candidate_correct"] - row["arms"][right]["candidate_correct"]
        for row in rows if row["group_id"] == group) for group in groups}
    return {"candidate_accuracy_gain": statistics.fmean(values.values()),
            "bootstrap_95ci": interval(values, seed), "subject_groups": len(groups)}


def audit_aggregate(rows, states):
    metrics = {arm: metric(rows, arm) for arm in ARMS}
    actions = {}
    for policy in STATE_POLICIES:
        decisions = [item for state in states for item in state["decisions"][policy]]
        actions[policy] = {"accuracy": statistics.fmean(row["correct"] for row in decisions),
                           "events": len(decisions), "by_event_kind": {
            kind: statistics.fmean(row["correct"] for row in decisions if row["event_kind"] == kind)
            for kind in ("primary_initial", "hard_retain", "primary_update")}}
    strongest = max(FIXED_CONTROLS, key=lambda arm: metrics[arm]["candidate_accuracy"])
    learned = compare(rows, "learned_auto", strongest, BOOTSTRAP_SEED)
    oracle = {arm: compare(rows, "oracle_auto", arm, BOOTSTRAP_SEED + index + 1)
              for index, arm in enumerate(("retain_auto", "global_auto", "oracle_dense", "zero"))}
    checks = {
        "oracle_oracle_overall_at_least_point_95": metrics["oracle_oracle"]["candidate_accuracy"] >= .95,
        "oracle_oracle_final_at_least_point_95": metrics["oracle_oracle"]["final_candidate_accuracy"] >= .95,
        "oracle_auto_overall_at_least_point_88": metrics["oracle_auto"]["candidate_accuracy"] >= .88,
        "oracle_auto_gap_at_most_point_08": metrics["oracle_oracle"]["candidate_accuracy"]
            - metrics["oracle_auto"]["candidate_accuracy"] <= .08,
        "learned_overall_at_least_point_68": metrics["learned_auto"]["candidate_accuracy"] >= .68,
        "learned_primary_auxiliary_at_least_point_65":
            min(metrics["learned_auto"]["by_chain_kind"].values()) >= .65,
        "learned_final_generation_at_least_point_55":
            metrics["learned_auto"]["final_generation_exact"] >= .55,
        "learned_beats_strongest_fixed_by_point_02": learned["candidate_accuracy_gain"] >= .02,
        "learned_gain_ci_lower_above_zero": learned["bootstrap_95ci"][0] > 0,
        "oracle_auto_beats_all_controls": all(
            value["candidate_accuracy_gain"] > 0 for value in oracle.values()),
        "oracle_control_ci_lowers_above_zero": all(
            value["bootstrap_95ci"][0] > 0 for value in oracle.values()),
    }
    return {"metrics": metrics, "actions": actions, "strongest_fixed_control": strongest,
            "learned_comparison": learned, "oracle_control_comparisons": oracle,
            "gate": {"checks": checks, "passed": all(checks.values())}}


def audit_summary(summary, recomputed, stage, rows, states, resources):
    keys = ("metrics", "actions", "strongest_fixed_control", "learned_comparison",
            "oracle_control_comparisons", "gate")
    return all((close({key: summary[key] for key in keys}, recomputed),
                summary["protocol"] == PROTOCOL, summary["stage"] == stage,
                summary["rows"] == len(rows), summary["sessions"] == len(states),
                summary["scale"] == SCALE, summary["scope_seed"] == SCOPE_SEED,
                summary["forgetting_threshold"] == FORGETTING_THRESHOLD,
                close(summary["resources"]["two_gpu_critical_path_seconds"],
                      max(row["seconds"] for row in resources)),
                not summary["test_loaded"], not summary["test_scored"],
                not summary["external_benchmarks_loaded"]))


def run(args):
    from transformers import AutoTokenizer

    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    dependency_ok, dependencies = audit_dependencies(root, args)
    data_ok, data_manifest = audit_data(root, args.source_root)
    cal = read_stage(root, root / "data", tokenizer, "calibration")
    cal_rows, cal_states, cal_resources = cal[:3]
    cal_recomputed = audit_aggregate(cal_rows, cal_states)
    cal_summary = json.loads((root / "calibration_summary.json").read_text())
    cal_summary_ok = audit_summary(
        cal_summary, cal_recomputed, "calibration", cal_rows, cal_states, cal_resources)
    dev_present = cal_recomputed["gate"]["passed"]
    dev_checks = (True, True, True, True); dev_recomputed = None; dev_summary_ok = True
    if dev_present:
        dev = read_stage(root, root / "data", tokenizer, "dev")
        dev_rows, dev_states, dev_resources = dev[:3]
        dev_checks = dev[3:]
        dev_recomputed = audit_aggregate(dev_rows, dev_states)
        dev_summary = json.loads((root / "dev_summary.json").read_text())
        dev_summary_ok = audit_summary(
            dev_summary, dev_recomputed, "dev", dev_rows, dev_states, dev_resources)
    expected_terminal = "completed_qwen_natural_causal_memory_v1" if dev_present \
        else "completed_qwen_natural_causal_memory_v1_calibration_stop"
    terminal_ok = all(((root / "terminal_status.txt").read_text().strip() == expected_terminal,
                       (root / "runner_exit_code.txt").read_text().strip() == "0"))
    checks = {
        "terminal_status_and_exit": terminal_ok,
        "frozen_dependency_hashes_and_release_evidence": dependency_ok,
        "calibration_development_corpus_reconstruction_and_isolation": data_ok,
        "calibration_shards_coverage": cal[3] and cal[4],
        "calibration_state_transition_replay": cal[5],
        "calibration_rows_tokenization_scores_and_selection": cal[6],
        "independent_calibration_aggregate_bootstrap_and_gate": cal_summary_ok,
        "conditional_development_shards_states_rows": all(dev_checks),
        "independent_development_aggregate_bootstrap_and_gate": dev_summary_ok,
        "test_and_external_benchmarks_sealed": all((not data_manifest["test_loaded"],
            not data_manifest["test_scored"], not data_manifest["external_benchmarks_loaded"])),
    }
    result = {
        "protocol": AUDIT_PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "dependencies": dependencies,
        "data_manifest_sha256": sha(root / "data/manifest.json"),
        "calibration_summary_sha256": sha(root / "calibration_summary.json"),
        "development_summary_sha256": sha(root / "dev_summary.json") if dev_present else None,
        "calibration_recomputed": cal_recomputed,
        "development_recomputed": dev_recomputed,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("natural causal-memory audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--context-root", required=True)
    parser.add_argument("--scope-root", required=True)
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--boundary-root", required=True)
    parser.add_argument("--value-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
