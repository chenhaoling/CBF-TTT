"""Diagnose reader and forgetting errors in an audited natural-memory run.

This script is deliberately score-free: it consumes the immutable JSONL rows
and state snapshots produced by ``evaluate_qwen_natural_causal_memory_v1`` and
never loads a model.  In particular, it reconstructs the frozen reader's top
record even when its retrieve/defer gate abstained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from tasks.evaluate_qwen_natural_causal_memory_v1 import read_stage
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "qwen_natural_causal_memory_v1_diagnosis"


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    def value(q):
        index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
        return values[index]
    return {"min": values[0], "p10": value(.10), "p25": value(.25),
            "median": value(.50), "p75": value(.75), "p90": value(.90),
            "max": values[-1]}


def accuracy(rows, key):
    return statistics.fmean(row[key] for row in rows) if rows else None


def snapshot_index(states):
    output = {}
    for state in states:
        for snapshot in state["snapshots"]:
            key = (state["session_id"], snapshot["phase"])
            if key in output:
                raise ValueError(f"duplicate snapshot {key}")
            output[key] = snapshot
    return output


def reader_records(rows, states, arm="oracle_auto", policy="oracle"):
    snapshots = snapshot_index(states); output = []
    for row in rows:
        result = row["arms"][arm]
        snapshot = snapshots[(row["session_id"], row["phase"])]
        memory = snapshot["states"][policy]
        top = result["scope"]["top"]
        if memory and top is None:
            raise ValueError("non-empty automatic memory has no ranked top")
        top_record = None if top is None else memory[top]
        top_correct = int(top_record is not None
                          and top_record["chain_id"] == row["chain_id"]
                          and top_record["version"] == row["active_version"])
        retrieve = int(not result["defer"])
        if top_correct and retrieve:
            category = "retrieve_correct_top"
        elif top_correct:
            category = "defer_correct_top"
        elif retrieve:
            category = "retrieve_wrong_top"
        else:
            category = "defer_wrong_top"
        oracle = row["arms"]["oracle_oracle"]
        output.append({
            "session_id": row["session_id"],
            "group_id": row["group_id"],
            "phase": row["phase"],
            "surface": row["surface"],
            "chain_kind": row["chain_kind"],
            "memory_size": len(memory),
            "retrieve": retrieve,
            "retrieve_probability": result["scope"]["retrieve_probability"],
            "top_correct": top_correct,
            "selection_correct": result["selection_correct"],
            "candidate_correct": result["candidate_correct"],
            "oracle_candidate_correct": oracle["candidate_correct"],
            "oracle_generation_exact": (
                oracle.get("generation", {}).get("token_exact") if row["final"] else None
            ),
            "category": category,
            "final": row["final"],
        })
    return output


def stratum(records, field):
    output = {}
    for value in sorted({row[field] for row in records}, key=str):
        part = [row for row in records if row[field] == value]
        output[str(value)] = {
            "rows": len(part),
            "top_accuracy": accuracy(part, "top_correct"),
            "retrieve_rate": accuracy(part, "retrieve"),
            "selection_accuracy": accuracy(part, "selection_correct"),
            "candidate_accuracy": accuracy(part, "candidate_correct"),
            "oracle_candidate_accuracy": accuracy(part, "oracle_candidate_correct"),
        }
    return output


def reader_diagnosis(rows, states):
    records = reader_records(rows, states)
    categories = Counter(row["category"] for row in records)
    correct_top = [row for row in records if row["top_correct"]]
    wrong_top = [row for row in records if not row["top_correct"]]
    final_correct_top = [row for row in correct_top if row["final"]]
    # When top is correct, force-top is identical to the already-scored
    # oracle-oracle residual, so this counterfactual requires no new model call.
    force_top_coverage = len(correct_top) / len(records)
    force_top_candidate_on_covered = accuracy(correct_top, "oracle_candidate_correct")
    force_top_generation_on_covered = accuracy(final_correct_top, "oracle_generation_exact")
    probabilities = {
        "correct_top": quantiles([row["retrieve_probability"] for row in correct_top]),
        "wrong_top": quantiles([row["retrieve_probability"] for row in wrong_top]),
        "false_negative": quantiles([
            row["retrieve_probability"] for row in correct_top if not row["retrieve"]
        ]),
    }
    return {
        "rows": len(records),
        "top_accuracy": accuracy(records, "top_correct"),
        "retrieve_rate": accuracy(records, "retrieve"),
        "selection_accuracy": accuracy(records, "selection_correct"),
        "candidate_accuracy": accuracy(records, "candidate_correct"),
        "categories": dict(sorted(categories.items())),
        "retrieve_probability": probabilities,
        "force_top_counterfactual": {
            "covered_fraction": force_top_coverage,
            "covered_rows": len(correct_top),
            "unknown_wrong_top_rows": len(wrong_top),
            "candidate_accuracy_on_covered": force_top_candidate_on_covered,
            "final_generation_exact_on_covered": force_top_generation_on_covered,
            "exact_if_covered_fraction_is_one": force_top_coverage == 1.0,
        },
        "by_surface": stratum(records, "surface"),
        "by_chain_kind": stratum(records, "chain_kind"),
        "by_phase": stratum(records, "phase"),
        "by_memory_size": stratum(records, "memory_size"),
    }


def forgetting_diagnosis(rows, states):
    row_counts = Counter()
    by_surface = defaultdict(Counter)
    for row in rows:
        learned = row["arms"]["learned_auto"]["candidate_correct"]
        cosine = row["arms"]["cosine_auto"]["candidate_correct"]
        outcome = ("both_correct" if learned and cosine else
                   "learned_only" if learned else
                   "cosine_only" if cosine else "both_wrong")
        row_counts[outcome] += 1; by_surface[row["surface"]][outcome] += 1

    action_counts = Counter(); action_by_kind = defaultdict(Counter)
    session_action_delta, session_final_delta = {}, {}
    for state in states:
        learned = state["decisions"]["learned"]
        cosine = state["decisions"]["cosine"]
        if len(learned) != len(cosine):
            raise ValueError("policy decision lengths differ")
        delta = 0
        for left, right in zip(learned, cosine):
            if (left["step"], left["chain_id"], left["version"]) != (
                    right["step"], right["chain_id"], right["version"]):
                raise ValueError("policy decisions are not aligned")
            outcome = ("both_correct" if left["correct"] and right["correct"] else
                       "learned_only" if left["correct"] else
                       "cosine_only" if right["correct"] else "both_wrong")
            action_counts[outcome] += 1; action_by_kind[left["event_kind"]][outcome] += 1
            delta += left["correct"] - right["correct"]
        session_action_delta[state["session_id"]] = delta / len(learned)
    for session_id in session_action_delta:
        part = [row for row in rows if row["session_id"] == session_id and row["final"]]
        session_final_delta[session_id] = statistics.fmean(
            row["arms"]["learned_auto"]["candidate_correct"]
            - row["arms"]["cosine_auto"]["candidate_correct"] for row in part
        )
    x = [session_action_delta[key] for key in sorted(session_action_delta)]
    y = [session_final_delta[key] for key in sorted(session_action_delta)]
    if statistics.pstdev(x) and statistics.pstdev(y):
        correlation = statistics.correlation(x, y)
    else:
        correlation = None
    return {
        "candidate_disagreement": dict(sorted(row_counts.items())),
        "candidate_disagreement_by_surface": {
            key: dict(sorted(value.items())) for key, value in sorted(by_surface.items())
        },
        "action_disagreement": dict(sorted(action_counts.items())),
        "action_disagreement_by_event_kind": {
            key: dict(sorted(value.items())) for key, value in sorted(action_by_kind.items())
        },
        "session_action_vs_final_accuracy_delta_correlation": correlation,
        "sessions": len(session_action_delta),
    }


def analyze(root, stage):
    rows, states, resources = read_stage(root, stage)
    return {
        "protocol": PROTOCOL,
        "source_protocol": rows[0]["protocol"],
        "stage": stage,
        "rows": len(rows),
        "sessions": len(states),
        "reader": reader_diagnosis(rows, states),
        "forgetting": forgetting_diagnosis(rows, states),
        "source_row_hashes": [row["rows_sha256"] for row in resources],
        "source_state_hashes": [row["states_sha256"] for row in resources],
        "model_loaded": False,
        "new_scores_produced": False,
        "development_loaded": stage == "dev",
        "test_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("calibration", "dev"), default="calibration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.stage)
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
