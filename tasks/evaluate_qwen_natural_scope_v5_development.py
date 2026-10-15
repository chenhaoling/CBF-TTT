"""One-time frozen natural-development evaluation of the V5 scope reader."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_qwen_natural_causal_memory_v1 import (
    SCALE,
    SCOPE_SEED,
    active_queries,
    advance,
    chain_map,
    checkpoint_label,
    primary_count,
)
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    candidate_scores,
    generate_fixed,
    prepare,
    value_rows,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_residual_gate_v4 import ResidualGate
from tasks.train_qwen_natural_scope_transfer_v2 import (
    bootstrap_group,
    build_examples,
    build_scorer,
)


PROTOCOL = "qwen_natural_scope_v5_development"
PLAN = "QWEN_NATURAL_SCOPE_V5_DEVELOPMENT_PLAN.md"
ARMS = ("v5_auto", "v1_auto", "oracle_oracle", "zero")
EXPECTED_CHECKPOINT_SHA256 = "3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc"


def build_trace(session):
    chains = chain_map(session); memory, active, snapshots, rows, decisions = [], {}, [], [], []
    count = primary_count(session)
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]; version = chain["versions"][event["version"]]
        memory, decision = advance("oracle", memory, event, chain, version, None, None)
        decision["step"] = step; decisions.append(decision)
        active[event["chain_id"]] = event["version"]
        phase = checkpoint_label(step, session)
        if phase is None:
            continue
        final = phase == f"update_{count}"
        snapshots.append({"phase": phase, "step": step, "active": dict(active),
                          "states": {"oracle": list(memory)}})
        for query in active_queries(session, active):
            rows.append({"protocol": PROTOCOL, "session_id": session["id"],
                         "split": "dev", "phase": phase, "final": final,
                         **query, "candidates": session["candidates"]})
    return rows, {"protocol": PROTOCOL, "session_id": session["id"], "split": "dev",
                  "decisions": {"oracle": decisions}, "snapshots": snapshots}


def load_v5_gate(path, base_state):
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("protocol") != "qwen_natural_scope_constrained_v5":
        raise ValueError("unexpected V5 reader checkpoint")
    selected = artifact["selected"]
    gate = ResidualGate(artifact["feature_dim"], selected["hidden"], base_state,
                        torch.device("cpu"))
    gate.first.load_state_dict(artifact["state"]["first"])
    gate.second.load_state_dict(artifact["state"]["second"])
    return gate, artifact["threshold"], selected


def selection(query, memory, scorer, gate, threshold):
    import torch

    frozen, feature = scorer.encode(query, [row["prompt"] for row in memory], SCOPE_SEED)
    with torch.no_grad():
        probability = float(gate(feature.unsqueeze(0)).sigmoid()[0])
    v5 = memory[frozen["top"]] if probability >= threshold else None
    v1 = memory[frozen["top"]] if frozen["retrieve"] else None
    return {"v5_auto": v5, "v1_auto": v1}, {
        "top": frozen["top"], "v5_probability": probability,
        "v5_retrieve": probability >= threshold,
        "v1_probability": frozen["retrieve_probability"],
        "v1_retrieve": bool(frozen["retrieve"]),
    }


def score_natural_rows(model, tokenizer, rows, examples, scorer, gate, threshold):
    import torch

    output = []; started = time.perf_counter()
    natural = sorted((row for row in examples if row["kind"] == "natural"),
                     key=lambda row: row["row_index"])
    if len(natural) != len(rows):
        raise ValueError("development natural examples do not align")
    with torch.no_grad():
        for index, (source, example) in enumerate(zip(rows, natural)):
            memory = example["memory"]
            automatic, scope = selection(source["query"], memory, scorer, gate, threshold)
            oracle = [row for row in memory if row["chain_id"] == source["chain_id"]
                      and row["version"] == source["active_version"]]
            if len(oracle) != 1:
                raise ValueError("development oracle record is not unique")
            selected = {**automatic, "oracle_oracle": oracle[0], "zero": None}
            prepared = prepare(model, tokenizer, source["query"], source["candidates"])
            cache = {}
            for record in selected.values():
                key = None if record is None else record["target"]
                if key in cache:
                    continue
                length = max(len(ids) for ids in prepared["encoded"].values())
                residual = (torch.zeros((length, model.config.hidden_size), dtype=torch.float32,
                                        device=model.device) if key is None else
                            value_rows(model, prepared["encoded"][key]))
                scores = candidate_scores(model, prepared, residual, SCALE)
                prediction = min(source["candidates"], key=lambda value: (scores[value], value))
                result = {"scores": scores, "prediction": prediction,
                          "candidate_correct": int(prediction == source["answer"]),
                          "answer_nll": scores[source["answer"]]}
                if source["final"]:
                    generated = generate_fixed(
                        model, tokenizer, prepared, residual, SCALE,
                        len(prepared["encoded"][source["answer"]]),
                    )
                    generated["token_exact"] = int(
                        generated["ids"] == prepared["encoded"][source["answer"]]
                    )
                    result["generation"] = generated
                cache[key] = result
            arms = {}
            for arm in ARMS:
                record = selected[arm]; key = None if record is None else record["target"]
                result = cache[key]
                arms[arm] = {
                    **result,
                    "selected_chain_id": None if record is None else record["chain_id"],
                    "selected_version": None if record is None else record["version"],
                    "selection_correct": int(record is not None
                        and record["chain_id"] == source["chain_id"]
                        and record["version"] == source["active_version"]),
                    "stale_selection": int(record is not None
                        and record["chain_id"] == source["chain_id"]
                        and record["version"] != source["active_version"]),
                    "wrong_chain_selection": int(record is not None
                        and record["chain_id"] != source["chain_id"]),
                    "defer": int(record is None),
                }
            output.append({**source, "arms": arms, "scope": scope,
                           "seconds": time.perf_counter() - started})
            if (index + 1) % 256 == 0:
                print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    return output


def score_no_match(examples, scorer, gate, threshold):
    import torch

    output = []
    with torch.no_grad():
        for row in examples:
            if row["kind"] != "no_match":
                continue
            frozen, feature = scorer.encode(row["query"], row["keys"], SCOPE_SEED)
            probability = float(gate(feature.unsqueeze(0)).sigmoid()[0])
            output.append({"session_id": row["session_id"], "phase": row["phase"],
                           "surface": row["surface"],
                           "v5_probability": probability,
                           "v5_defer": int(probability < threshold),
                           "v1_probability": frozen["retrieve_probability"],
                           "v1_defer": int(not frozen["retrieve"])})
    return output


def arm_metric(rows, arm):
    final = [row for row in rows if row["final"]]
    return {
        "candidate_accuracy": statistics.fmean(row["arms"][arm]["candidate_correct"] for row in rows),
        "answer_nll": statistics.fmean(row["arms"][arm]["answer_nll"] for row in rows),
        "selection_accuracy": statistics.fmean(row["arms"][arm]["selection_correct"] for row in rows),
        "defer_rate": statistics.fmean(row["arms"][arm]["defer"] for row in rows),
        "final_candidate_accuracy": statistics.fmean(
            row["arms"][arm]["candidate_correct"] for row in final),
        "final_generation_exact": statistics.fmean(
            row["arms"][arm]["generation"]["token_exact"] for row in final),
        "by_surface": {surface: statistics.fmean(
            row["arms"][arm]["candidate_correct"] for row in rows if row["surface"] == surface
        ) for surface in ("exact", "rephrase")},
    }


def aggregate(rows, negatives):
    metrics = {arm: arm_metric(rows, arm) for arm in ARMS}
    groups = {}
    for group in sorted({row["group_id"] for row in rows}):
        part = [row for row in rows if row["group_id"] == group]
        groups[group] = statistics.fmean(
            row["arms"]["v5_auto"]["candidate_correct"]
            - row["arms"]["v1_auto"]["candidate_correct"] for row in part
        )
    comparison = {"candidate_accuracy_gain": statistics.fmean(groups.values()),
                  "bootstrap_95ci": bootstrap_group(groups, 202610211),
                  "subject_groups": len(groups)}
    no_match = {arm: statistics.fmean(row[f"{arm}_defer"] for row in negatives)
                for arm in ("v5", "v1")}
    gap = (metrics["oracle_oracle"]["candidate_accuracy"]
           - metrics["v5_auto"]["candidate_accuracy"])
    checks = {
        "v5_no_match_specificity_at_least_point_90": no_match["v5"] >= .90,
        "v5_candidate_accuracy_at_least_point_90":
            metrics["v5_auto"]["candidate_accuracy"] >= .90,
        "v5_oracle_gap_at_most_point_07": gap <= .07,
        "v5_final_candidate_at_least_point_88":
            metrics["v5_auto"]["final_candidate_accuracy"] >= .88,
        "v5_final_generation_at_least_point_70":
            metrics["v5_auto"]["final_generation_exact"] >= .70,
        "v5_rephrase_at_least_point_85":
            metrics["v5_auto"]["by_surface"]["rephrase"] >= .85,
        "v5_gain_ci_lower_above_zero": comparison["bootstrap_95ci"][0] > 0,
    }
    return {"metrics": metrics, "no_match_specificity": no_match,
            "v5_oracle_gap": gap, "comparison": comparison,
            "gate": {"checks": checks, "passed": all(checks.values())}}


def run(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.v5_root) / "result" / "constrained_scope_gate.pt"
    if sha(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("unexpected V5 checkpoint")
    _, sessions = load(args.data_root, "dev")
    rows, states = [], []
    for session in sessions:
        part, state = build_trace(session); rows.extend(part); states.append(state)
    if len(rows) != 5120 or len(states) != 64:
        raise ValueError("unexpected development trace size")
    examples = build_examples(sessions, rows, states)
    source, scope = Path(args.context_root), Path(args.scope_root)
    source_summary = json.loads((source / "summary.json").read_text())
    transforms = torch.load(source / "transforms.pt", map_location="cpu", weights_only=False)
    scope_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(202610211); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    scorer, feature_meta = build_scorer(
        examples, model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact, args.batch_size,
    )
    gate, threshold, selected = load_v5_gate(checkpoint, scope_artifact["heads"]["7301"])
    scored = score_natural_rows(model, tokenizer, rows, examples, scorer, gate, threshold)
    negatives = score_no_match(examples, scorer, gate, threshold)
    with (root / "development_rows.jsonl").open("x") as sink:
        for row in scored: sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "development_no_match.jsonl").open("x") as sink:
        for row in negatives: sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = aggregate(scored, negatives)
    result.update({
        "protocol": PROTOCOL, "stage": "development", "sessions": len(sessions),
        "rows": len(scored), "no_match_rows": len(negatives),
        "reader_selected": selected, "reader_threshold": threshold,
        "feature_meta": feature_meta,
        "resources": {"seconds": time.perf_counter() - started,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30},
        "development_rows_sha256": sha(root / "development_rows.jsonl"),
        "development_no_match_sha256": sha(root / "development_no_match.jsonl"),
        "reader_checkpoint_sha256": sha(checkpoint), "plan_sha256": sha(PLAN),
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "calibration_loaded": False, "development_scored": True,
        "test_loaded": False, "external_benchmarks_loaded": False,
    })
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data_root", "v5_root", "model", "tokenizer", "context_root",
                 "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
