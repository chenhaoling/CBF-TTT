"""Trace joint actions through gap chunks with later writes enabled or frozen."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from cbf_ttt.experiment import load_scenarios
from cbf_ttt.runtime import CBFSession
from tasks.cbf_ttt import _load_model


ACTIONS = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
GAP_POLICIES = ("normal_11", "freeze_10")


def _memory_ratio(session: CBFSession) -> float:
    ratios = []
    for layer_idx in session.layers:
        memory = session.cache.cbf_memory.get(layer_idx)
        if memory is None:
            continue
        base = session.model.model.layers[layer_idx].mlp.down_proj.weight
        ratios.append((memory.float().square().sum() /
                       base.float().square().sum().clamp_min(1e-8)).item())
    return math.sqrt(sum(ratios))


def diagnose(model, scenarios: list[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            if scenario.get("objective") != "joint_v2":
                raise ValueError("gap diagnosis needs joint_v2 scenarios")
            session = CBFSession(model)
            chunk = session.chunk_size
            context = scenario["context_ids"]
            if len(context) != 2 * chunk or len(scenario["futures"]) != 2:
                raise ValueError("expected exactly two context chunks and two futures")
            future = scenario["futures"][1]
            gap_ids = future["continuation_ids"]
            gap_chunks = future["gap_chunks"]
            if gap_chunks < 1 or len(gap_ids) != gap_chunks * chunk:
                raise ValueError("gap continuation must contain full chunks")
            session.observe(context[:chunk])
            session.commit_both(1.0, 1.0)
            session.observe(context[chunk:])
            started = time.perf_counter()
            trajectories = {}
            for alpha, gate in ACTIONS:
                pending = session.clone()
                pending.commit_both(alpha, gate)
                for policy in GAP_POLICIES:
                    branch = pending.clone()
                    points = []
                    for gap_index in range(gap_chunks + 1):
                        losses = {query["kind"]: branch.score_answer(
                            query["query_ids"], query["answer_ids"]
                        ) for query in future["queries"]}
                        points.append({"gap_chunks": gap_index, "losses": losses,
                                       "mean_nll": sum(losses.values()) / len(losses),
                                       "memory_ratio": _memory_ratio(branch)})
                        if gap_index < gap_chunks:
                            branch.observe(gap_ids[gap_index * chunk:(gap_index + 1) * chunk])
                            branch.commit_both(1.0, 1.0 if policy == "normal_11" else 0.0)
                    trajectories[f"{int(alpha)}{int(gate)}/{policy}"] = points
            if next(model.parameters()).device.type == "cuda":
                torch.cuda.synchronize()
            row = {"protocol": "gap_diagnostic_v1", "id": scenario["id"],
                   "group_id": scenario["group_id"], "split": scenario["split"],
                   "regime": scenario["regime"], "gap_chunks": gap_chunks,
                   "trajectories": trajectories,
                   "diagnostic_time_s": time.perf_counter() - started,
                   "peak_reserved_gib": (torch.cuda.max_memory_reserved() / 1024**3
                                         if next(model.parameters()).device.type == "cuda" else None)}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
    return len(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    scenarios = load_scenarios(args.data, args.split)[:args.limit]
    model = _load_model(args.model, args.device, args.dtype)
    print(json.dumps({"diagnosed": diagnose(model, scenarios, args.output),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
