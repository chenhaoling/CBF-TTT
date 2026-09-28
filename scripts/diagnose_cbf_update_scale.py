"""Probe the direction and scale of a pending TTT update at fixed session state."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from cbf_ttt.experiment import load_scenarios
from cbf_ttt.runtime import CBFSession
from scripts.diagnose_cbf_gap import _memory_ratio
from tasks.cbf_ttt import _load_model


def parse_scales(raw: str) -> list[float]:
    values = [float(part) for part in raw.split(",")]
    if not values or any(not math.isfinite(value) for value in values) or len(set(values)) != len(values):
        raise ValueError("scales must be distinct finite values")
    if 0.0 not in values or 1.0 not in values:
        raise ValueError("scales must include 0 and 1 for label equivalence")
    return sorted(values)


def load_references(paths: list[Path]) -> dict[str, dict]:
    references = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "joint_postcutoff_v1" or row["id"] in references:
                raise ValueError("duplicate or incompatible reference label")
            references[row["id"]] = row
    return references


def _scaled_branch(session: CBFSession, scale: float) -> CBFSession:
    """Apply an unbounded scale only inside this diagnostic branch."""
    branch = session.clone()
    if set(branch.cache.cbf_candidates) != set(branch.layers):
        raise RuntimeError("missing candidate update")
    with torch.inference_mode():
        for layer_idx in branch.layers:
            delta = branch.cache.cbf_candidates[layer_idx]
            old = branch.cache.cbf_memory.get(layer_idx)
            branch.cache.cbf_memory[layer_idx] = scale * delta if old is None else old + scale * delta
    branch.cache.cbf_candidates = {}
    branch.cache.cbf_collect = False
    return branch


def diagnose(model, scenarios: list[dict], scales: list[float], output: Path,
             references: dict[str, dict] | None = None, tolerance: float = 1e-4) -> int:
    if output.exists() or tolerance < 0:
        raise ValueError("output must not already exist and tolerance must be nonnegative")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            if scenario.get("objective") != "joint_postcutoff_v1":
                raise ValueError("scale diagnostic needs postcutoff scenarios")
            session = CBFSession(model)
            chunk = session.chunk_size
            context = scenario["context_ids"]
            future = scenario["futures"]
            if len(context) != 2 * chunk or len(future) != 1 or len(future[0]["queries"]) != 1 or (
                future[0].get("gap_chunks") != 0 or future[0].get("continuation_ids")
            ):
                raise ValueError("expected two chunks and one zero-gap held-out query")
            on_cuda = session.device.type == "cuda"
            if on_cuda:
                torch.cuda.synchronize(session.device)
                torch.cuda.reset_peak_memory_stats(session.device)
            started = time.perf_counter()
            session.observe(context[:chunk])
            session.commit_both(1.0, 1.0)
            _, scalars = session.observe(context[chunk:])
            candidate_ratio = math.sqrt(sum(
                (session.cache.cbf_candidates[layer_idx].float().square().sum() /
                 model.model.layers[layer_idx].mlp.down_proj.weight.float().square().sum().clamp_min(1e-8)).item()
                for layer_idx in session.layers
            ))
            query = future[0]["queries"][0]
            losses, memory_ratios = [], []
            for scale in scales:
                branch = _scaled_branch(session, scale)
                losses.append(branch.score_answer(query["query_ids"], query["answer_ids"]))
                memory_ratios.append(_memory_ratio(branch))
            if references is not None:
                reference = references.get(scenario["id"])
                if reference is None:
                    raise ValueError("missing reference label")
                for scale, corner in ((0.0, "10"), (1.0, "11")):
                    if abs(losses[scales.index(scale)] - reference["corner_losses"][corner]) > tolerance:
                        raise ValueError(f"scale {scale} disagrees with {corner} label for {scenario['id']}")
            if on_cuda:
                torch.cuda.synchronize(session.device)
            row = {
                "protocol": "update_scale_diagnostic_v1", "id": scenario["id"],
                "group_id": scenario["group_id"], "split": scenario["split"],
                "regime": scenario["regime"], "scales": scales, "losses": losses,
                "memory_ratios": memory_ratios, "candidate_ratio": candidate_ratio,
                "chunk_surprise": float(scalars[0, 0]),
                "time_s": time.perf_counter() - started,
                "peak_reserved_gib": (torch.cuda.max_memory_reserved(session.device) / 1024**3
                                      if on_cuda else None),
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--regimes", default="new_only,old_only")
    parser.add_argument("--scales", default="-2,-1,-0.5,0,0.25,0.5,1,2")
    parser.add_argument("--reference-labels", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    regimes = set(args.regimes.split(","))
    if not regimes or not regimes <= {"new_only", "old_only"}:
        raise ValueError("regimes must be new_only and/or old_only")
    scenarios = [row for row in load_scenarios(args.data, args.split)
                 if row["regime"] in regimes]
    if not scenarios:
        raise ValueError("no selected scenarios")
    scales = parse_scales(args.scales)
    references = load_references(args.reference_labels) if args.reference_labels else None
    model = _load_model(args.model, args.device, args.dtype)
    count = diagnose(model, scenarios, scales, args.output, references)
    print(json.dumps({"diagnosed": count, "output": str(args.output), "scales": scales}))


if __name__ == "__main__":
    main()
