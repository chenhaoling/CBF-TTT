"""Compare a pending TTT update with the answer-loss gradient of session fast memory."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cbf_ttt.experiment import load_scenarios
from cbf_ttt.runtime import CBFSession
from tasks.cbf_ttt import _load_model


SCALES = (0.25, 1.0)


def load_references(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["protocol"] != "joint_title_recall_v1" or row["id"] in rows:
                raise ValueError("duplicate or incompatible title label")
            rows[row["id"]] = row
    return rows


def answer_gradient(branch: CBFSession, query: dict) -> tuple[float, dict[int, torch.Tensor]]:
    """Differentiate title NLL only through cloned session fast-memory leaves."""
    reader = branch.clone_memory_only()
    leaves = {}
    for layer_idx in reader.layers:
        memory = reader.cache.cbf_memory.get(layer_idx)
        if memory is None:
            raise RuntimeError("the first chunk must create fast memory")
        leaf = memory.detach().clone().requires_grad_(True)
        reader.cache.cbf_memory[layer_idx] = leaf
        leaves[layer_idx] = leaf
    query_ids, answer_ids = query["query_ids"], query["answer_ids"]
    token_ids = query_ids + answer_ids[:-1]
    input_ids = torch.tensor([token_ids], device=reader.device, dtype=torch.long)
    labels = torch.tensor(answer_ids, device=reader.device, dtype=torch.long)
    with torch.enable_grad():
        hidden = reader.model.model(input_ids=input_ids, past_key_values=reader.cache,
                                    use_cache=True).last_hidden_state
        logits = reader.model.lm_head(hidden[:, len(query_ids) - 1:]).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)
        gradients = torch.autograd.grad(loss, tuple(leaves.values()))
    return float(loss.detach()), {layer_idx: gradient.detach().float()
                                  for layer_idx, gradient in zip(leaves, gradients)}


def score_perturbation(branch: CBFSession, query: dict,
                       direction: dict[int, torch.Tensor], scale: float) -> float:
    reader = branch.clone_memory_only()
    with torch.inference_mode():
        for layer_idx in reader.layers:
            old = reader.cache.cbf_memory[layer_idx]
            reader.cache.cbf_memory[layer_idx] = (
                old.float() + scale * direction[layer_idx].float()
            ).to(old.dtype)
    return reader.score_answer(query["query_ids"], query["answer_ids"])


def diagnose(model, scenarios: list[dict], output: Path,
             references: dict[str, dict] | None = None, tolerance: float = 1e-4) -> int:
    if output.exists() or tolerance < 0:
        raise ValueError("output must not already exist and tolerance must be nonnegative")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as sink:
        for scenario in scenarios:
            if scenario.get("objective") != "joint_title_recall_v1" or scenario.get("regime") not in (
                "new_only", "old_only"
            ):
                raise ValueError("query gradient diagnostic needs title new_only/old_only scenarios")
            session = CBFSession(model)
            chunk = session.chunk_size
            context = scenario["context_ids"]
            futures = scenario["futures"]
            if len(context) != 2 * chunk or len(futures) != 2 or (
                [future.get("reset_kv") for future in futures] != [False, True]
                or futures[0]["queries"] != futures[1]["queries"]
                or len(futures[1]["queries"]) != 1
            ):
                raise ValueError("expected two chunks and one identical title query per read condition")
            query = futures[1]["queries"][0]
            if query.get("kind") not in ("new_title", "old_title"):
                raise ValueError("selected query must target an observed title")
            on_cuda = session.device.type == "cuda"
            if on_cuda:
                torch.cuda.synchronize(session.device)
                torch.cuda.reset_peak_memory_stats(session.device)
            started = time.perf_counter()
            session.observe(context[:chunk])
            session.commit_both(1.0, 1.0)
            session.observe(context[chunk:])
            candidates = {layer_idx: session.cache.cbf_candidates[layer_idx].detach().float()
                          for layer_idx in session.layers}
            base = session.clone()
            base.commit_both(1.0, 0.0)
            gradient_loss, gradients = answer_gradient(base, query)
            delta_norm = math.sqrt(sum(torch.sum(value.square()).item() for value in candidates.values()))
            gradient_norm = math.sqrt(sum(torch.sum(value.square()).item() for value in gradients.values()))
            if not math.isfinite(gradient_loss) or delta_norm <= 0 or gradient_norm <= 0:
                raise ValueError(f"invalid gradient or update norm for {scenario['id']}")
            dot = sum(torch.sum(gradients[layer_idx] * candidates[layer_idx]).item()
                      for layer_idx in session.layers)
            oracle = {layer_idx: -gradients[layer_idx] * (delta_norm / gradient_norm)
                      for layer_idx in session.layers}
            base_loss = score_perturbation(base, query, candidates, 0.0)
            raw_losses = {str(scale): score_perturbation(base, query, candidates, scale)
                          for scale in SCALES}
            oracle_losses = {str(scale): score_perturbation(base, query, oracle, scale)
                             for scale in SCALES}
            if abs(base_loss - gradient_loss) > tolerance:
                raise ValueError(f"autograd/inference NLL mismatch for {scenario['id']}")
            if references is not None:
                reference = references.get(scenario["id"])
                if reference is None or reference.get("split") != scenario["split"]:
                    raise ValueError(f"missing matching reference label for {scenario['id']}")
                actions = [tuple(action) for action in reference["actions"]]
                if [future.get("reset_kv") for future in reference["future_meta"]] != [False, True]:
                    raise ValueError("reference future order mismatch")
                for scale, actual in ((0.0, base_loss), (1.0, raw_losses["1.0"])):
                    expected = reference["losses_by_future"][actions.index((1.0, scale))][1]
                    if abs(actual - expected) > tolerance:
                        raise ValueError(f"scale {scale} disagrees with title label for {scenario['id']}")
            if on_cuda:
                torch.cuda.synchronize(session.device)
            row = {
                "protocol": "query_gradient_diagnostic_v1", "id": scenario["id"],
                "group_id": scenario["group_id"], "split": scenario["split"],
                "regime": scenario["regime"], "query_kind": query["kind"],
                "base_loss": base_loss, "gradient_loss": gradient_loss,
                "raw_losses": raw_losses, "oracle_losses": oracle_losses,
                "candidate_norm": delta_norm, "gradient_norm": gradient_norm,
                "gradient_dot_candidate": dot,
                "gradient_candidate_cosine": dot / (delta_norm * gradient_norm),
                "time_s": time.perf_counter() - started,
                "peak_allocated_gib": (torch.cuda.max_memory_allocated(session.device) / 1024**3
                                       if on_cuda else None),
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
    parser.add_argument("--split", choices=("train",), default="train",
                        help="mechanism audit is restricted to previously explored train groups")
    parser.add_argument("--reference-labels", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if args.max_scenarios is not None and args.max_scenarios < 1:
        parser.error("--max-scenarios must be positive")
    scenarios = [row for row in load_scenarios(args.data, args.split)
                 if row["regime"] in ("new_only", "old_only")]
    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]
    if not scenarios:
        raise ValueError("no selected title scenarios")
    references = load_references(args.reference_labels) if args.reference_labels else None
    model = _load_model(args.model, args.device, args.dtype)
    count = diagnose(model, scenarios, Path(args.output), references)
    print(json.dumps({"diagnosed": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
