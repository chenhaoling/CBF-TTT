"""Validate forgetting-only policies and the content carried by session fast memory."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("full_context", "correct_memory", "wrong_memory", "empty_memory")


def paired_scenarios(scenarios: list[dict]) -> list[tuple[dict, dict]]:
    """Pair adjacent source groups within each regime for deterministic wrong-memory controls."""
    by_regime: dict[str, dict[str, dict]] = defaultdict(dict)
    for scenario in scenarios:
        regime = scenario.get("regime")
        group = scenario["group_id"]
        if not isinstance(regime, str) or not regime:
            raise ValueError(f"scenario {scenario['id']} has no regime")
        if group in by_regime[regime]:
            raise ValueError(f"regime {regime} repeats source group {group}")
        if len(scenario["futures"]) != 1:
            raise ValueError("forgetting-only validation requires exactly one future per scenario")
        by_regime[regime][group] = scenario

    regimes = sorted(by_regime)
    if not regimes:
        raise ValueError("no scenarios to pair")
    expected_groups = sorted(by_regime[regimes[0]])
    if len(expected_groups) < 2 or len(expected_groups) % 2:
        raise ValueError("each regime needs an even number of at least two source groups")
    for regime in regimes[1:]:
        if sorted(by_regime[regime]) != expected_groups:
            raise ValueError("all regimes must contain the same source groups")

    pairs = []
    for group_index in range(0, len(expected_groups), 2):
        left_group, right_group = expected_groups[group_index : group_index + 2]
        for regime in regimes:
            pairs.append((by_regime[regime][left_group], by_regime[regime][right_group]))
    return pairs


def _score(session, queries: list[dict]) -> list[float]:
    return [session.score_answer(query["query_ids"], query["answer_ids"]) for query in queries]


def _memory_norm(session) -> float:
    import torch

    if not session.cache.cbf_memory:
        return 0.0
    squared = torch.zeros((), dtype=torch.float64, device=session.device)
    for memory in session.cache.cbf_memory.values():
        squared += memory.double().square().sum()
    return squared.sqrt().item()


def collect(model, scenarios: list[dict], output: Path, policy: str, policy_name: str,
            controller=None, shard: int = 0, num_shards: int = 1,
            limit_pairs: int | None = None) -> dict:
    """Score correct, swapped, and empty fast memory from identical query prompts."""
    from cbf_ttt.runtime import CBFSession

    if shard < 0 or num_shards < 1 or shard >= num_shards:
        raise ValueError("invalid shard selection")
    if policy == "controller" and controller is None:
        raise ValueError("controller policy requires a controller checkpoint")
    pairs = [pair for index, pair in enumerate(paired_scenarios(scenarios)) if index % num_shards == shard]
    if limit_pairs is not None:
        if limit_pairs < 1:
            raise ValueError("limit_pairs must be positive")
        pairs = pairs[:limit_pairs]
    if not pairs:
        raise ValueError("selected shard contains no scenario pairs")

    output.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    on_cuda = device.type == "cuda"
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    rows = 0
    started_all = time.perf_counter()
    max_allocated = 0
    max_reserved = 0
    with output.open("w", encoding="utf-8") as sink:
        for pair_index, (left, right) in enumerate(pairs):
            if on_cuda:
                import torch

                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            pair_started = time.perf_counter()
            sessions = []
            for scenario in (left, right):
                future = scenario["futures"][0]
                session = CBFSession(model, controller, update_rule="forget")
                coefficients = session.consume(
                    scenario["context_ids"] + future.get("continuation_ids", []), policy
                )
                sessions.append((session, coefficients))

            for target_index, scenario in enumerate((left, right)):
                donor_index = 1 - target_index
                target, coefficients = sessions[target_index]
                donor, _ = sessions[donor_index]
                future = scenario["futures"][0]
                queries = future["queries"]
                losses = {
                    "full_context": _score(target, queries),
                    "correct_memory": _score(target.clone_memory_only(), queries),
                    "wrong_memory": _score(donor.clone_memory_only(), queries),
                    "empty_memory": _score(CBFSession(model, update_rule="forget"), queries),
                }
                if any(not math.isfinite(value) for values in losses.values() for value in values):
                    raise ValueError(f"nonfinite validation loss for {scenario['id']}")
                row = {
                    "id": scenario["id"],
                    "group_id": scenario["group_id"],
                    "split": scenario["split"],
                    "regime": scenario["regime"],
                    "donor_id": (right if target_index == 0 else left)["id"],
                    "donor_group_id": (right if target_index == 0 else left)["group_id"],
                    "pair_index": pair_index,
                    "policy_name": policy_name,
                    "policy": policy,
                    "coefficients": coefficients,
                    "memory_norm": _memory_norm(target),
                    "donor_memory_norm": _memory_norm(donor),
                    "query_kinds": [query.get("kind", "unspecified") for query in queries],
                    "query_losses": losses,
                    "mean_nll": {name: statistics.mean(values) for name, values in losses.items()},
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                rows += 1
            if on_cuda:
                import torch

                torch.cuda.synchronize(device)
                max_allocated = max(max_allocated, torch.cuda.max_memory_allocated(device))
                max_reserved = max(max_reserved, torch.cuda.max_memory_reserved(device))
            pair_elapsed = time.perf_counter() - pair_started
            if pair_elapsed <= 0:
                raise RuntimeError("nonpositive pair runtime")

    if tuple(parameter._version for parameter in model.parameters()) != parameter_versions:
        raise RuntimeError("frozen model parameter versions changed during validation")
    elapsed = time.perf_counter() - started_all
    gib = 1024 ** 3
    result = {
        "policy_name": policy_name,
        "policy": policy,
        "shard": shard,
        "num_shards": num_shards,
        "pairs": len(pairs),
        "scenarios": rows,
        "elapsed_s": elapsed,
        "mean_s_per_scenario": elapsed / rows,
        "max_peak_allocated_gib": max_allocated / gib if on_cuda else None,
        "max_peak_reserved_gib": max_reserved / gib if on_cuda else None,
        "model_parameter_versions_unchanged": True,
    }
    Path(str(output) + ".summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _bootstrap_interval(group_values: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    size = len(group_values)
    estimates = sorted(
        statistics.mean(group_values[rng.randrange(size)] for _ in range(size))
        for _ in range(draws)
    )
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def summarize(inputs: list[Path], output: Path, draws: int = 5000, seed: int = 108) -> dict:
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    rows: dict[tuple[str, str], dict] = {}
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["policy_name"], row["id"])
                if key in rows:
                    raise ValueError(f"duplicate policy/scenario row: {key}")
                if set(row["mean_nll"]) != set(CONDITIONS):
                    raise ValueError(f"missing validation condition for {key}")
                if row["group_id"] == row["donor_group_id"]:
                    raise ValueError(f"wrong-memory donor shares source group for {key}")
                rows[key] = row
    if not rows:
        raise ValueError("no validation rows")

    policies = sorted({key[0] for key in rows})
    scenario_ids = {policy: {key[1] for key in rows if key[0] == policy} for policy in policies}
    reference_ids = scenario_ids[policies[0]]
    if any(ids != reference_ids for ids in scenario_ids.values()):
        raise ValueError("policies do not cover identical scenarios")

    result = {"scenarios": len(reference_ids), "policies": {}, "bootstrap_draws": draws}
    for policy in policies:
        policy_rows = [rows[(policy, sample_id)] for sample_id in sorted(reference_ids)]
        query_kinds = sorted({kind for row in policy_rows for kind in row["query_kinds"]})
        if "historical" not in query_kinds:
            raise ValueError("the memory-content gate requires historical queries")

        def values(condition: str, kind: str | None = None) -> list[float]:
            selected = []
            for row in policy_rows:
                for query_kind, loss in zip(row["query_kinds"], row["query_losses"][condition]):
                    if kind is None or query_kind == kind:
                        selected.append(loss)
            if not selected:
                raise ValueError(f"no {kind} queries for {policy}/{condition}")
            return selected

        item = {
            "mean_nll": {
                condition: statistics.mean(values(condition))
                for condition in CONDITIONS
            },
            "mean_nll_by_query_kind": {
                kind: {condition: statistics.mean(values(condition, kind)) for condition in CONDITIONS}
                for kind in query_kinds
            },
            "mean_last_coefficient": statistics.mean(
                row["coefficients"][-1] for row in policy_rows if row["coefficients"]
            ),
            "mean_memory_norm": statistics.mean(row["memory_norm"] for row in policy_rows),
        }
        for reference in ("wrong_memory", "empty_memory"):
            gains = []
            for row in policy_rows:
                historical = [index for index, kind in enumerate(row["query_kinds"]) if kind == "historical"]
                gains.append(statistics.mean(
                    row["query_losses"][reference][index]
                    - row["query_losses"]["correct_memory"][index]
                    for index in historical
                ))
            by_group: dict[str, list[float]] = defaultdict(list)
            for row, gain in zip(policy_rows, gains):
                by_group[row["group_id"]].append(gain)
            group_gains = [statistics.mean(values) for values in by_group.values()]
            item[f"correct_gain_vs_{reference}"] = {
                "query_kind": "historical",
                "mean": statistics.mean(gains),
                "source_group_bootstrap_95ci": _bootstrap_interval(group_gains, draws, seed),
                "fraction_positive": sum(gain > 0 for gain in gains) / len(gains),
            }
        item["memory_content_gate"] = all(
            item[f"correct_gain_vs_{reference}"]["mean"] > 0.005
            and item[f"correct_gain_vs_{reference}"]["source_group_bootstrap_95ci"][0] > 0
            and item[f"correct_gain_vs_{reference}"]["fraction_positive"] >= 0.6
            for reference in ("wrong_memory", "empty_memory")
        )
        result["policies"][policy] = item

    if "fixed_1" not in policies:
        raise ValueError("fixed_1 is required as the strongest historical baseline")
    fixed = {sample_id: rows[("fixed_1", sample_id)] for sample_id in reference_ids}
    controller_names = [name for name in policies if name.startswith("controller_seed")]
    controller_comparisons = {}
    for name in controller_names:
        comparisons = {}
        for condition, query_kind in (("full_context", None), ("correct_memory", "historical")):
            gains = []
            for sample_id in sorted(reference_ids):
                controller_row = rows[(name, sample_id)]
                fixed_row = fixed[sample_id]
                indices = [index for index, kind in enumerate(controller_row["query_kinds"])
                           if query_kind is None or kind == query_kind]
                gains.append(statistics.mean(
                    fixed_row["query_losses"][condition][index]
                    - controller_row["query_losses"][condition][index]
                    for index in indices
                ))
            by_group: dict[str, list[float]] = defaultdict(list)
            for sample_id, gain in zip(sorted(reference_ids), gains):
                by_group[rows[(name, sample_id)]["group_id"]].append(gain)
            group_gains = [statistics.mean(values) for values in by_group.values()]
            comparisons[condition] = {
                "query_kind": query_kind or "all",
                "mean_gain_vs_fixed_1": statistics.mean(gains),
                "source_group_bootstrap_95ci": _bootstrap_interval(group_gains, draws, seed),
                "fraction_positive": sum(gain > 0 for gain in gains) / len(gains),
            }
        controller_comparisons[name] = comparisons
    result["controller_vs_fixed_1"] = controller_comparisons
    result["adaptive_forgetting_gate"] = (
        len(controller_names) == 3
        and sum(controller_comparisons[name]["correct_memory"]["mean_gain_vs_fixed_1"] > 0.005
                for name in controller_names) >= 2
        and sum(controller_comparisons[name]["correct_memory"]["source_group_bootstrap_95ci"][0] > 0
                for name in controller_names) >= 2
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_model(path: str, device: str, dtype: str):
    import torch
    import inference_model  # noqa: F401
    from transformers import AutoModelForCausalLM

    resolved = torch.device(device)
    resolved_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=resolved_dtype)
    return model.to(resolved).eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--data", required=True)
    collect_parser.add_argument("--split", choices=("dev", "test"), default="test")
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--policy", required=True)
    collect_parser.add_argument("--policy-name", required=True)
    collect_parser.add_argument("--controller")
    collect_parser.add_argument("--device", default="cuda")
    collect_parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    collect_parser.add_argument("--shard", type=int, default=0)
    collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--limit-pairs", type=int)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--bootstrap-draws", type=int, default=5000)
    summary_parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()

    if args.command == "summarize":
        result = summarize(args.inputs, args.output, args.bootstrap_draws, args.seed)
    else:
        from cbf_ttt.experiment import load_controller, load_scenarios

        model = _load_model(args.model, args.device, args.dtype)
        controller = load_controller(args.controller, model, next(model.parameters()).device) if args.controller else None
        scenarios = load_scenarios(args.data, args.split)
        result = collect(model, scenarios, args.output, args.policy, args.policy_name, controller,
                         args.shard, args.num_shards, args.limit_pairs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
