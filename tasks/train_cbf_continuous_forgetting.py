"""Construct and probe continuous-alpha tradeoffs in low-rank fast memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import Embeddings


GRID = (0., .25, .5, .75, 1.)
MEMORY_SEEDS = (1901, 1902, 1903)


def generate_scenarios(split: str, groups: int, per_group: int, seed: int) -> list[dict]:
    output = []
    for group_index in range(groups):
        group_id = hashlib.sha256(f"{seed}|{split}|{group_index}".encode()).hexdigest()[:16]
        for scenario_index in range(per_group):
            scenario_id = f"{group_id}.{scenario_index}"
            rng = random.Random(int(hashlib.sha256(scenario_id.encode()).hexdigest()[:16], 16))
            old_count = rng.randint(6, 12)
            conflict_count = rng.randint(1, old_count - 1)
            remaining = old_count - conflict_count
            duplicate_count = rng.randint(0, min(3, remaining))
            new_count = rng.randint(0, 4)
            current_scale = rng.uniform(.55, 1.55)
            entities = [f"entity_{group_id}_{scenario_index}_{index}" for index in range(old_count + new_count)]
            old_values = [f"old_{group_id}_{scenario_index}_{index}" for index in range(old_count)]
            new_values = [f"new_{group_id}_{scenario_index}_{index}" for index in range(conflict_count + new_count)]
            indices = list(range(old_count)); rng.shuffle(indices)
            conflicts = set(indices[:conflict_count])
            duplicates = set(indices[conflict_count:conflict_count + duplicate_count])
            old = list(zip(entities[:old_count], old_values))
            delta, truth = [], dict(old)
            new_value_index = 0
            for index in sorted(conflicts):
                value = new_values[new_value_index]; new_value_index += 1
                delta.append((entities[index], value)); truth[entities[index]] = value
            for index in sorted(duplicates):
                delta.append((entities[index], old_values[index]))
            for offset in range(new_count):
                entity = entities[old_count + offset]; value = new_values[new_value_index]
                new_value_index += 1; delta.append((entity, value)); truth[entity] = value
            output.append({"id": scenario_id, "group_id": group_id, "split": split,
                           "old": old, "delta": delta, "truth": sorted(truth.items()),
                           "old_count": old_count, "conflict_count": conflict_count,
                           "duplicate_count": duplicate_count, "new_count": new_count,
                           "current_scale": current_scale})
    return output


def matrices(scenario: dict, embeddings: Embeddings):
    import torch

    memory = torch.zeros((embeddings.dim, embeddings.dim), dtype=torch.float32)
    delta = torch.zeros_like(memory)
    for entity, value in scenario["old"]:
        memory += torch.outer(embeddings.get("value", value), embeddings.get("entity", entity))
    for entity, value in scenario["delta"]:
        delta += scenario["current_scale"] * torch.outer(
            embeddings.get("value", value), embeddings.get("entity", entity))
    return memory, delta


def scenario_loss(scenario: dict, embeddings: Embeddings, alpha: float,
                  temperature: float = .2) -> dict:
    import torch

    memory, delta = matrices(scenario, embeddings)
    effective = (1. - alpha) * memory + delta
    candidates = sorted({value for _, value in scenario["old"] + scenario["delta"]})
    value_matrix = torch.stack([embeddings.get("value", value) for value in candidates])
    losses, correct = [], []
    for entity, truth in scenario["truth"]:
        output = effective @ embeddings.get("entity", entity)
        logits = value_matrix @ output / temperature
        target = candidates.index(truth)
        losses.append(float(-torch.log_softmax(logits, -1)[target]))
        correct.append(int(logits.argmax().item() == target))
    return {"nll": statistics.fmean(losses), "accuracy": statistics.fmean(correct)}


def scenario_features(scenario: dict, embeddings: Embeddings) -> list[float]:
    import torch

    memory, delta = matrices(scenario, embeddings)
    memory_norm, delta_norm = memory.norm(), delta.norm()
    alignment = float(torch.sum(memory * delta) / (memory_norm * delta_norm).clamp_min(1e-8))
    keys = [embeddings.get("entity", entity) for entity, _ in scenario["old"]]
    correlations = [abs(float(torch.dot(keys[i], keys[j])))
                    for i in range(len(keys)) for j in range(i)]
    old_count = scenario["old_count"]
    return [math.log1p(float(memory_norm)) / 5., math.log1p(float(delta_norm)) / 5., alignment,
            scenario["conflict_count"] / old_count,
            scenario["duplicate_count"] / old_count,
            scenario["new_count"] / 4., old_count / 12., len(scenario["delta"]) / 12.,
            (scenario["old_count"] - scenario["conflict_count"]) / old_count,
            scenario["current_scale"] / 1.55,
            statistics.fmean(correlations) if correlations else 0.,
            max(correlations) if correlations else 0.]


def label_scenario(scenario: dict, embeddings: Embeddings) -> dict:
    losses = {str(alpha): scenario_loss(scenario, embeddings, alpha) for alpha in GRID}
    best = min(row["nll"] for row in losses.values())
    winners = [alpha for alpha in GRID if losses[str(alpha)]["nll"] <= best + 1e-8]
    alpha = min(winners)
    return {"id": scenario["id"], "group_id": scenario["group_id"],
            "features": scenario_features(scenario, embeddings), "alpha": alpha,
            "losses": losses, "tied": len(winners) > 1}


def probe(args) -> dict:
    scenarios = generate_scenarios("probe", args.groups, args.per_group, args.seed)
    per_seed, combined = {}, Counter()
    for memory_seed in MEMORY_SEEDS:
        embeddings = Embeddings(memory_seed, args.dim)
        rows = [label_scenario(scenario, embeddings) for scenario in scenarios]
        counts = Counter(row["alpha"] for row in rows); combined.update(counts)
        per_seed[str(memory_seed)] = {str(alpha): counts[alpha] for alpha in GRID}
    total = sum(combined.values()); interior = sum(combined[alpha] for alpha in GRID[1:-1]) / total
    supported = sum(combined[alpha] / total >= .05 for alpha in GRID[1:-1])
    checks = {"interior_fraction": interior >= .30, "two_interior_points": supported >= 2}
    result = {"protocol": "continuous_forgetting_probe_v1", "scenarios": len(scenarios),
              "memory_seeds": MEMORY_SEEDS, "grid": GRID, "per_seed": per_seed,
              "combined": {str(alpha): combined[alpha] for alpha in GRID},
              "interior_fraction": interior, "supported_interior_points": supported,
              "checks": checks, "passed": all(checks.values()), "test_generated": False}
    write_json(args.output, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("probe"); item.add_argument("--output", required=True)
    item.add_argument("--groups", type=int, default=40); item.add_argument("--per-group", type=int, default=20)
    item.add_argument("--seed", type=int, default=20261101); item.add_argument("--dim", type=int, default=32)
    args = parser.parse_args(); result = probe(args); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
