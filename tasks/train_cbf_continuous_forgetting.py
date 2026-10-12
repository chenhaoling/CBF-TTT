"""Construct and probe continuous-alpha tradeoffs in low-rank fast memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import Embeddings


GRID = (0., .25, .5, .75, 1.)
MEMORY_SEEDS = (1901, 1902, 1903)
POLICIES = ("learned", "grid_learned", "structure_blind", "endpoint_learned",
            "physics", "train_mean", "fixed_0.0", "fixed_0.25", "fixed_0.5",
            "fixed_0.75", "fixed_1.0", "oracle_grid")
FEATURE_NAMES = ("memory_norm", "delta_norm", "alignment", "conflict_ratio",
                 "duplicate_ratio", "new_ratio", "old_count", "delta_count",
                 "retained_ratio", "current_scale", "mean_key_correlation",
                 "max_key_correlation")
STRUCTURE_BLIND_INDICES = (0, 1, 2, 10, 11)


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


def build_rows(scenarios: list[dict], dim: int,
               memory_seeds: tuple[int, ...] = MEMORY_SEEDS) -> list[dict]:
    rows = []
    for memory_seed in memory_seeds:
        embeddings = Embeddings(memory_seed, dim)
        for scenario in scenarios:
            row = label_scenario(scenario, embeddings)
            row["memory_seed"] = memory_seed
            rows.append(row)
    return rows


def _inputs(rows: list[dict], indices: tuple[int, ...] | None = None):
    import torch

    indices = tuple(range(len(FEATURE_NAMES))) if indices is None else indices
    return torch.tensor([[row["features"][index] for index in indices] for row in rows],
                        dtype=torch.float32)


def train_regressor(rows: list[dict], seed: int, epochs: int,
                    indices: tuple[int, ...] | None = None) -> tuple[dict, list[dict]]:
    import torch

    x = _inputs(rows, indices); y = torch.tensor([[row["alpha"]] for row in rows])
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5); x = (x - mean) / std
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, 16), torch.nn.Tanh(),
                                torch.nn.Linear(16, 1), torch.nn.Sigmoid())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, y); loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            curve.append({"epoch": epoch, "mse": float(loss.detach()),
                          "mae": float((prediction.detach() - y).abs().mean())})
    return {"kind": "regressor", "model": model.eval(), "mean": mean, "std": std,
            "indices": tuple(range(len(FEATURE_NAMES))) if indices is None else indices}, curve


def train_classifier(rows: list[dict], seed: int, epochs: int,
                     endpoints: bool = False) -> tuple[dict, list[dict]]:
    import torch

    x = _inputs(rows); mean, std = x.mean(0), x.std(0).clamp_min(1e-5); x = (x - mean) / std
    if endpoints:
        y = torch.tensor([int(row["losses"]["1.0"]["nll"] < row["losses"]["0.0"]["nll"])
                          for row in rows], dtype=torch.long)
        values = (0., 1.)
    else:
        y = torch.tensor([GRID.index(row["alpha"]) for row in rows], dtype=torch.long)
        values = GRID
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, len(values)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y); loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            curve.append({"epoch": epoch, "loss": float(loss.detach()),
                          "accuracy": float((logits.detach().argmax(-1) == y).float().mean())})
    return {"kind": "classifier", "model": model.eval(), "mean": mean, "std": std,
            "indices": tuple(range(len(FEATURE_NAMES))), "values": values}, curve


def controller_alpha(controller: dict, features: list[float]) -> float:
    import torch

    x = torch.tensor([[features[index] for index in controller["indices"]]], dtype=torch.float32)
    x = (x - controller["mean"]) / controller["std"]
    with torch.no_grad():
        output = controller["model"](x)
    if controller["kind"] == "regressor":
        return float(output[0, 0])
    return float(controller["values"][int(output.argmax(-1))])


def select_alpha(policy: str, scenario: dict, label: dict, controllers: dict,
                 train_mean: float) -> float:
    if policy in controllers:
        return controller_alpha(controllers[policy], label["features"])
    if policy.startswith("fixed_"):
        return float(policy.removeprefix("fixed_"))
    if policy == "train_mean":
        return train_mean
    if policy == "oracle_grid":
        return label["alpha"]
    if policy == "physics":
        # More conflicts call for stronger decay, while a stronger current write
        # already suppresses its stale competitor and therefore needs less decay.
        ratio = scenario["conflict_count"] / scenario["old_count"]
        return min(1., max(0., ratio * (2. - scenario["current_scale"])))
    raise ValueError(f"unknown continuous forgetting policy: {policy}")


def evaluate(scenarios: list[dict], controllers: dict, dim: int,
             train_mean: float, memory_seeds: tuple[int, ...] = MEMORY_SEEDS) -> list[dict]:
    records = []
    for memory_seed in memory_seeds:
        embeddings = Embeddings(memory_seed, dim)
        for scenario in scenarios:
            label = label_scenario(scenario, embeddings)
            for policy in POLICIES:
                alpha = select_alpha(policy, scenario, label, controllers, train_mean)
                metrics = scenario_loss(scenario, embeddings, alpha)
                records.append({"id": scenario["id"], "group_id": scenario["group_id"],
                                "memory_seed": memory_seed, "policy": policy, "alpha": alpha,
                                "oracle_alpha": label["alpha"], **metrics})
    return records


def aggregate(records: list[dict], memory_seeds: tuple[int, ...] = MEMORY_SEEDS) -> dict:
    output = {}
    for policy in POLICIES:
        rows = [row for row in records if row["policy"] == policy]
        per_seed = {}
        for memory_seed in memory_seeds:
            subset = [row for row in rows if row["memory_seed"] == memory_seed]
            per_seed[str(memory_seed)] = {"nll": statistics.fmean(row["nll"] for row in subset),
                                          "accuracy": statistics.fmean(row["accuracy"] for row in subset),
                                          "alpha_mae": statistics.fmean(abs(row["alpha"] - row["oracle_alpha"])
                                                                        for row in subset)}
        groups = {}
        for group in sorted({row["group_id"] for row in rows}):
            subset = [row for row in rows if row["group_id"] == group]
            groups[group] = {"nll": statistics.fmean(row["nll"] for row in subset),
                             "accuracy": statistics.fmean(row["accuracy"] for row in subset)}
        output[policy] = {"nll": statistics.fmean(row["nll"] for row in rows),
                          "accuracy": statistics.fmean(row["accuracy"] for row in rows),
                          "alpha_mae": statistics.fmean(abs(row["alpha"] - row["oracle_alpha"])
                                                        for row in rows),
                          "per_seed": per_seed, "groups": groups}
    return output


def comparison(primary: dict, baseline: dict, seed: int) -> dict:
    groups = sorted(primary["groups"])
    gains = [baseline["groups"][group]["nll"] - primary["groups"][group]["nll"]
             for group in groups]
    accuracy = [primary["groups"][group]["accuracy"] - baseline["groups"][group]["accuracy"]
                for group in groups]
    return {"nll_gain": statistics.fmean(gains), "nll_gain_bootstrap_95ci": _bootstrap(gains, seed),
            "accuracy_gain": statistics.fmean(accuracy),
            "accuracy_gain_bootstrap_95ci": _bootstrap(accuracy, seed + 1)}


def development_gate(summary: dict) -> dict:
    fixed = tuple(f"fixed_{alpha}" for alpha in GRID)
    best_fixed = min(fixed, key=lambda name: summary[name]["nll"])
    fixed_gain = comparison(summary["learned"], summary[best_fixed], 1931)
    endpoint_gain = comparison(summary["learned"], summary["endpoint_learned"], 1933)
    checks = {
        "alpha_mae": summary["learned"]["alpha_mae"] <= .15,
        "oracle_regret": summary["learned"]["nll"] - summary["oracle_grid"]["nll"] <= .03,
        "fixed_nll_margin": fixed_gain["nll_gain"] >= .05,
        "fixed_nll_confidence": fixed_gain["nll_gain_bootstrap_95ci"][0] > .02,
        "endpoint_nll_margin": endpoint_gain["nll_gain"] >= .02,
        "endpoint_nll_confidence": endpoint_gain["nll_gain_bootstrap_95ci"][0] > .005,
        "all_seeds_beat_fixed": all(summary["learned"]["per_seed"][str(seed)]["nll"] <
                                     summary[best_fixed]["per_seed"][str(seed)]["nll"]
                                     for seed in MEMORY_SEEDS),
    }
    return {"checks": checks, "best_fixed": best_fixed, "fixed_comparison": fixed_gain,
            "endpoint_comparison": endpoint_gain, "passed": all(checks.values())}


def serialize_controller(controller: dict) -> dict:
    return {key: value for key, value in controller.items() if key != "model"} | {
        "state_dict": controller["model"].state_dict()}


def restore_controller(saved: dict) -> dict:
    import torch

    if saved["kind"] == "regressor":
        model = torch.nn.Sequential(torch.nn.Linear(len(saved["indices"]), 32), torch.nn.Tanh(),
                                    torch.nn.Linear(32, 16), torch.nn.Tanh(),
                                    torch.nn.Linear(16, 1), torch.nn.Sigmoid())
    else:
        model = torch.nn.Sequential(torch.nn.Linear(len(saved["indices"]), 32), torch.nn.Tanh(),
                                    torch.nn.Linear(32, len(saved["values"])))
    model.load_state_dict(saved["state_dict"])
    return {**saved, "model": model.eval()}


def formal(args) -> dict:
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    probe_result = json.loads(Path(args.probe).read_text())
    if probe_result.get("protocol") != "continuous_forgetting_probe_v1" or not probe_result.get("passed"):
        raise ValueError("formal run requires the registered passing probe")
    train_scenarios = generate_scenarios("train", args.train_groups, args.per_group, args.seed)
    dev_scenarios = generate_scenarios("dev", args.dev_groups, args.per_group, args.seed)
    train_rows = build_rows(train_scenarios, args.dim)
    train_mean = statistics.fmean(row["alpha"] for row in train_rows)
    start = time.perf_counter()
    learned, learned_curve = train_regressor(train_rows, args.controller_seed, args.epochs)
    grid, grid_curve = train_classifier(train_rows, args.controller_seed + 1, args.epochs)
    blind, blind_curve = train_regressor(train_rows, args.controller_seed + 2, args.epochs,
                                         STRUCTURE_BLIND_INDICES)
    endpoint, endpoint_curve = train_classifier(train_rows, args.controller_seed + 3,
                                                 args.epochs, endpoints=True)
    controllers = {"learned": learned, "grid_learned": grid, "structure_blind": blind,
                   "endpoint_learned": endpoint}
    dev_records = evaluate(dev_scenarios, controllers, args.dim, train_mean)
    dev = aggregate(dev_records); gate = development_gate(dev)
    test_records, test, test_scored = None, None, False
    if gate["passed"]:
        test_scenarios = generate_scenarios("test", args.test_groups, args.per_group, args.seed)
        test_records = evaluate(test_scenarios, controllers, args.dim, train_mean)
        test = aggregate(test_records); test_scored = True
    checkpoint = root / "controllers.pt"
    torch.save({"controllers": {name: serialize_controller(value) for name, value in controllers.items()},
                "dim": args.dim, "epochs": args.epochs, "controller_seed": args.controller_seed,
                "train_mean": train_mean}, checkpoint)
    (root / "dev_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dev_records))
    if test_records is not None:
        (root / "test_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_records))
    training = {"seconds": time.perf_counter() - start, "examples": len(train_rows),
                "train_label_counts": dict(Counter(str(row["alpha"]) for row in train_rows)),
                "train_mean_alpha": train_mean, "curves": {"learned": learned_curve,
                "grid_learned": grid_curve, "structure_blind": blind_curve,
                "endpoint_learned": endpoint_curve}, "checkpoint_sha256": sha(checkpoint)}
    write_json(root / "training.json", training)
    manifest = {"protocol": "continuous_forgetting_formal_v1", "probe_sha256": sha(args.probe),
                "seed": args.seed, "controller_seed": args.controller_seed, "dim": args.dim,
                "epochs": args.epochs, "group_counts": {"train": args.train_groups,
                "dev": args.dev_groups, "test": args.test_groups}, "per_group": args.per_group,
                "memory_seeds": MEMORY_SEEDS, "grid": GRID, "feature_names": FEATURE_NAMES,
                "policies": POLICIES, "test_release_rule": "generate and score once only after development gate",
                "test_scored": test_scored}
    write_json(root / "training_manifest.json", manifest)
    result = {"protocol": "continuous_forgetting_formal_v1", "dev": dev,
              "dev_gate": gate, "test": test, "test_scored": test_scored,
              "stage_passed": gate["passed"], "training_seconds": training["seconds"],
              "terminal_status": "completed_continuous_forgetting"}
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("probe"); item.add_argument("--output", required=True)
    item.add_argument("--groups", type=int, default=40); item.add_argument("--per-group", type=int, default=20)
    item.add_argument("--seed", type=int, default=20261101); item.add_argument("--dim", type=int, default=32)
    item = sub.add_parser("formal"); item.add_argument("--probe", required=True)
    item.add_argument("--output", required=True); item.add_argument("--seed", type=int, default=20261101)
    item.add_argument("--controller-seed", type=int, default=1941)
    item.add_argument("--train-groups", type=int, default=128); item.add_argument("--dev-groups", type=int, default=32)
    item.add_argument("--test-groups", type=int, default=32); item.add_argument("--per-group", type=int, default=20)
    item.add_argument("--dim", type=int, default=32); item.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(); result = probe(args) if args.command == "probe" else formal(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
