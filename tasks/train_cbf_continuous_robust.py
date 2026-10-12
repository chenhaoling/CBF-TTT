"""Train and evaluate a domain-augmented continuous forgetting controller."""

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

from tasks.evaluate_cbf_continuous_stress import STRESS_SEEDS, SUITES, generate_stress_scenarios
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_continuous_forgetting import (
    FEATURE_NAMES, GRID, POLICIES, STRUCTURE_BLIND_INDICES, aggregate, build_rows,
    comparison, evaluate, generate_scenarios, restore_controller, serialize_controller,
    train_classifier, train_regressor,
)
from tasks.train_cbf_event_writer import write_json


TRAIN_SEEDS = (1901, 1902, 1903)
DEV_SEEDS = STRESS_SEEDS
TEST_SEEDS = (3901, 3902, 3903)
DOMAINS = ("base",) + SUITES
COMPOUND_DOMAINS = ("long_weak", "long_strong", "extreme_scale")


def domain_scenarios(domain: str, split: str, groups: int, per_group: int, seed: int) -> list[dict]:
    if domain == "base":
        return generate_scenarios(f"robust_{split}_base", groups, per_group, seed)
    if domain in SUITES:
        return generate_stress_scenarios(domain, groups, per_group, seed)
    if domain not in COMPOUND_DOMAINS:
        raise ValueError(f"unknown robust domain: {domain}")
    output = []
    for group_index in range(groups):
        group_id = hashlib.sha256(f"{seed}|robust|{split}|{domain}|{group_index}".encode()).hexdigest()[:16]
        for scenario_index in range(per_group):
            scenario_id = f"{domain}.{group_id}.{scenario_index}"
            rng = random.Random(int(hashlib.sha256(scenario_id.encode()).hexdigest()[:16], 16))
            old_count = rng.randint(16, 28) if domain.startswith("long_") else rng.randint(8, 20)
            current_scale = (rng.uniform(.20, .50) if domain == "long_weak" else
                             rng.uniform(1.60, 2.20) if domain == "long_strong" else
                             rng.choice((rng.uniform(.20, .50), rng.uniform(1.60, 2.20))))
            if domain == "extreme_scale":
                conflict_count = (rng.randint(1, max(1, old_count // 4)) if rng.random() < .5 else
                                  rng.randint(max(1, math.ceil(.70 * old_count)), old_count - 1))
            else:
                conflict_count = rng.randint(1, old_count - 1)
            remaining = old_count - conflict_count
            duplicate_count = rng.randint(0, min(5, remaining)); new_count = rng.randint(0, 8)
            entities = [f"entity_{group_id}_{scenario_index}_{index}"
                        for index in range(old_count + new_count)]
            old_values = [f"old_{group_id}_{scenario_index}_{index}" for index in range(old_count)]
            new_values = [f"new_{group_id}_{scenario_index}_{index}"
                          for index in range(conflict_count + new_count)]
            indices = list(range(old_count)); rng.shuffle(indices)
            conflicts = set(indices[:conflict_count])
            duplicates = set(indices[conflict_count:conflict_count + duplicate_count])
            old = list(zip(entities[:old_count], old_values)); delta, truth = [], dict(old)
            new_value_index = 0
            for index in sorted(conflicts):
                value = new_values[new_value_index]; new_value_index += 1
                delta.append((entities[index], value)); truth[entities[index]] = value
            for index in sorted(duplicates):
                delta.append((entities[index], old_values[index]))
            for offset in range(new_count):
                entity = entities[old_count + offset]; value = new_values[new_value_index]
                new_value_index += 1; delta.append((entity, value)); truth[entity] = value
            output.append({"id": scenario_id, "group_id": group_id, "split": f"robust_{split}_{domain}",
                           "old": old, "delta": delta, "truth": sorted(truth.items()),
                           "old_count": old_count, "conflict_count": conflict_count,
                           "duplicate_count": duplicate_count, "new_count": new_count,
                           "current_scale": current_scale})
    return output


def train_controllers(rows: list[dict], seed: int, epochs: int) -> tuple[dict, dict]:
    learned, curve = train_regressor(rows, seed, epochs)
    grid, grid_curve = train_classifier(rows, seed + 1, epochs)
    blind, blind_curve = train_regressor(rows, seed + 2, epochs, STRUCTURE_BLIND_INDICES)
    endpoint, endpoint_curve = train_classifier(rows, seed + 3, epochs, endpoints=True)
    return {"learned": learned, "grid_learned": grid, "structure_blind": blind,
            "endpoint_learned": endpoint}, {"learned": curve, "grid_learned": grid_curve,
            "structure_blind": blind_curve, "endpoint_learned": endpoint_curve}


def summarize_domains(records: dict[str, list[dict]], seeds: tuple[int, ...],
                      bootstrap_seed: int) -> tuple[dict, dict]:
    domains, combined = {}, []
    for index, (domain, rows) in enumerate(records.items()):
        metrics = aggregate(rows, seeds); fixed = min((f"fixed_{x}" for x in GRID), key=lambda x: metrics[x]["nll"])
        domains[domain] = {"metrics": metrics, "best_fixed": fixed,
                           "fixed_comparison": comparison(metrics["learned"], metrics[fixed],
                                                          bootstrap_seed + index * 20),
                           "endpoint_comparison": comparison(metrics["learned"], metrics["endpoint_learned"],
                                                             bootstrap_seed + index * 20 + 2)}
        combined.extend(rows)
    metrics = aggregate(combined, seeds); fixed = min((f"fixed_{x}" for x in GRID), key=lambda x: metrics[x]["nll"])
    macro = {"metrics": metrics, "best_fixed": fixed,
             "fixed_comparison": comparison(metrics["learned"], metrics[fixed], bootstrap_seed + 1000),
             "endpoint_comparison": comparison(metrics["learned"], metrics["endpoint_learned"], bootstrap_seed + 1002)}
    return domains, macro


def development_gate(domains: dict, macro: dict, frozen_macro: dict) -> dict:
    m = macro["metrics"]; fixed = macro["best_fixed"]
    improvement = comparison(m["learned"], frozen_macro["learned"], 3971)
    strong_gap = domains["strong_update"]["metrics"]["learned"]["nll"] - domains["strong_update"]["metrics"]["endpoint_learned"]["nll"]
    checks = {
        "macro_alpha_mae": m["learned"]["alpha_mae"] <= .12,
        "macro_fixed_margin": macro["fixed_comparison"]["nll_gain_bootstrap_95ci"][0] > .03,
        "macro_endpoint_margin": macro["endpoint_comparison"]["nll_gain_bootstrap_95ci"][0] > .01,
        "augmentation_improves_frozen": improvement["nll_gain_bootstrap_95ci"][0] > .003,
        "strong_update_endpoint_noninferiority": strong_gap <= .002,
        "all_domains_beat_hindsight_fixed": all(row["fixed_comparison"]["nll_gain_bootstrap_95ci"][0] > 0.
                                                 for row in domains.values()),
        "all_dev_seeds_beat_macro_fixed": all(m["learned"]["per_seed"][str(seed)]["nll"] <
                                               m[fixed]["per_seed"][str(seed)]["nll"] for seed in DEV_SEEDS),
    }
    return {"checks": checks, "frozen_comparison": improvement, "strong_endpoint_nll_gap": strong_gap,
            "passed": all(checks.values())}


def run(args) -> dict:
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    frozen_checkpoint = torch.load(Path(args.frozen) / "controllers.pt", map_location="cpu", weights_only=True)
    frozen = {name: restore_controller(value) for name, value in frozen_checkpoint["controllers"].items()}
    train_rows, train_counts = [], {}
    for domain in DOMAINS:
        scenarios = domain_scenarios(domain, "train", args.train_groups, args.per_group, args.train_seed)
        rows = build_rows(scenarios, args.dim, TRAIN_SEEDS); train_rows.extend(rows)
        train_counts[domain] = dict(Counter(str(row["alpha"]) for row in rows))
    start = time.perf_counter(); controllers, curves = train_controllers(train_rows, args.controller_seed, args.epochs)
    train_mean = statistics.fmean(row["alpha"] for row in train_rows)
    dev_records, frozen_records = {}, {}
    for domain in DOMAINS:
        scenarios = domain_scenarios(domain, "dev", args.dev_groups, args.per_group, args.dev_seed)
        dev_records[domain] = evaluate(scenarios, controllers, args.dim, train_mean, DEV_SEEDS)
        frozen_records[domain] = evaluate(scenarios, frozen, args.dim, frozen_checkpoint["train_mean"], DEV_SEEDS)
        (root / f"dev_{domain}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in dev_records[domain]))
    dev_domains, dev_macro = summarize_domains(dev_records, DEV_SEEDS, args.bootstrap_seed)
    _, frozen_macro = summarize_domains(frozen_records, DEV_SEEDS, args.bootstrap_seed + 10000)
    gate = development_gate(dev_domains, dev_macro, frozen_macro["metrics"])
    test_domains = test_macro = None; test_scored = False
    if gate["passed"]:
        test_records = {}
        for domain in DOMAINS + COMPOUND_DOMAINS:
            scenarios = domain_scenarios(domain, "test", args.test_groups, args.per_group, args.test_seed)
            test_records[domain] = evaluate(scenarios, controllers, args.dim, train_mean, TEST_SEEDS)
            (root / f"test_{domain}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in test_records[domain]))
        test_domains, test_macro = summarize_domains(test_records, TEST_SEEDS, args.bootstrap_seed + 20000)
        test_scored = True
    checkpoint = root / "controllers.pt"
    torch.save({"controllers": {name: serialize_controller(value) for name, value in controllers.items()},
                "dim": args.dim, "epochs": args.epochs, "controller_seed": args.controller_seed,
                "train_mean": train_mean}, checkpoint)
    training = {"seconds": time.perf_counter() - start, "examples": len(train_rows),
                "label_counts_by_domain": train_counts, "curves": curves,
                "checkpoint_sha256": sha(checkpoint)}
    write_json(root / "training.json", training)
    manifest = {"protocol": "continuous_forgetting_robust_v2", "frozen_checkpoint_sha256": sha(Path(args.frozen) / "controllers.pt"),
                "train_seeds": TRAIN_SEEDS, "dev_seeds": DEV_SEEDS, "test_seeds": TEST_SEEDS,
                "domains": DOMAINS, "compound_test_domains": COMPOUND_DOMAINS,
                "train_groups": args.train_groups, "dev_groups": args.dev_groups, "test_groups": args.test_groups,
                "per_group": args.per_group, "train_seed": args.train_seed, "dev_seed": args.dev_seed,
                "test_seed": args.test_seed, "controller_seed": args.controller_seed,
                "bootstrap_seed": args.bootstrap_seed, "dim": args.dim, "epochs": args.epochs,
                "feature_names": FEATURE_NAMES, "policies": POLICIES,
                "test_release_rule": "generate only after robust development gate", "test_scored": test_scored}
    write_json(root / "manifest.json", manifest)
    result = {"protocol": "continuous_forgetting_robust_v2", "dev_domains": dev_domains,
              "dev_macro": dev_macro, "dev_frozen_macro": frozen_macro, "dev_gate": gate,
              "test_domains": test_domains, "test_macro": test_macro, "test_scored": test_scored,
              "stage_passed": gate["passed"], "terminal_status": "completed_robust_continuous_forgetting"}
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--train-groups", type=int, default=64); parser.add_argument("--dev-groups", type=int, default=16)
    parser.add_argument("--test-groups", type=int, default=32); parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=20261105); parser.add_argument("--dev-seed", type=int, default=20261106)
    parser.add_argument("--test-seed", type=int, default=20261107); parser.add_argument("--controller-seed", type=int, default=3941)
    parser.add_argument("--bootstrap-seed", type=int, default=3951); parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
