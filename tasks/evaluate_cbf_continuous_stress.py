"""Evaluate a frozen continuous-alpha controller under registered shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_continuous_forgetting import (
    GRID, POLICIES, aggregate, comparison, evaluate, restore_controller,
)
from tasks.train_cbf_event_writer import write_json


STRESS_SEEDS = (2901, 2902, 2903)
SUITES = ("long_memory", "weak_update", "strong_update", "extreme_overlap")


def generate_stress_scenarios(suite: str, groups: int, per_group: int, seed: int) -> list[dict]:
    if suite not in SUITES:
        raise ValueError(f"unknown stress suite: {suite}")
    output = []
    for group_index in range(groups):
        group_id = hashlib.sha256(f"{seed}|stress|{suite}|{group_index}".encode()).hexdigest()[:16]
        for scenario_index in range(per_group):
            scenario_id = f"{suite}.{group_id}.{scenario_index}"
            rng = random.Random(int(hashlib.sha256(scenario_id.encode()).hexdigest()[:16], 16))
            if suite == "long_memory":
                old_count = rng.randint(16, 28); current_scale = rng.uniform(.55, 1.55)
                conflict_count = rng.randint(1, old_count - 1)
                duplicate_limit, new_count = 5, rng.randint(0, 8)
            elif suite == "weak_update":
                old_count = rng.randint(6, 12); current_scale = rng.uniform(.20, .50)
                conflict_count = rng.randint(1, old_count - 1)
                duplicate_limit, new_count = 3, rng.randint(0, 4)
            elif suite == "strong_update":
                old_count = rng.randint(6, 12); current_scale = rng.uniform(1.60, 2.20)
                conflict_count = rng.randint(1, old_count - 1)
                duplicate_limit, new_count = 3, rng.randint(0, 4)
            else:
                old_count = rng.randint(8, 20); current_scale = rng.uniform(.35, 1.90)
                if rng.random() < .5:
                    conflict_count = rng.randint(1, max(1, old_count // 4))
                else:
                    conflict_count = rng.randint(max(1, math.ceil(.70 * old_count)), old_count - 1)
                duplicate_limit, new_count = 5, rng.randint(0, 8)
            remaining = old_count - conflict_count
            duplicate_count = rng.randint(0, min(duplicate_limit, remaining))
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
            output.append({"id": scenario_id, "group_id": group_id, "split": f"stress_{suite}",
                           "old": old, "delta": delta, "truth": sorted(truth.items()),
                           "old_count": old_count, "conflict_count": conflict_count,
                           "duplicate_count": duplicate_count, "new_count": new_count,
                           "current_scale": current_scale})
    return output


def summarize_suite(records: list[dict], bootstrap_seed: int) -> dict:
    summary = aggregate(records, STRESS_SEEDS)
    fixed = tuple(f"fixed_{alpha}" for alpha in GRID)
    best_fixed = min(fixed, key=lambda name: summary[name]["nll"])
    comparisons = {name: comparison(summary["learned"], summary[name], bootstrap_seed + index * 2)
                   for index, name in enumerate((best_fixed, "endpoint_learned",
                                                  "structure_blind", "physics"))}
    labels = Counter(str(row["oracle_alpha"]) for row in records if row["policy"] == "learned")
    return {"metrics": summary, "best_fixed": best_fixed, "comparisons": comparisons,
            "label_counts": dict(labels)}


def stress_checks(suites: dict, macro: dict) -> dict:
    return {
        "all_suites_beat_hindsight_fixed": all(
            suites[suite]["comparisons"][suites[suite]["best_fixed"]]["nll_gain_bootstrap_95ci"][0] > 0.
            for suite in SUITES),
        "all_suites_beat_endpoint": all(
            suites[suite]["comparisons"]["endpoint_learned"]["nll_gain_bootstrap_95ci"][0] > 0.
            for suite in SUITES),
        "macro_alpha_mae": macro["metrics"]["learned"]["alpha_mae"] <= .20,
        "every_new_seed_beats_macro_fixed": all(
            macro["metrics"]["learned"]["per_seed"][str(seed)]["nll"] <
            macro["metrics"][macro["best_fixed"]]["per_seed"][str(seed)]["nll"]
            for seed in STRESS_SEEDS),
    }


def run(args) -> dict:
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    formal = Path(args.formal)
    checkpoint = torch.load(formal / "controllers.pt", map_location="cpu", weights_only=True)
    controllers = {name: restore_controller(value) for name, value in checkpoint["controllers"].items()}
    suites, all_records = {}, []
    for index, suite in enumerate(SUITES):
        scenarios = generate_stress_scenarios(suite, args.groups, args.per_group, args.seed)
        records = evaluate(scenarios, controllers, checkpoint["dim"], checkpoint["train_mean"], STRESS_SEEDS)
        (root / f"{suite}_records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))
        suites[suite] = summarize_suite(records, args.bootstrap_seed + index * 100)
        all_records.extend(records)
    macro = summarize_suite(all_records, args.bootstrap_seed + 1000)
    checks = stress_checks(suites, macro)
    manifest = {"protocol": "continuous_forgetting_stress_v1", "formal_checkpoint_sha256": sha(formal / "controllers.pt"),
                "formal_summary_sha256": sha(formal / "summary.json"), "stress_seeds": STRESS_SEEDS,
                "suites": SUITES, "groups": args.groups, "per_group": args.per_group,
                "seed": args.seed, "bootstrap_seed": args.bootstrap_seed, "policies": POLICIES,
                "controller_retrained": False}
    write_json(root / "manifest.json", manifest)
    result = {"protocol": "continuous_forgetting_stress_v1", "suites": suites,
              "macro": macro, "checks": checks, "passed": all(checks.values()),
              "terminal_status": "completed_frozen_controller_stress"}
    write_json(root / "summary.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=32); parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103); parser.add_argument("--bootstrap-seed", type=int, default=2951)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
