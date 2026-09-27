"""Compare paired CBF rollouts with source-group bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def _interval(group_values: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    size = len(group_values)
    estimates = sorted(
        sum(group_values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(draws)
    )
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def analyze(rollouts: dict[str, Path], scenarios: Path, output: Path,
            draws: int = 2000, seed: int = 42) -> dict:
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    scenario_info = {}
    with scenarios.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["id"] in scenario_info:
                    raise ValueError(f"duplicate scenario {row['id']}")
                scenario_info[row["id"]] = (row["group_id"], row["regime"])

    by_policy = defaultdict(dict)
    coefficient_by_policy = defaultdict(list)
    group_by_key = {}
    regime_by_key = {}
    update_rules = set()
    for controller_name, path in rollouts.items():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                update_rules.add(row.get("update_rule", "forget"))
                coefficients = row.get("coefficients")
                if coefficients is None:
                    coefficients = row["alphas"]
                policy = controller_name if row["policy"] == "controller" else row["policy"]
                key = (row["id"], row["future_index"])
                if key in by_policy[policy]:
                    raise ValueError(f"duplicate {policy} rollout for {key}")
                info = scenario_info.get(row["id"])
                if info is None or info[0] != row["group_id"]:
                    raise ValueError(f"rollout has unknown or mismatched scenario {row['id']}")
                if not math.isfinite(row["mean_loss"]) or any(not math.isfinite(x) for x in coefficients):
                    raise ValueError(f"nonfinite loss or coefficient for {row['id']}")
                by_policy[policy][key] = row["mean_loss"]
                group_by_key[key], regime_by_key[key] = info
                if coefficients:
                    coefficient_by_policy[policy].append(coefficients[-1])

    if len(update_rules) != 1 or not update_rules <= {"forget", "write"}:
        raise ValueError("rollouts must share one known update rule")
    update_rule = update_rules.pop()
    required_fixed = "0" if update_rule == "write" else "1"
    if "baseline" not in by_policy or required_fixed not in by_policy:
        raise ValueError(f"baseline and fixed {required_fixed} rollouts are required")
    keys = set(by_policy["baseline"])
    if any(set(values) != keys for values in by_policy.values()):
        raise ValueError("policies do not cover the same scenario/future keys")
    groups = sorted(set(group_by_key.values()))
    if not keys or not groups:
        raise ValueError("no comparable rollouts")

    metrics = {}
    references = [("baseline", "vs_gate1"), ("0", "vs_gate0")] if update_rule == "write" else [
        ("baseline", "vs_alpha0"), ("1", "vs_alpha1")]
    references.extend((name, f"vs_{name}") for name in sorted(by_policy)
                      if name not in ("baseline", required_fixed) and not name.startswith("controller"))
    for policy, losses in by_policy.items():
        item = {
            "scenarios": len(losses), "source_groups": len(groups),
            "mean_nll": statistics.mean(losses.values()),
            "mean_last_coefficient": statistics.mean(coefficient_by_policy[policy]),
            "std_last_coefficient": statistics.pstdev(coefficient_by_policy[policy]),
        }
        if update_rule == "forget":
            item["mean_last_alpha"] = item["mean_last_coefficient"]
            item["std_last_alpha"] = item["std_last_coefficient"]
        for reference, label in references:
            paired_gain = {key: by_policy[reference][key] - losses[key] for key in keys}
            by_group = defaultdict(list)
            for key, gain in paired_gain.items():
                by_group[group_by_key[key]].append(gain)
            group_gains = [statistics.mean(by_group[group]) for group in groups]
            item[label] = {
                "mean_gain": statistics.mean(paired_gain.values()),
                "source_group_bootstrap_95ci": _interval(group_gains, draws, seed),
                "harmful_scenario_fraction": sum(value < 0 for value in paired_gain.values()) / len(keys),
            }
        item["mean_nll_by_regime"] = {
            regime: statistics.mean(losses[key] for key in keys if regime_by_key[key] == regime)
            for regime in sorted(set(regime_by_key.values()))
        }
        metrics[policy] = item

    result = {"scenarios": len(keys), "source_groups": len(groups), "update_rule": update_rule,
              "bootstrap_draws": draws, "policies": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", action="append", required=True,
                        help="CONTROLLER_NAME=PATH; non-controller policies retain their own names")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = {}
    for entry in args.rollout:
        name, separator, path = entry.partition("=")
        if not separator or not name or not path or name in paths:
            parser.error("each --rollout must be a unique CONTROLLER_NAME=PATH")
        paths[name] = Path(path)
    print(json.dumps(analyze(paths, args.scenarios, args.output,
                             args.bootstrap_draws, args.seed), indent=2))


if __name__ == "__main__":
    main()
