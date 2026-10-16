"""Audit the g=1 slice of a native joint fast-weight counterfactual grid."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics


MEMORY_SUBSTRATE = "native_dense_ttt_fast_weight"


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _cluster_interval(group_values: dict[str, list[float]], seed: int,
                      samples: int) -> list[float]:
    groups = sorted(group_values)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [rng.choice(groups) for _ in groups]
        draws.append(statistics.fmean(
            value for group in chosen for value in group_values[group]
        ))
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def _read(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("no joint labels were provided")
    identities = [row["id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("joint label inputs contain duplicate scenario IDs")
    return rows


def analyze(rows: list[dict], *, tie_tolerance: float = 1e-6,
            bootstrap_samples: int = 10000, seed: int = 202610231) -> dict:
    if tie_tolerance < 0 or bootstrap_samples < 100:
        raise ValueError("invalid tolerance or bootstrap sample count")
    protocols = {row.get("protocol") for row in rows}
    if not protocols <= {"joint_natural_v1", "joint_postcutoff_v1"}:
        raise ValueError("V15 slice requires native joint natural-text labels")
    grids = {tuple(float(value) for value in row["grid"]) for row in rows}
    if grids != {(0.0, 0.5, 1.0)}:
        raise ValueError("V15 requires the frozen 0/0.5/1 grid")

    forgetting_grid = (0.0, 0.5, 1.0)
    enriched = []
    for row in rows:
        action_losses = {tuple(float(value) for value in action): float(loss)
                         for action, loss in zip(row["actions"], row["losses"])}
        expected = {(retention, gate) for retention in forgetting_grid
                    for gate in forgetting_grid}
        if set(action_losses) != expected:
            raise ValueError(f"{row['id']} does not contain the complete joint grid")
        # Runtime commit_both uses retention, while the method defines alpha as forgetting.
        g1 = {forgetting: action_losses[(1.0 - forgetting, 1.0)]
              for forgetting in forgetting_grid}
        minimum = min(g1.values())
        # Ties prefer less forgetting, matching the conservative deployment rule.
        best = min(value for value in forgetting_grid
                   if g1[value] <= minimum + tie_tolerance)
        best_g0 = min(action_losses[(retention, 0.0)] for retention in forgetting_grid)
        enriched.append({
            "id": row["id"], "group_id": row["group_id"],
            "split": row["split"], "regime": row["regime"],
            "losses": g1, "best_forgetting": best,
            "spread": max(g1.values()) - min(g1.values()),
            "write_gain_vs_best_g0": best_g0 - min(g1.values()),
        })

    fixed = {str(value): statistics.fmean(row["losses"][value] for row in enriched)
             for value in forgetting_grid}
    best_fixed = min(forgetting_grid, key=lambda value: (fixed[str(value)], value))
    oracle_nll = statistics.fmean(min(row["losses"].values()) for row in enriched)
    oracle_gains = {
        row["group_id"]: [] for row in enriched
    }
    write_gains = {row["group_id"]: [] for row in enriched}
    for row in enriched:
        oracle_gains[row["group_id"]].append(
            row["losses"][best_fixed] - min(row["losses"].values())
        )
        write_gains[row["group_id"]].append(row["write_gain_vs_best_g0"])

    by_regime = {}
    for regime in sorted({row["regime"] for row in enriched}):
        part = [row for row in enriched if row["regime"] == regime]
        by_regime[regime] = {
            "rows": len(part),
            "fixed_mean_nll": {
                str(value): statistics.fmean(row["losses"][value] for row in part)
                for value in forgetting_grid
            },
            "oracle_counts": dict(sorted(Counter(
                str(row["best_forgetting"]) for row in part
            ).items())),
            "mean_write_gain_vs_best_g0": statistics.fmean(
                row["write_gain_vs_best_g0"] for row in part
            ),
        }

    best_counts = Counter(str(row["best_forgetting"]) for row in enriched)
    full_forget_fraction = best_counts[str(1.0)] / len(enriched)
    headroom = fixed[str(best_fixed)] - oracle_nll
    headroom_ci = _cluster_interval(oracle_gains, seed, bootstrap_samples)
    mean_write_gain = statistics.fmean(
        row["write_gain_vs_best_g0"] for row in enriched
    )
    write_ci = _cluster_interval(write_gains, seed + 1, bootstrap_samples)
    source_groups_with_non_full_forgetting = len({
        row["group_id"] for row in enriched if row["best_forgetting"] < 1.0
    })
    checks = {
        "nonflat_fraction_at_least_point_25": (
            sum(row["spread"] > 0.005 for row in enriched) / len(enriched) >= 0.25
        ),
        "oracle_headroom_at_least_point_005_nll": headroom >= 0.005,
        "oracle_headroom_cluster_ci_lower_above_zero": headroom_ci[0] > 0.0,
        "full_forgetting_not_more_than_point_75_of_oracle_rows": full_forget_fraction <= 0.75,
        "at_least_two_groups_need_non_full_forgetting": (
            source_groups_with_non_full_forgetting >= 2
        ),
        "current_native_write_mean_gain_above_zero": mean_write_gain > 0.0,
        "current_native_write_cluster_ci_lower_above_zero": write_ci[0] > 0.0,
    }
    return {
        "protocol": "native_fast_weight_forgetting_v15_reanalysis",
        "memory_substrate": MEMORY_SUBSTRATE,
        "candidate_update_source": "native_ttt_proj_ttt_conv",
        "base_weights_frozen": True,
        "coefficient_semantics": "forgetting_fraction",
        "current_write_gate": 1.0,
        "fast_weight_equation": "M_t=(1-alpha_t)*M_t-1+DeltaW_t",
        "source_protocols": sorted(protocols),
        "rows": len(enriched),
        "source_groups": len({row["group_id"] for row in enriched}),
        "forgetting_grid": list(forgetting_grid),
        "fixed_mean_nll": fixed,
        "best_fixed_forgetting": best_fixed,
        "oracle_mean_nll": oracle_nll,
        "oracle_headroom_vs_best_fixed_nll": headroom,
        "oracle_headroom_group_bootstrap_95ci": headroom_ci,
        "oracle_counts": dict(sorted(best_counts.items())),
        "full_forgetting_oracle_fraction": full_forget_fraction,
        "source_groups_with_non_full_forgetting": source_groups_with_non_full_forgetting,
        "nonflat_fraction_spread_gt_point_005": (
            sum(row["spread"] > 0.005 for row in enriched) / len(enriched)
        ),
        "mean_current_write_gain_vs_best_g0": mean_write_gain,
        "current_write_gain_group_bootstrap_95ci": write_ci,
        "rows_with_positive_current_write_gain": sum(
            row["write_gain_vs_best_g0"] > 0.0 for row in enriched
        ),
        "by_regime": by_regime,
        "gate": {"checks": checks, "passed": all(checks.values())},
        "decision": (
            "train_native_forgetting_controller" if all(checks.values())
            else "stop_controller_training_and_repair_native_writer"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=202610231)
    args = parser.parse_args()
    result = analyze(_read(args.inputs), bootstrap_samples=args.bootstrap_samples,
                     seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
