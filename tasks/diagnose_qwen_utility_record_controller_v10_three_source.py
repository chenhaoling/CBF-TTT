"""Frozen post-hoc diagnosis of the failed V10 effect gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time

from scripts.audit_qwen_utility_record_controller_v5_residual_training import load_checkpoint
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.diagnose_qwen_utility_record_controller_v7_cross_source import head_diagnostic
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    fact_key_projected_actions,
)
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.train_qwen_utility_record_controller_v5_residual import (
    SEEDS, predict_state, residual_actions,
)
from tasks.train_qwen_utility_record_controller_v10_three_source import (
    ARMS, PROTOCOL as TRAINING_PROTOCOL, _load_inputs, add_arguments,
    equal_domain_normalization,
)


PROTOCOL = "qwen_utility_record_controller_v10_failure_diagnostic"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V10_FAILURE_DIAGNOSTIC_PLAN.md"


def source_geometry(train_domains: dict, development: dict) -> dict:
    domain_rows = {name: domain["examples"] for name, domain in train_domains.items()}
    means, stds = equal_domain_normalization(domain_rows)

    def standardized(rows):
        return [[(value - mean) / std
                 for value, mean, std in zip(row["features"], means, stds)]
                for row in rows]

    centroids = {
        name: [statistics.fmean(row[index] for row in standardized(rows))
               for index in range(len(FEATURE_NAMES))]
        for name, rows in domain_rows.items()
    }
    distances = {}
    names = sorted(centroids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            distances[f"{left}:{right}"] = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(centroids[left], centroids[right])
            ))
    assignments = {}
    for domain_name, domain in development.items():
        nearest = Counter()
        mean_distance = {name: [] for name in names}
        for row in standardized(domain["examples"]):
            values = {
                name: math.sqrt(sum((value - center) ** 2
                                    for value, center in zip(row, centroids[name])))
                for name in names
            }
            nearest[min(values, key=lambda name: (values[name], name))] += 1
            for name, value in values.items():
                mean_distance[name].append(value)
        assignments[domain_name] = {
            "examples": len(domain["examples"]),
            "nearest_source": dict(sorted(nearest.items())),
            "mean_distance": {
                name: statistics.fmean(values)
                for name, values in mean_distance.items()
            },
        }
    return {"training_centroid_distances": distances,
            "development_assignments": assignments}


def target_support(train_domains: dict) -> dict:
    result = {}
    for name, domain in train_domains.items():
        rows = domain["examples"]
        mechanisms = {}
        for mechanism in sorted({row["mechanism"] for row in rows}):
            part = [row for row in rows if row["mechanism"] == mechanism]
            mechanisms[mechanism] = {
                "examples": len(part),
                "benefit_prevalence": statistics.fmean(
                    row["benefit_target"] for row in part
                ),
                "risk_prevalence": statistics.fmean(
                    row["risk_target"] for row in part
                ),
            }
        result[name] = {
            "examples": len(rows),
            "benefit_prevalence": statistics.fmean(
                row["benefit_target"] for row in rows
            ),
            "risk_prevalence": statistics.fmean(
                row["risk_target"] for row in rows
            ),
            "per_mechanism": mechanisms,
        }
    return result


def action_comparison(domain: dict, primary: dict, ablation: dict) -> dict:
    raw_primary = residual_actions(
        domain["labels"], domain["features"], primary["predictions"],
        primary["config"],
    )
    raw_ablation = residual_actions(
        domain["labels"], domain["features"], ablation["predictions"],
        ablation["config"],
    )
    primary_actions = fact_key_projected_actions(
        domain["labels"], domain["features"], raw_primary, domain["matches"]
    )
    ablation_actions = fact_key_projected_actions(
        domain["labels"], domain["features"], raw_ablation, domain["matches"]
    )
    label_map = {(row["episode_id"], row["boundary_index"]): row
                 for row in domain["labels"]}
    rows = []
    for identity in sorted(primary_actions):
        if primary_actions[identity] == ablation_actions[identity]:
            continue
        label = label_map[identity]
        left = label["actions"][primary_actions[identity]]
        right = label["actions"][ablation_actions[identity]]
        rows.append({
            "mechanism": label["mechanism"],
            "correct_delta": left["correct"] - right["correct"],
            "unrelated_delta": (
                left["unrelated_correct"] - right["unrelated_correct"]
            ),
        })

    def aggregate(values):
        return {
            "changed_boundaries": len(values),
            "favorable": sum(row["correct_delta"] > 0 for row in values),
            "unfavorable": sum(row["correct_delta"] < 0 for row in values),
            "correct_delta": sum(row["correct_delta"] for row in values),
            "unrelated_correct_delta": sum(
                row["unrelated_delta"] for row in values
            ),
        }
    return {
        "overall": aggregate(rows),
        "per_mechanism": {
            mechanism: aggregate([
                row for row in rows if row["mechanism"] == mechanism
            ]) for mechanism in sorted({row["mechanism"] for row in rows})
        },
    }


def probability_shift(examples: list[dict], primary: dict, ablation: dict) -> dict:
    output = {}
    for head, target in (("benefit", "benefit_target"), ("risk", "risk_target")):
        deltas = []
        positives, negatives = [], []
        for row in examples:
            identity = (row["episode_id"], row["boundary_index"], row["slot"])
            delta = primary[identity][head] - ablation[identity][head]
            deltas.append(delta)
            (positives if row[target] else negatives).append(delta)
        output[head] = {
            "mean_delta": statistics.fmean(deltas),
            "mean_absolute_delta": statistics.fmean(map(abs, deltas)),
            "positive_target_mean_delta": statistics.fmean(positives),
            "negative_target_mean_delta": statistics.fmean(negatives),
        }
    return output


def diagnose(args) -> dict:
    started = time.perf_counter()
    training_root = Path(args.training_root)
    summary_path = training_root / "summary.json"
    audit_path = Path(args.training_audit)
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != TRAINING_PROTOCOL or not summary.get("passed") \
            or not audit.get("passed") \
            or audit.get("development_release_passed"):
        raise ValueError("diagnostic requires the audited failed V10 release")
    if audit.get("summary_sha256") != sha256(summary_path):
        raise ValueError("V10 summary changed after audit")
    train_domains, development, hashes, _ = _load_inputs(args)
    checkpoint_hashes, seeds = {}, {}
    for seed in SEEDS:
        seed_data = {}
        for arm in ARMS:
            path = training_root / arm / f"seed_{seed}" / "controller.pt"
            checkpoint = load_checkpoint(path)
            checkpoint_hashes[f"{arm}:{seed}"] = sha256(path)
            predictions = {
                domain_name: predict_state(
                    checkpoint["state_dict"], domain["examples"],
                    checkpoint["means"], checkpoint["stds"],
                ) for domain_name, domain in development.items()
            }
            seed_data[arm] = {
                "checkpoint": checkpoint, "predictions": predictions,
            }
        domains = {}
        for domain_name, domain in development.items():
            primary = seed_data["three_source_equal"]
            ablation = seed_data["fewrel_removed_equal"]
            domains[domain_name] = {
                "ranking": {
                    arm: {
                        "benefit": head_diagnostic(
                            domain["examples"], seed_data[arm]["predictions"][domain_name],
                            "benefit", "benefit_target",
                        )["overall"],
                        "risk": head_diagnostic(
                            domain["examples"], seed_data[arm]["predictions"][domain_name],
                            "risk", "risk_target",
                        )["overall"],
                    } for arm in ARMS
                },
                "probability_shift": probability_shift(
                    domain["examples"], primary["predictions"][domain_name],
                    ablation["predictions"][domain_name],
                ),
                "action_comparison": action_comparison(
                    domain,
                    {"predictions": primary["predictions"][domain_name],
                     "config": primary["checkpoint"]["selected_config"]},
                    {"predictions": ablation["predictions"][domain_name],
                     "config": ablation["checkpoint"]["selected_config"]},
                ),
            }
        seeds[str(seed)] = domains
    lama_changes = [
        seeds[str(seed)]["lama"]["action_comparison"]["overall"]
        for seed in SEEDS
    ]
    change_fraction = statistics.fmean(
        row["changed_boundaries"] / len(development["lama"]["labels"])
        for row in lama_changes
    )
    net = statistics.fmean(row["correct_delta"] for row in lama_changes)
    benefit_ap_delta = statistics.fmean(
        seeds[str(seed)]["lama"]["ranking"]["three_source_equal"]["benefit"][
            "average_precision"
        ] - seeds[str(seed)]["lama"]["ranking"]["fewrel_removed_equal"][
            "benefit"
        ]["average_precision"] for seed in SEEDS
    )
    diagnosis = (
        "weak_positive_data_effect"
        if audit["fewrel_removal_lama_drop"] > 0 and net > 0
        else "policy_discretization_limited"
        if benefit_ap_delta > .01 and change_fraction < .05
        else "supervision_mismatch"
        if change_fraction >= .05 and net <= 0
        else "representation_shift_limited"
    )
    return {
        "protocol": PROTOCOL, "plan": PLAN, "passed": True,
        "training_summary_sha256": sha256(summary_path),
        "training_audit_sha256": sha256(audit_path),
        "upstream_hashes": hashes, "checkpoint_hashes": checkpoint_hashes,
        "target_support": target_support(train_domains),
        "source_geometry": source_geometry(train_domains, development),
        "seeds": seeds,
        "aggregate": {
            "lama_action_change_fraction": change_fraction,
            "lama_changed_action_net_correct": net,
            "lama_benefit_ap_delta": benefit_ap_delta,
            "fewrel_removal_lama_drop": audit["fewrel_removal_lama_drop"],
        },
        "diagnosis": diagnosis,
        "templama_test_loaded": False, "lama_confirmation_loaded": False,
        "longmemeval_loaded": False, "locomo_loaded": False,
        "next_stage": "close_v10_constructed_corpus_branch",
        "seconds": time.perf_counter() - started,
        "terminal_status": "v10_failure_diagnostic_completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--training-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = diagnose(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
