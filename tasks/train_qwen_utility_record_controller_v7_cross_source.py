"""Train and select the preregistered V7 cross-source residual controller."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import time

from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.evaluate_qwen_utility_record_controller_v1_development import (
    fixed_policy_actions,
)
from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    metric,
)
from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    fact_key_matches,
    fact_key_projected_actions,
    safe_fact_key_oracle_actions,
)
from tasks.extract_qwen_utility_record_controller_v1_features import FEATURE_NAMES
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl
from tasks.train_qwen_utility_record_controller_v5_residual import (
    CHECKPOINT_EPOCHS,
    CONFORMAL_LEVELS,
    LOCALITY_RULE,
    RISK_PENALTIES,
    SEEDS,
    UTILITY_THRESHOLDS,
    config_name,
    conformal_risk_threshold,
    fit,
    load_feature_rows,
    load_labels,
    predict_state,
    residual_actions,
    residual_examples,
    residual_inference_candidates,
    residual_precision,
)


PROTOCOL = "qwen_utility_record_controller_v7_cross_source_training"
AUDIT_PROTOCOL = "qwen_utility_record_controller_v7_cross_source_training_audit"
PLAN = "QWEN_UTILITY_RECORD_CONTROLLER_V7_CROSS_SOURCE_PLAN.md"

EXPECTED_TRAIN = {
    "templama": {"examples": 4392, "benefit": 685, "risk": 201, "groups": 512},
    "lama": {"examples": 4684, "benefit": 747, "risk": 160, "groups": 512},
}
EXPECTED_DEVELOPMENT = {
    "templama": {"examples": 2204, "groups": 256},
    "lama": {"examples": 588, "groups": 64},
}
AUDIT_FILES = {
    "templama_train_labels": "train_labels_audit.json",
    "templama_train_features": "train_feature_audit.json",
    "lama_train_labels": "formal_labels_audit.json",
    "lama_train_features": "feature_audit.json",
    "templama_dev_data": "data_audit.json",
    "templama_dev_labels": "development_labels_audit.json",
    "templama_dev_features": "development_feature_audit.json",
    "lama_dev_data": "development_data_audit.json",
    "lama_dev_labels": "development_labels_audit.json",
    "lama_dev_features": "development_feature_audit.json",
}


def passed_audit(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not value.get("passed"):
        raise ValueError(f"upstream audit did not pass: {path}")
    return value


def attach_relations(labels: list[dict], feature_rows: list[dict]) -> None:
    features = {
        (row["episode_id"], row["boundary_index"]): row for row in feature_rows
    }
    if len(features) != len(feature_rows):
        raise ValueError("duplicate V7 feature identity")
    for row in labels:
        identity = (row["episode_id"], row["boundary_index"])
        relation = features[identity]["audit"]["target_relation"]
        if "target_relation" in row and row["target_relation"] != relation:
            raise ValueError("label/feature relation disagreement")
        row["target_relation"] = relation


def load_training_domain(label_root: Path, feature_root: Path) -> dict:
    labels = load_labels(label_root)
    features = load_feature_rows(feature_root)
    attach_relations(labels, features)
    examples = residual_examples(labels, features)
    return {"labels": labels, "features": features, "examples": examples}


def load_development_domain(label_root: Path, feature_root: Path,
                            data_root: Path, episode_name: str) -> dict:
    value = load_training_domain(label_root, feature_root)
    episodes = load_jsonl(data_root / episode_name)
    matches = fact_key_matches(value["labels"], episodes)
    feature_map = {
        (row["episode_id"], row["boundary_index"]): row
        for row in value["features"]
    }
    physics_actions = fixed_policy_actions(
        value["labels"], feature_map, "explicit_lifecycle_physics"
    )
    oracle_actions = safe_fact_key_oracle_actions(
        value["labels"], value["features"], matches
    )
    physics = metric(value["labels"], physics_actions, feature_map)
    oracle = metric(value["labels"], oracle_actions, feature_map)
    value.update({
        "episodes": episodes,
        "matches": matches,
        "feature_map": feature_map,
        "physics_actions": physics_actions,
        "physics": physics,
        "oracle": oracle,
        "oracle_headroom": oracle["accuracy"] - physics["accuracy"],
        "candidates": residual_inference_candidates(
            value["labels"], value["features"]
        ),
    })
    return value


def domain_metric(domain: dict, predictions: dict, config: dict) -> dict:
    raw = residual_actions(
        domain["labels"], domain["features"], predictions, config
    )
    actions = fact_key_projected_actions(
        domain["labels"], domain["features"], raw, domain["matches"]
    )
    value = metric(domain["labels"], actions, domain["feature_map"])
    precision, fraction, selected = residual_precision(
        domain["examples"], actions, domain["labels"]
    )
    gain = value["accuracy"] - domain["physics"]["accuracy"]
    mechanism_gain = {
        name: value["per_mechanism_accuracy"][name]
        - domain["physics"]["per_mechanism_accuracy"][name]
        for name in value["per_mechanism_accuracy"]
    }
    return {
        **value,
        "gain_over_lifecycle_physics": gain,
        "oracle_gap_closure": (
            gain / domain["oracle_headroom"]
            if domain["oracle_headroom"] > 0 else 0.0
        ),
        "residual_deletion_precision": precision,
        "residual_deletion_fraction": fraction,
        "residual_deletions": selected,
        "filtered_cross_fact_deletions": sum(
            raw[identity] != actions[identity] for identity in actions
        ),
        "per_mechanism_gain": mechanism_gain,
    }


def select_config(candidates: dict[str, dict]) -> str | None:
    feasible = [
        name for name, row in candidates.items()
        if all(value["added_unrelated_errors"] == 0
               for value in row["domains"].values())
    ]
    if not feasible:
        return None

    def key(name):
        row = candidates[name]
        values = list(row["domains"].values())
        config = row["config"]
        closures = [value["oracle_gap_closure"] for value in values]
        gains = [value["gain_over_lifecycle_physics"] for value in values]
        return (
            min(closures), statistics.fmean(closures),
            min(gains), statistics.fmean(gains),
            statistics.fmean(value["residual_deletion_precision"] for value in values),
            -statistics.fmean(value["residual_deletion_fraction"] for value in values),
            -config["epoch"], config["utility_threshold"],
            config["risk_penalty"], -config["conformal_level"], name,
        )

    return max(feasible, key=key)


def load_inputs(args) -> tuple[dict, dict, dict]:
    roots = {
        name: Path(getattr(args, name)) for name in (
            "templama_train_label_root", "templama_train_feature_root",
            "lama_train_label_root", "lama_train_feature_root",
            "templama_dev_data_root", "templama_dev_label_root",
            "templama_dev_feature_root", "lama_dev_data_root",
            "lama_dev_label_root", "lama_dev_feature_root",
        )
    }
    audit_paths = {
        "templama_train_labels": roots["templama_train_label_root"]
        / AUDIT_FILES["templama_train_labels"],
        "templama_train_features": roots["templama_train_feature_root"]
        / AUDIT_FILES["templama_train_features"],
        "lama_train_labels": roots["lama_train_label_root"]
        / AUDIT_FILES["lama_train_labels"],
        "lama_train_features": roots["lama_train_feature_root"]
        / AUDIT_FILES["lama_train_features"],
        "templama_dev_data": roots["templama_dev_data_root"]
        / AUDIT_FILES["templama_dev_data"],
        "templama_dev_labels": roots["templama_dev_label_root"]
        / AUDIT_FILES["templama_dev_labels"],
        "templama_dev_features": roots["templama_dev_feature_root"]
        / AUDIT_FILES["templama_dev_features"],
        "lama_dev_data": roots["lama_dev_data_root"]
        / AUDIT_FILES["lama_dev_data"],
        "lama_dev_labels": roots["lama_dev_label_root"]
        / AUDIT_FILES["lama_dev_labels"],
        "lama_dev_features": roots["lama_dev_feature_root"]
        / AUDIT_FILES["lama_dev_features"],
    }
    for path in audit_paths.values():
        passed_audit(path)
    train = {
        "templama": load_training_domain(
            roots["templama_train_label_root"], roots["templama_train_feature_root"]
        ),
        "lama": load_training_domain(
            roots["lama_train_label_root"], roots["lama_train_feature_root"]
        ),
    }
    development = {
        "templama": load_development_domain(
            roots["templama_dev_label_root"], roots["templama_dev_feature_root"],
            roots["templama_dev_data_root"], "development_episodes.jsonl",
        ),
        "lama": load_development_domain(
            roots["lama_dev_label_root"], roots["lama_dev_feature_root"],
            roots["lama_dev_data_root"], "development_episodes.jsonl",
        ),
    }
    return train, development, {
        name: sha256(path) for name, path in audit_paths.items()
    }


def evaluate_state(state: dict, means: list[float], stds: list[float],
                   development: dict, epoch: int) -> dict[str, dict]:
    predictions_by_domain = {
        name: predict_state(state, domain["examples"], means, stds)
        for name, domain in development.items()
    }
    output = {}
    for level in CONFORMAL_LEVELS:
        threshold = min(
            conformal_risk_threshold(
                predictions_by_domain[name], domain["examples"], level
            )
            for name, domain in development.items()
        )
        for utility in UTILITY_THRESHOLDS:
            for penalty in RISK_PENALTIES:
                name = config_name(epoch, utility, penalty, level)
                config = {
                    "epoch": epoch,
                    "utility_threshold": utility,
                    "risk_penalty": penalty,
                    "conformal_level": level,
                    "risk_threshold": threshold,
                    "locality_rule": LOCALITY_RULE,
                    "maximum_extra_deletions": 1,
                    "risk_threshold_rule": "minimum_across_development_domains",
                }
                output[name] = {
                    "config": config,
                    "domains": {
                        domain_name: domain_metric(
                            domain, predictions_by_domain[domain_name], config
                        )
                        for domain_name, domain in development.items()
                    },
                }
    return output


def development_gate(seed_results: list[dict], development: dict,
                     upstream_passed: bool) -> dict:
    mean_gain = {
        domain: statistics.fmean(
            seed["selected_metric"]["domains"][domain][
                "gain_over_lifecycle_physics"
            ] for seed in seed_results
        ) for domain in development
    }
    mean_closure = {
        domain: statistics.fmean(
            seed["selected_metric"]["domains"][domain]["oracle_gap_closure"]
            for seed in seed_results
        ) for domain in development
    }
    mean_mechanism = {
        domain: {
            mechanism: statistics.fmean(
                seed["selected_metric"]["domains"][domain][
                    "per_mechanism_gain"
                ][mechanism] for seed in seed_results
            )
            for mechanism in development[domain]["physics"]["per_mechanism_accuracy"]
        } for domain in development
    }
    checks = {
        "oracle_headroom_at_least_point_015_each_domain": all(
            domain["oracle_headroom"] >= .015 for domain in development.values()
        ),
        "every_seed_positive_each_domain": all(
            seed["selected_metric"]["domains"][domain][
                "gain_over_lifecycle_physics"
            ] > 0
            for seed in seed_results for domain in development
        ),
        "mean_gain_at_least_point_015_each_domain": all(
            value >= .015 for value in mean_gain.values()
        ),
        "zero_collateral_every_seed_domain": all(
            seed["selected_metric"]["domains"][domain][
                "added_unrelated_errors"
            ] == 0
            for seed in seed_results for domain in development
        ),
        "mean_oracle_closure_at_least_point_50_each_domain": all(
            value >= .50 for value in mean_closure.values()
        ),
        "overwrite_and_stale_nonnegative_each_domain": all(
            values[mechanism] >= 0
            for values in mean_mechanism.values()
            for mechanism in ("semantic_overwrite", "stale_repetition")
        ),
        "upstream_audits_passed": upstream_passed,
    }
    return {
        "checks": checks,
        "passed_pending_independent_audit": all(checks.values()),
        "mean_gain": mean_gain,
        "mean_oracle_gap_closure": mean_closure,
        "mean_per_mechanism_gain": mean_mechanism,
    }


def train(args) -> dict:
    import torch

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    train_domains, development, audit_hashes = load_inputs(args)
    combined = []
    identity_domains = {}
    for domain_name, domain in train_domains.items():
        for row in domain["examples"]:
            identity = (row["episode_id"], row["boundary_index"], row["slot"])
            if identity in identity_domains:
                raise ValueError("cross-domain training identity collision")
            identity_domains[identity] = domain_name
            combined.append({**row, "training_domain": domain_name})
    counts = {
        name: {
            "examples": len(domain["examples"]),
            "benefit": sum(row["benefit_target"] for row in domain["examples"]),
            "risk": sum(row["risk_target"] for row in domain["examples"]),
            "groups": len({row["source_group"] for row in domain["labels"]}),
            "relations": len({row["target_relation"] for row in domain["labels"]}),
        } for name, domain in train_domains.items()
    }
    development_counts = {
        name: {
            "examples": len(domain["examples"]),
            "groups": len({row["source_group"] for row in domain["labels"]}),
            "relations": len({row["target_relation"] for row in domain["labels"]}),
        } for name, domain in development.items()
    }
    if any(
        any(counts[name][key] != value for key, value in expected.items())
        for name, expected in EXPECTED_TRAIN.items()
    ):
        raise ValueError("V7 training support differs from preregistration")
    if any(
        any(development_counts[name][key] != value
            for key, value in expected.items())
        for name, expected in EXPECTED_DEVELOPMENT.items()
    ):
        raise ValueError("V7 development support differs from preregistration")

    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    seed_results = []
    for seed in SEEDS:
        fitted = fit(combined, seed, device)
        candidates = {}
        for epoch in CHECKPOINT_EPOCHS:
            candidates.update(evaluate_state(
                fitted["states"][epoch], fitted["means"], fitted["stds"],
                development, epoch,
            ))
        selected = select_config(candidates)
        if selected is None:
            raise ValueError(f"V7 seed {seed} has no dual-domain safe config")
        selected_config = candidates[selected]["config"]
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        checkpoint = {
            "protocol": PROTOCOL,
            "seed": seed,
            "feature_names": list(FEATURE_NAMES),
            "input_dim": len(FEATURE_NAMES),
            "state_dict": fitted["states"][selected_config["epoch"]],
            "means": fitted["means"],
            "stds": fitted["stds"],
            "class_weights": fitted["class_weights"],
            "selected_config": selected_config,
            "upstream_audit_hashes": audit_hashes,
            "training_domains": sorted(train_domains),
        }
        checkpoint_path = seed_root / "controller.pt"
        torch.save(checkpoint, checkpoint_path)
        result = {
            "seed": seed,
            "selected": selected,
            "selected_config": selected_config,
            "selected_metric": candidates[selected],
            "class_weights": fitted["class_weights"],
            "trace": fitted["trace"],
            "candidate_count": len(candidates),
            "checkpoint_sha256": sha256(checkpoint_path),
            "passed": True,
        }
        (seed_root / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        seed_results.append(result)
    gate = development_gate(seed_results, development, True)
    checks = {
        "registered_training_support": counts == {
            "templama": {**EXPECTED_TRAIN["templama"], "relations": 9},
            "lama": {**EXPECTED_TRAIN["lama"], "relations": 17},
        },
        "registered_development_support": development_counts == {
            "templama": {**EXPECTED_DEVELOPMENT["templama"], "relations": 9},
            "lama": {**EXPECTED_DEVELOPMENT["lama"], "relations": 5},
        },
        "near_equal_domain_mass": all(
            .45 <= counts[name]["examples"] / len(combined) <= .55
            for name in counts
        ),
        "all_checkpoints_written": all(
            (output_root / f"seed_{seed}" / "controller.pt").is_file()
            for seed in SEEDS
        ),
        "all_selected_configs_dual_domain_safe": all(
            all(value["added_unrelated_errors"] == 0
                for value in row["selected_metric"]["domains"].values())
            for row in seed_results
        ),
        "finite_training": all(
            math.isfinite(epoch["loss"]) and math.isfinite(epoch["gradient_norm"])
            for row in seed_results for epoch in row["trace"]
        ),
        "heldout_sources_not_loaded": True,
    }
    passed = all(checks.values())
    summary = {
        "protocol": PROTOCOL,
        "plan": PLAN,
        "checks": checks,
        "passed": passed,
        "counts": counts,
        "development_counts": development_counts,
        "development_baselines": {
            name: {
                "lifecycle_physics": domain["physics"],
                "safe_single_fact_key_oracle": domain["oracle"],
                "safe_single_fact_key_oracle_headroom": domain["oracle_headroom"],
            } for name, domain in development.items()
        },
        "seeds": seed_results,
        "development_gate": gate,
        "seconds": time.perf_counter() - started,
        "upstream_audit_hashes": audit_hashes,
        "feature_names": list(FEATURE_NAMES),
        "templama_test_loaded": False,
        "lama_confirmation_loaded": False,
        "longmemeval_loaded": False,
        "locomo_loaded": False,
        "next_stage": "v7_independent_training_audit" if passed else "stop_v7",
        "terminal_status": (
            "v7_cross_source_training_completed" if passed
            else "v7_cross_source_training_failed"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def add_arguments(parser) -> None:
    parser.add_argument("--templama-train-label-root", required=True)
    parser.add_argument("--templama-train-feature-root", required=True)
    parser.add_argument("--lama-train-label-root", required=True)
    parser.add_argument("--lama-train-feature-root", required=True)
    parser.add_argument("--templama-dev-data-root", required=True)
    parser.add_argument("--templama-dev-label-root", required=True)
    parser.add_argument("--templama-dev-feature-root", required=True)
    parser.add_argument("--lama-dev-data-root", required=True)
    parser.add_argument("--lama-dev-label-root", required=True)
    parser.add_argument("--lama-dev-feature-root", required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
