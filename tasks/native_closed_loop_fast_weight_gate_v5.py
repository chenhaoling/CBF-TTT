"""Closed-loop rollout of frozen V4 native fast-weight admission policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from tasks.build_native_layerwise_write_gate_v3 import PROTOCOL as DATA_PROTOCOL
from tasks.build_native_support_write_gate_v1 import sha256
from tasks.cbf_checkpoint_native import NativeSession, load_model
from tasks.cbf_checkpoint_readout import parameter_digest
from tasks.native_layerwise_write_gate_v2 import (
    LAYERS,
    SINGLETON_ACTIONS,
    compose_singleton,
    layer_hashes,
)
from tasks.native_support_write_gate_v1 import (
    _measure_cuda,
    branch_boundary,
    fast_weight_digest,
    fresh_from_weights,
    geometry_features,
    prediction_statistics,
    score_queries,
    support_features,
)
from tasks.train_native_eos_safe_layerwise_write_gate_v4 import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PROTOCOL as CONTROLLER_PROTOCOL,
    SEEDS,
    WEIGHT_DECAY,
    build_model,
    choose_actions,
    feature_indices,
    predict,
)


PROTOCOL = "native_closed_loop_fastweight_gate_v5"
PLAN = "NATIVE_CLOSED_LOOP_FAST_WEIGHT_GATE_V5_PLAN.md"
PHASE_B_PLAN = "NATIVE_CLOSED_LOOP_FAST_WEIGHT_GATE_V5_PHASE_B_PLAN.md"
PROJECTION_SEED = 2026101601
BOOTSTRAP_SEED = 2026101901
BOOTSTRAP_DRAWS = 20_000
VARIANTS = ("full", "no_support")
BASELINES = ("always_write", "never_write")


def canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def policy_name(variant: str, seed: int) -> str:
    return f"{variant}_seed{seed}"


def policy_specs() -> list[dict]:
    return [
        {"policy": "always_write", "kind": "baseline"},
        {"policy": "never_write", "kind": "baseline"},
        *[
            {"policy": policy_name(variant, seed), "kind": "controller",
             "variant": variant, "seed": seed}
            for variant in VARIANTS for seed in SEEDS
        ],
    ]


def load_inputs(data_root: Path, stage: str) -> tuple[list[dict], dict, dict]:
    manifest_path, audit_path = data_root / "manifest.json", data_root / "data_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("protocol") != DATA_PROTOCOL or not audit.get("passed") \
            or audit.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("V3 data dependency did not pass independent audit")
    if manifest.get("sealed_test_scored") or audit.get("test_opened"):
        raise ValueError("sealed test was opened")
    if stage not in ("smoke", "development"):
        raise ValueError("V5 may score only smoke or development")
    split = "confirmation" if stage == "smoke" else "development"
    path = data_root / f"{split}_model_inputs.jsonl"
    if manifest["files_sha256"].get(path.name) != sha256(path):
        raise ValueError(f"V3 {split} inputs changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    dependency = {
        "data_manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path),
        f"{split}_inputs_sha256": sha256(path),
    }
    if stage == "smoke":
        allowed = set(manifest["smoke_groups"])
        rows = [row for row in rows if row["group_id"] in allowed]
        if len(rows) != 12 or {row["group_id"] for row in rows} != allowed:
            raise ValueError("V5 smoke group contract changed")
        dependency["smoke_groups_sha256"] = canonical({"group_ids": sorted(allowed)})
    elif len(rows) != 96:
        raise ValueError("V5 development group contract changed")
    if any(len(row["boundaries"]) != 3 for row in rows):
        raise ValueError("V5 boundary contract changed")
    return rows, manifest, {
        **dependency,
    }


def load_smoke_inputs(data_root: Path) -> tuple[list[dict], dict, dict]:
    """Compatibility helper for the frozen Phase-A auditor."""
    return load_inputs(data_root, "smoke")


def load_controllers(root: Path) -> tuple[dict, dict]:
    import torch

    summary_path = root / "training_summary.json"
    audit_path = root / "training_independent_audit.json"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != CONTROLLER_PROTOCOL or summary.get("test_opened") \
            or summary.get("development_released_test") \
            or not audit.get("passed") or audit.get("test_released") or audit.get("test_opened") \
            or audit.get("summary_sha256") != sha256(summary_path):
        raise ValueError("expected independently audited, development-stopped V4 controllers")
    run_index = {(row["variant"], row["seed"]): row for row in summary["runs"]}
    controllers = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            recorded = run_index[variant, seed]
            checkpoint_path = Path(recorded["checkpoint"])
            if sha256(checkpoint_path) != recorded["checkpoint_sha256"]:
                raise ValueError("V4 checkpoint hash changed")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            indices = feature_indices(variant)
            if checkpoint["protocol"] != CONTROLLER_PROTOCOL \
                    or checkpoint["variant"] != variant or checkpoint["seed"] != seed \
                    or checkpoint["indices"] != indices or checkpoint["epochs"] != EPOCHS \
                    or checkpoint["batch_size"] != BATCH_SIZE \
                    or checkpoint["learning_rate"] != LEARNING_RATE \
                    or checkpoint["weight_decay"] != WEIGHT_DECAY:
                raise ValueError("V4 controller contract changed")
            model = build_model(len(indices)); model.load_state_dict(checkpoint["state_dict"])
            model.eval().requires_grad_(False)
            controllers[policy_name(variant, seed)] = {
                "model": model, "stats": checkpoint["stats"], "indices": indices,
                "variant": variant, "seed": seed,
                "checkpoint_sha256": recorded["checkpoint_sha256"],
            }
    return controllers, {
        "controller_summary_sha256": sha256(summary_path),
        "controller_independent_audit_sha256": sha256(audit_path),
    }


def select_branch(reject: NativeSession, admit: NativeSession, action: str):
    if action == "reject_all":
        selected = fresh_from_weights(reject)
        sources = {str(layer): "reject" for layer in LAYERS}
    elif action == "admit_all":
        selected = fresh_from_weights(admit)
        sources = {str(layer): "admit" for layer in LAYERS}
    elif action in SINGLETON_ACTIONS:
        layer = LAYERS[SINGLETON_ACTIONS.index(action)]
        selected, sources = compose_singleton(reject, admit, layer)
    else:
        raise ValueError(f"unknown native action: {action}")
    return selected, sources


def collect(args) -> dict:
    import torch

    data_root, controller_root = Path(args.data_root), Path(args.controller_root)
    rows, _, data_dependency = load_inputs(data_root, args.stage)
    controllers, controller_dependency = load_controllers(controller_root)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    if not rows:
        raise ValueError("empty V5 shard")
    output, decisions_output = Path(args.output), Path(args.decisions_output)
    if output.exists() or decisions_output.exists():
        raise FileExistsError("refusing to overwrite V5 artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PROJECTION_SEED); torch.backends.cuda.matmul.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    if tuple(model.config.ttt_layers) != LAYERS or model.config.ttt_chunk != 4096:
        raise ValueError("native checkpoint layer/chunk contract changed")
    before = parameter_digest(model); versions = [parameter._version for parameter in model.parameters()]
    device = next(model.parameters()).device; resources = []
    specs = policy_specs()
    with output.open("x") as label_sink, decisions_output.open("x") as decision_sink:
        for group_index, row in enumerate(rows):
            torch.cuda.empty_cache()
            setup = NativeSession(model); setup._forward(row["setup_chunk_ids"])
            setup = fresh_from_weights(setup)
            for spec in specs:
                parent = fresh_from_weights(setup); recent = None
                for boundary in row["boundaries"]:
                    def step():
                        nonlocal parent, recent
                        is_controller = spec["kind"] == "controller"
                        reject, admit, hidden, current, branch_audit = branch_boundary(
                            parent, boundary["chunk_ids"], collect_support=is_controller
                        )
                        reject_hashes, admit_hashes = layer_hashes(reject), layer_hashes(admit)
                        decision = {
                            "protocol": PROTOCOL, "stage": args.stage, "group_id": row["group_id"],
                            "boundary": boundary["boundary"], "policy": spec["policy"],
                            "kind": spec["kind"], "parent_fast_sha256": fast_weight_digest(parent),
                        }
                        recent_next = recent
                        if is_controller:
                            labels = torch.tensor(
                                boundary["chunk_ids"][1:], dtype=torch.long, device=device
                            )
                            uncertainty = prediction_statistics(model, hidden[:, :-1], labels)
                            support, recent_next = support_features(current, recent, list(LAYERS))
                            geometry, geometry_audit = geometry_features(model, parent, admit)
                            features = [
                                *uncertainty.values(), *geometry, *support,
                                boundary["boundary"] / 2.0, 1.0,
                            ]
                            controller = controllers[spec["policy"]]
                            x = torch.tensor([features], dtype=torch.float32)
                            prediction = predict(
                                controller["model"], x[:, controller["indices"]], controller["stats"]
                            )
                            action = choose_actions(*prediction)[0]
                            decision.update({
                                "variant": spec["variant"], "seed": spec["seed"],
                                "checkpoint_sha256": controller["checkpoint_sha256"],
                                "features": features,
                                "feature_schema": {
                                    "uncertainty": list(uncertainty), "geometry_per_layer": 5,
                                    "support_per_layer": 259,
                                    "tail": ["boundary_fraction", "bias"],
                                },
                                "online_memory": recent_next, "uncertainty": uncertainty,
                                "geometry": geometry_audit,
                                "predicted_benefit": prediction[0][0].tolist(),
                                "predicted_protected_nll_change": prediction[1][0].tolist(),
                                "predicted_accuracy_harm_probability": prediction[2][0].tolist(),
                                "predicted_eos_harm_probability": prediction[3][0].tolist(),
                                "action": action,
                            })
                        else:
                            action = "admit_all" if spec["policy"] == "always_write" else "reject_all"
                            decision.update({"online_memory": {}, "action": action})
                        selected, sources = select_branch(reject, admit, action)
                        selected_hashes = layer_hashes(selected)
                        selected_fast = fast_weight_digest(selected)
                        outcome = score_queries(selected, boundary["queries"], args.eos_id)
                        if fast_weight_digest(selected) != selected_fast:
                            raise RuntimeError("query scoring changed selected fast weights")
                        next_parent = fresh_from_weights(selected)
                        decision_sha = canonical(decision)
                        label = {
                            "protocol": PROTOCOL, "stage": args.stage, "group_id": row["group_id"],
                            "boundary": boundary["boundary"], "policy": spec["policy"],
                            "decision_sha256": decision_sha, "action": action,
                            "branch_audit": branch_audit,
                            "base_layer_sha256": {"reject": reject_hashes, "admit": admit_hashes},
                            "composition": {"sources": sources, "layer_sha256": selected_hashes},
                            "next_fast_sha256": fast_weight_digest(next_parent), "outcome": outcome,
                        }
                        parent, recent = next_parent, recent_next
                        return decision, label

                    result, resource = _measure_cuda(step, device)
                    decision, label = result; label["resource"] = resource; resources.append(resource)
                    decision_sink.write(json.dumps(decision) + "\n"); decision_sink.flush()
                    label_sink.write(json.dumps(label) + "\n"); label_sink.flush()
                    print(json.dumps({
                        "group": row["group_id"], "policy": spec["policy"],
                        "boundary": boundary["boundary"], **resource,
                    }), flush=True)
            print(json.dumps({"completed_groups": group_index + 1, "total_groups": len(rows)}), flush=True)
    if before != parameter_digest(model) or versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("frozen Qwen parameters changed")
    sidecar = {
        "protocol": PROTOCOL, "stage": args.stage, "shard": args.shard,
        "num_shards": args.num_shards, "groups": len(rows), "policies": len(specs),
        "transitions": len(resources), "rows_sha256": sha256(output),
        "decisions_sha256": sha256(decisions_output), "decisions_path": str(decisions_output),
        "plan_sha256": sha256(Path(args.plan)), **data_dependency, **controller_dependency,
        "model": model_meta, "weights_sha256": before, "backbone_unchanged": True,
        "mean_transition_seconds": statistics.fmean(row["seconds"] for row in resources),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in resources),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in resources),
        "test_opened": False,
    }
    Path(str(output) + ".audit.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def bootstrap(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed); size = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return [estimates[int(.025 * BOOTSTRAP_DRAWS)], estimates[int(.975 * BOOTSTRAP_DRAWS) - 1]]


def load_bound_artifacts(label_paths: list[Path], decision_paths: list[Path]):
    labels = [json.loads(line) for path in label_paths for line in path.read_text().splitlines() if line.strip()]
    decisions = [json.loads(line) for path in decision_paths for line in path.read_text().splitlines() if line.strip()]
    key = lambda row: (row["group_id"], row["policy"], row["boundary"])
    decision_index = {key(row): row for row in decisions}
    if len(decision_index) != len(decisions) or len(labels) != len(decisions):
        raise ValueError("V5 decision/label coverage")
    for row in labels:
        if key(row) not in decision_index or canonical(decision_index[key(row)]) != row["decision_sha256"]:
            raise ValueError("V5 decision hash dependency")
    return labels, [decision_index[key(row)] for row in labels]


def summarize(label_paths: list[Path], decision_paths: list[Path], output: Path) -> dict:
    labels, decisions = load_bound_artifacts(label_paths, decision_paths)
    stages = {row["stage"] for row in labels}
    if len(stages) != 1:
        raise ValueError("mixed V5 stages")
    stage = stages.pop()
    specs = policy_specs(); names = [row["policy"] for row in specs]
    groups = sorted({row["group_id"] for row in labels})
    expected = len(groups) * len(names) * 3
    expected_groups = {"smoke": 12, "development": 96}[stage]
    if len(groups) != expected_groups or len(labels) != expected \
            or set(row["policy"] for row in labels) != set(names):
        raise ValueError(f"incomplete V5 {stage} matrix")
    index = {(row["group_id"], row["policy"], row["boundary"]): row for row in labels}
    trajectory_ok = True
    for group in groups:
        first_parents = set()
        for name in names:
            part = [index[group, name, boundary] for boundary in range(3)]
            first_parents.add(part[0]["branch_audit"]["parent_fast_sha256"])
            trajectory_ok &= all(
                part[boundary]["next_fast_sha256"]
                == part[boundary + 1]["branch_audit"]["parent_fast_sha256"]
                for boundary in range(2)
            )
        trajectory_ok &= len(first_parents) == 1

    def mean(name: str, key: str) -> float:
        return statistics.fmean(row["outcome"][key] for row in labels if row["policy"] == name)

    def group_values(name: str, key: str) -> list[float]:
        return [statistics.fmean(
            index[group, name, boundary]["outcome"][key] for boundary in range(3)
        ) for group in groups]

    metrics = {}
    never_groups = group_values("never_write", "J")
    for offset, name in enumerate(names):
        policy_groups = group_values(name, "J")
        gains = [base - value for base, value in zip(never_groups, policy_groups)]
        actions = [row["action"] for row in labels if row["policy"] == name]
        metrics[name] = {
            "J": mean(name, "J"), "protected_accuracy": mean(name, "protected_accuracy"),
            "protected_nll": mean(name, "protected_nll"), "eos_rate": mean(name, "eos_rate"),
            "accuracy": mean(name, "accuracy"), "write_rate": sum(a != "reject_all" for a in actions) / len(actions),
            "actions": {action: actions.count(action) for action in ("reject_all", *SINGLETON_ACTIONS, "admit_all")},
            "gain_vs_never": statistics.fmean(gains),
            "gain_vs_never_group_bootstrap_95ci": bootstrap(gains, BOOTSTRAP_SEED + offset),
        }
    never = metrics["never_write"]
    seed_results = {}
    for seed in SEEDS:
        full_name, no_name = policy_name("full", seed), policy_name("no_support", seed)
        full_groups, no_groups = group_values(full_name, "J"), group_values(no_name, "J")
        support_gains = [base - value for base, value in zip(no_groups, full_groups)]
        checks = {
            "better_than_never": metrics[full_name]["J"] < never["J"],
            "better_than_matched_no_support": metrics[full_name]["J"] < metrics[no_name]["J"],
            "protected_accuracy_safe": metrics[full_name]["protected_accuracy"] >= never["protected_accuracy"] - .005,
            "protected_nll_safe": metrics[full_name]["protected_nll"] <= never["protected_nll"] + .01,
            "eos_safe": metrics[full_name]["eos_rate"] <= never["eos_rate"] + .01,
        }
        seed_results[str(seed)] = {
            "checks": checks,
            "gain_vs_matched_no_support": statistics.fmean(support_gains),
            "gain_vs_matched_no_support_group_bootstrap_95ci": bootstrap(
                support_gains, BOOTSTRAP_SEED + seed % 1000 + 100
            ),
        }
    better_never = sum(row["checks"]["better_than_never"] for row in seed_results.values())
    better_no_support = sum(row["checks"]["better_than_matched_no_support"] for row in seed_results.values())
    mean_full_gain = statistics.fmean(metrics[policy_name("full", seed)]["gain_vs_never"] for seed in SEEDS)
    common_checks = {
        "all_full_safety_checks": all(
            all(row["checks"][key] for key in ("protected_accuracy_safe", "protected_nll_safe", "eos_safe"))
            for row in seed_results.values()
        ),
        "trajectory_continuity": trajectory_ok,
        "branch_and_composition": all(
            row["branch_audit"]["parent_fast_sha256"] == row["branch_audit"]["reject_fast_sha256"]
            and all(
                row["composition"]["layer_sha256"][str(layer)]
                == row["base_layer_sha256"][row["composition"]["sources"][str(layer)]][str(layer)]
                for layer in LAYERS
            ) for row in labels
        ),
    }
    if stage == "smoke":
        checks = {
            "at_least_two_full_better_than_never": better_never >= 2,
            "mean_full_gain_vs_never_positive": mean_full_gain > 0,
            "at_least_two_full_better_than_no_support": better_no_support >= 2,
            **common_checks,
        }
        next_stage = "freeze_phase_b_closed_loop_development" if all(checks.values()) \
                     else "stop_before_phase_b_and_test"
    else:
        checks = {
            "all_full_better_than_never": better_never == len(SEEDS),
            "all_full_bootstrap_vs_never_lower_positive": all(
                metrics[policy_name("full", seed)]["gain_vs_never_group_bootstrap_95ci"][0] > 0
                for seed in SEEDS
            ),
            "all_full_better_than_no_support": better_no_support == len(SEEDS),
            **common_checks,
        }
        next_stage = "freeze_fresh_source_formal_v5" if all(checks.values()) \
                     else "stop_before_formal_corpus_and_test"
    result = {
        "protocol": PROTOCOL, "stage": stage, "groups": len(groups),
        "policies": len(names), "transitions": len(labels), "metrics": metrics,
        "seed_results": seed_results, "mean_full_gain_vs_never": mean_full_gain,
        "checks": checks, "passed": all(checks.values()), "test_opened": False,
        "next_stage": next_stage,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True); collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--controller-root", required=True); collect_parser.add_argument("--plan", default=PLAN)
    collect_parser.add_argument("--stage", choices=("smoke", "development"), default="smoke")
    collect_parser.add_argument("--shard", type=int, default=0); collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--eos-id", type=int, default=151645)
    collect_parser.add_argument("--output", required=True); collect_parser.add_argument("--decisions-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--labels", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--decisions", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps(collect(args), indent=2))
    else:
        print(json.dumps(summarize(args.labels, args.decisions, args.output), indent=2))


if __name__ == "__main__":
    main()
