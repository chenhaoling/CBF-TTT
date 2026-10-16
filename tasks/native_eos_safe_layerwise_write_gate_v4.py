"""EOS-safe formal labels and support features for native layer admission V4."""

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
    ALL_ACTIONS,
    BENEFIT_EPS,
    LAYERS,
    SINGLETON_ACTIONS,
    compose_singleton,
    layer_hashes,
)
from tasks.native_support_write_gate_v1 import (
    _measure_cuda,
    branch_boundary,
    fresh_from_weights,
    geometry_features,
    prediction_statistics,
    score_queries,
    support_features,
)


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026101702
PROJECTION_SEED = 2026101601
PROTOCOL = "native_eos_safe_layerwise_fastweight_admission_v4"
PLAN = "NATIVE_EOS_SAFE_LAYERWISE_WRITE_GATE_V4_PLAN.md"


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_inputs(data_root: Path, stage: str) -> tuple[list[dict], dict, dict]:
    manifest = json.loads((data_root / "manifest.json").read_text())
    audit = json.loads((data_root / "data_audit.json").read_text())
    if manifest.get("protocol") != DATA_PROTOCOL or not audit.get("passed"):
        raise ValueError("V3 corpus did not pass independent audit")
    if audit.get("manifest_sha256") != sha256(data_root / "manifest.json"):
        raise ValueError("V3 corpus changed after audit")
    if stage not in ("train", "development"):
        raise ValueError("V4 may score only train or development before test release")
    path = data_root / f"{stage}_model_inputs.jsonl"
    if manifest["files_sha256"].get(path.name) != sha256(path):
        raise ValueError(f"{stage} input hash mismatch")
    if manifest.get("sealed_test_scored") or audit.get("test_opened"):
        raise ValueError("sealed test was opened before confirmation")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if stage == "smoke":
        allowed = set(manifest["smoke_groups"])
        rows = [row for row in rows if row["group_id"] in allowed]
    return rows, manifest, audit


def split_artifacts(result: dict) -> tuple[dict, dict]:
    identity = {key: result[key] for key in ("protocol", "stage", "group_id", "boundary")}
    feature = {
        **identity,
        "features": result["features"],
        "feature_schema": result["feature_schema"],
        "online_memory": result["online_memory"],
        "uncertainty": result["uncertainty"],
        "geometry": result["geometry"],
    }
    feature_sha = canonical_hash(feature)
    label = {
        **identity,
        "feature_sha256": feature_sha,
        "branch_audit": result["branch_audit"],
        "base_layer_sha256": result["base_layer_sha256"],
        "composition": result["composition"],
        "actions": result["actions"],
        "benefit": result["benefit"],
        "protected_nll_change": result["protected_nll_change"],
        "protected_accuracy_harm": result["protected_accuracy_harm"],
        "eos_rate_change": result["eos_rate_change"],
        "eos_increase_harm": result["eos_increase_harm"],
        "resource": result["resource"],
    }
    return feature, label


def collect(args) -> dict:
    import torch

    data_root = Path(args.data_root)
    rows, manifest, data_audit = load_inputs(data_root, args.stage)
    plan_path, v3_root = Path(args.plan), Path(args.v3_root)
    diagnostic_path = v3_root / "confirmation_failure_diagnostic.json"
    confirmation_audit_path = v3_root / "confirmation_independent_audit.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    confirmation_audit = json.loads(confirmation_audit_path.read_text())
    constrained = diagnostic.get("accuracy_and_eos_safe_oracle", {})
    if not confirmation_audit.get("passed") or confirmation_audit.get("test_opened") \
            or not diagnostic.get("v3_gate_remains_failed") \
            or constrained.get("gain_group_bootstrap_95ci", [0])[0] <= 0 \
            or constrained.get("eos_rate_delta", 1) > 0:
        raise ValueError("V4 design prerequisite did not pass")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    if not rows:
        raise ValueError("empty V3 shard")
    output, feature_output = Path(args.output), Path(args.features_output)
    if output.exists() or feature_output.exists():
        raise FileExistsError("refusing to overwrite V3 artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PROJECTION_SEED); torch.backends.cuda.matmul.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    if tuple(model.config.ttt_layers) != LAYERS or model.config.ttt_chunk != 4096:
        raise ValueError("native checkpoint layer/chunk contract changed")
    before = parameter_digest(model); versions = [parameter._version for parameter in model.parameters()]
    device = next(model.parameters()).device; resources = []
    with output.open("x") as label_sink, feature_output.open("x") as feature_sink:
        for group_index, row in enumerate(rows):
            torch.cuda.empty_cache()
            parent = NativeSession(model); parent._forward(row["setup_chunk_ids"])
            parent = fresh_from_weights(parent); recent = None
            for boundary in row["boundaries"]:
                def label_one():
                    nonlocal parent, recent
                    reject, admit, hidden, current, branch_audit = branch_boundary(
                        parent, boundary["chunk_ids"], collect_support=True
                    )
                    labels = torch.tensor(boundary["chunk_ids"][1:], dtype=torch.long, device=device)
                    uncertainty = prediction_statistics(model, hidden[:, :-1], labels)
                    support, recent_next = support_features(current, recent, list(LAYERS))
                    geometry, geometry_audit = geometry_features(model, parent, admit)
                    reject_hashes, admit_hashes = layer_hashes(reject), layer_hashes(admit)
                    actions = {
                        "reject_all": score_queries(reject, boundary["queries"], args.eos_id),
                        "admit_all": score_queries(admit, boundary["queries"], args.eos_id),
                    }
                    composition = {}
                    for layer, action in zip(LAYERS, SINGLETON_ACTIONS):
                        singleton, sources = compose_singleton(reject, admit, layer)
                        hashes = layer_hashes(singleton)
                        if any(hashes[str(item)] != (admit_hashes if item == layer else reject_hashes)[str(item)]
                               for item in LAYERS):
                            raise RuntimeError("V3 singleton composition mismatch")
                        actions[action] = score_queries(singleton, boundary["queries"], args.eos_id)
                        composition[action] = {"sources": sources, "layer_sha256": hashes}
                    benefits = {
                        action: actions["reject_all"]["J"] - actions[action]["J"]
                        for action in SINGLETON_ACTIONS
                    }
                    protected_nll_change = {
                        action: actions[action]["protected_nll"] - actions["reject_all"]["protected_nll"]
                        for action in SINGLETON_ACTIONS
                    }
                    protected_accuracy_harm = {
                        action: int(actions[action]["protected_accuracy"] < actions["reject_all"]["protected_accuracy"])
                        for action in SINGLETON_ACTIONS
                    }
                    eos_rate_change = {
                        action: actions[action]["eos_rate"] - actions["reject_all"]["eos_rate"]
                        for action in SINGLETON_ACTIONS
                    }
                    eos_increase_harm = {
                        action: int(actions[action]["eos_rate"] > actions["reject_all"]["eos_rate"])
                        for action in SINGLETON_ACTIONS
                    }
                    feature_values = [
                        *uncertainty.values(), *geometry, *support,
                        boundary["boundary"] / 2.0, 1.0,
                    ]
                    parent = fresh_from_weights(admit); recent = recent_next
                    return {
                        "protocol": PROTOCOL, "stage": args.stage,
                        "group_id": row["group_id"], "boundary": boundary["boundary"],
                        "features": feature_values,
                        "feature_schema": {
                            "uncertainty": list(uncertainty), "geometry_per_layer": 5,
                            "support_per_layer": 259,
                            "tail": ["boundary_fraction", "bias"],
                        },
                        "online_memory": recent_next, "uncertainty": uncertainty,
                        "geometry": geometry_audit, "branch_audit": branch_audit,
                        "base_layer_sha256": {"reject": reject_hashes, "admit": admit_hashes},
                        "composition": composition, "actions": actions, "benefit": benefits,
                        "protected_nll_change": protected_nll_change,
                        "protected_accuracy_harm": protected_accuracy_harm,
                        "eos_rate_change": eos_rate_change,
                        "eos_increase_harm": eos_increase_harm,
                    }

                result, resource = _measure_cuda(label_one, device)
                result["resource"] = resource; resources.append(resource)
                feature, label = split_artifacts(result)
                feature_sink.write(json.dumps(feature) + "\n"); feature_sink.flush()
                label_sink.write(json.dumps(label) + "\n"); label_sink.flush()
                print(json.dumps({"group": row["group_id"], "boundary": boundary["boundary"], **resource}), flush=True)
            print(json.dumps({"completed_groups": group_index + 1, "total_groups": len(rows)}), flush=True)
    if before != parameter_digest(model) or versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("frozen Qwen parameters changed")
    sidecar = {
        "protocol": PROTOCOL, "stage": args.stage, "shard": args.shard,
        "num_shards": args.num_shards, "groups": len(rows), "labels": len(resources),
        "rows_sha256": sha256(output), "features_sha256": sha256(feature_output),
        "features_path": str(feature_output),
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "data_audit_sha256": sha256(data_root / "data_audit.json"),
        "model": model_meta, "weights_sha256": before, "backbone_unchanged": True,
        "mean_label_seconds": statistics.fmean(item["seconds"] for item in resources),
        "peak_allocated_gib": max(item["peak_allocated_gib"] for item in resources),
        "peak_reserved_gib": max(item["peak_reserved_gib"] for item in resources),
        "sealed_test_scored": False,
        "plan_sha256": sha256(plan_path),
        "v3_failure_diagnostic_sha256": sha256(diagnostic_path),
        "v3_confirmation_audit_sha256": sha256(confirmation_audit_path),
    }
    Path(str(output) + ".audit.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED); count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return [estimates[int(.025 * BOOTSTRAP_DRAWS)], estimates[int(.975 * BOOTSTRAP_DRAWS) - 1]]


def safe_action(row: dict) -> str:
    reject = row["actions"]["reject_all"]
    eligible = [
        action for action in SINGLETON_ACTIONS
        if row["actions"][action]["J"] < reject["J"]
        and row["actions"][action]["protected_accuracy"] >= reject["protected_accuracy"]
        and row["actions"][action]["eos_rate"] <= reject["eos_rate"]
    ]
    return min(eligible, key=lambda action: row["actions"][action]["J"]) if eligible else "reject_all"


def summarize(inputs: list[Path], output: Path) -> dict:
    rows = [json.loads(line) for path in inputs for line in path.read_text().splitlines() if line.strip()]
    expected = {"train": 1152, "development": 288}[rows[0]["stage"]]
    if len(rows) != expected or len({(row["group_id"], row["boundary"]) for row in rows}) != expected:
        raise ValueError("incomplete/duplicate V3 action rows")
    if any(set(row["actions"]) != set(ALL_ACTIONS) for row in rows):
        raise ValueError("V3 action matrix changed")
    choices = [safe_action(row) for row in rows]
    groups = sorted({row["group_id"] for row in rows})
    group_gains = [statistics.fmean(
        row["actions"]["reject_all"]["J"] - row["actions"][safe_action(row)]["J"]
        for row in rows if row["group_id"] == group
    ) for group in groups]
    interval = bootstrap(group_gains)
    headroom = [max(row["benefit"].values()) for row in rows]
    def metric(names, key):
        return statistics.fmean(row["actions"][name][key] for row, name in zip(rows, names))
    rejects = ["reject_all"] * len(rows)
    checks = {
        "singleton_headroom_fraction_at_least_15pct": sum(value > BENEFIT_EPS for value in headroom) / len(rows) >= .15,
        "safe_oracle_mean_gain_positive": statistics.fmean(group_gains) > 0,
        "safe_oracle_bootstrap_lower_positive": interval[0] > 0,
        "protected_accuracy_safe": metric(choices, "protected_accuracy") >= metric(rejects, "protected_accuracy"),
        "protected_nll_safe": metric(choices, "protected_nll") - metric(rejects, "protected_nll") <= .01,
        "eos_safe": metric(choices, "eos_rate") - metric(rejects, "eos_rate") <= .01,
        "branch_replay_and_composition": all(
            row["branch_audit"]["parent_fast_sha256"] == row["branch_audit"]["reject_fast_sha256"]
            and all(sum(value == "admit" for value in item["sources"].values()) == 1
                    for item in row["composition"].values()) for row in rows
        ),
    }
    result = {
        "protocol": PROTOCOL, "stage": rows[0]["stage"], "groups": len(groups), "labels": len(rows),
        "headroom_fraction": sum(value > BENEFIT_EPS for value in headroom) / len(rows),
        "mean_max_singleton_benefit": statistics.fmean(headroom),
        "mean_J": {name: statistics.fmean(row["actions"][name]["J"] for row in rows) for name in ALL_ACTIONS},
        "safe_oracle_J": metric(choices, "J"),
        "safe_oracle_gain_vs_reject": statistics.fmean(group_gains),
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "safe_oracle_actions": {name: choices.count(name) for name in ALL_ACTIONS if choices.count(name)},
        "per_layer": {str(layer): {
            "positive_boundaries": sum(row["benefit"][action] > BENEFIT_EPS for row in rows),
            "mean_benefit": statistics.fmean(row["benefit"][action] for row in rows),
        } for layer, action in zip(LAYERS, SINGLETON_ACTIONS)},
        "checks": checks, "passed": all(checks.values()),
        "next_stage": "train_controller" if rows[0]["stage"] == "development" and all(checks.values()) \
                      else "score_development" if rows[0]["stage"] == "train" and all(checks.values()) \
                      else "stop_before_controller_and_test",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True); collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--stage", choices=("train", "development"), required=True)
    collect_parser.add_argument("--plan", default=PLAN); collect_parser.add_argument("--v3-root", required=True)
    collect_parser.add_argument("--shard", type=int, default=0); collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--eos-id", type=int, default=151645)
    collect_parser.add_argument("--output", required=True); collect_parser.add_argument("--features-output", required=True)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        print(json.dumps(collect(args), indent=2))
    else:
        print(json.dumps(summarize(args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
