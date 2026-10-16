"""Independent action, trajectory, metric, and resource audit for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.native_closed_loop_fast_weight_gate_v5 import (
    BASELINES,
    LAYERS,
    PROTOCOL,
    canonical,
    load_bound_artifacts,
    load_controllers,
    load_inputs,
    policy_specs,
    predict,
    choose_actions,
    summarize,
)


def close(left, right, tolerance=1e-7):
    if isinstance(left, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)), abs(float(right)))
    return left == right


def assert_numeric(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            assert_numeric(child)
    elif isinstance(value, list):
        for child in value:
            assert_numeric(child)
    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("non-numeric online memory")


def fast_digest_from_layer_hashes(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for layer in LAYERS:
        digest.update(str(layer).encode())
        digest.update(bytes.fromhex(hashes[str(layer)]))
    return digest.hexdigest()


def outcome_J(outcome: dict) -> float:
    related = [row["answer_nll"] for row in outcome["queries"] if row["role"] == "related"]
    protected = [row["answer_nll"] for row in outcome["queries"] if row["role"] == "protected"]
    if len(related) != 2 or len(protected) != 3:
        raise ValueError("V5 query role contract changed")
    return .5 * sum(related) / len(related) + .5 * sum(protected) / len(protected)


def audit(args) -> dict:
    import torch

    root, data_root = Path(args.root), Path(args.data_root)
    controller_root, plan = Path(args.controller_root), Path(args.plan)
    _, _, data_dependency = load_inputs(data_root, args.stage)
    controllers, controller_dependency = load_controllers(controller_root)
    label_paths = sorted(root.glob(f"{args.stage}_shard*_labels.jsonl"))
    decision_paths = sorted(root.glob(f"{args.stage}_shard*_decisions.jsonl"))
    if len(label_paths) != args.num_shards or len(decision_paths) != args.num_shards:
        raise ValueError("V5 shard coverage")
    labels, decisions = load_bound_artifacts(label_paths, decision_paths)
    if any(row.get("stage") != args.stage for row in [*labels, *decisions]):
        raise ValueError("V5 artifact stage mismatch")
    expected_dependencies = {
        "plan_sha256": sha256(plan), **data_dependency, **controller_dependency,
    }
    sidecars = []
    for label_path, decision_path in zip(label_paths, decision_paths):
        sidecar = json.loads(Path(str(label_path) + ".audit.json").read_text())
        if sidecar.get("rows_sha256") != sha256(label_path) \
                or sidecar.get("decisions_sha256") != sha256(decision_path) \
                or Path(sidecar.get("decisions_path", "")) != decision_path \
                or not sidecar.get("backbone_unchanged") or sidecar.get("test_opened"):
            raise ValueError("V5 collector sidecar")
        if any(sidecar.get(key) != value for key, value in expected_dependencies.items()):
            raise ValueError("V5 sidecar dependency")
        sidecars.append(sidecar)
    spec_index = {row["policy"]: row for row in policy_specs()}
    key = lambda row: (row["group_id"], row["policy"], row["boundary"])
    label_index = {key(row): row for row in labels}
    if len(label_index) != len(labels):
        raise ValueError("duplicate V5 transitions")
    for label, decision in zip(labels, decisions):
        if canonical(decision) != label["decision_sha256"] \
                or decision["parent_fast_sha256"] != label["branch_audit"]["parent_fast_sha256"] \
                or label["branch_audit"]["parent_fast_sha256"] != label["branch_audit"]["reject_fast_sha256"]:
            raise ValueError("V5 decision/branch identity")
        if any(name in decision for name in (
            "mechanism", "subject", "relation", "value", "answer", "future_loss", "oracle_action"
        )):
            raise ValueError("prohibited V5 decision field")
        assert_numeric(decision["online_memory"])
        spec = spec_index[decision["policy"]]
        if spec["kind"] == "baseline":
            expected_action = "admit_all" if decision["policy"] == "always_write" else "reject_all"
            if decision["action"] != expected_action or "features" in decision:
                raise ValueError("V5 baseline action")
        else:
            if len(decision.get("features", [])) != 1854 \
                    or any(not math.isfinite(value) for value in decision["features"]):
                raise ValueError("V5 feature contract")
            controller = controllers[decision["policy"]]
            x = torch.tensor([decision["features"]], dtype=torch.float32)
            prediction = predict(controller["model"], x[:, controller["indices"]], controller["stats"])
            expected_action = choose_actions(*prediction)[0]
            observed_prediction = [
                decision["predicted_benefit"], decision["predicted_protected_nll_change"],
                decision["predicted_accuracy_harm_probability"],
                decision["predicted_eos_harm_probability"],
            ]
            if decision["action"] != expected_action \
                    or decision["checkpoint_sha256"] != controller["checkpoint_sha256"] \
                    or not close(observed_prediction, [value[0].tolist() for value in prediction]):
                raise ValueError("V5 controller action reconstruction")
        sources = label["composition"]["sources"]
        selected_hashes = label["composition"]["layer_sha256"]
        if any(selected_hashes[str(layer)] != label["base_layer_sha256"][sources[str(layer)]][str(layer)]
               for layer in LAYERS):
            raise ValueError("V5 layer composition")
        if label["action"] == "reject_all" and set(sources.values()) != {"reject"}:
            raise ValueError("V5 reject composition")
        if label["action"] == "admit_all" and set(sources.values()) != {"admit"}:
            raise ValueError("V5 admit composition")
        if label["action"].startswith("admit_layer_") \
                and sum(value == "admit" for value in sources.values()) != 1:
            raise ValueError("V5 singleton composition")
        if fast_digest_from_layer_hashes(selected_hashes) != label["next_fast_sha256"]:
            raise ValueError("V5 next-state digest")
        outcome = label["outcome"]
        if abs(outcome_J(outcome) - outcome["J"]) > 1e-9:
            raise ValueError("V5 J reconstruction")
        resource = label["resource"]
        if any(not math.isfinite(resource[name]) or resource[name] < 0
               for name in ("seconds", "peak_allocated_gib", "peak_reserved_gib")):
            raise ValueError("V5 resource record")
    groups = sorted({row["group_id"] for row in labels})
    names = list(spec_index)
    for group in groups:
        first = set()
        for name in names:
            part = [label_index[group, name, boundary] for boundary in range(3)]
            first.add(part[0]["branch_audit"]["parent_fast_sha256"])
            if any(part[index]["next_fast_sha256"]
                   != part[index + 1]["branch_audit"]["parent_fast_sha256"] for index in range(2)):
                raise ValueError("V5 trajectory continuity")
        if len(first) != 1:
            raise ValueError("V5 policy initial-state mismatch")
    summary_path = root / f"{args.stage}_summary.json"
    summary = json.loads(summary_path.read_text())
    with tempfile.TemporaryDirectory() as directory:
        recomputed = summarize(label_paths, decision_paths, Path(directory) / "summary.json")
    if not close(summary, recomputed):
        raise ValueError("V5 summary reconstruction")
    result = {
        "protocol": PROTOCOL + "_independent_audit", "stage": args.stage, "passed": True,
        "groups": len(groups), "policies": len(names), "transitions": len(labels),
        "summary_sha256": sha256(summary_path), "result_files": {
            path.name: sha256(path) for path in [*label_paths, *decision_paths]
        },
        "collector_audits": len(sidecars), "recomputed_checks": recomputed["checks"],
        "stage_gate_passed": recomputed["passed"], "test_opened": False,
        **expected_dependencies,
    }
    if args.stage == "smoke":
        result["smoke_gate_passed"] = recomputed["passed"]
    (root / f"{args.stage}_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--controller-root", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--stage", choices=("smoke", "development"), default="smoke")
    args = parser.parse_args(); print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
