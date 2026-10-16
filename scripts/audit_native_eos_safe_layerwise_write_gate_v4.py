"""Independent audit for V4 EOS-safe formal admission artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from tasks.native_layerwise_write_gate_v2 import ALL_ACTIONS, BENEFIT_EPS, LAYERS, SINGLETON_ACTIONS


PROTOCOL = "native_eos_safe_layerwise_fastweight_admission_v4"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(paths) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def action_J(action: dict) -> float:
    related = [row["answer_nll"] for row in action["queries"] if row["role"] == "related"]
    protected = [row["answer_nll"] for row in action["queries"] if row["role"] == "protected"]
    if len(related) != 2 or len(protected) != 3:
        raise ValueError("query role contract changed")
    return .5 * statistics.fmean(related) + .5 * statistics.fmean(protected)


def safe_action(row: dict) -> str:
    reject = row["actions"]["reject_all"]
    eligible = [name for name in SINGLETON_ACTIONS
                if row["actions"][name]["J"] < reject["J"]
                and row["actions"][name]["protected_accuracy"] >= reject["protected_accuracy"]]
    eligible = [name for name in eligible if row["actions"][name]["eos_rate"] <= reject["eos_rate"]]
    return min(eligible, key=lambda name: row["actions"][name]["J"]) if eligible else "reject_all"


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(2026101702); count = len(values); draws = 20_000
    estimates = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count)) for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]]


def assert_numeric(value) -> None:
    if isinstance(value, dict):
        for child in value.values(): assert_numeric(child)
    elif isinstance(value, list):
        for child in value: assert_numeric(child)
    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("non-numeric online memory")


def audit(args) -> dict:
    root, data_root = Path(args.root), Path(args.data_root)
    v3_root, plan = Path(args.v3_root), Path(args.plan)
    diagnostic_path = v3_root / "confirmation_failure_diagnostic.json"
    confirmation_audit_path = v3_root / "confirmation_independent_audit.json"
    manifest = json.loads((data_root / "manifest.json").read_text())
    data_audit = json.loads((data_root / "data_audit.json").read_text())
    if not data_audit.get("passed") or data_audit.get("test_opened") or manifest.get("sealed_test_scored"):
        raise ValueError("data audit/test seal failed")
    release_dependency = {}
    if args.stage in ("development", "sealed_test"):
        train_summary_path = root / "train_summary.json"
        train_audit_path = root / "train_independent_audit.json"
        train_summary = json.loads(train_summary_path.read_text())
        train_audit = json.loads(train_audit_path.read_text())
        if not train_summary.get("passed") or not train_audit.get("passed") \
                or train_audit.get("test_opened") \
                or train_audit.get("summary_sha256") != digest(train_summary_path):
            raise ValueError("train release dependency")
        release_dependency.update({
            "train_summary_sha256": digest(train_summary_path),
            "train_independent_audit_sha256": digest(train_audit_path),
        })
    if args.stage == "sealed_test":
        if not args.controller_root:
            raise ValueError("sealed test audit requires controller root")
        dev_summary_path = root / "development_summary.json"
        dev_audit_path = root / "development_independent_audit.json"
        dev_summary = json.loads(dev_summary_path.read_text())
        dev_audit = json.loads(dev_audit_path.read_text())
        if not dev_summary.get("passed") or not dev_audit.get("passed") \
                or dev_audit.get("test_opened") \
                or dev_audit.get("summary_sha256") != digest(dev_summary_path):
            raise ValueError("development release dependency")
        controller_root = Path(args.controller_root)
        controller_summary_path = controller_root / "training_summary.json"
        controller_audit_path = controller_root / "training_independent_audit.json"
        marker_path = controller_root / "sealed_test_release.json"
        controller_summary = json.loads(controller_summary_path.read_text())
        controller_audit = json.loads(controller_audit_path.read_text())
        marker = json.loads(marker_path.read_text())
        if not controller_summary.get("development_released_test") \
                or controller_summary.get("test_opened") or not controller_audit.get("passed") \
                or not controller_audit.get("test_released") or controller_audit.get("test_opened") \
                or controller_audit.get("summary_sha256") != digest(controller_summary_path):
            raise ValueError("controller release dependency")
        if marker.get("artifact_root") != str(root.resolve()) \
                or marker.get("controller_summary_sha256") != digest(controller_summary_path) \
                or marker.get("controller_independent_audit_sha256") != digest(controller_audit_path):
            raise ValueError("sealed-test one-shot marker")
        release_dependency.update({
            "development_summary_sha256": digest(dev_summary_path),
            "development_independent_audit_sha256": digest(dev_audit_path),
            "controller_summary_sha256": digest(controller_summary_path),
            "controller_independent_audit_sha256": digest(controller_audit_path),
            "sealed_test_release_marker_sha256": digest(marker_path),
        })
    paths = sorted(root.glob(f"{args.stage}_shard*_labels.jsonl"))
    feature_paths = sorted(root.glob(f"{args.stage}_shard*_features.jsonl"))
    rows, features = load(paths), load(feature_paths)
    expected = {"train": 1152, "development": 288, "sealed_test": 288}[args.stage]
    if len(rows) != expected or len(features) != expected:
        raise ValueError("V3 row coverage")
    feature_index = {(row["group_id"], row["boundary"]): row for row in features}
    if len(feature_index) != expected:
        raise ValueError("duplicate V3 features")
    sidecars = []
    for path in paths:
        item = json.loads(Path(str(path) + ".audit.json").read_text())
        if item["rows_sha256"] != digest(path) or not item["backbone_unchanged"] \
                or item["sealed_test_scored"] != (args.stage == "sealed_test"):
            raise ValueError("V4 collector sidecar")
        if item["plan_sha256"] != digest(plan) \
                or item["v3_failure_diagnostic_sha256"] != digest(diagnostic_path) \
                or item["v3_confirmation_audit_sha256"] != digest(confirmation_audit_path):
            raise ValueError("V4 prerequisite sidecar")
        if item["features_sha256"] != digest(Path(item["features_path"])):
            raise ValueError("V3 feature sidecar")
        if any(item.get(key) != value for key, value in release_dependency.items()):
            raise ValueError("V4 stage release sidecar dependency")
        sidecars.append(item)
    for row in rows:
        key = (row["group_id"], row["boundary"]); feature = feature_index[key]
        if row["protocol"] != PROTOCOL or canonical(feature) != row["feature_sha256"]:
            raise ValueError("V3 feature dependency")
        if len(feature["features"]) != 1854 or any(not math.isfinite(value) for value in feature["features"]):
            raise ValueError("V3 feature contract")
        if any(key in feature for key in ("actions", "benefit", "mechanism", "answer", "oracle_action")):
            raise ValueError("offline field leaked into V3 feature")
        assert_numeric(feature["online_memory"])
        if set(row["actions"]) != set(ALL_ACTIONS):
            raise ValueError("V3 action matrix")
        for action, value in row["actions"].items():
            if abs(action_J(value) - value["J"]) > 1e-9:
                raise ValueError("V3 action J")
        reject = row["actions"]["reject_all"]
        for layer, action in zip(LAYERS, SINGLETON_ACTIONS):
            if abs(reject["J"] - row["actions"][action]["J"] - row["benefit"][action]) > 1e-9:
                raise ValueError("V3 benefit")
            eos_change = row["actions"][action]["eos_rate"] - reject["eos_rate"]
            if abs(eos_change - row["eos_rate_change"][action]) > 1e-12 \
                    or row["eos_increase_harm"][action] != int(eos_change > 0):
                raise ValueError("V4 EOS targets")
            item = row["composition"][action]
            expected_sources = {str(value): ("admit" if value == layer else "reject") for value in LAYERS}
            if item["sources"] != expected_sources:
                raise ValueError("V3 composition sources")
            for value in LAYERS:
                donor = item["sources"][str(value)]
                if item["layer_sha256"][str(value)] != row["base_layer_sha256"][donor][str(value)]:
                    raise ValueError("V3 composition hash")
        if row["branch_audit"]["parent_fast_sha256"] != row["branch_audit"]["reject_fast_sha256"]:
            raise ValueError("V3 reject branch")
    choices = [safe_action(row) for row in rows]
    groups = sorted({row["group_id"] for row in rows})
    gains = [statistics.fmean(
        row["actions"]["reject_all"]["J"] - row["actions"][safe_action(row)]["J"]
        for row in rows if row["group_id"] == group
    ) for group in groups]
    summary = json.loads((root / f"{args.stage}_summary.json").read_text())
    recomputed = {
        "headroom_fraction": sum(max(row["benefit"].values()) > BENEFIT_EPS for row in rows) / len(rows),
        "safe_oracle_gain_vs_reject": statistics.fmean(gains),
        "safe_oracle_gain_group_bootstrap_95ci": bootstrap(gains),
        "safe_oracle_actions": {name: choices.count(name) for name in ALL_ACTIONS if choices.count(name)},
    }
    if any(summary[name] != value for name, value in recomputed.items()):
        raise ValueError("V3 summary reconstruction")
    result = {
        "protocol": PROTOCOL + "_independent_audit", "stage": args.stage, "passed": True,
        "groups": len(groups), "labels": len(rows), "feature_dimension": 1854,
        "plan_sha256": digest(plan), "data_manifest_sha256": digest(data_root / "manifest.json"),
        "data_audit_sha256": digest(data_root / "data_audit.json"),
        "v3_failure_diagnostic_sha256": digest(diagnostic_path),
        "v3_confirmation_audit_sha256": digest(confirmation_audit_path),
        "result_files": {path.name: digest(path) for path in [*paths, *feature_paths]},
        "summary_sha256": digest(root / f"{args.stage}_summary.json"),
        "recomputed": recomputed, "collector_audits": len(sidecars),
        "test_opened": args.stage == "sealed_test",
        **release_dependency,
    }
    (root / f"{args.stage}_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--v3-root", required=True)
    parser.add_argument("--stage", choices=("train", "development", "sealed_test"), required=True)
    parser.add_argument("--controller-root")
    args = parser.parse_args(); print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
