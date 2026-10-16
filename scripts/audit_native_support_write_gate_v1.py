"""Independent artifact audit for native support-aware admission V1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


PROTOCOL = "native_support_aware_fastweight_admission_v1"
PLAN_SHA256 = "__FILLED_AT_RUNTIME_FROM_MANIFEST__"
BENEFIT_EPS = 1e-4


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return statistics.fmean(values)


def percentile_bootstrap(values: list[float], draws=20_000, seed=2026101602):
    rng = random.Random(seed); count = len(values)
    estimates = sorted(mean(values[rng.randrange(count)] for _ in range(count)) for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[min(draws - 1, int(.975 * draws))]]


def assert_numeric_tree(value, path="online_memory"):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str): raise ValueError(f"non-string key in {path}")
            assert_numeric_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value): assert_numeric_tree(child, f"{path}[{index}]")
    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"forbidden/nonfinite online memory value at {path}")


def recompute_action(action: dict) -> tuple[float, float, float, float]:
    related = [row for row in action["queries"] if row["role"] == "related"]
    protected = [row for row in action["queries"] if row["role"] == "protected"]
    if len(related) != 2 or len(protected) != 3:
        raise ValueError("query role/count contract changed")
    for query in action["queries"]:
        if len(query["candidate_nll"]) != 6 or not all(math.isfinite(x) for x in query["candidate_nll"]):
            raise ValueError("invalid candidate losses")
        if query["prediction"] != min(range(6), key=query["candidate_nll"].__getitem__):
            raise ValueError("prediction mismatch")
        if query["answer_nll"] != query["candidate_nll"][query["label"]]:
            raise ValueError("answer NLL mismatch")
    protected_nll = mean(row["answer_nll"] for row in protected)
    J = .5 * mean(row["answer_nll"] for row in related) + .5 * protected_nll
    protected_accuracy = mean(row["correct"] for row in protected)
    eos = mean(row["eos"] for row in action["queries"])
    return J, protected_accuracy, protected_nll, eos


def safe_action(row):
    return ("admit" if row["J1"] < row["J0"]
            and row["actions"]["admit"]["protected_accuracy"]
            >= row["actions"]["reject"]["protected_accuracy"] else "reject")


def audit(args) -> dict:
    root, data_root, plan = Path(args.root), Path(args.data_root), Path(args.plan)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL or manifest["plan_sha256"] != digest(plan):
        raise ValueError("plan/protocol mismatch")
    for name, expected in manifest["files_sha256"].items():
        if digest(data_root / name) != expected: raise ValueError(f"data hash mismatch: {name}")
    model_inputs = load_jsonl(data_root / "model_inputs.jsonl")
    if len(model_inputs) != 48 or len({row["group_id"] for row in model_inputs}) != 48:
        raise ValueError("pilot input group contract changed")
    paths = sorted(root.glob(f"{args.stage}_shard*_labels.jsonl"))
    rows = [row for path in paths for row in load_jsonl(path)]
    feature_paths = sorted(root.glob(f"{args.stage}_shard*_features.jsonl"))
    feature_rows = [row for path in feature_paths for row in load_jsonl(path)]
    expected_groups = 12 if args.stage == "smoke" else 48
    expected_labels = expected_groups * 3
    if len(rows) != expected_labels or len({(r["group_id"], r["boundary"]) for r in rows}) != expected_labels:
        raise ValueError("missing/duplicate result rows")
    if len(feature_rows) != expected_labels or len({(r["group_id"], r["boundary"]) for r in feature_rows}) != expected_labels:
        raise ValueError("missing/duplicate feature rows")
    feature_index = {(row["group_id"], row["boundary"]): row for row in feature_rows}
    audits = []
    for path in paths:
        sidecar = Path(str(path) + ".audit.json"); item = json.loads(sidecar.read_text())
        if item["rows_sha256"] != digest(path) or not item["backbone_unchanged"]:
            raise ValueError("collector audit mismatch")
        feature_path = Path(item["features_path"])
        if item["features_sha256"] != digest(feature_path):
            raise ValueError("feature shard hash mismatch")
        audits.append(item)
    forbidden = {"mechanism", "subject", "relation", "value", "answer", "future_loss", "oracle_action"}
    for row in rows:
        if row["protocol"] != PROTOCOL or row["stage"] != args.stage:
            raise ValueError("row identity mismatch")
        if {"features", "online_memory", "uncertainty", "geometry"} & set(row):
            raise ValueError("online features leaked into label artifact")
        feature = feature_index[row["group_id"], row["boundary"]]
        if {"actions", "J0", "J1", "B", "mechanism_hash"} & set(feature):
            raise ValueError("offline labels leaked into feature artifact")
        if forbidden & set(feature): raise ValueError("forbidden controller field leaked")
        expected_feature_hash = hashlib.sha256(json.dumps(feature, sort_keys=True).encode()).hexdigest()
        if row["feature_sha256"] != expected_feature_hash:
            raise ValueError("feature/label join hash mismatch")
        if len(feature["features"]) != 1854 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in feature["features"]):
            raise ValueError("feature dimension/value mismatch")
        assert_numeric_tree(feature["online_memory"])
        for name, key in (("reject", "J0"), ("admit", "J1")):
            J, accuracy, nll, eos = recompute_action(row["actions"][name])
            if abs(J - row[key]) > 1e-9 or abs(J - row["actions"][name]["J"]) > 1e-9:
                raise ValueError("J recomputation mismatch")
            if abs(accuracy - row["actions"][name]["protected_accuracy"]) > 1e-12:
                raise ValueError("protected accuracy mismatch")
            if abs(nll - row["actions"][name]["protected_nll"]) > 1e-9 or abs(eos - row["actions"][name]["eos_rate"]) > 1e-12:
                raise ValueError("safety metric mismatch")
        if abs(row["J0"] - row["J1"] - row["B"]) > 1e-9:
            raise ValueError("benefit mismatch")
        branch = row["branch_audit"]
        if branch["parent_fast_sha256"] != branch["reject_fast_sha256"] or branch["admit_fast_sha256"] == branch["parent_fast_sha256"]:
            raise ValueError("native branch invariant failed")
        if branch["current_hidden_max_abs_error"] > 1e-5:
            raise ValueError("branch current output mismatch")
        if any(not math.isfinite(row["resource"][key]) or row["resource"][key] <= 0
               for key in ("seconds", "peak_allocated_gib", "peak_reserved_gib")):
            raise ValueError("invalid resource profile")

    groups = sorted({row["group_id"] for row in rows})
    summaries = {}
    for group in groups:
        part = [row for row in rows if row["group_id"] == group]
        summaries[group] = {
            "always": mean(row["J1"] for row in part), "never": mean(row["J0"] for row in part),
            "oracle": mean(row["actions"][safe_action(row)]["J"] for row in part),
        }
    always, never = mean(x["always"] for x in summaries.values()), mean(x["never"] for x in summaries.values())
    stronger = "always" if always <= never else "never"
    gains = [x[stronger] - x["oracle"] for x in summaries.values()]
    recomputed = {
        "positive_fraction": sum(row["B"] > BENEFIT_EPS for row in rows) / len(rows),
        "negative_fraction": sum(row["B"] < -BENEFIT_EPS for row in rows) / len(rows),
        "stronger_baseline": stronger + "_write",
        "oracle_gain": mean(gains),
        "oracle_gain_group_bootstrap_95ci": percentile_bootstrap(gains),
    }
    summary_path = root / f"{args.stage}_summary.json"
    summary = json.loads(summary_path.read_text())
    for key, value in recomputed.items():
        if summary[key] != value: raise ValueError(f"summary mismatch: {key}")
    result = {"protocol": PROTOCOL + "_independent_audit", "stage": args.stage,
              "passed": True, "groups": len(groups), "labels": len(rows),
              "plan_sha256": digest(plan), "data_manifest_sha256": digest(data_root/"manifest.json"),
              "result_files": {path.name: digest(path) for path in [*paths, *feature_paths]},
              "summary_sha256": digest(summary_path), "collector_audits": len(audits),
              "feature_dimension": 1854, "plaintext_online_memory_fields": 0,
              "recomputed": recomputed, "test_opened": manifest["test_opened"]}
    output = root / f"{args.stage}_independent_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args(); print(json.dumps(audit(args), indent=2))


if __name__ == "__main__": main()
