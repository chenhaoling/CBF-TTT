"""Independent recomputation audit for continuous fast-weight forgetting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_continuous_forgetting import (
    FEATURE_NAMES, GRID, MEMORY_SEEDS, POLICIES, STRUCTURE_BLIND_INDICES,
    aggregate, build_rows, development_gate, evaluate, generate_scenarios,
    restore_controller,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root: str | Path, probe: str | Path) -> dict:
    import torch

    root, probe = Path(root), Path(probe)
    manifest = json.loads((root / "training_manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    checkpoint = torch.load(root / "controllers.pt", map_location="cpu", weights_only=True)
    if (manifest["protocol"] != "continuous_forgetting_formal_v1"
            or tuple(manifest["grid"]) != GRID
            or tuple(manifest["memory_seeds"]) != MEMORY_SEEDS
            or tuple(manifest["feature_names"]) != FEATURE_NAMES
            or tuple(manifest["policies"]) != POLICIES
            or manifest["probe_sha256"] != sha(probe)
            or sha(root / "controllers.pt") != training["checkpoint_sha256"]
            or checkpoint["dim"] != manifest["dim"]
            or checkpoint["epochs"] != manifest["epochs"]):
        raise ValueError("continuous forgetting manifest mismatch")
    counts = manifest["group_counts"]
    train_scenarios = generate_scenarios("train", counts["train"], manifest["per_group"], manifest["seed"])
    train_rows = build_rows(train_scenarios, manifest["dim"])
    label_counts = dict(Counter(str(row["alpha"]) for row in train_rows))
    if label_counts != training["train_label_counts"] or len(train_rows) != training["examples"]:
        raise ValueError("continuous forgetting train labels mismatch")
    controllers = {name: restore_controller(value) for name, value in checkpoint["controllers"].items()}
    if tuple(controllers["structure_blind"]["indices"]) != STRUCTURE_BLIND_INDICES:
        raise ValueError("continuous forgetting ablation mismatch")
    dev_scenarios = generate_scenarios("dev", counts["dev"], manifest["per_group"], manifest["seed"])
    dev_records = evaluate(dev_scenarios, controllers, manifest["dim"], checkpoint["train_mean"])
    if dev_records != read_jsonl(root / "dev_records.jsonl"):
        raise ValueError("continuous forgetting dev records mismatch")
    dev = aggregate(dev_records); gate = development_gate(dev)
    if dev != summary["dev"] or gate != summary["dev_gate"]:
        raise ValueError("continuous forgetting dev summary mismatch")
    expected_test = None
    if gate["passed"]:
        scenarios = generate_scenarios("test", counts["test"], manifest["per_group"], manifest["seed"])
        records = evaluate(scenarios, controllers, manifest["dim"], checkpoint["train_mean"])
        if records != read_jsonl(root / "test_records.jsonl"):
            raise ValueError("continuous forgetting test records mismatch")
        expected_test = aggregate(records)
    elif (root / "test_records.jsonl").exists():
        raise ValueError("test records exist despite failed development gate")
    if (summary["test"] != expected_test or summary["test_scored"] != gate["passed"]
            or summary["stage_passed"] != gate["passed"]
            or manifest["test_scored"] != gate["passed"]):
        raise ValueError("continuous forgetting conditional test mismatch")
    result = {"protocol": "continuous_forgetting_formal_v1_audit", "passed": True,
              "probe_sha256": sha(probe), "checkpoint_sha256": sha(root / "controllers.pt"),
              "dev_records_sha256": sha(root / "dev_records.jsonl"),
              "dev_gate": gate, "test_scored": gate["passed"]}
    if gate["passed"]:
        result["test_records_sha256"] = sha(root / "test_records.jsonl")
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--probe", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.probe), indent=2))


if __name__ == "__main__":
    main()
