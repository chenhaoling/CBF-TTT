"""Independent reconstruction audit for frozen-controller stress evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.evaluate_cbf_continuous_stress import (
    STRESS_SEEDS, SUITES, generate_stress_scenarios, stress_checks, summarize_suite,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_continuous_forgetting import evaluate, restore_controller


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root: str | Path, formal: str | Path) -> dict:
    import torch

    root, formal = Path(root), Path(formal)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    checkpoint = torch.load(formal / "controllers.pt", map_location="cpu", weights_only=True)
    if (manifest["protocol"] != "continuous_forgetting_stress_v1"
            or tuple(manifest["stress_seeds"]) != STRESS_SEEDS
            or tuple(manifest["suites"]) != SUITES
            or manifest["controller_retrained"]
            or manifest["formal_checkpoint_sha256"] != sha(formal / "controllers.pt")
            or manifest["formal_summary_sha256"] != sha(formal / "summary.json")):
        raise ValueError("continuous stress manifest mismatch")
    controllers = {name: restore_controller(value) for name, value in checkpoint["controllers"].items()}
    suites, all_records, hashes = {}, [], {}
    for index, suite in enumerate(SUITES):
        scenarios = generate_stress_scenarios(suite, manifest["groups"], manifest["per_group"], manifest["seed"])
        records = evaluate(scenarios, controllers, checkpoint["dim"], checkpoint["train_mean"], STRESS_SEEDS)
        path = root / f"{suite}_records.jsonl"
        if records != read_jsonl(path):
            raise ValueError(f"continuous stress records mismatch: {suite}")
        suites[suite] = summarize_suite(records, manifest["bootstrap_seed"] + index * 100)
        hashes[suite] = sha(path); all_records.extend(records)
    macro = summarize_suite(all_records, manifest["bootstrap_seed"] + 1000)
    checks = stress_checks(suites, macro)
    if (suites != summary["suites"] or macro != summary["macro"] or checks != summary["checks"]
            or summary["passed"] != all(checks.values())):
        raise ValueError("continuous stress summary mismatch")
    result = {"protocol": "continuous_forgetting_stress_v1_audit", "passed": True,
              "formal_checkpoint_sha256": sha(formal / "controllers.pt"),
              "record_sha256": hashes, "stress_checks": checks}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--formal", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.formal), indent=2))


if __name__ == "__main__":
    main()
