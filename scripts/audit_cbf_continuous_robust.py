"""Independent reconstruction audit for robust continuous forgetting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_continuous_forgetting import build_rows, evaluate, restore_controller
from tasks.train_cbf_continuous_robust import (
    COMPOUND_DOMAINS, DEV_SEEDS, DOMAINS, TEST_SEEDS, TRAIN_SEEDS,
    development_gate, domain_scenarios, summarize_domains,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(root: str | Path, frozen: str | Path) -> dict:
    import torch

    root, frozen = Path(root), Path(frozen)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    training = json.loads((root / "training.json").read_text())
    checkpoint = torch.load(root / "controllers.pt", map_location="cpu", weights_only=True)
    old_checkpoint = torch.load(frozen / "controllers.pt", map_location="cpu", weights_only=True)
    if (manifest["protocol"] != "continuous_forgetting_robust_v2"
            or tuple(manifest["train_seeds"]) != TRAIN_SEEDS
            or tuple(manifest["dev_seeds"]) != DEV_SEEDS
            or tuple(manifest["test_seeds"]) != TEST_SEEDS
            or tuple(manifest["domains"]) != DOMAINS
            or tuple(manifest["compound_test_domains"]) != COMPOUND_DOMAINS
            or manifest["frozen_checkpoint_sha256"] != sha(frozen / "controllers.pt")
            or training["checkpoint_sha256"] != sha(root / "controllers.pt")):
        raise ValueError("robust continuous manifest mismatch")
    rebuilt_counts, examples = {}, 0
    for domain in DOMAINS:
        scenarios = domain_scenarios(domain, "train", manifest["train_groups"], manifest["per_group"], manifest["train_seed"])
        rows = build_rows(scenarios, manifest["dim"], TRAIN_SEEDS); examples += len(rows)
        rebuilt_counts[domain] = dict(Counter(str(row["alpha"]) for row in rows))
    if rebuilt_counts != training["label_counts_by_domain"] or examples != training["examples"]:
        raise ValueError("robust continuous train labels mismatch")
    controllers = {name: restore_controller(value) for name, value in checkpoint["controllers"].items()}
    old = {name: restore_controller(value) for name, value in old_checkpoint["controllers"].items()}
    dev_records, old_records, hashes = {}, {}, {}
    for domain in DOMAINS:
        scenarios = domain_scenarios(domain, "dev", manifest["dev_groups"], manifest["per_group"], manifest["dev_seed"])
        rows = evaluate(scenarios, controllers, manifest["dim"], checkpoint["train_mean"], DEV_SEEDS)
        path = root / f"dev_{domain}.jsonl"
        if rows != read_jsonl(path):
            raise ValueError(f"robust continuous dev records mismatch: {domain}")
        dev_records[domain] = rows; hashes[f"dev_{domain}"] = sha(path)
        old_records[domain] = evaluate(scenarios, old, manifest["dim"], old_checkpoint["train_mean"], DEV_SEEDS)
    dev_domains, dev_macro = summarize_domains(dev_records, DEV_SEEDS, manifest["bootstrap_seed"])
    _, old_macro = summarize_domains(old_records, DEV_SEEDS, manifest["bootstrap_seed"] + 10000)
    gate = development_gate(dev_domains, dev_macro, old_macro["metrics"])
    if (dev_domains != summary["dev_domains"] or dev_macro != summary["dev_macro"]
            or old_macro != summary["dev_frozen_macro"] or gate != summary["dev_gate"]):
        raise ValueError("robust continuous development mismatch")
    test_domains = test_macro = None
    if gate["passed"]:
        records = {}
        for domain in DOMAINS + COMPOUND_DOMAINS:
            scenarios = domain_scenarios(domain, "test", manifest["test_groups"], manifest["per_group"], manifest["test_seed"])
            rows = evaluate(scenarios, controllers, manifest["dim"], checkpoint["train_mean"], TEST_SEEDS)
            path = root / f"test_{domain}.jsonl"
            if rows != read_jsonl(path):
                raise ValueError(f"robust continuous test records mismatch: {domain}")
            records[domain] = rows; hashes[f"test_{domain}"] = sha(path)
        test_domains, test_macro = summarize_domains(records, TEST_SEEDS, manifest["bootstrap_seed"] + 20000)
    elif list(root.glob("test_*.jsonl")):
        raise ValueError("robust test records exist despite failed development gate")
    if (test_domains != summary["test_domains"] or test_macro != summary["test_macro"]
            or summary["test_scored"] != gate["passed"] or summary["stage_passed"] != gate["passed"]
            or manifest["test_scored"] != gate["passed"]):
        raise ValueError("robust continuous conditional test mismatch")
    result = {"protocol": "continuous_forgetting_robust_v2_audit", "passed": True,
              "checkpoint_sha256": sha(root / "controllers.pt"), "record_sha256": hashes,
              "dev_gate": gate, "test_scored": gate["passed"]}
    (root / "execution_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--frozen", required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.root, args.frozen), indent=2))


if __name__ == "__main__":
    main()
