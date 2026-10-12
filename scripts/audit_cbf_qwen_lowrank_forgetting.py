"""Independent result aggregation audit for Qwen-grounded forgetting."""

import argparse
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_lowrank_forgetting import POLICIES, aggregate
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS, development_gate, multi_seed_summary


def audit(root, data_root):
    root, data_root = Path(root), Path(data_root)
    summary = json.loads((root / "summary.json").read_text()); training = json.loads((root / "training.json").read_text())
    manifest = json.loads((root / "training_manifest.json").read_text()); data = json.loads((data_root / "manifest.json").read_text())
    checks = {"protocol": summary["protocol"] == manifest["protocol"] == "qwen_lowrank_forgetting_v1",
              "data_hash": manifest["data_manifest_sha256"] == sha(data_root / "manifest.json"),
              "splits": data["group_counts"] == {"train": 128, "dev": 32, "test": 32},
              "seeds": tuple(manifest["memory_seeds"]) == SEEDS,
              "embedding_hashes": set(training["embedding_meta"]["hashes"]) == set(map(str, SEEDS))}
    dev_seed = {}
    for seed in SEEDS:
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            path = root / f"dev_seed{seed}_{policy}.jsonl"; rows = list(map(json.loads, path.read_text().splitlines()))
            checks[f"dev_hash_{seed}_{policy}"] = training["raw_rows_sha256"][path.name] == sha(path)
            dev_seed[str(seed)][policy] = aggregate(rows)
    rebuilt_dev = multi_seed_summary(dev_seed)
    checks["dev_recomputed"] = rebuilt_dev == summary["dev"]
    checks["gate_recomputed"] = development_gate(rebuilt_dev) == summary["dev_gate"]
    test_exists = all((root / f"test_seed{seed}_{policy}.jsonl").exists() for seed in SEEDS for policy in POLICIES)
    checks["test_lock"] = test_exists == bool(summary["dev_gate"]["passed"]) == bool(summary["test_scored"])
    if test_exists:
        test_seed = {}
        for seed in SEEDS:
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                path = root / f"test_seed{seed}_{policy}.jsonl"; rows = list(map(json.loads, path.read_text().splitlines()))
                checks[f"test_hash_{seed}_{policy}"] = training["raw_rows_sha256"][path.name] == sha(path)
                test_seed[str(seed)][policy] = aggregate(rows)
        checks["test_recomputed"] = multi_seed_summary(test_seed) == summary["test"]
    result = {"protocol": "qwen_lowrank_forgetting_v1", "audit_passed": all(checks.values()), "checks": checks,
              "dev_gate_passed": summary["dev_gate"]["passed"], "test_scored": summary["test_scored"],
              "summary_sha256": sha(root / "summary.json"), "training_sha256": sha(root / "training.json"),
              "data_manifest_sha256": sha(data_root / "manifest.json")}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", required=True); p.add_argument("--data-root", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); result = audit(a.root, a.data_root); Path(a.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
