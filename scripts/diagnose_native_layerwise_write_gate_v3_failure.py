"""Post-stop diagnostic using only already-scored V3 confirmation actions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics

from tasks.native_layerwise_write_gate_v2 import SINGLETON_ACTIONS


def load(root: Path) -> list[dict]:
    return [json.loads(line) for path in sorted(root.glob("confirmation_shard*_labels.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def choose(row: dict, eos_barrier: bool) -> str:
    reject = row["actions"]["reject_all"]
    eligible = [name for name in SINGLETON_ACTIONS
                if row["actions"][name]["J"] < reject["J"]
                and row["actions"][name]["protected_accuracy"] >= reject["protected_accuracy"]
                and (not eos_barrier or row["actions"][name]["eos_rate"] <= reject["eos_rate"])]
    return min(eligible, key=lambda name: row["actions"][name]["J"]) if eligible else "reject_all"


def metrics(rows: list[dict], choices: list[str]) -> dict:
    reject = ["reject_all"] * len(rows)
    def mean(names, key):
        return statistics.fmean(row["actions"][name][key] for row, name in zip(rows, names))
    groups = sorted({row["group_id"] for row in rows})
    gains = [statistics.fmean(
        row["actions"]["reject_all"]["J"] - row["actions"][action]["J"]
        for row, action in zip(rows, choices) if row["group_id"] == group
    ) for group in groups]
    rng = random.Random(2026101707); draws = 20_000; count = len(gains)
    samples = sorted(statistics.fmean(gains[rng.randrange(count)] for _ in range(count)) for _ in range(draws))
    return {
        "J": mean(choices, "J"), "gain_vs_reject": statistics.fmean(gains),
        "gain_group_bootstrap_95ci": [samples[int(.025 * draws)], samples[int(.975 * draws) - 1]],
        "protected_accuracy": mean(choices, "protected_accuracy"),
        "protected_accuracy_delta": mean(choices, "protected_accuracy") - mean(reject, "protected_accuracy"),
        "protected_nll": mean(choices, "protected_nll"),
        "protected_nll_delta": mean(choices, "protected_nll") - mean(reject, "protected_nll"),
        "eos_rate": mean(choices, "eos_rate"),
        "eos_rate_delta": mean(choices, "eos_rate") - mean(reject, "eos_rate"),
        "actions": dict(Counter(choices)),
    }


def diagnose(root: Path, output: Path) -> dict:
    summary = json.loads((root / "confirmation_summary.json").read_text())
    audit = json.loads((root / "confirmation_independent_audit.json").read_text())
    if summary.get("passed") or not audit.get("passed") or audit.get("test_opened"):
        raise ValueError("diagnostic requires the audited stopped V3 confirmation")
    rows = load(root)
    original = [choose(row, False) for row in rows]
    constrained = [choose(row, True) for row in rows]
    result = {
        "protocol": "native_layerwise_fastweight_admission_v3_failure_diagnostic",
        "status": "post_stop_descriptive_only", "groups": len({row["group_id"] for row in rows}),
        "labels": len(rows), "original_accuracy_safe_oracle": metrics(rows, original),
        "accuracy_and_eos_safe_oracle": metrics(rows, constrained),
        "actions_changed_by_eos_barrier": sum(a != b for a, b in zip(original, constrained)),
        "v3_gate_remains_failed": True, "controller_training_authorized": False,
        "test_opened": False,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(diagnose(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
