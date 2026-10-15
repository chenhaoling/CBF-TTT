"""Independent selection and resource audit for the S3 projection grid."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_projection_v1"
PILOT_PROTOCOL = "qwen_retrieval_evidence_forgetting_s3_projection_pilot_v1"


def key(summary: dict) -> tuple:
    config = summary["config"]
    return (
        summary["retrieval"]["recall_at_16"], summary["direct_pair_roc_auc"],
        -config["dimension"], -config["epoch"],
        -int(config["pubmed_pretraining"]), -config["temperature"], -config["margin"],
    )


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return True


def audit(root: str | Path) -> dict:
    import torch

    root = Path(root)
    selection_path, weights_path, pilot_path = (
        root / "selection.json", root / "projection.pt", root / "resource_pilot.json"
    )
    selection = json.loads(selection_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    summaries = selection["summaries"]
    expected = set(itertools.product(
        (False, True), (256, 512), (.05, .10), (.10, .20), (10, 25, 50)
    ))
    actual = {(row["config"]["pubmed_pretraining"], row["config"]["dimension"],
               row["config"]["temperature"], row["config"]["margin"],
               row["config"]["epoch"]) for row in summaries}
    selected = max(summaries, key=key)
    subset = {
        flag: max((row for row in summaries
                   if row["config"]["pubmed_pretraining"] is flag), key=key)
        for flag in (False, True)
    }
    checks = {
        "protocol": selection.get("protocol") == PROTOCOL
            and weights.get("protocol") == PROTOCOL,
        "grid": selection.get("grid_size") == 48 and actual == expected,
        "finite": finite(summaries),
        "metric_bounds": all(
            0 <= row["retrieval"][name] <= 1
            for row in summaries for name in
            ("recall_at_4", "recall_at_8", "recall_at_16", "mrr_at_16")
        ) and all(0 <= row["direct_pair_roc_auc"] <= 1 for row in summaries),
        "selection": selection["selected"] == selected
            and weights["config"] == selected["config"],
        "pretraining_ablation": all(
            selection["best_by_pubmed_pretraining"][str(flag).lower()] == subset[flag]
            and weights["pretraining_ablation"][str(flag).lower()]["config"]
            == subset[flag]["config"] for flag in (False, True)
        ),
        "weights": sha(weights_path) == selection["weights_sha256"]
            and set(weights["state_dict"])
            == {"base.weight", "norm.weight", "norm.bias",
                "residual_in.weight", "residual_out.weight"},
        "pilot": pilot.get("protocol") == PILOT_PROTOCOL
            and pilot.get("passed") is True and pilot.get("steps") == 2
            and pilot.get("support_queries") == 1000
            and pilot.get("finite_losses") is True
            and pilot.get("peak_reserved_gib", math.inf) < 28,
        "resource": finite(selection["resource"])
            and selection["resource"]["peak_reserved_gib"] < 28
            and selection["resource"]["support_queries"] == 1000,
        "sealed": selection.get("dialogre_dev_loaded") is False
            and selection.get("dialogre_test_loaded") is False
            and selection.get("pubmed_validation_loaded") is False,
    }
    passed = all(checks.values())
    result = {
        "protocol": "qwen_retrieval_evidence_forgetting_s3_projection_audit_v1",
        "passed": passed,
        "checks": checks,
        "selected": selected,
        "selection_sha256": sha(selection_path),
        "weights_sha256": sha(weights_path),
        "pilot_sha256": sha(pilot_path),
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
