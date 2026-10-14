"""Independent audit for the post-hoc V18 locality-constrained diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.diagnose_v18_locality_constrained_oracle import PROTOCOL, aggregate
from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


def audit(formal_root, inputs, summary_path, output):
    formal_root, summary_path = Path(formal_root), Path(summary_path)
    formal_summary = json.loads((formal_root / "summary.json").read_text())
    formal_audit = json.loads((formal_root / "audit.json").read_text())
    summary = json.loads(summary_path); rows, hashes = [], {}
    for value in inputs:
        path = Path(value); rows.extend(load_jsonl(path)); hashes[path.name] = sha(path)
    recomputed = aggregate(rows)
    source_ok = (
        formal_summary.get("terminal_status") == "formal_no_sufficient_headroom"
        and formal_summary.get("headroom_gate", {}).get("passed") is False
        and formal_summary.get("causal_headroom_supported") is False
        and formal_audit.get("audit_passed") is True
        and formal_audit.get("summary_sha256") == sha(formal_root / "summary.json")
        and formal_audit.get("recomputed_headroom_gate", {}).get("passed") is False
        and (formal_root / "runner_exit_code.txt").read_text().strip() == "0")
    checks = {
        "audited_failed_original_v18_source": source_ok,
        "complete_unique_raw_coverage_and_hashes": (
            len(rows) == len({row["episode_id"] for row in rows}) == 320
            and summary.get("raw_rows_sha256") == hashes
            and hashes == formal_summary.get("raw_rows_sha256")),
        "independent_budget_dp_pareto_and_gate_recomputation": all(
            summary.get(key) == value for key, value in recomputed.items()),
        "posthoc_status_and_no_release_lock": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 320
            and summary.get("formal_summary_sha256") == sha(formal_root / "summary.json")
            and summary.get("formal_audit_sha256") == sha(formal_root / "audit.json")
            and summary.get("model_rescored") is False and summary.get("posthoc") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("original_v18_status_unchanged")
                == "formal_no_sufficient_headroom"
            and summary.get("stage_b_released") is False
            and summary.get("terminal_status")
                == "completed_v18_locality_constrained_diagnostic"),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(summary_path),
              "raw_rows_sha256": hashes,
              "recomputed_diagnostic_gate": recomputed["diagnostic_gate"],
              "original_v18_status_unchanged": "formal_no_sufficient_headroom",
              "stage_b_released": False, "posthoc": True}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18-LC diagnostic audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
