"""Independent recomputation audit for the Qwen-grounded memory bridge."""

import argparse
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_qwen_lowrank_bridge import (load, summarize_dev, summarize_fixed)


def audit(root):
    root = Path(root); rows, design = load(root); checks = {"episodes": len(rows) == 192}
    dev_paths = [root / "dev_0.jsonl", root / "dev_1.jsonl"]
    q_paths = [root / "q_0.jsonl", root / "q_1.jsonl"]
    rebuilt_dev_path = root / ".audit_dev.json"
    rebuilt_q_path = root / ".audit_q.json"
    dev = summarize_dev(root, dev_paths, rebuilt_dev_path)
    saved_dev = json.loads((root / "dev_selection.json").read_text())
    checks["dev_recomputed"] = dev == saved_dev
    q = summarize_fixed(root, q_paths, "q", root / "dev_selection.json", rebuilt_q_path)
    saved_q = json.loads((root / "q_summary.json").read_text())
    checks["q_recomputed"] = q == saved_q
    rebuilt_dev_path.unlink(); rebuilt_q_path.unlink()
    v_paths = [root / "v_0.jsonl", root / "v_1.jsonl"]
    v_exists = all(path.exists() for path in v_paths)
    checks["v_lock"] = v_exists == bool(q["passed"])
    saved_v = None
    if v_exists:
        rebuilt_v_path = root / ".audit_v.json"
        rebuilt_v = summarize_fixed(root, v_paths, "v", root / "dev_selection.json", rebuilt_v_path)
        saved_v = json.loads((root / "v_summary.json").read_text())
        checks["v_recomputed"] = rebuilt_v == saved_v; rebuilt_v_path.unlink()
    checks["source_hash"] = all(row["context_sha256"] for row in rows)
    result = {"protocol": "qwen_lowrank_bridge_v1", "audit_passed": all(checks.values()), "checks": checks,
              "selected_scale": dev["selected_scale"], "q_passed": q["passed"], "v_scored": v_exists,
              "decision": saved_v if saved_v is not None else q,
              "source_data_sha256": design["source_data_sha256"],
              "dev_selection_sha256": file_digest(root / "dev_selection.json"),
              "q_summary_sha256": file_digest(root / "q_summary.json"),
              "v_summary_sha256": file_digest(root / "v_summary.json") if v_exists else None}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); result = audit(a.root); Path(a.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
