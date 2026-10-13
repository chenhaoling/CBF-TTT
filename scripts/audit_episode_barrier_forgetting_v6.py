"""Audit V6 with the V5 cross-fit recomputation plus objective locks."""

import argparse
import json
from pathlib import Path

from scripts.audit_crossfit_calibrated_forgetting_v5 import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True); parser.add_argument("--v1-root", required=True)
    parser.add_argument("--v3-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.root, args.data_root, args.source_root, args.v1_root, args.v3_root)
    summary = json.loads((Path(args.root) / "summary.json").read_text())
    manifest = json.loads((Path(args.root) / "training_manifest.json").read_text())
    objective_ok = (summary["protocol"] == "episode_barrier_forgetting_v6"
                    and manifest["protocol"] == "episode_barrier_forgetting_v6"
                    and manifest["objective"]
                    == "episode_max_retain_barrier_margin_0.5_plus_0.25_bce")
    result["protocol"] = "episode_barrier_forgetting_v6_audit"
    result["checks"]["barrier_objective_lock"] = objective_ok
    result["audit_passed"] = result["audit_passed"] and objective_ok
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["audit_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
