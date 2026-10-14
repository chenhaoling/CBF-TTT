#!/usr/bin/env bash
set -euo pipefail

EXAMPLES_ROOT=${EXAMPLES_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_examples_20261014}
POLICY_ROOT=${POLICY_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v18_examples_to_policy_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"
while [[ ! -f "$EXAMPLES_ROOT/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"
  sleep 60
done
if [[ "$(cat "$EXAMPLES_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_examples_runner_failed > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi
if "$PYTHON" - "$EXAMPLES_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); manifest_path = root / "data/manifest.json"
manifest = json.loads(manifest_path.read_text()); audit = json.loads((root / "audit.json").read_text())
checks = (manifest.get("protocol") == "v18_causal_deletion_examples_v1",
          manifest.get("rows") == 9280, audit.get("audit_passed") is True,
          audit.get("manifest_sha256") == hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          (root / "terminal_status.txt").read_text().strip()
              == "completed_v18_causal_deletion_examples")
raise SystemExit(0 if all(checks) else 29)
PY
then released=1; else released=0; fi
if [[ "$released" -ne 1 ]]; then
  echo stopped_examples_audit_unsupported > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi
test ! -e "$POLICY_ROOT"
echo examples_audit_passed_launching_policy > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"
cd "$REPO"
ROOT="$POLICY_ROOT" EXAMPLES_ROOT="$EXAMPLES_ROOT" \
  bash scripts/run_v18_causal_deletion_policy.sh > "$CHAIN_ROOT/policy_runner.log" 2>&1
echo completed_policy_after_examples_release > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
