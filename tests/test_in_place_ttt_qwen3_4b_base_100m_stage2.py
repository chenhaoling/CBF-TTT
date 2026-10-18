from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_100m_stage2 import audit
from scripts.create_in_place_ttt_qwen3_4b_base_stage2_release import (
    MODEL_REVISION,
    create_release,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_100M_STAGE2_PLAN.md"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_100m_stage2.sh"


def timing_fixture(root: Path) -> tuple[Path, Path]:
    timing = root / "timing.json"
    timing.write_text(json.dumps({
        "protocol": "in_place_ttt_2x5090_stage2_timing_audit_v1",
        "passed": True, "errors": [],
        "projected_sequential_stage2_seconds": 500.0,
        "arms": {
            arm: {"projected_formal_stage2_seconds": 250.0,
                  "warm_excluded_median_seconds": 50.0}
            for arm in ("no_ttt", "ttt")
        },
    }))
    corrected = root / "corrected.json"
    corrected.write_text(json.dumps({
        "protocol": "in_place_ttt_qwen3_4b_base_stage2_timing_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "formal_release_created": False,
        "underlying_timing_audit_sha256": sha256(timing),
        "projected_sequential_stage2_seconds": 500.0,
    }))
    return timing, corrected


class CorrectedBaseFormalStage2Test(unittest.TestCase):
    def test_plan_freezes_host_counts_order_and_scope(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("41,943,040 training tokens", text)
        self.assertLess(text.index("Run no-TTT"), text.index("native TTT"))
        self.assertIn("never inferred or created by a watcher", text)
        self.assertIn("public-data 100M-tier reproduction", text)

    def test_watcher_has_valid_bash_and_waits_for_corrected_release(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_stage2_timing", text)
        self.assertIn("waiting_for_explicit_corrected_stage2_release", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertLess(text.index("completed_corrected_stage2_timing)"),
                        text.index("launching_corrected_formal_stage2"))

    def test_release_requires_accept_and_binds_both_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing, corrected = timing_fixture(root)
            output = root / "release.json"
            with self.assertRaises(PermissionError):
                create_release(
                    timing_audit=timing, corrected_timing_audit=corrected,
                    output=output, accept=False, reviewed_by="reviewer",
                    decision_note="measured projection accepted",
                )
            self.assertFalse(output.exists())
            release = create_release(
                timing_audit=timing, corrected_timing_audit=corrected,
                output=output, accept=True, reviewed_by="reviewer",
                decision_note="measured projection accepted",
            )
            self.assertEqual(release["stage"], "stage2")
            self.assertEqual(release["timing_audit_sha256"], sha256(timing))
            self.assertEqual(release["corrected_timing_audit_sha256"],
                             sha256(corrected))
            self.assertEqual(release["model_revision"], MODEL_REVISION)

    def test_wrapper_audit_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            result = audit(argparse.Namespace(
                root=missing, model=missing, model_identity=missing / "id.json",
                corpus_root=missing, corpus_audit=missing / "corpus.json",
                stage1_root=missing,
                corrected_stage1_audit=missing / "stage1.json",
                ladder_root=missing,
                corrected_ladder_audit=missing / "ladder.json",
                timing_root=missing,
                corrected_timing_audit=missing / "timing.json",
                release=missing / "release.json", formal_plan=PLAN,
                execution_addendum=missing / "addendum.md",
                runner=missing / "runner.sh",
                underlying_auditor=missing / "auditor.py",
                no_ttt_config=missing / "no.yaml",
                ttt_config=missing / "ttt.yaml",
            ))
            self.assertFalse(result["passed"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
