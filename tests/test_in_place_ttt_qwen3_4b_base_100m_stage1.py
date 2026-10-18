from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_100m_stage1 import audit
from scripts.create_in_place_ttt_qwen3_4b_base_stage1_release import (
    MODEL_REVISION,
    create_release,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_100M_STAGE1_PLAN.md"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_100m_stage1.sh"


def timing_fixture(root: Path) -> tuple[Path, Path]:
    timing = root / "timing.json"
    timing.write_text(json.dumps({
        "protocol": "in_place_ttt_2x5090_stage1_timing_audit_v1",
        "passed": True, "errors": [],
        "projected_sequential_stage1_seconds": 300.0,
        "arms": {
            arm: {"projected_formal_stage1_seconds": 150.0,
                  "warm_excluded_median_seconds": 5.0}
            for arm in ("no_ttt", "ttt")
        },
    }))
    corrected = root / "corrected.json"
    corrected.write_text(json.dumps({
        "protocol": "in_place_ttt_qwen3_4b_base_stage1_timing_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "formal_release_created": False,
        "underlying_timing_audit_sha256": sha256(timing),
        "projected_sequential_stage1_seconds": 300.0,
    }))
    return timing, corrected


def selected_stream_fixture(root: Path) -> Path:
    path = root / "selected_stream.json"
    path.write_text(json.dumps({
        "protocol": "in_place_ttt_qwen3_4b_base_100m_selected_stream_audit_v1",
        "passed": True, "errors": [], "checks": {"all": True},
        "bindings": {"corpus_manifest_sha256": "a" * 64},
        "independent_stages": {
            "stage1_32k": {
                "selected_records": 1728,
                "selected_training_tokens": 56_623_104,
                "ordered_stage_indices_sha256": "b" * 64,
                "ordered_training_hashes_sha256": "c" * 64,
            }
        },
    }))
    return path


class CorrectedBaseFormalStage1Test(unittest.TestCase):
    def test_plan_freezes_host_counts_order_and_scope(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("56,623,104 training tokens", text)
        self.assertLess(text.index("Run no-TTT"), text.index("before\nTTT"))
        self.assertIn("never inferred or created by a watcher", text)

    def test_watcher_has_valid_bash_and_waits_for_corrected_release(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_stage1_timing", text)
        self.assertIn("waiting_for_explicit_corrected_stage1_release", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertLess(text.index("completed_corrected_stage1_timing)"),
                        text.index("launching_corrected_formal_stage1"))

    def test_release_requires_accept_and_binds_both_audits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing, corrected = timing_fixture(root)
            selected = selected_stream_fixture(root)
            output = root / "release.json"
            with self.assertRaises(PermissionError):
                create_release(
                    timing_audit=timing, corrected_timing_audit=corrected,
                    selected_stream_audit=selected,
                    output=output, accept=False, reviewed_by="reviewer",
                    decision_note="measured projection accepted",
                )
            self.assertFalse(output.exists())
            release = create_release(
                timing_audit=timing, corrected_timing_audit=corrected,
                selected_stream_audit=selected,
                output=output, accept=True, reviewed_by="reviewer",
                decision_note="measured projection accepted",
            )
            self.assertEqual(release["timing_audit_sha256"], sha256(timing))
            self.assertEqual(release["corrected_timing_audit_sha256"],
                             sha256(corrected))
            self.assertEqual(release["model_revision"], MODEL_REVISION)
            self.assertEqual(release["selected_stream_audit_sha256"], sha256(selected))
            self.assertEqual(release["selected_stream_stage"], "stage1_32k")

    def test_wrapper_audit_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            result = audit(argparse.Namespace(
                root=missing, model=missing, model_identity=missing / "id.json",
                corpus_root=missing, corpus_audit=missing / "audit.json",
                timing_root=missing,
                corrected_timing_audit=missing / "corrected.json",
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
