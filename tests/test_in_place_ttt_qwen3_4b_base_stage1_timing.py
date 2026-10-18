import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_stage1_timing import audit


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE1_TIMING_PLAN.md"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_stage1_timing.sh"


class CorrectedBaseStage1TimingTest(unittest.TestCase):
    def test_plan_freezes_host_order_steps_and_release_boundary(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("exactly three optimizer steps", text)
        self.assertLess(text.index("Run no-TTT"), text.index("native TTT"))
        self.assertIn("never creates a formal release automatically", text)

    def test_watcher_has_valid_bash_and_fail_closed_dependencies(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_stage1_ladder", text)
        self.assertIn("completed_corrected_stage1_ladder", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertIn('EXECUTION_ADDENDUM="$STAGE1_PLAN"', text)

    def test_wrapper_audit_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = audit(argparse.Namespace(
                timing_root=root / "timing",
                model=root / "model",
                model_identity=root / "identity.json",
                corpus_root=root / "corpus",
                corpus_audit=root / "corpus" / "independent_audit.json",
                ladder_audit=root / "ladder" / "independent_ladder_audit.json",
                timing_plan=PLAN,
                stage1_plan=root / "stage1_plan.md",
                runner=root / "runner.sh",
            ))
            self.assertFalse(result["passed"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
