import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_stage1_ladder import (
    LENGTHS,
    audit_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE1_RESOURCE_PLAN.md"
RUNNER = ROOT / "scripts/run_in_place_ttt_qwen3_4b_base_stage1_ladder.sh"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_stage1_ladder.sh"


class CorrectedBaseStage1LadderTest(unittest.TestCase):
    def test_plan_freezes_host_lengths_order_and_boundary(self):
        text = PLAN.read_text()
        self.assertEqual(LENGTHS, (16384, 32768))
        self.assertIn("hku-gpu2", text)
        offsets = [text.index(value) for value in (
            "16K no-TTT", "16K native TTT", "32K no-TTT", "32K native TTT"
        )]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("resource evidence only", text)

    def test_runner_and_watcher_have_valid_bash(self):
        subprocess.run(["bash", "-n", str(RUNNER)], check=True)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_level_a", watcher)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', watcher)
        runner = RUNNER.read_text()
        self.assertLess(runner.index("for length in 16384 32768"),
                        runner.index("write_state all_children_completed"))
        self.assertLess(runner.index("for arm in no_ttt ttt"),
                        runner.index("write_state all_children_completed"))

    def test_prerequisite_fails_closed_when_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = audit_prerequisite(
                root / "level_a", root / "level_a" / "independent_audit.json",
                root / "inputs" / "manifest.json",
                root / "inputs" / "independent_audit.json",
            )
            self.assertFalse(result["passed"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
