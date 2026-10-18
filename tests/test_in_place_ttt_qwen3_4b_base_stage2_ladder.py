import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_stage2_ladder import audit


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE2_RESOURCE_PLAN.md"
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_stage2_ladder.sh"


class CorrectedBaseStage2LadderTest(unittest.TestCase):
    def test_plan_freezes_host_order_models_and_scope(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        order = [text.index(value) for value in (
            "65,536 no-TTT", "65,536 native TTT",
            "131,072 no-TTT", "131,072 native TTT",
        )]
        self.assertEqual(order, sorted(order))
        self.assertIn("Each arm starts from its own", text)
        self.assertIn("evidence only", text)

    def test_watcher_has_valid_bash_and_defers_future_inputs(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_formal_stage1", text)
        self.assertIn("completed_corrected_formal_stage1", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertGreater(text.index('test -s "$RESOURCE_INPUT_ROOT/manifest.json"'),
                           text.index("completed_corrected_formal_stage1)"))

    def test_wrapper_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing"
            result = audit(argparse.Namespace(
                root=root, stage1_root=root,
                corrected_stage1_audit=root / "corrected.json",
                model_identity=root / "identity.json",
                resource_input_root=root, resource_plan=PLAN,
                execution_addendum=root / "formal.md",
                input_addendum=root / "input.md", runner=root / "runner.sh",
                underlying_auditor=root / "auditor.py",
                no_ttt_config=root / "no.yaml", ttt_config=root / "ttt.yaml",
            ))
            self.assertFalse(result["passed"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
