import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_level_a import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    audit,
)


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_level_a.sh"
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_LEVEL_A_EXECUTION_ADDENDUM.md"


class CorrectedBaseLevelATest(unittest.TestCase):
    def test_plan_freezes_exact_model_and_host(self):
        text = PLAN.read_text()
        self.assertIn("frozen before corrected-Base GPU execution", text)
        self.assertIn(MODEL_REPOSITORY, text)
        self.assertIn(MODEL_REVISION, text)
        self.assertIn("hku-gpu2", text)
        self.assertIn("no_ttt` before `ttt", text)

    def test_watcher_has_valid_bash_and_orders_arms(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertIn("WAITING_BASE_CORPUS", text)
        self.assertLess(text.index("run_arm no_ttt"), text.index("run_arm ttt"))
        self.assertLess(text.index("CONVERTING_CHECKPOINTS"),
                        text.index("RUNNING_RULER_SMOKE"))
        self.assertIn("REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS=1", text)

    def test_auditor_fails_closed_on_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                root=root / "run", model=root / "model",
                identity=root / "identity.json", corpus_root=root / "corpus",
                corpus_audit=root / "corpus" / "independent_audit.json",
                resource_inputs=root / "inputs",
                resource_audit=root / "inputs" / "independent_audit.json",
                parent_plan=root / "parent.md",
                correction_addendum=root / "correction.md",
                execution_addendum=root / "execution.md",
                resource_input_addendum=root / "resource.md",
                output=root / "audit.json",
            )
            result = audit(args)
            self.assertFalse(result["passed"])
            self.assertIn("missing required files", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
