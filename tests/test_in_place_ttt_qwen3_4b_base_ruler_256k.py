import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_ruler_256k_resource import audit


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_256K_PLAN.md"
WATCHER = (
    ROOT /
    "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_ruler_256k_resource.sh"
)


class CorrectedBaseRuler256KTest(unittest.TestCase):
    def test_plan_freezes_scope_host_and_formal_boundary(self):
        text = PLAN.read_text()
        self.assertIn("262,144 tokens", text)
        self.assertIn("exactly one example", text)
        self.assertIn("all 13 tasks and 100 examples", text)
        self.assertIn("Frobenius `1e-5`", text)
        self.assertIn("hku-gpu2", text)
        self.assertIn("separate non-authorizing review packet", text)

    def test_watcher_waits_for_corrected_primary_matrix(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_ruler_matrix", text)
        self.assertIn("completed_corrected_ruler_matrix", text)
        self.assertLess(text.index("completed_corrected_ruler_matrix)"),
                        text.index("launching_corrected_ruler_256k_resource"))
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}',
                      text)
        self.assertIn('PAPER_INFERENCE_ADDENDUM="$PAPER_INFERENCE_ADDENDUM"',
                      text)

    def test_wrapper_audit_fails_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            result = audit(argparse.Namespace(
                root=missing, training_root=missing,
                corrected_training_audit=missing / "training.json",
                matrix_root=missing,
                corrected_matrix_audit=missing / "matrix.json",
                model_identity=missing / "identity.json", tokenizer=missing,
                public_corpus_plan=missing / "public.md",
                evaluation_plan=missing / "evaluation.md",
                extrapolation_plan=missing / "extrapolation.md",
                corrected_primary_plan=missing / "corrected-primary.md",
                corrected_256k_plan=PLAN,
                paper_inference_addendum=missing / "inference.md",
                runner=missing / "runner.sh",
                underlying_auditor=missing / "auditor.py",
                single_runner=missing / "single.sh",
            ))
            self.assertFalse(result["passed"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
