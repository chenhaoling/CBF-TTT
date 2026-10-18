import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_ruler_matrix import (
    audit as audit_matrix,
)
from scripts.audit_in_place_ttt_qwen3_4b_base_ruler_resource import (
    audit as audit_resource,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_PLAN.md"
RESOURCE_WATCHER = (
    ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_ruler_resource.sh"
)
MATRIX_WATCHER = (
    ROOT / "scripts/watch_and_run_in_place_ttt_qwen3_4b_base_ruler_matrix.sh"
)


class CorrectedBaseRulerTest(unittest.TestCase):
    def test_plan_freezes_host_tasks_samples_contexts_and_claim_boundary(self):
        text = PLAN.read_text()
        self.assertIn("hku-gpu2", text)
        self.assertIn("all 13 registered RULER tasks", text)
        self.assertIn("exactly one example per", text)
        self.assertIn("exactly 100 examples", text)
        for value in ("4K", "8K", "16K", "32K", "64K", "128K"):
            self.assertIn(value, text)
        self.assertIn("Frobenius norm `1e-5`", text)
        self.assertIn("Update-admission experiments remain sealed", text)

    def test_resource_watcher_waits_for_corrected_formal_stage2(self):
        subprocess.run(["bash", "-n", str(RESOURCE_WATCHER)], check=True)
        text = RESOURCE_WATCHER.read_text()
        self.assertIn("waiting_for_corrected_formal_stage2", text)
        self.assertIn("completed_corrected_formal_stage2", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertLess(text.index("completed_corrected_formal_stage2)"),
                        text.index("launching_corrected_ruler_resource"))

    def test_matrix_watcher_waits_for_corrected_resource(self):
        subprocess.run(["bash", "-n", str(MATRIX_WATCHER)], check=True)
        text = MATRIX_WATCHER.read_text()
        self.assertIn("waiting_for_corrected_ruler_resource", text)
        self.assertIn("completed_corrected_ruler_resource", text)
        self.assertLess(text.index("completed_corrected_ruler_resource)"),
                        text.index("launching_corrected_ruler_matrix"))

    def test_wrapper_audits_fail_closed_when_inputs_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            common = dict(
                root=missing, training_root=missing,
                corrected_training_audit=missing / "training.json",
                model_identity=missing / "identity.json", tokenizer=missing,
                public_corpus_plan=missing / "corpus.md",
                evaluation_plan=missing / "evaluation.md",
                corrected_plan=PLAN,
                paper_inference_addendum=missing / "inference.md",
                runner=missing / "runner.sh",
                underlying_auditor=missing / "auditor.py",
            )
            resource = audit_resource(argparse.Namespace(
                **common, single_runner=missing / "single.sh",
            ))
            self.assertFalse(resource["passed"])
            matrix = audit_matrix(argparse.Namespace(
                **common, resource_root=missing,
                corrected_resource_audit=missing / "resource.json",
                summarizer=missing / "summarizer.py",
            ))
            self.assertFalse(matrix["passed"])


if __name__ == "__main__":
    unittest.main()
