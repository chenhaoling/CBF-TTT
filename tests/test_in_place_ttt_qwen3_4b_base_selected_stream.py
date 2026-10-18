from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_SELECTED_STREAM_ADDENDUM.md"
REPLAY = ROOT / "scripts/replay_in_place_ttt_qwen3_4b_base_100m_selected_stream.py"
AUDITOR = ROOT / "scripts/audit_in_place_ttt_qwen3_4b_base_100m_selected_stream.py"
WATCHER = ROOT / "scripts/watch_and_audit_in_place_ttt_qwen3_4b_base_100m_selected_stream.sh"


class CorrectedSelectedStreamTest(unittest.TestCase):
    def test_plan_freezes_runtime_and_release_gate(self):
        text = PLAN.read_text()
        for value in (
            "hku-gpu2", "StatefulDataLoader", "num_workers=2",
            "56,623,104", "41,943,040", "15,000", "explicit human",
        ):
            self.assertIn(value, text)

    def test_python_sources_parse(self):
        ast.parse(REPLAY.read_text())
        ast.parse(AUDITOR.read_text())

    def test_watcher_is_cpu_only_and_fail_closed(self):
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        text = WATCHER.read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES=", text)
        self.assertIn("waiting_for_corrected_base_corpus", text)
        self.assertIn("completed_selected_stream_audit", text)
        self.assertIn('EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}', text)
        self.assertLess(text.index("replaying_native_selected_stream"),
                        text.index("independently_auditing_selected_stream"))

    def test_replay_artifact_never_serializes_rows(self):
        text = REPLAY.read_text()
        self.assertNotIn('"selected_rows"', text)
        self.assertNotIn('"input_ids":', text)
        self.assertIn("ordered_training_hashes_sha256", text)


if __name__ == "__main__":
    unittest.main()
