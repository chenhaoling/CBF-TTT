from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_in_place_ttt_qwen3_4b_base_scale_feasibility import audit
from scripts.summarize_in_place_ttt_qwen3_4b_base_scale_feasibility import (
    MODEL_REVISION,
    render_markdown,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_SCALE_FEASIBILITY_PLAN.md"
REPORTER = ROOT / "scripts/summarize_in_place_ttt_qwen3_4b_base_scale_feasibility.py"
WATCHER = ROOT / "scripts/watch_and_prepare_in_place_ttt_qwen3_4b_base_scale_feasibility.sh"


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def timing(root: Path, stage: str, speeds: dict[str, float]) -> Path:
    selected = 56_623_104 if stage == "stage1" else 41_943_040
    arms = {
        arm: {
            "steady_tokens_per_second": speed,
            f"projected_formal_{stage}_seconds": selected / speed,
            "gpu": {"peak_memory_mib": {"0": 20000, "1": 21000},
                    "minimum_free_fraction": {"0": 0.3, "1": 0.3}},
            "host": {"minimum_memory_available_mib": 20000,
                     "peak_swap_used_mib": 100},
        }
        for arm, speed in speeds.items()
    }
    return write(root / f"{stage}.json", {
        "protocol": f"in_place_ttt_qwen3_4b_base_{stage}_timing_audit_v1",
        "passed": True, "errors": [], "model_revision": MODEL_REVISION,
        "arms": arms,
        f"projected_sequential_{stage}_seconds": sum(
            selected / speed for speed in speeds.values()
        ),
    })


class CorrectedBaseScaleFeasibilityTest(unittest.TestCase):
    def test_plan_and_watcher_are_projection_only(self):
        text = PLAN.read_text()
        self.assertIn("20,000,000,000", text)
        self.assertIn("15,000,000,000", text)
        self.assertIn("70B processed tokens", text)
        self.assertIn("paper_scale_executed=false", text)
        subprocess.run(["bash", "-n", str(WATCHER)], check=True)
        watcher = WATCHER.read_text()
        self.assertIn("waiting_for_corrected_stage1_and_stage2_timing", watcher)
        self.assertIn("completed_scale_feasibility_report", watcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=", watcher)

    def test_projection_and_independent_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage1 = timing(root, "stage1", {"no_ttt": 1000.0, "ttt": 800.0})
            stage2 = timing(root, "stage2", {"no_ttt": 500.0, "ttt": 400.0})
            result = summarize(stage1, stage2, PLAN)
            self.assertEqual(result["matched_pair_processed_tokens"], 70_000_000_000)
            self.assertFalse(result["paper_scale_executed"])
            self.assertTrue(result["projection_only"])
            summary = write(root / "summary.json", result)
            report = root / "SCALE_FEASIBILITY.md"
            report.write_text(render_markdown(result))
            args = argparse.Namespace(
                stage1_audit=stage1, stage2_audit=stage2, plan=PLAN,
                reporter=REPORTER, summary_json=summary, report_md=report,
                output=root / "audit.json",
            )
            passed = audit(args)
            self.assertTrue(passed["passed"], passed["errors"])
            changed = json.loads(stage2.read_text())
            changed["arms"]["ttt"]["steady_tokens_per_second"] = 401.0
            write(stage2, changed)
            failed = audit(args)
            self.assertFalse(failed["passed"])
            self.assertIn("independent recomputation failed", failed["errors"][0])


if __name__ == "__main__":
    unittest.main()
