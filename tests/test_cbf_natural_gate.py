import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_cbf_natural_gate import evaluate
from scripts.summarize_cbf_gap_diagnostic import summarize


class NaturalGateTest(unittest.TestCase):
    def test_group_gate_uses_unique_groups_and_four_regimes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            rows = []
            for group in range(3):
                for regime in ("both_relevant", "old_only", "new_only", "neither_relevant"):
                    gain = 0.01 if group < 2 and regime in ("both_relevant", "new_only") else -0.01
                    rows.append({"protocol": "joint_natural_v1", "group_id": str(group),
                                 "regime": regime,
                                 "corner_losses": {"00": 1.0, "01": 1.1, "10": 1.1,
                                                   "11": 1.0 - gain}})
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = evaluate([path])
            self.assertEqual(result["positive_groups"], 2)
            self.assertFalse(result["passed"])
            self.assertEqual(result["positive_groups_by_regime"],
                             {"both_relevant": 2, "new_only": 2})
            rows.pop()
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                evaluate([path])

    def test_gap_diagnostic_requires_complete_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gap.jsonl"
            trajectories = {
                f"{action}/{policy}": [
                    {"gap_chunks": step, "mean_nll": float(step + 1),
                     "memory_ratio": step / 100.0} for step in range(3)
                ]
                for action in ("00", "01", "10", "11")
                for policy in ("normal_11", "freeze_10")
            }
            row = {"protocol": "gap_diagnostic_v1", "regime": "test",
                   "gap_chunks": 2, "diagnostic_time_s": 4.0,
                   "peak_reserved_gib": 20.0, "trajectories": trajectories}
            path.write_text(json.dumps(row) + "\n")
            result = summarize(path)
            self.assertEqual(result["mean_nll_by_trajectory"]["11/normal_11"],
                             [1.0, 2.0, 3.0])
            del trajectories["11/normal_11"]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                summarize(path)


if __name__ == "__main__":
    unittest.main()
