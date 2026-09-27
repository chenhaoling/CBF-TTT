"""Joint-grid pilot summary checks without PyTorch."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_cbf_joint_pilot import summarize


class JointPilotSummaryTests(unittest.TestCase):
    def test_corner_and_interior_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            actions = [[a, g] for a in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
            losses = [0.7, 0.8, 1.0, 0.8, 0.5, 0.9, 0.9, 0.8, 1.1]
            row = {
                "id": "s", "group_id": "g", "regime": "old_relevant_new_noise", "boundary": 2,
                "protocol": "joint_v1", "grid": [0.0, 0.5, 1.0],
                "actions": actions, "losses": losses, "interaction": -0.1,
                "label_time_s": 1.2, "peak_allocated_gib": 12.0, "peak_reserved_gib": 14.0,
            }
            path.write_text(json.dumps(row) + "\n")
            result = summarize([path], flat_tolerance=0.01)
            self.assertEqual(result["best_corner_counts"], {"00": 1})
            self.assertEqual(result["interior_better_fraction"], 1.0)
            self.assertAlmostEqual(result["mean_interior_gain"], 0.2)
            self.assertEqual(result["max_peak_reserved_gib"], 14.0)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize([path, path])


if __name__ == "__main__":
    unittest.main()
