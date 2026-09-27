"""The v2 stage gate distinguishes write, skip, and retention signals."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_cbf_joint_v2_gate import evaluate


class JointV2GateTests(unittest.TestCase):
    def test_balanced_diagnostic_actions_pass(self):
        actions = [[a, g] for a in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
        regimes = {
            "old_relevant_new_informative": {"00": 0.3, "01": 0.2, "10": 0.2, "11": 0.1},
            "old_relevant_new_noise": {"00": 0.2, "01": 0.3, "10": 0.1, "11": 0.3},
            "old_conflict_new_correction": {"00": 0.3, "01": 0.1, "10": 0.4, "11": 0.2},
            "old_conflict_new_noise": {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            with path.open("w") as sink:
                for group in range(3):
                    for variant, (regime, corner_losses) in enumerate(regimes.items()):
                        losses = [corner_losses.get(f"{int(a)}{int(g)}", 0.5) if a in (0, 1) and g in (0, 1)
                                  else 0.5 for a, g in actions]
                        row = {
                            "id": f"{group}-v{variant}", "group_id": str(group),
                            "regime": regime, "boundary": 2, "protocol": "joint_v2",
                            "actions": actions, "future_meta": [{"gap_chunks": 0}, {"gap_chunks": 2}],
                            "losses_by_future": [[loss, loss] for loss in losses],
                            "peak_reserved_gib": 12.0,
                        }
                        sink.write(json.dumps(row) + "\n")
            result = evaluate([path])
            self.assertTrue(result["passed_before_repeat_check"])
            self.assertEqual(result["write_gain_groups_by_regime"]["old_conflict_new_correction"], 3)
            self.assertEqual(result["skip_write_groups_by_regime"]["old_relevant_new_noise"], 3)
            self.assertEqual(result["meaningful_corner_winner_groups"],
                             {"00": 3, "01": 3, "10": 3, "11": 3})


if __name__ == "__main__":
    unittest.main()
