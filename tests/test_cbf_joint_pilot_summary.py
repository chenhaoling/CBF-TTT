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
                "corner_losses": {"00": 0.7, "01": 1.0, "10": 0.9, "11": 1.1},
                "energy_terms": {"A": 0.1, "B": 0.0, "C": 0.2},
                "label_time_s": 1.2, "peak_allocated_gib": 12.0, "peak_reserved_gib": 14.0,
            }
            path.write_text(json.dumps(row) + "\n")
            result = summarize([path], flat_tolerance=0.01)
            self.assertEqual(result["best_corner_counts"], {"00": 1})
            self.assertEqual(result["interior_better_fraction"], 1.0)
            self.assertAlmostEqual(result["mean_interior_gain"], 0.2)
            self.assertAlmostEqual(result["mean_group_11_minus_00"], 0.4)
            self.assertEqual(result["groups_00_better_than_11"], 1)
            self.assertAlmostEqual(result["mean_energy_C"], 0.2)
            self.assertEqual(result["max_peak_reserved_gib"], 14.0)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize([path, path])

    def test_v2_short_and_long_gap_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            actions = [[a, g] for a in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
            short = [0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8]
            long = [1.0, 0.9, 0.8, 1.1, 1.0, 0.9, 1.2, 1.1, 1.0]
            losses = [(a + b) / 2 for a, b in zip(short, long)]
            row = {
                "id": "s", "group_id": "g", "regime": "old_relevant_new_informative", "boundary": 2,
                "protocol": "joint_v2", "grid": [0.0, 0.5, 1.0],
                "actions": actions, "losses": losses,
                "corner_losses": {"00": losses[0], "01": losses[2],
                                  "10": losses[6], "11": losses[8]},
                "interaction": losses[8] - losses[6] - losses[2] + losses[0],
                "future_meta": [{"gap_chunks": 0, "query_kinds": ["old_heldout"]},
                                {"gap_chunks": 2, "query_kinds": ["old_heldout"]}],
                "losses_by_future": [[a, b] for a, b in zip(short, long)],
                "query_losses_by_future": [[[a], [b]] for a, b in zip(short, long)],
                "energy_terms": {"A": 0.1, "B": 0.0, "C": 0.2},
                "label_time_s": 2.0, "peak_allocated_gib": 12.0, "peak_reserved_gib": 14.0,
            }
            path.write_text(json.dumps(row) + "\n")
            result = summarize([path])
            self.assertEqual(result["protocol"], "joint_v2")
            self.assertEqual(set(result["mean_corner_losses_by_gap"]), {"0", "2"})
            self.assertAlmostEqual(result["mean_corner_losses_by_gap"]["0"]["01"], 0.6)
            self.assertAlmostEqual(result["mean_corner_losses_by_gap"]["2"]["01"], 0.8)
            self.assertAlmostEqual(result["mean_corner_losses_by_query_kind"]["gap_2/old_heldout"]["01"], 0.8)

    def test_title_recall_conditions_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            actions = [[a, g] for a in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
            kv = [1.0] * 9
            memory = [1.2] * 9
            memory[-1] = 0.9
            losses = [(a + b) / 2 for a, b in zip(kv, memory)]
            row = {
                "id": "title", "group_id": "paper-group", "regime": "new_only", "boundary": 2,
                "protocol": "joint_title_recall_v1", "grid": [0.0, 0.5, 1.0],
                "actions": actions, "losses": losses,
                "corner_losses": {"00": losses[0], "01": losses[2],
                                  "10": losses[6], "11": losses[8]},
                "interaction": losses[8] - losses[6] - losses[2] + losses[0],
                "future_meta": [{"gap_chunks": 0, "reset_kv": False, "query_kinds": ["new_title"]},
                                {"gap_chunks": 0, "reset_kv": True, "query_kinds": ["new_title"]}],
                "losses_by_future": [[a, b] for a, b in zip(kv, memory)],
                "query_losses_by_future": [[[a], [b]] for a, b in zip(kv, memory)],
                "energy_terms": {"A": 0.1, "B": 0.0, "C": 0.2},
                "label_time_s": 2.0, "peak_allocated_gib": 12.0, "peak_reserved_gib": 14.0,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = summarize([path])
            self.assertEqual(set(result["mean_corner_losses_by_gap"]), {"kv_intact", "memory_only"})
            self.assertAlmostEqual(result["mean_corner_losses_by_gap"]["memory_only"]["11"], 0.9)
            self.assertAlmostEqual(result["mean_corner_losses_by_gap"]["kv_intact"]["11"], 1.0)


if __name__ == "__main__":
    unittest.main()
