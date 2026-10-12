"""Unit tests for the development-only forgetting threshold diagnostic."""

import unittest

from tasks.diagnose_zsre_forgetting_dev_threshold import rank_threshold, threshold_key


class ThresholdDiagnosticTests(unittest.TestCase):
    @staticmethod
    def row(threshold, action, hard, update, query):
        return {"threshold": threshold, "controller": {
            "mean_action_accuracy": action,
            "mean_accuracy": query,
            "by_event_kind": {
                "hard_retain": {"accuracy": hard},
                "primary_update": {"accuracy": update},
            },
        }}

    def test_rank_prioritizes_action_then_worst_difficult_type(self):
        stronger_action = self.row(.6, .95, .91, .92, .80)
        stronger_query = self.row(.4, .94, .99, .99, .99)
        self.assertGreater(rank_threshold(stronger_action), rank_threshold(stronger_query))
        balanced = self.row(.5, .95, .93, .93, .79)
        self.assertGreater(rank_threshold(balanced), rank_threshold(stronger_action))

    def test_threshold_key_is_stable(self):
        self.assertEqual(threshold_key(.4), "0.4")


if __name__ == "__main__":
    unittest.main()
