"""Unit tests for CounterFact zero-shot failure diagnostics."""

import unittest

from scripts.analyze_counterfact_zero_shot_failure import event_summary, quantile


class CounterFactFailureDiagnosticTests(unittest.TestCase):
    def test_quantile_is_deterministic_order_statistic(self):
        self.assertEqual(quantile([4, 1, 3, 2], .5), 3)

    def test_triggered_candidate_accuracy_uses_only_deletions(self):
        rows = [
            {"controller_probability": .9, "predicted": "local", "correct": 1},
            {"controller_probability": .8, "predicted": "local", "correct": 0},
            {"controller_probability": .1, "predicted": "retain", "correct": 1},
        ]
        result = event_summary(rows)
        self.assertAlmostEqual(result["predicted_local_rate"], 2 / 3)
        self.assertEqual(result["triggered_candidate_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
