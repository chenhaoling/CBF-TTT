import unittest

from scripts.audit_natural_semantic_pair_cross_controller import (
    binary_metrics, close_tree, complete_metrics, rank_metrics,
)


class NaturalSemanticPairCrossControllerAuditTest(unittest.TestCase):
    def test_binary_and_rank_metrics_are_independently_reconstructed(self):
        rows = [
            {"same_relation": 1, "prediction": 1, "decision_score": .9},
            {"same_relation": 1, "prediction": 0, "decision_score": .4},
            {"same_relation": 0, "prediction": 1, "decision_score": .6},
            {"same_relation": 0, "prediction": 0, "decision_score": .1},
        ]
        metric = complete_metrics(rows)
        self.assertEqual(metric["tp"], 1)
        self.assertEqual(metric["tn"], 1)
        self.assertEqual(metric["fp"], 1)
        self.assertEqual(metric["fn"], 1)
        self.assertEqual(metric["balanced_accuracy"], .5)
        self.assertEqual(metric["auroc"], .75)
        self.assertAlmostEqual(metric["average_precision"], 5 / 6)

    def test_tied_rank_metrics_match_average_rank_convention(self):
        metric = rank_metrics([1, 0, 1, 0], [.5, .5, .2, .1])
        self.assertEqual(metric["auroc"], .625)
        self.assertTrue(0 <= metric["average_precision"] <= 1)

    def test_empty_and_mismatched_binary_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([], [])
        with self.assertRaises(ValueError):
            binary_metrics([1], [1, 0])

    def test_metric_tree_accepts_int_float_but_not_key_or_value_change(self):
        self.assertTrue(close_tree({"a": 1, "b": .5}, {"a": 1.0, "b": .5}))
        self.assertFalse(close_tree({"a": 1}, {"a": 1, "b": 2}))
        self.assertFalse(close_tree({"a": .5}, {"a": .6}))


if __name__ == "__main__":
    unittest.main()
