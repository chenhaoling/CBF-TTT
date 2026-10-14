import unittest

from tasks.evaluate_natural_semantic_embedding_pairs import (
    binary_metrics,
    rank_metrics,
    select_threshold,
)


class NaturalSemanticEmbeddingPairsTest(unittest.TestCase):
    def test_perfect_metrics(self):
        labels = [0, 0, 1, 1]
        scores = [.1, .2, .8, .9]
        threshold, metric, feasible = select_threshold(labels, scores)
        self.assertTrue(feasible)
        self.assertEqual(metric["balanced_accuracy"], 1.0)
        self.assertEqual(binary_metrics(labels, [int(x >= threshold) for x in scores])["f1"], 1.0)
        self.assertEqual(rank_metrics(labels, scores)["auroc"], 1.0)
        self.assertEqual(rank_metrics(labels, scores)["average_precision"], 1.0)

    def test_rank_metrics_handle_ties(self):
        metric = rank_metrics([0, 1], [.5, .5])
        self.assertEqual(metric["auroc"], .5)
        self.assertEqual(metric["average_precision"], .5)

    def test_infeasible_constraints_fall_back_deterministically(self):
        threshold, metric, feasible = select_threshold([0, 1], [.5, .5])
        self.assertFalse(feasible)
        self.assertTrue(math_is_finite(threshold))
        self.assertGreaterEqual(metric["balanced_accuracy"], .5)


def math_is_finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    unittest.main()
