import unittest
import random

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

    def test_fast_threshold_matches_brute_force(self):
        rng = random.Random(17)
        labels = [index % 2 for index in range(40)]
        scores = [rng.choice([.1, .2, .4, .8, .9]) for _ in labels]
        threshold, metric, feasible = select_threshold(labels, scores)
        candidates = sorted(set([
            float("-inf"), float("inf"),
            *[(left + right) / 2 for left, right in
              zip(sorted(set(scores)), sorted(set(scores))[1:])],
        ]))
        brute = []
        for value in candidates:
            result = binary_metrics(labels, [int(score >= value) for score in scores])
            brute.append((value, result))
        feasible_rows = [value for value in brute
                         if value[1]["positive_recall"] >= .8
                         and value[1]["negative_specificity"] >= .95]
        selected = max(feasible_rows or brute, key=lambda value: (
            value[1]["balanced_accuracy"], value[1]["f1"],
            value[1]["negative_specificity"], -value[0]))
        self.assertEqual(feasible, bool(feasible_rows))
        self.assertEqual(metric, selected[1])
        self.assertEqual(
            [int(score >= threshold) for score in scores],
            [int(score >= selected[0]) for score in scores])


def math_is_finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    unittest.main()
