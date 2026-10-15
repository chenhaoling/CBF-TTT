import unittest

from tasks.select_natural_relation_reranker_s2 import (
    delete_counts, select_retrieval_k, select_threshold,
)


class NaturalRelationRerankerS2SelectionTests(unittest.TestCase):
    def test_delete_counts_reflect_two_safe_actions_per_positive(self):
        gold = {"p": True, "n": False}
        value = delete_counts(gold, {"p": True, "n": True})
        self.assertEqual((value["tp"], value["fp"], value["fn"], value["tn"]),
                         (2, 1, 0, 2))

    def test_threshold_respects_delete_precision(self):
        rows = [{"pair_id": "p0", "same_relation": 1},
                {"pair_id": "n0", "same_relation": 0},
                {"pair_id": "p1", "same_relation": 1},
                {"pair_id": "n1", "same_relation": 0}]
        value = select_threshold(rows, {"p0": .9, "n0": .8, "p1": .7, "n1": .6})
        self.assertEqual(value["threshold"], .9)
        self.assertEqual(value["precision"], 1.0)
        self.assertEqual(value["recall"], .5)

    def test_retrieval_k_prefers_smallest_passing_value(self):
        metrics = {"recall_at_4": .94, "recall_at_8": .96, "recall_at_16": .98}
        self.assertEqual(select_retrieval_k(metrics), 8)
        metrics = {"recall_at_4": .4, "recall_at_8": .6, "recall_at_16": .8}
        self.assertEqual(select_retrieval_k(metrics), 16)


if __name__ == "__main__":
    unittest.main()
