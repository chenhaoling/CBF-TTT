import unittest

from tasks.score_natural_relation_reranker_s2 import candidates, partition


class RetrievalEvidenceForgettingS3PriorScoreTests(unittest.TestCase):
    def test_binary_candidate_order_is_false_then_true(self):
        self.assertEqual(candidates(), [False, True])

    def test_partition_preserves_global_indices(self):
        rows = [{"pair_id": str(index)} for index in range(7)]
        self.assertEqual([index for index, _ in partition(rows, 1, 2)], [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
