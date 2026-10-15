import unittest

from tasks.score_natural_relation_reranker_s2 import candidates, partition


class NaturalRelationRerankerS2ScoreTests(unittest.TestCase):
    def test_candidates_are_binary(self):
        self.assertEqual(candidates(), [False, True])

    def test_partition_is_complete_and_disjoint(self):
        rows = list(range(9))
        left, right = partition(rows, 0, 2), partition(rows, 1, 2)
        self.assertEqual(sorted(index for index, _ in left + right), list(range(9)))
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})


if __name__ == "__main__":
    unittest.main()
