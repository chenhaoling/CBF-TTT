import unittest

from tasks.extract_fewrel_selective_pair_embeddings import partition_pairs


class FewRelSelectivePairEmbeddingsTest(unittest.TestCase):
    def test_shards_are_disjoint_and_complete(self):
        pairs = [{"pair_id": str(index)} for index in range(17)]
        left = partition_pairs(pairs, 0, 2)
        right = partition_pairs(pairs, 1, 2)
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})
        self.assertEqual(sorted(index for index, _ in left + right), list(range(17)))

    def test_invalid_shard_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_pairs([], 2, 2)
        with self.assertRaises(ValueError):
            partition_pairs([], 0, 0)


if __name__ == "__main__":
    unittest.main()
