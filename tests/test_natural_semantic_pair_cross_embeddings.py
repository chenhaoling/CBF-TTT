import unittest

from tasks.extract_natural_semantic_pair_cross_embeddings import (
    ordered_prompts, pair_prompt, partition_pairs,
)


class NaturalSemanticPairCrossEmbeddingsTest(unittest.TestCase):
    def test_ordered_prompts_are_symmetric_and_label_free(self):
        pair = {"left_surface": "[SUBJECT] was born in",
                "right_surface": "birthplace of [SUBJECT] is"}
        forward, reverse = ordered_prompts(pair)
        self.assertEqual(forward, pair_prompt(pair["left_surface"], pair["right_surface"]))
        self.assertEqual(reverse, pair_prompt(pair["right_surface"], pair["left_surface"]))
        self.assertNotIn("same_relation", forward)
        self.assertNotIn("same_relation", reverse)

    def test_partition_is_complete_and_disjoint(self):
        pairs = [{"pair_id": str(index)} for index in range(11)]
        left = partition_pairs(pairs, 0, 2); right = partition_pairs(pairs, 1, 2)
        self.assertEqual({index for index, _ in left} | {index for index, _ in right},
                         set(range(11)))
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})


if __name__ == "__main__":
    unittest.main()
