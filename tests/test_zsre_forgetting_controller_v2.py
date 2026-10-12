"""Tests for low-dimensional cosine-top forgetting gate."""

import unittest

from tasks.train_zsre_forgetting_controller_v2 import relation_tokens, set_similarity


class TopGateTests(unittest.TestCase):
    def test_relation_tokens_remove_capitalized_entity_and_stopwords(self):
        self.assertEqual(relation_tokens("What is the publisher of Silversands?"), ["publisher"])

    def test_relation_similarity_separates_relation_words(self):
        same = set_similarity(["mother"], ["name", "mother"])
        different = set_similarity(["occupation"], ["name", "mother"])
        self.assertGreater(same[0], different[0])
        self.assertGreater(same[1], different[1])


if __name__ == "__main__": unittest.main()
