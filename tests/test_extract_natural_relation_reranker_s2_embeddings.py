import unittest

from tasks.extract_natural_relation_reranker_s2_embeddings import partition, unique_mentions


class NaturalRelationRerankerS2EmbeddingTests(unittest.TestCase):
    def test_mentions_are_deduplicated_and_sorted(self):
        rows = [{"left_source_row_sha256": "b", "left_text": "B", "left_relation_audit": "r1",
                 "right_source_row_sha256": "a", "right_text": "A", "right_relation_audit": "r2"},
                {"left_source_row_sha256": "b", "left_text": "B", "left_relation_audit": "r1",
                 "right_source_row_sha256": "c", "right_text": "C", "right_relation_audit": "r1"}]
        values = unique_mentions(rows)
        self.assertEqual([row["source_id"] for row in values], ["a", "b", "c"])

    def test_inconsistent_source_is_rejected(self):
        rows = [{"left_source_row_sha256": "a", "left_text": "A", "left_relation_audit": "r",
                 "right_source_row_sha256": "a", "right_text": "changed",
                 "right_relation_audit": "r"}]
        with self.assertRaises(ValueError):
            unique_mentions(rows)

    def test_partition_is_complete(self):
        rows = list(range(7))
        values = partition(rows, 0, 2) + partition(rows, 1, 2)
        self.assertEqual(sorted(index for index, _ in values), list(range(7)))


if __name__ == "__main__":
    unittest.main()
