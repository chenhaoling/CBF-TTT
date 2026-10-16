import unittest

from scripts.inspect_dart_v13_capacity import aligned_pattern, deduplicate_pairs


class DartV13CapacityTest(unittest.TestCase):
    def test_aligned_pattern_requires_one_literal_pair(self):
        self.assertEqual(
            aligned_pattern("Alice was born in Paris.", "Alice", "Paris"),
            "[X] was born in [Y].",
        )
        self.assertIsNone(aligned_pattern("Alice met Alice in Paris.", "Alice", "Paris"))
        self.assertIsNone(aligned_pattern("Alice visited France.", "Alice", "Paris"))

    def test_pair_deduplication_is_deterministic(self):
        base = {
            "relation_id": "born in",
            "entity_pair": ["alice", "paris"],
            "source_index": 0,
        }
        rows = [
            {**base, "source_row_sha256": "a" * 64},
            {**base, "source_row_sha256": "b" * 64},
        ]
        first = deduplicate_pairs(rows, 17)
        second = deduplicate_pairs(list(reversed(rows)), 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 1)
        self.assertEqual(first[1][0]["reason"], "duplicate_relation_entity_pair")


if __name__ == "__main__":
    unittest.main()
