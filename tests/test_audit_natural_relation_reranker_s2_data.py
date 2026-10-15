import unittest

from scripts.audit_natural_relation_reranker_s2_data import (
    apply_threshold, binary, lexical_bin, lexical_similarity, select_threshold,
)


class NaturalRelationRerankerS2AuditTests(unittest.TestCase):
    def test_binary_counts(self):
        value = binary([True, True, False, False], [True, False, True, False])
        self.assertEqual((value["tp"], value["fp"], value["fn"], value["tn"]),
                         (1, 1, 1, 1))

    def test_threshold_keeps_precision_constraint(self):
        rows = [
            {"pair_id": "p0", "same_relation": 1, "trigram_similarity": .9},
            {"pair_id": "n0", "same_relation": 0, "trigram_similarity": .8},
            {"pair_id": "p1", "same_relation": 1, "trigram_similarity": .7},
            {"pair_id": "n1", "same_relation": 0, "trigram_similarity": .6},
        ]
        selected = select_threshold(rows)
        self.assertEqual(selected["threshold"], .9)
        self.assertEqual(selected["precision"], 1.0)
        self.assertEqual(selected["recall"], .5)
        applied, mapping = apply_threshold(rows, selected["threshold"])
        self.assertEqual(applied["tp"], 1)
        self.assertTrue(mapping["p0"])

    def test_lexical_score_and_bin(self):
        self.assertEqual(lexical_similarity("same", "same"), 1.0)
        self.assertEqual(lexical_bin(1.0), 19)
        self.assertGreaterEqual(lexical_similarity("[SUBJECT] founded [VALUE]",
                                                   "[SUBJECT] founded [VALUE] today"), .5)


if __name__ == "__main__":
    unittest.main()
