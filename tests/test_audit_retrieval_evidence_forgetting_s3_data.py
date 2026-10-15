import unittest

from scripts.audit_retrieval_evidence_forgetting_s3_data import (
    binary,
    bm25_pair_scores,
    threshold_at_specificity,
    trigram,
)


class RetrievalEvidenceForgettingS3AuditTests(unittest.TestCase):
    def test_threshold_respects_specificity(self):
        scores = [.9, .8, .7, .6]
        gold = [True, False, True, False]
        value = threshold_at_specificity(scores, gold, .95)
        self.assertEqual(value["threshold"], .9)
        self.assertEqual(value["specificity"], 1.0)
        self.assertEqual(value["recall"], .5)

    def test_bm25_prefers_related_pair(self):
        rows = [
            {"left_text": "[ARG1] founded [ARG2] today",
             "right_text": "[ARG1] founded [ARG2] yesterday"},
            {"left_text": "[ARG1] founded [ARG2] today",
             "right_text": "[ARG1] lives near [ARG2]"},
        ]
        scores = bm25_pair_scores(rows)
        self.assertGreater(scores[0], scores[1])

    def test_trigram_and_binary(self):
        self.assertEqual(trigram("same", "same"), 1.0)
        value = binary([True, False], [True, True])
        self.assertEqual((value["tp"], value["fp"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
