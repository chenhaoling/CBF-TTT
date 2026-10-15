import unittest

from scripts.audit_retrieval_evidence_forgetting_s3_fusion import close
from scripts.audit_retrieval_evidence_forgetting_s3_projection import key


class RetrievalEvidenceForgettingS3AuditTests(unittest.TestCase):
    def test_nested_close(self):
        self.assertTrue(close({"a": 1.0, "b": {"x": 2}},
                              {"a": 1.0 + 1e-10, "b": {"x": 2}, "extra": 3}))
        self.assertFalse(close({"a": 1.0}, {"a": 1.1}))

    def test_projection_key_prefers_primary_metric(self):
        def row(recall, auc, dim=256):
            return {"retrieval": {"recall_at_16": recall}, "direct_pair_roc_auc": auc,
                    "config": {"dimension": dim, "epoch": 10,
                               "pubmed_pretraining": False,
                               "temperature": .05, "margin": .1}}
        self.assertGreater(key(row(.8, .5)), key(row(.7, .9)))


if __name__ == "__main__":
    unittest.main()
