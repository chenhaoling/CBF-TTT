import unittest

from scripts.audit_natural_relation_reranker_s2_selection import close, weighted_delete


class NaturalRelationRerankerS2SelectionAuditTests(unittest.TestCase):
    def test_nested_close(self):
        self.assertTrue(close({"a": [1., {"b": 2.}]}, {"a": [1. + 1e-12, {"b": 2.}]}))
        self.assertFalse(close({"a": 1}, {"a": 2}))

    def test_weighted_delete_counts(self):
        value = weighted_delete({"p": True, "n": False}, {"p": True, "n": False})
        self.assertEqual((value["tp"], value["fp"], value["fn"], value["tn"]),
                         (2, 0, 0, 3))


if __name__ == "__main__":
    unittest.main()
