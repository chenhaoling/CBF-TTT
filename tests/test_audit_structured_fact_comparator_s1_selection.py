import unittest

from scripts.audit_structured_fact_comparator_s1_selection import binary, close, threshold


class StructuredFactComparatorS1SelectionAuditTests(unittest.TestCase):
    def test_independent_binary_counts(self):
        result = binary([True, True, False, False], [True, False, True, False])
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]),
                         (1, 1, 1, 1))

    def test_nested_float_comparison(self):
        self.assertTrue(close({"a": [1., {"b": .2}]}, {"a": [1., {"b": .2 + 1e-13}]}))

    def test_threshold_has_finite_retain_all_candidate(self):
        rows = [{"gold": {"operation": "update"},
                 "prediction": {"operation": "retain"}, "margin": .1}]
        result = threshold(rows)
        self.assertEqual(result["threshold"], 1.)
        self.assertEqual(result["recall"], 0.)


if __name__ == "__main__":
    unittest.main()
