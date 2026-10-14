"""Focused tests for the independent V16 data audit helpers."""

import unittest

from scripts.audit_mquake_qwen_outcome_v16_data import assign_folds, parse_text


class OutcomeDataAuditTest(unittest.TestCase):
    def test_fold_assignment_is_order_invariant_and_balanced(self):
        rows = [{"primary_subject_hash": subject} for subject, count in
                (("a", 8), ("b", 7), ("c", 6), ("d", 5), ("e", 4), ("f", 3))
                for _ in range(count)]
        first, loads = assign_folds(rows)
        second, reverse_loads = assign_folds(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(loads, reverse_loads)
        self.assertEqual(sum(loads), len(rows))

    def test_parse_text(self):
        self.assertEqual(parse_text("Memory statement: Who?\nValue: Alice"),
                         ("Who?", "Alice"))
        with self.assertRaises(ValueError):
            parse_text("Who? Alice")


if __name__ == "__main__":
    unittest.main()
