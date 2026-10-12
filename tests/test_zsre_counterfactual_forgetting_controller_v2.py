"""Tests for grouped inner OOF threshold protocol."""

import unittest

from tasks.train_zsre_counterfactual_forgetting_controller_v2 import grouped_folds


class GroupedOOFTests(unittest.TestCase):
    def test_folds_are_complete_disjoint_and_deterministic(self):
        sessions = [f"s{i}" for i in range(13)]
        first = grouped_folds(sessions, "held", 5)
        second = grouped_folds(list(reversed(sessions)), "held", 5)
        self.assertEqual(first, second)
        flattened = [value for fold in first for value in fold]
        self.assertEqual(sorted(flattened), sorted(sessions))
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__": unittest.main()
