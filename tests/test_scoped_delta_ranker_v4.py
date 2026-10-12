"""Pure tests for the V4 factorized scope ranker."""

import unittest

from tasks.train_scoped_delta_ranker_v4 import development_gate, local_groups


class ScopedDeltaRankerV4Tests(unittest.TestCase):
    def test_local_groups_excludes_retain_decisions(self):
        rows = [
            {"target": 8, "desired_action": "retain"},
            {"target": 1, "desired_action": "local"},
        ]
        self.assertEqual(local_groups(rows), [rows[1]])

    def test_gate_requires_all_registered_checks(self):
        metric = {"top1_accuracy": .89, "mrr": .94,
                  "target_beats_same_subject": .86,
                  "target_beats_all_different_subject": .96}
        comparison = {"top1_gain": .06, "bootstrap_95ci": [.01, .10]}
        self.assertTrue(development_gate(metric, comparison)["passed"])
        metric["target_beats_same_subject"] = .84
        self.assertFalse(development_gate(metric, comparison)["passed"])


if __name__ == "__main__":
    unittest.main()

