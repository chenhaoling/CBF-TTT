import unittest

from tasks.train_crossfit_calibrated_forgetting_v5 import oof_gate, subject_folds


class CrossfitV5Test(unittest.TestCase):
    def test_subject_folds_are_balanced_and_subject_disjoint(self):
        episodes = [{"primary_subject_hash": f"s{i // 4}"} for i in range(100)]
        folds = subject_folds(episodes)
        counts = [sum(value == fold for value in folds.values()) for fold in range(5)]
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertEqual(len(folds), 25)

    def test_oof_gate_requires_deletion_and_safety(self):
        metric = {"current_retention": .98, "non_target_retention": .96,
                  "stale_deletion": .86, "pair_state_accuracy": .85,
                  "by_event_kind": {"hard_retain": {"accuracy": .96}}}
        self.assertTrue(oof_gate(metric)["passed"])
        metric["stale_deletion"] = .84
        self.assertFalse(oof_gate(metric)["passed"])


if __name__ == "__main__": unittest.main()
