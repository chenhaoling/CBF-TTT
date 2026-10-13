import unittest

from tasks.train_mquake_robust_calibration_v11 import oof_gate, subject_folds


class MquakeRobustCalibrationV11Test(unittest.TestCase):
    def test_subject_folds_are_disjoint_and_count_balanced(self):
        episodes = []
        for subject, count in (("a", 20), ("b", 12), ("c", 8), ("d", 8),
                               ("e", 4), ("f", 4), ("g", 4), ("h", 4), ("i", 4), ("j", 4)):
            episodes.extend({"primary_subject_hash": subject} for _ in range(count))
        folds, loads = subject_folds(episodes)
        self.assertEqual(set(folds), set("abcdefghij"))
        self.assertEqual(sum(loads), len(episodes))
        self.assertTrue(all(load > 0 for load in loads))

    def test_oof_gate_requires_safety_and_effectiveness(self):
        metric = {"current_retention": .995, "non_target_retention": .98,
                  "stale_deletion": .93, "pair_state_accuracy": .92,
                  "by_event_kind": {"hard_retain": {"accuracy": .98}}}
        candidate = {"candidate_top1": .995}
        folds = {str(i): {"pair_state_accuracy": .88} for i in range(5)}
        comparison = {"bootstrap_95ci": [.1, .2]}
        self.assertTrue(oof_gate(metric, candidate, folds, comparison)["passed"])
        metric["non_target_retention"] = .96
        self.assertFalse(oof_gate(metric, candidate, folds, comparison)["passed"])


if __name__ == "__main__": unittest.main()
