import unittest

try:
    import torch
except ImportError:  # Local documentation/test environments need not install torch.
    torch = None

from tasks.train_fewrel_selective_controller import (
    apply_decisions, cluster_false_deletion_upper, full_binary_selection,
    relation_bootstrap_draws, score_boundaries,
)


class FewRelSelectiveControllerTest(unittest.TestCase):
    def test_score_boundaries_include_tied_groups(self):
        rows = score_boundaries([1, 0, 1, 0], [.8, .8, .4, .1])
        self.assertEqual(len(rows), 3)
        self.assertEqual((rows[0]["tp"], rows[0]["fp"]), (1, 1))
        self.assertEqual((rows[-1]["tp"], rows[-1]["fp"]), (2, 2))

    def test_relation_bootstrap_zero_false_deletion_has_zero_upper_bound(self):
        labels = [0, 0, 1, 1]
        predictions = [0, 0, 1, 0]
        relations = ["r1", "r2", "r1", "r2"]
        draws = relation_bootstrap_draws(relations, draws=100, seed=1)
        self.assertEqual(cluster_false_deletion_upper(
            labels, predictions, relations, draws), 0.0)

    def test_full_binary_selection_recovers_separable_probabilities(self):
        if torch is None:
            self.skipTest("torch is not installed")
        labels = [1, 1, 0, 0]
        probabilities = torch.tensor([[.95, .05], [.9, .1], [.1, .9], [.05, .95]])
        selected, _ = full_binary_selection(labels, probabilities)
        self.assertTrue(selected["constraints_feasible"])
        self.assertEqual(selected["metrics"]["balanced_accuracy"], 1.0)

    def test_apply_decisions_respects_defer_band(self):
        if torch is None:
            self.skipTest("torch is not installed")
        probabilities = torch.tensor([[.9, .1], [.5, .5], [.1, .9]])
        full = {"risk_penalty": 0.0, "threshold": .5}
        selective = {"risk_penalty": 0.0, "delete": {"threshold": .8},
                     "retain": {"threshold": .2}}
        _, predictions, _, actions = apply_decisions(probabilities, full, selective)
        self.assertEqual(predictions, [1, 1, 0])
        self.assertEqual(actions, ["delete", "defer", "retain"])


if __name__ == "__main__":
    unittest.main()
