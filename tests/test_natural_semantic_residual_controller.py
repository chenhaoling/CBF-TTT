import unittest

from tasks.train_natural_semantic_residual_controller import (
    search_decision, select_snapshot,
)


class NaturalSemanticResidualControllerTest(unittest.TestCase):
    def test_snapshot_prefers_feasible_then_metric(self):
        def row(epoch, feasible, balanced):
            return {"epoch": epoch, "selection": {
                "constraints_feasible": feasible,
                "metrics": {"balanced_accuracy": balanced, "positive_recall": .9,
                            "negative_specificity": .98}}}
        self.assertEqual(select_snapshot([
            row(25, False, .9), row(50, True, .8), row(100, True, .79)
        ])["epoch"], 50)

    def test_registered_grid_is_searched(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        labels = torch.tensor([1., 1., 0., 0.])
        probabilities = torch.tensor([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
        selected, candidates = search_decision(labels, probabilities)
        self.assertEqual(len(candidates), 12)
        self.assertTrue(selected["constraints_feasible"])


if __name__ == "__main__":
    unittest.main()
