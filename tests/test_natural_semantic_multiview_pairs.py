import unittest

from tasks.evaluate_natural_semantic_multiview_pairs import select_arm


class NaturalSemanticMultiviewPairsTest(unittest.TestCase):
    def test_selection_prefers_feasibility_then_balanced_accuracy(self):
        candidates = [
            {"arm": "mean", "inner_constraints_feasible": False,
             "inner_metrics": {"balanced_accuracy": .9, "auroc": .9, "f1": .9}},
            {"arm": "last_token", "inner_constraints_feasible": True,
             "inner_metrics": {"balanced_accuracy": .7, "auroc": .8, "f1": .7}},
            {"arm": "last8_mean", "inner_constraints_feasible": True,
             "inner_metrics": {"balanced_accuracy": .8, "auroc": .75, "f1": .8}},
        ]
        self.assertEqual(select_arm(candidates)["arm"], "last8_mean")


if __name__ == "__main__":
    unittest.main()
