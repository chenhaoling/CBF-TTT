import unittest

from tasks.train_qwen_natural_scope_transfer_v2 import decision_metrics, fold_for


class NaturalScopeTransferTest(unittest.TestCase):
    def test_registered_three_stratum_metrics(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is exercised in the remote training environment")

        metadata = [
            {"stratum": "natural_correct_top"},
            {"stratum": "natural_correct_top"},
            {"stratum": "natural_wrong_top"},
            {"stratum": "no_match"},
        ]
        logits = torch.tensor([3., -3., -3., -3.])
        result = decision_metrics(logits, metadata, .5)
        self.assertEqual(result["correct_top_recall"], .5)
        self.assertEqual(result["natural_wrong_top_specificity"], 1.)
        self.assertEqual(result["no_match_specificity"], 1.)
        self.assertAlmostEqual(result["balanced_accuracy"], 5 / 6)

    def test_fold_is_stable_and_bounded(self):
        self.assertEqual(fold_for("session-a"), fold_for("session-a"))
        self.assertIn(fold_for("session-a"), range(5))


if __name__ == "__main__":
    unittest.main()
