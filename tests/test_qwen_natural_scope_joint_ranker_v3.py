import unittest

from tasks.train_qwen_natural_scope_joint_ranker_v3 import ranker_metrics


class NaturalScopeJointRankerTest(unittest.TestCase):
    def test_metrics_separate_wrong_key_and_defer(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is exercised in the remote training environment")
        strata = (
            ("primary_exact", "natural", "exact", 0, 0),
            ("primary_rephrase", "natural", "rephrase", 1, 8),
            ("auxiliary_exact", "natural", "exact", 2, 3),
            ("auxiliary_rephrase", "natural", "rephrase", 3, 3),
            ("no_match_exact", "no_match", "exact", 8, 8),
            ("no_match_rephrase", "no_match", "rephrase", 8, 8),
        )
        metadata = [{"stratum": s, "kind": k, "surface": q, "target": t}
                    for s, k, q, t, _ in strata]
        logits = torch.full((len(strata), 9), -10.)
        for index, (*_, prediction) in enumerate(strata):
            logits[index, prediction] = 10.
        result, prediction = ranker_metrics(logits, metadata, 0.)
        self.assertEqual(prediction.tolist(), [0, 8, 3, 3, 8, 8])
        self.assertAlmostEqual(result["natural_selection_accuracy"], .5)
        self.assertAlmostEqual(result["natural_wrong_key_rate"], .25)
        self.assertAlmostEqual(result["natural_defer_rate"], .25)
        self.assertEqual(result["no_match_specificity"], 1.)


if __name__ == "__main__":
    unittest.main()
