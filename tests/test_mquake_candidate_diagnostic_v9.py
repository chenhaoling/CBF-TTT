import unittest

from tasks.diagnose_mquake_candidate_scope_v9 import select_arm


class MquakeCandidateDiagnosticV9Test(unittest.TestCase):
    def test_selects_smallest_eligible_anchor(self):
        metrics = {"learned_full": {"top1": .91},
                   "max_8": {"top1": .85, "target_recall": .94},
                   "anchor_8": {"top1": .85, "target_recall": .95},
                   "anchor_4": {"top1": .79, "target_recall": .95}}
        result = select_arm(metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(result["selected"], "learned_full")

    def test_rejects_low_recall_or_top1(self):
        metrics = {"max_16": {"top1": .85, "target_recall": .89},
                   "anchor_16": {"top1": .79, "target_recall": .99}}
        self.assertFalse(select_arm(metrics)["passed"])


if __name__ == "__main__":
    unittest.main()
