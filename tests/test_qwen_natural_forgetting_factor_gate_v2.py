import unittest

try:
    import torch
except ImportError:
    torch = None

from tasks.train_qwen_natural_forgetting_factor_gate_v2 import (
    ENGINEERED,
    NO_SCHEDULE,
    factor_actions,
    factor_targets,
)


def row(local_correct=6, retain_correct=5, local_unrelated=4, retain_unrelated=5,
        local_nll=3., retain_nll=4.):
    return {"actions": {
        "retain_all": {"correct": retain_correct, "total_answer_nll": retain_nll,
                       "unrelated_correct": retain_unrelated},
        "scope_top1_a0": {"correct": local_correct, "total_answer_nll": local_nll,
                          "unrelated_correct": local_unrelated},
    }}


class FactorGateV2Test(unittest.TestCase):
    def test_feature_sets_exclude_raw_reader_and_schedule_ablation(self):
        self.assertEqual(len(ENGINEERED), 13)
        self.assertEqual(len(NO_SCHEDULE), 10)
        self.assertFalse(any(name.startswith("reader_") for name in ENGINEERED))
        self.assertNotIn("event_position", NO_SCHEDULE)

    @unittest.skipIf(torch is None, "torch is available in the experiment environment")
    def test_targets_factor_benefit_from_unrelated_risk(self):
        benefit, risk = factor_targets([row(), row(local_correct=5, retain_correct=5,
                                                    local_unrelated=5, retain_unrelated=5,
                                                    local_nll=3., retain_nll=4.)])
        self.assertEqual(benefit.tolist(), [1., 1.])
        self.assertEqual(risk.tolist(), [1., 0.])

    @unittest.skipIf(torch is None, "torch is available in the experiment environment")
    def test_policy_requires_benefit_and_safety(self):
        benefit = torch.tensor([.9, .9, .2])
        risk = torch.tensor([.01, .5, .01])
        guarded = factor_actions(benefit, risk, .7, .1, True)
        self.assertNotEqual(guarded[0], 0)
        self.assertEqual(guarded[1:], [0, 0])
        unguarded = factor_actions(benefit, risk, .7, .1, False)
        self.assertNotEqual(unguarded[1], 0)


if __name__ == "__main__":
    unittest.main()
