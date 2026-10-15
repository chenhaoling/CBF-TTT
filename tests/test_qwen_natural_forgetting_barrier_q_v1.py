import unittest

try:
    import torch
except ImportError:
    torch = None

from tasks.train_qwen_natural_forgetting_barrier_q_v1 import (
    ACTIONS,
    fold_for,
    immediate_mass,
    inner_fold_for,
    lexical_features,
    policy_metrics,
    select_q_actions,
)


def label():
    actions = {}
    for index, action in enumerate(ACTIONS):
        actions[action] = {
            "correct": 8 + int(index == 1), "queries": 10,
            "total_answer_nll": 10. - index / 10,
            "unrelated_correct": 5 - int(index == 2),
            "mean_memory_mass": 4.,
            "intervention": {"indices": ([0, 1] if action != "retain_all" else []),
                             "alpha": (.5 if action.endswith("a05") else 0.)},
        }
    actions["retain_all"]["unrelated_correct"] = 5
    return {"memory_size_before": 4, "actions": actions}


class BarrierQTest(unittest.TestCase):
    def test_group_folds_are_deterministic_and_bounded(self):
        values = [fold_for(f"session-{index}") for index in range(50)]
        self.assertTrue(set(values) <= set(range(5)))
        self.assertEqual(values, [fold_for(f"session-{index}") for index in range(50)])
        inner = [inner_fold_for(f"session-{index}", 2) for index in range(50)]
        self.assertTrue(set(inner) <= set(range(4)))

    def test_lexical_and_immediate_mass(self):
        jaccard, containment, ratio = lexical_features("Who is Ada Lovelace", "Ada Lovelace")
        self.assertAlmostEqual(jaccard, .5)
        self.assertEqual(containment, 1.)
        self.assertGreater(ratio, 0.)
        row = label()
        self.assertEqual(immediate_mass(row, "retain_all"), 5.)
        self.assertEqual(immediate_mass(row, "global_a0"), 3.)
        self.assertEqual(immediate_mass(row, "global_a05"), 4.)

    def test_policy_metrics_uses_stored_counterfactual_outcome(self):
        row = label(); result = policy_metrics([row], [1])
        self.assertEqual(result["candidate_accuracy"], .9)
        self.assertEqual(result["queries"], 10)
        self.assertEqual(result["unrelated_errors_added"], 0)

    @unittest.skipIf(torch is None, "torch is available in the experiment environment")
    def test_barrier_filters_risky_high_gain_action(self):
        gain = torch.zeros((1, len(ACTIONS))); nll = torch.zeros_like(gain)
        risk = torch.zeros_like(gain); mass = torch.ones_like(gain)
        gain[0, 1] = .2; risk[0, 1] = .9
        gain[0, 2] = .1; risk[0, 2] = .01
        chosen = select_q_actions(gain, nll, risk, mass, .05, 0., 0., True)
        self.assertEqual(chosen, [2])
        unguarded = select_q_actions(gain, nll, risk, mass, .05, 0., 0., False)
        self.assertEqual(unguarded, [1])


if __name__ == "__main__":
    unittest.main()
