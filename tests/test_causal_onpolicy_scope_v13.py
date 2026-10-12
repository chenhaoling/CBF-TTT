"""Pure tests for V13 state collection and final gate."""

import unittest

from tasks.train_causal_onpolicy_scope_v13 import development_gate


class CausalOnpolicyScopeV13Tests(unittest.TestCase):
    def test_gate(self):
        result = {"mean_action_accuracy": .96, "mean_accuracy": .518,
                  "by_event_kind": {"hard_retain": {"accuracy": .96},
                                     "primary_update": {"accuracy": .93}}}
        candidate = {"top1_accuracy": .97}
        comparison = {"query_bootstrap_95ci": [-.001, .01]}
        self.assertTrue(development_gate(result, candidate, comparison, .519)["passed"])
        comparison["query_bootstrap_95ci"] = [-.002, .01]
        self.assertFalse(development_gate(result, candidate, comparison, .519)["passed"])


if __name__ == "__main__":
    unittest.main()
