"""Tests for frozen development release gate."""

import unittest

from tasks.evaluate_zsre_counterfactual_forgetting_controller import development_gate


class DevelopmentGateTests(unittest.TestCase):
    def metrics(self):
        return {"mean_action_accuracy": .97, "by_event_kind": {
            "primary_initial": {"accuracy": .99}, "hard_retain": {"accuracy": .96},
            "primary_update": {"accuracy": .93}}}

    def test_gate_requires_both_confident_gains_and_action_floors(self):
        comparison = {"query_bootstrap_95ci": [.001, .02],
                      "action_bootstrap_95ci": [.001, .03]}
        self.assertTrue(development_gate(self.metrics(), comparison)["passed"])
        comparison["action_bootstrap_95ci"][0] = 0.0
        self.assertFalse(development_gate(self.metrics(), comparison)["passed"])


if __name__ == "__main__": unittest.main()
