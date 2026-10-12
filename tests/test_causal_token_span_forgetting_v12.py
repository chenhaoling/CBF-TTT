"""Pure protocol tests for V12 causal gate selection."""

import unittest

from tasks.train_causal_token_span_forgetting_v12 import (
    development_gate, select_threshold, update_examples,
)


def metric(overall, retain, update):
    return {"action_accuracy": overall, "by_event_kind": {
        "primary_initial": {"accuracy": 1.0},
        "hard_retain": {"accuracy": retain},
        "primary_update": {"accuracy": update}}}


class CausalTokenSpanForgettingV12Tests(unittest.TestCase):
    def test_threshold_prefers_feasible_point(self):
        from tasks.train_causal_token_span_forgetting_v12 import THRESHOLDS
        grid = {value: metric(.8, .8, .8) for value in THRESHOLDS}
        grid[.45] = metric(.96, .96, .91)
        grid[.50] = metric(.97, .94, .98)
        value, status = select_threshold(grid)
        self.assertEqual((value, status), (.45, "constraints_satisfied"))

    def test_update_examples_rejects_ambiguous_text(self):
        sessions = [{"chains": [
            {"subject": "A", "versions": [{"prompt": "same"}]},
            {"subject": "B", "versions": [{"prompt": "same"}]},
        ]}]
        with self.assertRaises(ValueError):
            update_examples(sessions)

    def test_gate(self):
        result = {"mean_action_accuracy": .97, "mean_accuracy": .52,
                  "by_event_kind": {"hard_retain": {"accuracy": .96},
                                     "primary_update": {"accuracy": .94}}}
        candidate = {"top1_accuracy": .97}
        comparison = {"query_bootstrap_95ci": [.002, .04]}
        self.assertTrue(development_gate(result, candidate, comparison, .525)["passed"])
        comparison["query_bootstrap_95ci"] = [-.001, .04]
        self.assertFalse(development_gate(result, candidate, comparison, .525)["passed"])


if __name__ == "__main__":
    unittest.main()
