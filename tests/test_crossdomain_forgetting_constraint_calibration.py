"""Selection-rule tests for constraint-calibrated forgetting."""

import unittest

from tasks.evaluate_crossdomain_forgetting_reranker_v2_constraints import (
    select_constrained_threshold,
)


def metric(action, hard, update):
    return {"action_accuracy": action, "by_event_kind": {
        "hard_retain": {"accuracy": hard}, "primary_update": {"accuracy": update}}}


class ConstraintCalibrationTests(unittest.TestCase):
    def test_prefers_best_overall_feasible_threshold(self):
        grid = {"0.3": metric(.90, .84, .90), "0.4": metric(.86, .86, .76),
                "0.5": metric(.88, .90, .75)}
        self.assertEqual(select_constrained_threshold(grid), (.5, "feasible"))

    def test_fallback_minimizes_total_deficit(self):
        grid = {"0.3": metric(.90, .84, .76), "0.4": metric(.85, .86, .745)}
        self.assertEqual(select_constrained_threshold(grid),
                         (.4, "minimum_constraint_deficit"))


if __name__ == "__main__":
    unittest.main()
