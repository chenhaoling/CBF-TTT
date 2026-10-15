import unittest

from tasks.pilot_qwen_utility_record_controller_v10_fewrel import (
    pilot_gate, single_candidate_benefit,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import action_grid


def row(positive=True, risky=False):
    actions = {}
    for spec in action_grid(1):
        actions[spec["name"]] = {
            "spec": spec, "correct": 4, "unrelated_correct": 3,
            "queries": 5, "total_answer_nll": 2.0,
            "mean_memory_mass": 1.0,
        }
    actions["grid_0"]["correct"] = 5 if positive else 4
    actions["grid_0"]["unrelated_correct"] = 2 if risky else 3
    return {"candidate_count": 1, "retain_action": "grid_1", "actions": actions}


class V10FewRelPilotTest(unittest.TestCase):
    def test_single_candidate_positive_requires_safety_and_correct_gain(self):
        result = single_candidate_benefit([
            row(positive=True), row(positive=True, risky=True), row(positive=False),
        ])
        self.assertEqual(result["candidates"], 3)
        self.assertEqual(result["positive_safe_benefit"], 1)
        self.assertEqual(result["risky"], 1)

    def test_gate_requires_every_registered_condition(self):
        values = dict(
            finite_complete=True, peak_reserved_gib=27.9,
            answer_coverage=.95, positive_fraction=.05,
            oracle_headroom=.015, heldouts_sealed=True,
        )
        self.assertTrue(pilot_gate(**values)["passed"])
        values["positive_fraction"] = .049
        self.assertFalse(pilot_gate(**values)["passed"])


if __name__ == "__main__":
    unittest.main()
