import unittest

from scripts.audit_qwen_utility_semantic_record_gate_v1_pilot import (
    expected_actions,
    independent_label,
)


class TestUtilitySemanticRecordPilotAudit(unittest.TestCase):
    def test_action_reconstruction(self):
        assert len(expected_actions(3)) == 28
        assert expected_actions(1)["grid_05"]["alphas"] == [.5]

    def test_independent_barrier_label(self):
        actions = {
            "grid_0": {"spec": {"alphas": [0.0]}, "correct": 10,
                       "total_answer_nll": 1.0, "mean_memory_mass": 0,
                       "unrelated_correct": 2},
            "grid_05": {"spec": {"alphas": [.5]}, "correct": 9,
                        "total_answer_nll": .8, "mean_memory_mass": .5,
                        "unrelated_correct": 4},
            "grid_1": {"spec": {"alphas": [1.0]}, "correct": 8,
                       "total_answer_nll": 1.2, "mean_memory_mass": 1,
                       "unrelated_correct": 4},
        }
        row = {"actions": actions, "retain_action": "grid_1"}
        assert independent_label(row) == "grid_05"
