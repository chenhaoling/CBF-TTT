from tasks.pilot_qwen_utility_semantic_record_gate_v1 import (
    action_grid,
    choose_safe_oracle,
    retain_action_name,
)
import unittest


def test_grid_has_all_ternary_actions_plus_global_clear():
    assert len(action_grid(0)) == 2
    assert len(action_grid(1)) == 4
    assert len(action_grid(3)) == 28
    assert retain_action_name(3) == "grid_1_1_1"


def test_safe_oracle_enforces_unrelated_barrier_before_utility():
    actions = action_grid(1)
    outcomes = {
        "grid_0": {"correct": 10, "total_answer_nll": 1.0,
                   "mean_memory_mass": 0.0, "unrelated_correct": 3},
        "grid_05": {"correct": 9, "total_answer_nll": .8,
                    "mean_memory_mass": .5, "unrelated_correct": 4},
        "grid_1": {"correct": 8, "total_answer_nll": 1.2,
                   "mean_memory_mass": 1.0, "unrelated_correct": 4},
        "global_clear": {"correct": 11, "total_answer_nll": .5,
                         "mean_memory_mass": 0.0, "unrelated_correct": 2},
    }
    assert choose_safe_oracle(outcomes, actions, "grid_1") == "grid_05"


class TestUtilitySemanticRecordResourcePilot(unittest.TestCase):
    def test_grid(self):
        test_grid_has_all_ternary_actions_plus_global_clear()

    def test_barrier(self):
        test_safe_oracle_enforces_unrelated_barrier_before_utility()
