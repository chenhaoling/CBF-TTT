import itertools
import unittest

from scripts.diagnose_qwen_utility_record_controller_v4_residual import analyze


def make_action(values, correct, unrelated, nll):
    name = "grid_" + "_".join(map(str, values))
    return name, {
        "spec": {
            "name": name, "kind": "grid",
            "alphas": [float(value) for value in values],
        },
        "queries": 5, "correct": correct, "unrelated_correct": unrelated,
        "total_answer_nll": nll,
    }


class TestDiagnoseUtilityRecordControllerV4Residual(unittest.TestCase):
    def test_counts_single_and_pair_only_safe_gains(self):
        actions = dict(
            make_action(values, 3, 3, 5.0)
            for values in itertools.product((0, 1), repeat=2)
        )
        actions["grid_1_1"].update(correct=3)
        actions["grid_0_1"].update(correct=3)
        actions["grid_1_0"].update(correct=3, unrelated_correct=2)
        actions["grid_0_0"].update(correct=4)
        label = {
            "episode_id": "e", "boundary_index": 0, "source_group": "g",
            "retain_action": "grid_1_1", "candidate_count": 2,
            "mechanism": "semantic_overwrite", "target_relation": "P1",
            "actions": actions,
        }
        values = [[0.0] * 66 for _ in range(2)]
        for row in values:
            row[0] = 1.0; row[2] = 1.0; row[4] = .995
        feature = {
            "episode_id": "e", "boundary_index": 0,
            "features": values,
        }
        result = analyze([label], [feature])
        self.assertEqual(result["eligible_records"], 2)
        self.assertEqual(result["single_residual_actions"], 2)
        self.assertEqual(result["single_collateral_actions"], 1)
        self.assertEqual(result["pair_or_higher_only_gain"], 1)
        self.assertEqual(result["additional_correct_from_best_safe"], 1)
        self.assertEqual(result["safe_residual_headroom"], .2)
        self.assertEqual(result["safe_single_residual_headroom"], 0.0)

    def test_rule_argument_changes_candidate_support(self):
        actions = dict(
            make_action(values, 3 + int(values == (0,)), 3, 5.0)
            for values in ((0,), (1,))
        )
        label = {
            "episode_id": "e", "boundary_index": 0, "source_group": "g",
            "retain_action": "grid_1", "candidate_count": 1,
            "mechanism": "semantic_overwrite", "target_relation": "P1",
            "actions": actions,
        }
        values = [0.0] * 66
        values[0] = 1.0; values[2] = 1.0; values[4] = .9
        feature = {"episode_id": "e", "boundary_index": 0,
                   "features": [values]}
        self.assertEqual(
            analyze([label], [feature], "conflict_cosine_099")[
                "eligible_records"
            ],
            0,
        )
        self.assertEqual(
            analyze([label], [feature], "exact_subject")["eligible_records"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
