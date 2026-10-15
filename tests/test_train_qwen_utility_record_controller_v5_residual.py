import itertools
import unittest

from tasks.train_qwen_utility_record_controller_v5_residual import (
    conformal_risk_threshold,
    residual_actions,
    residual_examples,
    select_config,
)


def action(values, correct=3, unrelated=3, nll=5.0):
    name = "grid_" + "_".join(map(str, values))
    return name, {
        "spec": {
            "name": name, "kind": "grid",
            "alphas": [float(value) for value in values],
        },
        "queries": 5,
        "correct": correct,
        "total_answer_nll": nll,
        "unrelated_queries": 3,
        "unrelated_correct": unrelated,
        "mean_memory_mass": sum(values),
        "selected_null": 0,
        "per_query_role": {},
    }


def fixture():
    actions = dict(action(values) for values in itertools.product((0, 1), repeat=2))
    actions["grid_0_1"]["correct"] = 4
    actions["grid_1_0"]["unrelated_correct"] = 2
    label = {
        "episode_id": "e", "boundary_index": 0, "source_group": "g",
        "target_relation": "P1", "mechanism": "semantic_overwrite",
        "candidate_count": 2, "retain_action": "grid_1_1",
        "safe_oracle_action": "grid_0_1", "actions": actions,
    }
    values = [[0.0] * 66 for _ in range(2)]
    for row in values:
        row[0] = 1.0
    feature = {
        "episode_id": "e", "boundary_index": 0, "source_group": "g",
        "features": values,
        "targets": {"delete_targets": [1, 0]},
    }
    return label, feature


class TestTrainUtilityRecordControllerV5Residual(unittest.TestCase):
    def test_residual_examples_use_single_counterfactual_targets(self):
        label, feature = fixture()
        rows = residual_examples([label], [feature])
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(row["benefit_target"], row["risk_target"]) for row in rows],
            [(1, 0), (0, 1)],
        )
        self.assertTrue(all(row["physics_action"] == "grid_1_1" for row in rows))

    def test_policy_selects_at_most_one_residual_record(self):
        label, feature = fixture()
        predictions = {
            ("e", 0, 0): {"benefit": .8, "risk": .1},
            ("e", 0, 1): {"benefit": .9, "risk": .1},
        }
        selected = residual_actions([label], [feature], predictions, {
            "utility_threshold": .4,
            "risk_penalty": 0.0,
            "risk_threshold": .5,
        })
        self.assertEqual(selected[("e", 0)], "grid_1_0")
        self.assertEqual(
            label["actions"][selected[("e", 0)]]["spec"]["alphas"].count(0.0),
            1,
        )

    def test_selection_requires_zero_collateral(self):
        base = {
            "accuracy": .8,
            "residual_deletion_precision": .7,
            "residual_deletion_fraction": .2,
            "added_unrelated_errors": 0,
            "config": {
                "epoch": 25, "utility_threshold": .5,
                "risk_penalty": 1.0, "conformal_level": .03,
            },
        }
        candidates = {
            "unsafe": {**base, "accuracy": .9, "added_unrelated_errors": 1},
            "selected": base,
        }
        self.assertEqual(select_config(candidates), "selected")

    def test_conformal_threshold_uses_low_risk_positive_quantile(self):
        rows = [
            {"episode_id": "e", "boundary_index": 0, "slot": index,
             "risk_target": target}
            for index, target in enumerate((1, 1, 0))
        ]
        predictions = {
            ("e", 0, 0): {"risk": .2},
            ("e", 0, 1): {"risk": .7},
            ("e", 0, 2): {"risk": .1},
        }
        self.assertEqual(conformal_risk_threshold(predictions, rows, .05), .2)


if __name__ == "__main__":
    unittest.main()
