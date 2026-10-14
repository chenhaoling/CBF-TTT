import unittest

try:
    import torch
except ImportError:
    torch = None

from tasks.train_v19_nested_forgetting_policy import (
    EPOCHS, RISK_PENALTIES, THRESHOLDS, fit_snapshots, policy_states,
    predict, select_hyperparameters, selection_metrics,
)


def rows():
    values = []
    for index, (target, risk) in enumerate(((1.0, 0), (1.0, 1), (.5, 0), (0.0, 0))):
        values.append({"example_id": f"x{index}", "episode_id": "e", "subject": "s",
                       "template_group": "t", "fold": 0,
                       "current_session_id": "current",
                       "component_session_id": f"h{index}", "target": target,
                       "locality_risk_target": risk,
                       "conservation_target": int(target <= .5)})
    return values


class TrainV19NestedForgettingPolicyTest(unittest.TestCase):
    def test_risk_penalty_suppresses_risky_positive(self):
        utility, risk = [.9, .9, .1, .1], [.0, .9, .0, .0]
        no_penalty = selection_metrics(rows(), utility, risk, .5, 0)
        penalized = selection_metrics(rows(), utility, risk, .5, 1)
        self.assertEqual(no_penalty["risk_violation_rate"], 1)
        self.assertEqual(penalized["risk_violation_rate"], 0)
        self.assertLess(penalized["positive_recall"], no_penalty["positive_recall"])

    def test_selection_requires_all_registered_constraints(self):
        grid = {}
        for epoch in EPOCHS:
            grid[str(epoch)] = {}
            for threshold in THRESHOLDS:
                grid[str(epoch)][str(threshold)] = {}
                for penalty in RISK_PENALTIES:
                    good = (epoch, threshold, penalty) == (50, .6, 1.0)
                    grid[str(epoch)][str(threshold)][str(penalty)] = {
                        "positive_recall": .9 if good else .7,
                        "conservation_specificity": .98 if good else .9,
                        "risk_violation_rate": .01 if good else .2,
                        "balanced_utility_accuracy": .94 if good else .8,
                        "action_accuracy": .9, "mse": .1,
                        "delete_rate": .2, "rows": 4, "positive_rows": 2,
                        "conservation_rows": 2, "locality_risk_rows": 1}
        selected = select_hyperparameters(grid)
        self.assertEqual((selected["epoch"], selected["threshold"],
                          selected["risk_penalty"]), (50, .6, 1.0))
        self.assertTrue(selected["constraints_satisfied"])

    def test_policy_state_never_deletes_current_update(self):
        prediction = {"utility": [.9, .9, .1, .1], "risk": [0, 1, 0, 0]}
        state = policy_states(rows(), prediction, .5, 1)[0]
        self.assertIn("current", state["retained_session_ids"])
        self.assertNotIn("current", state["deleted_session_ids"])
        self.assertEqual(state["deleted_session_ids"], ["h0"])

    @unittest.skipUnless(torch is not None, "PyTorch unavailable")
    def test_two_head_training_is_finite(self):
        training = []
        for index in range(20):
            target = 1.0 if index % 4 == 0 else .5
            training.append({"features": [float((index + j) % 5) for j in range(17)],
                             "target": target, "locality_risk_target": int(index % 7 == 0),
                             "conservation_target": int(target <= .5),
                             "episode_id": f"e{index // 4}"})
        snapshots, curve = fit_snapshots(training, 5, "cpu")
        self.assertEqual(set(snapshots), set(EPOCHS))
        result = predict(snapshots[25], training, "cpu")
        self.assertEqual(set(result), {"utility", "risk"})
        self.assertTrue(all(0 <= value <= 1 for values in result.values()
                            for value in values))
        self.assertEqual(len(curve), max(EPOCHS))


if __name__ == "__main__":
    unittest.main()
