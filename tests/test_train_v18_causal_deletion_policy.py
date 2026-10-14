import unittest

try:
    import torch
except ImportError:
    torch = None

from tasks.train_v18_causal_deletion_policy import (
    EPOCHS, THRESHOLDS, fit_snapshots, policy_states, predict,
    select_hyperparameters, selection_metrics,
)


def rows():
    output = []
    for index, target in enumerate((1.0, 1.0, .5, .5)):
        output.append({"episode_id": "e", "subject": "s", "template_group": "t",
                       "fold": 0, "current_session_id": "current",
                       "component_session_id": f"h{index}", "target": target,
                       "conservation_target": int(target == .5)})
    return output


class TrainV18CausalDeletionPolicyTest(unittest.TestCase):
    def test_selection_metrics_distinguish_helpful_and_conserved(self):
        value = selection_metrics(rows(), [.9, .8, .1, .2], .5)
        self.assertEqual(value["positive_recall"], 1.0)
        self.assertEqual(value["conservation_specificity"], 1.0)
        self.assertEqual(value["action_accuracy"], 1.0)

    def test_hyperparameter_selection_uses_registered_constraints(self):
        grid = {}
        for epoch in EPOCHS:
            grid[str(epoch)] = {}
            for threshold in THRESHOLDS:
                good = epoch == 50 and threshold == .5
                grid[str(epoch)][str(threshold)] = {
                    "positive_recall": 1.0 if good else .7,
                    "conservation_specificity": 1.0 if good else .9,
                    "balanced_accuracy": 1.0 if good else .8,
                    "action_accuracy": 1.0 if good else .8,
                    "mse": .01 if good else .1, "delete_rate": .5, "rows": 4,
                    "positive_rows": 2, "conservation_rows": 2}
        selected = select_hyperparameters(grid)
        self.assertEqual((selected["epoch"], selected["threshold"]), (50, .5))
        self.assertTrue(selected["constraints_satisfied"])

    def test_policy_states_never_delete_current_update(self):
        states = policy_states(rows(), [.9, .8, .1, .2], .5)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["deleted_session_ids"], ["h0", "h1"])
        self.assertEqual(states[0]["retained_session_ids"], ["h2", "h3", "current"])

    @unittest.skipUnless(torch is not None, "PyTorch unavailable")
    def test_mlp_training_snapshots_are_finite(self):
        values = []
        for index in range(12):
            target = 1.0 if index < 4 else .5
            values.append({"features": [float((index + offset) % 3) for offset in range(13)],
                           "target": target, "conservation_target": int(target == .5),
                           "useful_repetition": int(index in (8, 9)),
                           "episode_id": f"e{index // 3}",
                           "component_key_id": "repeat" if index in (8, 9) else f"k{index}"})
        snapshots, curve = fit_snapshots(values, 7, "cpu")
        self.assertEqual(set(snapshots), set(EPOCHS))
        scores = predict(snapshots[25], values, "cpu")
        self.assertTrue(all(0 <= value <= 1 for value in scores))
        self.assertEqual(len(curve), max(EPOCHS))


if __name__ == "__main__":
    unittest.main()
