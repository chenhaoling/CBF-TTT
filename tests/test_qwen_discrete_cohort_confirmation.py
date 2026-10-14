import unittest

from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_qwen_cohort_forgetting import REFERENCE_INDICES
from tasks.train_qwen_discrete_cohort_confirmation import (
    DATA_SEED,
    GROUP_COUNTS,
    POLICIES,
    blind_examples,
    development_gate,
    snap_retention,
    test_gate,
)


class QwenDiscreteCohortConfirmationTest(unittest.TestCase):
    def test_registered_release_and_snapping(self):
        self.assertEqual(DATA_SEED, 202610153)
        self.assertEqual(GROUP_COUNTS, {"train": 256, "dev": 64, "test": 128})
        self.assertIn("learned_snapped", POLICIES)
        self.assertEqual(snap_retention((0.1, 0.74, 0.9)), (0.0, 0.5, 1.0))

    def test_blind_rows_zero_only_reference_features(self):
        features = list(range(20))
        rows = [{"features": features, "retention": [1.0, 0.0, 1.0]}]
        blind = blind_examples(rows)
        self.assertEqual(rows[0]["features"], features)
        for index, value in enumerate(blind[0]["features"]):
            self.assertEqual(value, 0.0 if index in REFERENCE_INDICES else features[index])

    @staticmethod
    def policy(accuracy, nll, historical=0.97, minimum_family=0.96,
               action=0.9):
        return {
            "mean_accuracy": accuracy,
            "mean_nll": nll,
            "mean_grid_action_accuracy": action,
            "roles": {"current": 1.0, "historical": historical, "anchor": 1.0},
            "families": {"a": minimum_family, "b": 0.99},
            "per_seed": {
                str(seed): {
                    "accuracy": accuracy,
                    "groups": {
                        "g0": {"accuracy": accuracy, "nll": nll},
                        "g1": {"accuracy": accuracy, "nll": nll},
                    },
                }
                for seed in SEEDS
            },
        }

    def passing_summary(self):
        return {
            "grid_classifier": self.policy(0.99, 0.15),
            "physics_cohort": self.policy(0.93, 0.35, historical=0.79,
                                            minimum_family=0.72, action=None),
            "oracle_grid": self.policy(0.995, 0.14, action=1.0),
            "reference_blind": self.policy(0.65, 1.2, historical=0.25,
                                             minimum_family=0.4, action=None),
        }

    def test_registered_gates(self):
        training = {str(seed): {"interior_fraction": 0.2} for seed in SEEDS}
        self.assertTrue(development_gate(self.passing_summary(), training)["passed"])
        self.assertTrue(test_gate(self.passing_summary())["passed"])
        failed = self.passing_summary()
        failed["grid_classifier"] = self.policy(0.97, 0.2)
        self.assertFalse(development_gate(failed, training)["passed"])


if __name__ == "__main__":
    unittest.main()
