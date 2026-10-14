import unittest

from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_qwen_sparse_cohort_forgetting import (
    DATA_SEED,
    GROUP_COUNTS,
    READER,
    development_gate,
    test_gate,
)
from scripts.audit_qwen_sparse_cohort_forgetting import (
    development_gate as audit_development_gate,
    test_gate as audit_test_gate,
)


class QwenSparseCohortForgettingTest(unittest.TestCase):
    def test_registered_release(self):
        self.assertEqual(DATA_SEED, 202610152)
        self.assertEqual(GROUP_COUNTS, {"train": 256, "dev": 64, "test": 128})
        self.assertEqual(READER, "component_top1")

    @staticmethod
    def policy(accuracy, nll, historical=0.96, minimum_family=0.94):
        return {
            "mean_accuracy": accuracy,
            "mean_nll": nll,
            "roles": {
                "current": 0.99,
                "historical": historical,
                "anchor": 0.99,
            },
            "families": {"a": minimum_family, "b": 0.98},
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
            "learned": self.policy(0.98, 0.2),
            "physics_cohort": self.policy(0.93, 0.4, historical=0.8, minimum_family=0.72),
            "oracle_grid": self.policy(0.99, 0.18),
        }

    def test_development_gate_requires_gain_over_physics(self):
        training = {str(seed): {"interior_fraction": 0.2} for seed in SEEDS}
        gate = development_gate(self.passing_summary(), training)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate, audit_development_gate(self.passing_summary(), training))
        summary = self.passing_summary()
        summary["learned"] = self.policy(0.945, 0.35)
        self.assertFalse(development_gate(summary, training)["passed"])

    def test_one_time_test_gate(self):
        gate = test_gate(self.passing_summary())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate, audit_test_gate(self.passing_summary()))


if __name__ == "__main__":
    unittest.main()
