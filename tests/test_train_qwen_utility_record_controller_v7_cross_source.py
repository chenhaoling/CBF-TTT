import unittest

from tests.test_train_qwen_utility_record_controller_v5_residual import fixture
from tasks.train_qwen_utility_record_controller_v7_cross_source import (
    compatible_metric,
    development_gate,
    select_config,
)


def domain(gain, closure, errors=0, precision=1.0, fraction=.1):
    return {
        "gain_over_lifecycle_physics": gain,
        "oracle_gap_closure": closure,
        "added_unrelated_errors": errors,
        "residual_deletion_precision": precision,
        "residual_deletion_fraction": fraction,
        "per_mechanism_gain": {
            "semantic_overwrite": gain,
            "stale_repetition": gain,
        },
    }


class TestV7CrossSourceController(unittest.TestCase):
    def test_legacy_metric_does_not_require_query_roles(self):
        row, feature = fixture()
        for action in row["actions"].values():
            action.pop("per_query_role")
        value = compatible_metric(
            [row], {("e", 0): row["retain_action"]}, {("e", 0): feature}
        )
        self.assertEqual(value["accuracy"], .6)
        self.assertNotIn("per_query_role", value)

    def test_selection_prioritizes_worst_domain_closure_and_safety(self):
        config = {
            "epoch": 25, "utility_threshold": .4, "risk_penalty": 1.0,
            "conformal_level": .01,
        }
        candidates = {
            "balanced": {
                "config": config,
                "domains": {
                    "templama": domain(.02, .60),
                    "lama": domain(.018, .55),
                },
            },
            "high_mean_low_worst": {
                "config": config,
                "domains": {
                    "templama": domain(.04, .90),
                    "lama": domain(.005, .20),
                },
            },
            "unsafe": {
                "config": config,
                "domains": {
                    "templama": domain(.05, 1.0),
                    "lama": domain(.05, 1.0, errors=1),
                },
            },
        }
        self.assertEqual(select_config(candidates), "balanced")

    def test_dual_domain_gate_requires_each_domain(self):
        seed_results = [{
            "selected_metric": {
                "domains": {
                    "templama": domain(.02, .60),
                    "lama": domain(.016, .55),
                }
            }
        } for _ in range(3)]
        development = {
            "templama": {
                "oracle_headroom": .03,
                "physics": {"per_mechanism_accuracy": {
                    "semantic_overwrite": .5, "stale_repetition": .5,
                }},
            },
            "lama": {
                "oracle_headroom": .02,
                "physics": {"per_mechanism_accuracy": {
                    "semantic_overwrite": .5, "stale_repetition": .5,
                }},
            },
        }
        gate = development_gate(seed_results, development, True)
        self.assertTrue(gate["passed_pending_independent_audit"])
        seed_results[0]["selected_metric"]["domains"]["lama"][
            "added_unrelated_errors"
        ] = 1
        gate = development_gate(seed_results, development, True)
        self.assertFalse(gate["passed_pending_independent_audit"])


if __name__ == "__main__":
    unittest.main()
