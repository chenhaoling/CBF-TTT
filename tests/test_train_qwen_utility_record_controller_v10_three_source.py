import unittest

from tasks.train_qwen_utility_record_controller_v10_three_source import (
    equal_domain_normalization, extended_gate,
)
from tests.test_train_qwen_utility_record_controller_v7_cross_source import domain


class V10ThreeSourceControllerTest(unittest.TestCase):
    def test_normalization_weights_domains_equally(self):
        domains = {
            "small": [{"features": [0.0] * 66}],
            "large": [{"features": [10.0] * 66} for _ in range(9)],
        }
        means, stds = equal_domain_normalization(domains)
        self.assertEqual(means, [5.0] * 66)
        self.assertEqual(stds, [5.0] * 66)

    def test_extended_gate_requires_fewrel_ablation_and_v8_margin(self):
        def seeds(tt, lama):
            return [{"selected_metric": {"domains": {
                "templama": domain(tt, .8), "lama": domain(lama, .8),
            }}} for _ in range(3)]
        development = {
            name: {
                "oracle_headroom": .05,
                "physics": {"per_mechanism_accuracy": {
                    "semantic_overwrite": .5, "stale_repetition": .5,
                }},
            } for name in ("templama", "lama")
        }
        v8 = {"development_gate": {"mean_gain": {
            "templama": .042, "lama": .012,
        }}}
        gate = extended_gate(
            seeds(.04, .017), seeds(.03, .014), development, v8, True,
        )
        self.assertTrue(gate["passed_pending_independent_audit"])
        gate = extended_gate(
            seeds(.04, .017), seeds(.03, .016), development, v8, True,
        )
        self.assertFalse(gate["passed_pending_independent_audit"])


if __name__ == "__main__":
    unittest.main()
