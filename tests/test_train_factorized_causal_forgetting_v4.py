import unittest

from tasks.train_factorized_causal_forgetting_v4 import payload, release_gate


class FactorizedCausalV4Test(unittest.TestCase):
    def test_payload_parser_keeps_value_outside_scope(self):
        self.assertEqual(payload("Memory statement: Q\nValue: A"), "A")
        self.assertIsNone(payload("Memory statement: Q"))

    def test_release_gate_requires_shortlist_and_safety(self):
        metric = {
            "current_retention": .98, "non_target_retention": .96,
            "stale_deletion": .91, "pair_state_accuracy": .91,
            "by_event_kind": {"hard_retain": {"accuracy": .96}},
            "by_length": {str(n): {"pair_state_accuracy": .9}
                          for n in (16, 32, 48, 64)},
        }
        comparison = {"bootstrap_95ci": [.3, .4]}
        random_comparison = {"bootstrap_95ci": [.11, .2]}
        self.assertTrue(release_gate(metric, .995, .96, comparison, random_comparison)["passed"])
        self.assertFalse(release_gate(metric, .98, .96, comparison, random_comparison)["passed"])


if __name__ == "__main__": unittest.main()
