import unittest

from tasks.train_factorized_scope_candidate_v3 import release_gate, scope_key


class FactorizedScopeV3Test(unittest.TestCase):
    def test_scope_key_excludes_changed_payload(self):
        self.assertEqual(scope_key("Memory statement: Who wrote X?\nValue: old"), "Who wrote X?")
        self.assertEqual(scope_key("Memory statement: Who wrote X?"), "Who wrote X?")

    def test_gate_requires_factorization_control(self):
        mean = {"top1": .96, "mrr": .98,
                "by_length": {str(n): {"top1": .95} for n in (16, 32, 48, 64)}}
        per_seed = {str(n): {"top1": .94} for n in range(3)}
        comparison = {"bootstrap_95ci": [.16, .2]}
        self.assertTrue(release_gate(mean, per_seed, comparison, {"top1": .71}, {"top1": .4})["passed"])
        self.assertFalse(release_gate(mean, per_seed, comparison, {"top1": .69}, {"top1": .4})["passed"])


if __name__ == "__main__": unittest.main()
