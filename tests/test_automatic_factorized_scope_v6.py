"""Pure tests for automatic factorized scope V6."""

import unittest

from tasks.evaluate_automatic_factorized_scope_v6 import (
    automatic_pair_view, development_gate,
)


class AutomaticFactorizedScopeV6Tests(unittest.TestCase):
    def test_rare_shared_entity_beats_common_relation_phrase(self):
        idf = {"where": 1., "was": 1., "born": 1., "ada": 6., "lovelace": 6.,
               "london": 4., "worked": 2.}
        view = automatic_pair_view("Where was Ada Lovelace born",
                                   "Ada Lovelace worked in London", idf, window=3)
        self.assertEqual(view["matched_tokens"], ["ada", "lovelace"])
        self.assertIn("<ENTITY>", view["query_relation"])
        self.assertIn("<ENTITY>", view["key_relation"])

    def test_gate_requires_text_only_floor(self):
        metric = {"top1_accuracy": .96, "mrr": .98}
        comparison = {"top1_gain": .12, "bootstrap_95ci": [.09, .15]}
        self.assertTrue(development_gate(metric, comparison, .978)["passed"])
        metric["top1_accuracy"] = .94
        self.assertFalse(development_gate(metric, comparison, .978)["passed"])


if __name__ == "__main__":
    unittest.main()

