"""Pure tests for relation-supervised V5."""

import unittest

from tasks.train_relation_supervised_scope_v5 import development_gate, focus_relation


class RelationSupervisedScopeV5Tests(unittest.TestCase):
    def test_focus_relation_removes_subject_and_long_prefix(self):
        text = "one two three four five six seven eight nine ten eleven Ada was born in"
        focused, found = focus_relation(text, "Ada", window=3)
        self.assertTrue(found)
        self.assertEqual(focused, "nine ten eleven <ENTITY> was born in")

    def test_gate(self):
        metric = {"top1_accuracy": .91, "mrr": .96,
                  "query_relation_accuracy": .92}
        gain = {"top1_gain": .06, "bootstrap_95ci": [.02, .09]}
        self.assertTrue(development_gate(metric, gain)["passed"])
        metric["query_relation_accuracy"] = .89
        self.assertFalse(development_gate(metric, gain)["passed"])


if __name__ == "__main__":
    unittest.main()

