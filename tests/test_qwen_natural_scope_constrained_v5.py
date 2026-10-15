import unittest

from tasks.select_qwen_natural_scope_constrained_v5 import select_record


class NaturalScopeConstrainedSelectionTest(unittest.TestCase):
    def test_filters_every_release_constraint_before_ranking(self):
        base = {"hidden": 32, "penalty": 1., "learning_rate": .005,
                "epochs": 50, "natural_correct_top_examples": 10,
                "natural_wrong_top_examples": 2, "no_match_examples": 4}
        records = [
            {**base, "threshold": .25, "correct_top_recall": .96,
             "natural_wrong_top_specificity": .86, "no_match_specificity": .94,
             "balanced_accuracy": .919},
            {**base, "threshold": .30, "correct_top_recall": .95,
             "natural_wrong_top_specificity": .89, "no_match_specificity": .95,
             "balanced_accuracy": .93},
        ]
        self.assertEqual(select_record(records)["threshold"], .30)


if __name__ == "__main__":
    unittest.main()
