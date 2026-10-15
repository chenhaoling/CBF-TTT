import unittest

from tasks.evaluate_pair_localized_relation_memory_s4_oracle import selection_key


class PairLocalizedRelationMemoryS4OracleTests(unittest.TestCase):
    def test_selection_rule(self):
        def row(recall, auc, epoch):
            return {"retrieval": {"recall_at_16": recall},
                    "direct_pair_roc_auc": auc, "epoch": epoch}
        self.assertGreater(selection_key(row(.8, .6, 25)),
                           selection_key(row(.7, .9, 10)))
        self.assertGreater(selection_key(row(.8, .7, 25)),
                           selection_key(row(.8, .6, 10)))
        self.assertGreater(selection_key(row(.8, .7, 10)),
                           selection_key(row(.8, .7, 25)))


if __name__ == "__main__":
    unittest.main()
