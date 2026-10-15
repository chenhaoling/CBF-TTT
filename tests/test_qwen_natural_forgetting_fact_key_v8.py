import unittest

from tasks.train_qwen_natural_forgetting_fact_key_v8 import (
    FEATURE_NAMES,
    pair_features,
    pair_metrics,
    relation_view,
    subject_fold,
)


class FactKeyV8Test(unittest.TestCase):
    def test_pair_feature_width_and_finiteness(self):
        values = pair_features("Where was Ada Lovelace born?",
                               "What is the birthplace of Ada Lovelace?")
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertTrue(all(value == value for value in values))

    def test_relation_view_masks_inference_anchors(self):
        self.assertEqual(relation_view("Where was Ada Lovelace born?"),
                         "where was <entity> <entity> born")

    def test_subject_fold_is_stable(self):
        self.assertEqual(subject_fold("Ada Lovelace"), subject_fold("ada lovelace"))
        self.assertIn(subject_fold("Ada Lovelace"), range(5))

    def test_pair_metrics_reports_both_negative_specificities(self):
        rows = [
            {"label": 1, "pair_type": "positive_same_fact_key"},
            {"label": 0, "pair_type": "negative_same_relation_different_subject"},
            {"label": 0, "pair_type": "negative_same_subject_different_relation"},
        ]
        value = pair_metrics(rows, [.9, .1, .2], .5)
        self.assertEqual(value["precision"], 1.0)
        self.assertEqual(value["recall"], 1.0)
        self.assertEqual(value["by_type"][
            "negative_same_relation_different_subject"]["specificity"], 1.0)
        self.assertEqual(value["by_type"][
            "negative_same_subject_different_relation"]["specificity"], 1.0)


if __name__ == "__main__":
    unittest.main()
