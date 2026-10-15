import unittest

from tasks.select_structured_fact_comparator_s1 import select_threshold, summarize


def record(gold, prediction, margin, relation="r", kind="k"):
    def value(delete):
        return {"same_entity": delete, "same_relation": delete, "same_value": False,
                "operation": "update" if delete else "retain"}
    return {"gold": value(gold), "prediction": value(prediction), "margin": margin,
            "relation": relation, "kind": kind}


class StructuredFactComparatorS1SelectionTests(unittest.TestCase):
    def test_threshold_maximizes_recall_under_precision(self):
        rows = [record(True, True, .9), record(True, True, .8),
                record(False, True, .7), record(True, False, .1)]
        selected = select_threshold(rows, minimum_precision=1.)
        self.assertEqual(selected["threshold"], .8)
        self.assertEqual(selected["tp"], 2)
        self.assertEqual(selected["fp"], 0)

    def test_summary_reports_relation_and_exact_metrics(self):
        rows = [record(True, True, .9, "a", "update"),
                record(False, False, .2, "a", "retain")]
        result = summarize(rows)
        self.assertEqual(result["exact_four_field_accuracy"], 1.)
        self.assertEqual(result["same_relation"]["specificity"], 1.)


if __name__ == "__main__":
    unittest.main()
