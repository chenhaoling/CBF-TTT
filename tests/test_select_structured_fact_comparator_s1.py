import unittest

from tasks.select_structured_fact_comparator_s1 import (
    baseline_row, relation_surface, select_threshold, summarize, trigram_similarity,
)


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

    def test_model_free_baseline_extracts_fields_without_metadata(self):
        row = {"pair_id": "p", "kind": "update", "left_relation_audit": "r",
               "left_text": "[SUBJECT] Alice [/SUBJECT] works at [VALUE] X [/VALUE]",
               "right_text": "[SUBJECT] Alice [/SUBJECT] works at [VALUE] Y [/VALUE]",
               "same_entity": 1, "same_relation": 1, "same_value": 0,
               "operation": "update"}
        scored = baseline_row(row, True, .8)
        self.assertEqual(scored["prediction"]["operation"], "update")
        self.assertEqual(relation_surface(row["left_text"]),
                         "[subject] works at [value]")
        self.assertGreater(trigram_similarity(row["left_text"], row["right_text"]), .99)


if __name__ == "__main__":
    unittest.main()
