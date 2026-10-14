import unittest

from tasks.train_natural_semantic_projection import (
    build_relation_triples,
    select_epoch,
)


def pair(pair_id, fold, relation, anchor, kind, label, left, right,
         other_relation=None):
    return {
        "pair_id": pair_id,
        "outer_fold": fold,
        "anchor_relation_id": relation,
        "other_relation_id": other_relation or relation,
        "anchor_source_row_sha256": anchor,
        "kind": kind,
        "same_relation": label,
        "left_surface_id": left,
        "right_surface_id": right,
    }


class NaturalSemanticProjectionTest(unittest.TestCase):
    def test_triples_join_registered_hard_negative(self):
        pairs = [
            pair("p0", 0, "r0", "a0", "within_row_positive", 1, "s0", "s1"),
            pair("n0", 0, "r0", "a0", "lexical_hard_negative", 0,
                 "s0", "s2", "r1"),
            pair("p1", 0, "r1", "a1", "within_row_positive", 1, "s3", "s4"),
            pair("n1", 0, "r1", "a1", "lexical_hard_negative", 0,
                 "s3", "s5", "r0"),
        ]
        index = {f"s{value}": value for value in range(6)}
        grouped = build_relation_triples(pairs, index, {0})
        self.assertEqual(grouped["r0"], [(0, 1, 2)])
        self.assertEqual(grouped["r1"], [(3, 4, 5)])

    def test_missing_hard_negative_is_rejected(self):
        pairs = [
            pair("p0", 0, "r0", "a0", "within_row_positive", 1, "s0", "s1"),
            pair("p1", 0, "r1", "a1", "within_row_positive", 1, "s2", "s3"),
        ]
        with self.assertRaises(ValueError):
            build_relation_triples(pairs, {f"s{x}": x for x in range(4)}, {0})

    def test_selection_prefers_feasible_then_metric_then_shorter_epoch(self):
        rows = [
            {"epoch": 25, "inner_constraints_feasible": False,
             "inner_metrics": {"balanced_accuracy": .9, "f1": .9,
                               "negative_specificity": .9}},
            {"epoch": 50, "inner_constraints_feasible": True,
             "inner_metrics": {"balanced_accuracy": .8, "f1": .7,
                               "negative_specificity": .96}},
            {"epoch": 100, "inner_constraints_feasible": True,
             "inner_metrics": {"balanced_accuracy": .8, "f1": .7,
                               "negative_specificity": .96}},
        ]
        self.assertEqual(select_epoch(rows)["epoch"], 50)


if __name__ == "__main__":
    unittest.main()
