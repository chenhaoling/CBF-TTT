import unittest

from tasks.build_natural_relation_reranker_s2 import (
    FIELDS, KINDS, add_masked, construct, mask_spans, operation, similarity_bin,
)


def row(name, relation, text, head, tail):
    tokens = text.split()
    return add_masked({"source_row_sha256": name, "relation": relation,
                       "tokens": tokens, "head": head, "tail": tail})


class NaturalRelationRerankerS2BuilderTests(unittest.TestCase):
    def test_mask_spans_supports_both_entity_orders(self):
        self.assertEqual(mask_spans("a b c d".split(), (0, 1), (3, 4)),
                         "[SUBJECT] b c [VALUE]")
        self.assertEqual(mask_spans("a b c d".split(), (3, 4), (0, 1)),
                         "[VALUE] b c [SUBJECT]")

    def test_similarity_bin_is_bounded(self):
        self.assertEqual(similarity_bin(0.0), 0)
        self.assertEqual(similarity_bin(0.999), 19)
        self.assertEqual(similarity_bin(1.0), 19)

    def test_natural_pairs_have_matched_bins_and_complete_scenarios(self):
        rows = [
            row("a0", "a", "Alice founded Acme today", (0, 1), (2, 3)),
            row("a1", "a", "Bob founded Beta yesterday", (0, 1), (2, 3)),
            row("a2", "a", "Cara founded Core recently", (0, 1), (2, 3)),
            row("b0", "b", "Dan joined Delta today", (0, 1), (2, 3)),
            row("b1", "b", "Eve joined Echo yesterday", (0, 1), (2, 3)),
            row("b2", "b", "Finn joined Fox recently", (0, 1), (2, 3)),
        ]
        pairs, scenarios, _ = construct(rows, "train", negative_pool=6)
        by_left = {}
        for value in pairs:
            by_left.setdefault(value["left_source_row_sha256"], []).append(value)
            self.assertNotEqual(value["left_text"], value["right_text"])
        for values in by_left.values():
            self.assertEqual({value["label"] for value in values}, {"positive", "negative"})
            self.assertEqual(len({value["similarity_bin"] for value in values}), 1)
        grouped = {}
        for value in scenarios:
            grouped.setdefault(value["group_id"], []).append(value)
        for values in grouped.values():
            self.assertEqual({value["kind"] for value in values}, set(KINDS))
            for value in values:
                expected = FIELDS[value["kind"]]
                self.assertEqual((value["same_entity"], value["same_relation"],
                                  value["same_value"]), expected)
                self.assertEqual(value["operation"], operation(*map(bool, expected)))

    def test_mask_rejects_overlap(self):
        with self.assertRaises(ValueError):
            mask_spans("a b c".split(), (0, 2), (1, 3))


if __name__ == "__main__":
    unittest.main()
