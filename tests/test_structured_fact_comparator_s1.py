import json
import tempfile
import unittest
from pathlib import Path

from tasks.build_structured_fact_comparator_s1 import (
    KINDS, construct_pairs, filter_eligible_relations, load_semeval, operation,
    primary_span, render,
)


def row(index, relation, head, tail, template=None):
    tokens = (template or [head, "has", "relation", "to", tail])
    hs, ts = tokens.index(head), tokens.index(tail)
    return {
        "source": "toy", "source_split": "train", "relation": relation,
        "source_index": index, "source_row_sha256": f"{index:064x}",
        "tokens": tokens, "head": (hs, hs + 1), "tail": (ts, ts + 1),
        "head_text": head, "tail_text": tail, "head_id": head, "tail_id": tail,
        "text": render(tokens, (hs, hs + 1), (ts, ts + 1)),
    }


class StructuredFactComparatorS1Tests(unittest.TestCase):
    def test_render_marks_both_entity_orders(self):
        self.assertEqual(
            render(["tail", "near", "head"], (2, 3), (0, 1)),
            "[VALUE] tail [/VALUE] near [SUBJECT] head [/SUBJECT]",
        )

    def test_operation_is_fixed_physical_rule(self):
        self.assertEqual(operation(True, True, True), "duplicate")
        self.assertEqual(operation(True, True, False), "update")
        for values in ((False, True, True), (True, False, False), (False, False, False)):
            self.assertEqual(operation(*values), "retain")

    def test_fewrel_inclusive_indices_become_half_open(self):
        self.assertEqual(primary_span(["New York", "id", [[3, 4]]]), (3, 5))

    def test_constructs_five_balanced_kinds_per_anchor(self):
        rows = [
            row(1, "r1", "a", "x"), row(2, "r1", "b", "y"),
            row(3, "r1", "c", "z"), row(4, "r2", "d", "u"),
            row(5, "r2", "e", "v"), row(6, "r2", "f", "w"),
        ]
        pairs = construct_pairs(rows, "train")
        self.assertEqual(len(pairs), len(rows) * len(KINDS))
        for kind in KINDS:
            self.assertEqual(sum(item["kind"] == kind for item in pairs), len(rows))
        updates = [item for item in pairs if item["kind"] == "update"]
        self.assertTrue(all(item["same_entity"] and item["same_relation"]
                            and not item["same_value"] for item in updates))
        hard = [item for item in pairs if item["kind"] == "same_entity_other_relation"]
        self.assertTrue(all(item["same_entity"] and not item["same_relation"]
                            and item["operation"] == "retain" for item in hard))

    def test_undercovered_relation_is_excluded_before_pairing(self):
        rows = [row(1, "singleton", "a", "x"),
                row(2, "covered", "b", "y"), row(3, "covered", "c", "z")]
        kept, excluded = filter_eligible_relations(rows)
        self.assertEqual({item["relation"] for item in kept}, {"covered"})
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "undercovered_relation_for_structured_pair")

    def test_sealed_test_cannot_be_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semeval_test.txt"
            path.write_text(json.dumps({"token": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "sealed"):
                load_semeval(directory, "semeval_test.txt", "test")


if __name__ == "__main__":
    unittest.main()
