import unittest

from tasks.build_fewrel_selective_forgetting import (
    construct_partition_pairs, masked_surface, relation_partitions,
)


def instance(relation, subject, index):
    return {
        "partition": "train", "relation_id": relation,
        "head_entity_id": subject, "tail_entity_id": f"v{relation}{index}",
        "source_row_sha256": f"{relation}-{subject}-{index}",
        "instance_id": f"{relation}-{subject}-{index}",
        "surface": f"[SUBJECT] relation {relation} marker {index} [VALUE]",
    }


class FewRelSelectiveForgettingTest(unittest.TestCase):
    def test_relation_partition_is_deterministic_and_disjoint(self):
        train = [f"P{index}" for index in range(64)]
        external = [f"E{index}" for index in range(16)]
        first = relation_partitions(train, external)
        second = relation_partitions(reversed(train), reversed(external))
        self.assertEqual(first, second)
        self.assertEqual([len(first[key]) for key in (
            "train", "development", "internal_test", "external_test")],
            [48, 8, 8, 16])
        self.assertEqual(len(set().union(*map(set, first.values()))), 80)

    def test_masked_surface_replaces_all_mentions(self):
        row = {
            "tokens": ["Alice", "met", "Bob", "and", "Alice", "thanked", "Bob"],
            "h": ["alice", "Q1", [[0], [4]]],
            "t": ["bob", "Q2", [[2], [6]]],
        }
        self.assertEqual(masked_surface(row),
                         "[SUBJECT] met [VALUE] and [SUBJECT] thanked [VALUE]")

    def test_out_of_range_and_overlapping_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            masked_surface({"tokens": ["x"], "h": ["x", "Q1", [[0]]],
                            "t": ["x", "Q2", [[0]]]})
        with self.assertRaises(ValueError):
            masked_surface({"tokens": ["x"], "h": ["x", "Q1", [[1]]],
                            "t": ["x", "Q2", [[0]]]})

    def test_pair_construction_balances_semantics_and_hard_negatives(self):
        rows = []
        for relation in ("P1", "P2", "P3"):
            for index, subject in enumerate(("shared", f"{relation}-a", f"{relation}-b",
                                             f"{relation}-c")):
                rows.append(instance(relation, subject, index))
        pairs = construct_partition_pairs(rows, "train")
        self.assertEqual(sum(row["same_relation"] for row in pairs),
                         sum(not row["same_relation"] for row in pairs))
        self.assertIn("same_subject_hard_negative", {row["kind"] for row in pairs})
        by_id = {row["instance_id"]: row for row in rows}
        for pair in pairs:
            left, right = by_id[pair["left_instance_id"]], by_id[pair["right_instance_id"]]
            self.assertEqual(pair["same_relation"],
                             int(left["relation_id"] == right["relation_id"]))
            if pair["kind"] == "same_relation_positive":
                self.assertNotEqual(left["head_entity_id"], right["head_entity_id"])


if __name__ == "__main__":
    unittest.main()
