import unittest

from tasks.score_structured_fact_comparator_s1 import candidates, parse_structured, partition


class StructuredFactComparatorS1ScoreTests(unittest.TestCase):
    def test_candidates_cover_all_field_states_with_physical_operation(self):
        values = candidates()
        self.assertEqual(len(values), 8)
        self.assertEqual(len({tuple(item[field] for field in
                                    ("same_entity", "same_relation", "same_value"))
                              for item in values}), 8)
        self.assertEqual(sum(item["operation"] == "duplicate" for item in values), 1)
        self.assertEqual(sum(item["operation"] == "update" for item in values), 1)

    def test_parser_is_strict_and_physics_consistent(self):
        valid = '{"same_entity":true,"same_relation":true,"same_value":false,"operation":"update"}'
        self.assertIsNotNone(parse_structured(valid))
        self.assertIsNone(parse_structured(valid + " trailing"))
        self.assertIsNone(parse_structured(valid.replace('"update"', '"retain"')))

    def test_partition_is_complete_and_disjoint(self):
        rows = list(range(11))
        left = partition(rows, 0, 2); right = partition(rows, 1, 2)
        self.assertEqual(sorted(index for index, _ in left + right), list(range(11)))
        self.assertFalse({index for index, _ in left} & {index for index, _ in right})


if __name__ == "__main__":
    unittest.main()
