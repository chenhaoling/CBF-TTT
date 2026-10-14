import unittest

from tasks.prepare_semantic_controller_balancing import (
    prepare,
    relation_source_weights,
    validate_balance,
)


def rows():
    output = []
    for fold in range(5):
        for relation, counts in ((f"P{fold}a", {"counterfact": 7, "mquake": 1}),
                                 (f"P{fold}b", {"counterfact": 2})):
            for source, count in counts.items():
                for index in range(count):
                    output.append({"row_id": f"{fold}-{relation}-{source}-{index}",
                                   "outer_fold": fold, "relation_id": relation,
                                   "source": source,
                                   "subject_normalized": f"s-{fold}-{relation}-{index}"})
    return output


class SemanticControllerBalancingTest(unittest.TestCase):
    def test_inverse_sqrt_group_weights_are_exact(self):
        values = rows()
        weights = relation_source_weights(values)
        checks = validate_balance(values, weights)
        self.assertTrue(checks["inverse_sqrt_group_frequency_exact"])
        self.assertTrue(checks["mean_row_weight_one"])
        self.assertTrue(checks["resource_gate_passed"])
        shared = [row for row in values if row["relation_id"] == "P0a"]
        cf_weight = weights[next(row["row_id"] for row in shared
                                 if row["source"] == "counterfact")]
        mq_weight = weights[next(row["row_id"] for row in shared
                                 if row["source"] == "mquake")]
        self.assertAlmostEqual(mq_weight / cf_weight, 7 ** .5)

    def test_nested_roles_do_not_leak_outer_or_inner(self):
        output, summaries = prepare(rows(), 5)
        self.assertEqual(set(output), set(range(5)))
        for outer, weighted in output.items():
            inner = (outer + 1) % 5
            source = {row["row_id"]: row for row in rows()}
            for item in weighted:
                fold = source[item["row_id"]]["outer_fold"]
                expected = "outer_test" if fold == outer else (
                    "inner_validation" if fold == inner else "train")
                self.assertEqual(item["role"], expected)
                self.assertEqual(item["balance_weight"] is None, expected != "train")
            self.assertEqual(summaries[str(outer)]["train_folds"],
                             sorted(set(range(5)) - {outer, inner}))

    def test_balancing_is_deterministic(self):
        first = prepare(rows(), 5)
        second = prepare(list(reversed(rows())), 5)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
