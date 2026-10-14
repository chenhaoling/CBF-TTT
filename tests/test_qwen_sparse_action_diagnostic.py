import unittest

from scripts.summarize_qwen_sparse_action_diagnostic import combine, summarize_rows


class QwenSparseActionDiagnosticTest(unittest.TestCase):
    def rows(self):
        return [
            {
                "policy": "grid_classifier",
                "reader": "component_top1",
                "decisions": [
                    {
                        "oracle_retention": [1, 1, 0],
                        "retention": [1, 1, 0],
                        "retention_mae": 0.0,
                    },
                    {
                        "oracle_retention": [0.5, 1, 0],
                        "retention": [0, 1, 0],
                        "retention_mae": 1 / 6,
                    },
                ],
                "queries": [
                    {"correct": 1}, {"correct": 1}, {"correct": 1},
                    {"correct": 1}, {"correct": 0}, {"correct": 1},
                ],
            }
        ]

    def test_endpoint_and_interior_statistics(self):
        summary = summarize_rows(self.rows())
        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["oracle_interior_fraction"], 0.5)
        self.assertEqual(summary["predicted_interior_fraction"], 0.0)
        self.assertEqual(summary["buckets"]["endpoint"]["action_accuracy"], 1.0)
        self.assertEqual(summary["buckets"]["interior"]["action_accuracy"], 0.0)
        self.assertAlmostEqual(summary["buckets"]["interior"]["query_accuracy"], 2 / 3)

    def test_combination_is_count_weighted(self):
        first = summarize_rows(self.rows())
        second = summarize_rows(self.rows() * 2)
        macro = combine({"2301": first, "2302": second})
        self.assertEqual(macro["decisions"], 6)
        self.assertEqual(macro["oracle_interior_fraction"], 0.5)
        self.assertEqual(macro["buckets"]["endpoint"]["action_accuracy"], 1.0)
        self.assertAlmostEqual(macro["buckets"]["interior"]["query_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
