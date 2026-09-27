"""Natural-continuation pilot preserves source groups and held-out suffixes."""

import unittest

from tasks.build_cbf_natural_scenarios import build_scenarios


class NaturalScenarioTests(unittest.TestCase):
    def test_four_cases_share_group_without_target_leakage(self):
        records = [[100 * record + token for token in range(12)] for record in range(12)]
        rows, metadata = build_scenarios(records, {"train": 1, "dev": 1, "test": 1},
                                         context_tokens=4, query_tokens=2, answer_tokens=3, seed=7)
        self.assertEqual(len(rows), 12)
        self.assertEqual(metadata["record_count"], 12)
        for group_index in range(3):
            group = [row for row in rows if row["group_id"] == f"natural-pack-{group_index:05d}"]
            self.assertEqual(len(group), 4)
            self.assertEqual(len({row["split"] for row in group}), 1)
            self.assertEqual({row["regime"] for row in group},
                             {"both_relevant", "old_only", "new_only", "neither_relevant"})
            for row in group:
                self.assertEqual(len(row["context_ids"]), 8)
                self.assertEqual(row["objective"], "joint_natural_v1")
                for query in row["futures"][0]["queries"]:
                    self.assertEqual(len(query["query_ids"]), 2)
                    self.assertEqual(len(query["answer_ids"]), 3)
                    self.assertFalse(any(query["answer_ids"] == row["context_ids"][start:start + 3]
                                         for start in range(6)))
        with self.assertRaisesRegex(ValueError, "four distinct records"):
            build_scenarios(records[:-1], {"train": 1, "dev": 1, "test": 1}, 4, 2, 3, 7)

        postcutoff, metadata = build_scenarios(
            records, {"train": 1, "dev": 1, "test": 1}, 4, 2, 3, 7,
            protocol="joint_postcutoff_v1", group_prefix="postcutoff-paper"
        )
        self.assertEqual(metadata["protocol"], "joint_postcutoff_v1")
        self.assertTrue(all(row["objective"] == "joint_postcutoff_v1" and
                            row["group_id"].startswith("postcutoff-paper-")
                            for row in postcutoff))


if __name__ == "__main__":
    unittest.main()
