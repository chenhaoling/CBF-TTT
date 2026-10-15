import unittest

from tasks.train_qwen_natural_forgetting_state_equivalent_v10 import (
    duplicate_equivalent,
    fact_value_identity,
)


class StateEquivalentV10Test(unittest.TestCase):
    def test_fact_identity_requires_source_and_target(self):
        self.assertEqual(
            fact_value_identity({"source_row_sha256": "abc", "target": "x"}),
            fact_value_identity({"source_row_sha256": "abc", "target": "x"}),
        )
        self.assertNotEqual(
            fact_value_identity({"source_row_sha256": "abc", "target": "x"}),
            fact_value_identity({"source_row_sha256": "abc", "target": "y"}),
        )
        self.assertIsNone(fact_value_identity({"target": "x"}))

    def test_duplicate_equivalence_ignores_chain_identity(self):
        session = {"chains": [
            {"id": "incoming", "versions": [{"source_row_sha256": "abc",
                                                  "target": "same"}]},
            {"id": "stored", "versions": [{"source_row_sha256": "abc",
                                                "target": "same"}]},
        ]}
        row = {"top": 0, "incoming_chain_id": "incoming", "incoming_version": 0,
               "memory": [["stored", 0]]}
        self.assertTrue(duplicate_equivalent(row, session))
        session["chains"][1]["versions"][0]["target"] = "different"
        self.assertFalse(duplicate_equivalent(row, session))

    def test_empty_top_is_not_equivalent(self):
        self.assertFalse(duplicate_equivalent({"top": None}, {"chains": []}))


if __name__ == "__main__":
    unittest.main()
