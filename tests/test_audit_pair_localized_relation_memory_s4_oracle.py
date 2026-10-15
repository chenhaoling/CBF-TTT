import unittest

from scripts.audit_pair_localized_relation_memory_s4_oracle import close


class PairLocalizedRelationMemoryS4OracleAuditTests(unittest.TestCase):
    def test_nested_close(self):
        self.assertTrue(close({"x": [1.0, {"y": 2}]},
                              {"x": [1.0 + 1e-10, {"y": 2}], "z": 3}))
        self.assertFalse(close({"x": 1.0}, {"x": 1.2}))


if __name__ == "__main__":
    unittest.main()
