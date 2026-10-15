import unittest

from scripts.audit_structured_fact_comparator_s1_data import expected_train_relations, stable


class StructuredFactComparatorS1AuditTests(unittest.TestCase):
    def test_expected_train_partition_uses_exact_registered_hash(self):
        train = {f"p{index}": [] for index in range(64)}
        external = {f"x{index}": [] for index in range(16)}
        expected = sorted(train, key=lambda value: __import__("hashlib").sha256(
            f"2311:relation:{value}".encode()).hexdigest())[:48]
        self.assertEqual(expected_train_relations(train, external), set(expected))

    def test_pair_id_hash_is_order_sensitive(self):
        self.assertNotEqual(stable("left", "right"), stable("right", "left"))


if __name__ == "__main__":
    unittest.main()
