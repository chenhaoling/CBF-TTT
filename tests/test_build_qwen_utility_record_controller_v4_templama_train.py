import unittest

from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    hamilton_quotas,
)
from tasks.build_qwen_utility_record_controller_v4_templama_train import (
    EXPECTED_QUOTAS,
    GROUPS,
    MINIMUM_PER_RELATION,
)


class TestBuildUtilityRecordControllerV4TempLamaTrain(unittest.TestCase):
    def test_registered_hamilton_allocation(self):
        capacities = {
            "P102": 134, "P108": 159, "P127": 47, "P286": 104,
            "P39": 112, "P488": 79, "P54": 124, "P6": 92,
            "P69": 33,
        }
        self.assertEqual(
            hamilton_quotas(
                capacities,
                total=GROUPS,
                minimum=MINIMUM_PER_RELATION,
            ),
            EXPECTED_QUOTAS,
        )
        self.assertEqual(sum(EXPECTED_QUOTAS.values()), GROUPS)
        self.assertTrue(all(
            EXPECTED_QUOTAS[key] <= capacity
            for key, capacity in capacities.items()
        ))

    def test_allocation_refuses_insufficient_capacity(self):
        with self.assertRaises(ValueError):
            hamilton_quotas(
                {"P1": 15, "P2": 1000}, total=512, minimum=16
            )


if __name__ == "__main__":
    unittest.main()
