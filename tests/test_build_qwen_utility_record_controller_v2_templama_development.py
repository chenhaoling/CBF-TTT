import unittest

from tasks.build_qwen_utility_record_controller_v2_templama_development import (
    hamilton_quotas,
)


class TestBuildUtilityRecordControllerV2TempLamaDevelopment(unittest.TestCase):
    def test_registered_hamilton_allocation(self):
        capacities = {
            "P102": 45, "P108": 79, "P127": 22, "P286": 42, "P39": 44,
            "P488": 36, "P54": 54, "P6": 33, "P69": 16,
        }
        self.assertEqual(hamilton_quotas(capacities), {
            "P102": 31, "P108": 52, "P127": 17, "P286": 29, "P39": 30,
            "P488": 25, "P54": 36, "P6": 23, "P69": 13,
        })

    def test_allocation_refuses_insufficient_capacity(self):
        with self.assertRaises(ValueError):
            hamilton_quotas({"P1": 7, "P2": 300})


if __name__ == "__main__":
    unittest.main()
