"""Focused tests for V11 corpus partition and parameter locks."""

import unittest

from tasks.build_qwen_utility_record_controller_v11_fewrel_development import (
    EXPECTED_RELATIONS, GROUP_PREFIX, PARTITION_SEED, SEED,
)
from tasks.build_qwen_utility_record_controller_v10_fewrel_corpus import (
    GROUPS_PER_RELATION,
)


class V11CorpusTest(unittest.TestCase):
    def test_registered_scale(self):
        self.assertEqual(EXPECTED_RELATIONS * GROUPS_PER_RELATION, 112)
        self.assertEqual(EXPECTED_RELATIONS * GROUPS_PER_RELATION * 2 * 3, 672)

    def test_frozen_identifiers(self):
        self.assertEqual(PARTITION_SEED, 2311)
        self.assertEqual(SEED, 202610191)
        self.assertTrue(GROUP_PREFIX.startswith("usr-v11-"))


if __name__ == "__main__":
    unittest.main()
