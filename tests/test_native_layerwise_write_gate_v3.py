import unittest

from tasks.build_native_layerwise_write_gate_v3 import (
    SOURCE_GROUPS,
    SPLIT_PER_MECHANISM,
    assign_splits,
)


class NativeLayerwiseWriteGateV3Test(unittest.TestCase):
    def test_registered_partition_is_exhaustive_and_balanced(self):
        source_mechanisms = (
            "same_subject_different_relation", "semantic_duplicate", "explicit_reset_revoke",
            "different_subject_same_relation", "semantic_overwrite", "stale_repetition",
        )
        episodes = []
        for mechanism in source_mechanisms:
            for index in range(104):
                episodes.append({
                    "source_group": f"{mechanism}-{index}", "mechanism": mechanism, "twin": 0,
                })
                episodes.append({
                    "source_group": f"{mechanism}-{index}", "mechanism": mechanism, "twin": 1,
                })
        splits, assignment = assign_splits(episodes)
        self.assertEqual(len(assignment), SOURCE_GROUPS)
        self.assertEqual(
            {name: len(rows) for name, rows in splits.items()},
            {name: count * 6 for name, count in SPLIT_PER_MECHANISM.items()},
        )
        self.assertEqual(set(assignment.values()), set(SPLIT_PER_MECHANISM))


if __name__ == "__main__":
    unittest.main()
