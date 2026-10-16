"""Tests for the native fast-weight forgetting-only slice audit."""

import unittest

from scripts.analyze_native_forgetting_slice_v15 import analyze


class NativeForgettingSliceV15Test(unittest.TestCase):
    def test_retention_is_converted_to_forgetting_and_write_is_fixed(self):
        rows = []
        actions = [[r, g] for r in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
        for index in range(4):
            # g=1 losses indexed by retention are 3, 2, 1, so forgetting=0 wins.
            losses = [4.0, 3.5, 3.0, 4.0, 3.0, 2.0, 4.0, 2.0, 1.0]
            rows.append({
                "id": f"row-{index}", "group_id": f"group-{index // 2}",
                "split": "train", "regime": "old_only",
                "protocol": "joint_natural_v1", "grid": [0.0, 0.5, 1.0],
                "actions": actions, "losses": losses,
            })
        result = analyze(rows, bootstrap_samples=100, seed=7)
        self.assertEqual(result["memory_substrate"], "native_dense_ttt_fast_weight")
        self.assertEqual(result["best_fixed_forgetting"], 0.0)
        self.assertEqual(result["oracle_counts"], {"0.0": 4})
        self.assertEqual(result["current_write_gate"], 1.0)

    def test_rejects_incomplete_joint_grid(self):
        row = {
            "id": "bad", "group_id": "group", "split": "train",
            "regime": "old_only", "protocol": "joint_natural_v1",
            "grid": [0.0, 0.5, 1.0], "actions": [[0.0, 1.0]], "losses": [1.0],
        }
        with self.assertRaisesRegex(ValueError, "complete joint grid"):
            analyze([row], bootstrap_samples=100)


if __name__ == "__main__":
    unittest.main()
