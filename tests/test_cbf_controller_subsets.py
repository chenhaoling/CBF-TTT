"""Group-safe learning-curve subset checks without model weights."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_cbf_controller_subsets import prepare_subsets


class ControllerSubsetTests(unittest.TestCase):
    def test_write_labels_are_counted_without_mixing_forget_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev = root / "train.jsonl", root / "dev.jsonl"
            train.write_text(json.dumps({"id": "a", "group_id": "a", "split": "train",
                                         "update_rule": "write", "write_gate_star": 0.0}) + "\n")
            dev.write_text(json.dumps({"id": "b", "group_id": "b", "split": "dev",
                                       "update_rule": "write", "write_gate_star": 1.0}) + "\n")
            result = prepare_subsets([train], [dev], root / "out", [1])
            self.assertEqual(result["update_rule"], "write")
            self.assertEqual(result["subsets"]["1"]["gate_counts"], {"0.0": 1})
            dev.write_text(json.dumps({"id": "b", "group_id": "b", "split": "dev",
                                       "alpha_star": 0.0}) + "\n")
            with self.assertRaisesRegex(ValueError, "share one known update rule"):
                prepare_subsets([train], [dev], root / "out", [1])

    def test_nested_groups_and_dev_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text("".join(
                json.dumps({"id": f"g{group}-v{variant}", "group_id": f"g{group}",
                            "split": "train", "alpha_star": float(variant) / 2}) + "\n"
                for group in range(4) for variant in range(3)
            ), encoding="utf-8")
            dev.write_text(json.dumps({"id": "dev", "group_id": "dev", "split": "dev"}) + "\n")
            output = root / "subsets"
            result = prepare_subsets([train], [dev], output, [2, 4], seed=7)
            self.assertEqual(result["subsets"]["2"]["labels"], 6)
            self.assertEqual(result["subsets"]["4"]["labels"], 12)
            small = {json.loads(line)["group_id"] for line in
                     (output / "train_groups_2.jsonl").read_text().splitlines()}
            large = {json.loads(line)["group_id"] for line in
                     (output / "train_groups_4.jsonl").read_text().splitlines()}
            self.assertEqual(len(small), 2)
            self.assertTrue(small < large)
            self.assertNotIn("dev", large)
            self.assertEqual(result["subsets"]["2"]["alpha_counts"],
                             {"0.0": 2, "0.5": 2, "1.0": 2})


if __name__ == "__main__":
    unittest.main()
