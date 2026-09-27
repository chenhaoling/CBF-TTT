"""Group-safe learning-curve subset checks without model weights."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_cbf_controller_subsets import prepare_subsets


class ControllerSubsetTests(unittest.TestCase):
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
