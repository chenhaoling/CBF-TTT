import unittest

from tasks.train_cbf_continuous_forgetting import GRID, generate_scenarios


class ContinuousForgettingTests(unittest.TestCase):
    def test_scenarios_are_deterministic_and_split_disjoint(self):
        first = generate_scenarios("train", 2, 3, 7)
        second = generate_scenarios("train", 2, 3, 7)
        dev = generate_scenarios("dev", 2, 3, 7)
        self.assertEqual(first, second)
        self.assertFalse({row["group_id"] for row in first} & {row["group_id"] for row in dev})

    def test_scenario_contains_preservation_and_conflict(self):
        rows = generate_scenarios("train", 2, 10, 11)
        self.assertTrue(all(0 < row["conflict_count"] < row["old_count"] for row in rows))
        self.assertTrue(all(set(dict(row["truth"])) >= set(dict(row["old"])) for row in rows))
        self.assertEqual(GRID, (0., .25, .5, .75, 1.))

    def test_grid_label_is_observed_loss_minimum(self):
        try:
            from tasks.train_cbf_continuous_forgetting import Embeddings, label_scenario
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")
        row = generate_scenarios("probe", 1, 1, 13)[0]
        label = label_scenario(row, Embeddings(1901, 8))
        selected = label["losses"][str(label["alpha"])]["nll"]
        self.assertEqual(selected, min(value["nll"] for value in label["losses"].values()))


if __name__ == "__main__":
    unittest.main()
