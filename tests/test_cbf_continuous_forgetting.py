import unittest

from tasks.train_cbf_continuous_forgetting import (
    GRID, POLICIES, generate_scenarios, select_alpha,
)


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

    def test_physics_rule_is_bounded_and_fixed_controls_are_registered(self):
        row = generate_scenarios("train", 1, 1, 17)[0]
        alpha = select_alpha("physics", row, {"features": []}, {}, .5)
        self.assertGreaterEqual(alpha, 0.)
        self.assertLessEqual(alpha, 1.)
        self.assertTrue(all(f"fixed_{value}" in POLICIES for value in GRID))

    def test_formal_smoke_keeps_test_sealed_when_gate_fails(self):
        try:
            import torch  # noqa: F401
            from tasks.train_cbf_continuous_forgetting import formal
        except ImportError:
            self.skipTest("torch unavailable")
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            probe = directory / "probe.json"
            probe.write_text(json.dumps({"protocol": "continuous_forgetting_probe_v1", "passed": True}))
            args = SimpleNamespace(output=str(directory / "formal"), probe=str(probe), seed=23,
                                   controller_seed=29, train_groups=1, dev_groups=1,
                                   test_groups=1, per_group=2, dim=8, epochs=2)
            result = formal(args)
            if not result["dev_gate"]["passed"]:
                self.assertFalse((directory / "formal" / "test_records.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
