import unittest

from tasks.evaluate_cbf_continuous_stress import SUITES, generate_stress_scenarios


class ContinuousStressTests(unittest.TestCase):
    def test_suites_are_deterministic_and_disjoint(self):
        rows = {suite: generate_stress_scenarios(suite, 2, 3, 7) for suite in SUITES}
        for suite in SUITES:
            self.assertEqual(rows[suite], generate_stress_scenarios(suite, 2, 3, 7))
        ids = [{row["group_id"] for row in rows[suite]} for suite in SUITES]
        self.assertEqual(sum(map(len, ids)), len(set().union(*ids)))

    def test_registered_shifts_are_outside_training_support(self):
        long_rows = generate_stress_scenarios("long_memory", 2, 10, 11)
        weak_rows = generate_stress_scenarios("weak_update", 2, 10, 11)
        strong_rows = generate_stress_scenarios("strong_update", 2, 10, 11)
        extreme = generate_stress_scenarios("extreme_overlap", 2, 20, 11)
        self.assertTrue(all(row["old_count"] >= 16 for row in long_rows))
        self.assertTrue(all(row["current_scale"] <= .50 for row in weak_rows))
        self.assertTrue(all(row["current_scale"] >= 1.60 for row in strong_rows))
        ratios = [row["conflict_count"] / row["old_count"] for row in extreme]
        self.assertTrue(all(ratio <= .25 or ratio >= .70 for ratio in ratios))


if __name__ == "__main__":
    unittest.main()
