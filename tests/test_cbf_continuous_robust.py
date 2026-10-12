import unittest

from tasks.train_cbf_continuous_robust import COMPOUND_DOMAINS, domain_scenarios


class ContinuousRobustTests(unittest.TestCase):
    def test_compound_suites_are_deterministic_and_disjoint(self):
        rows = {name: domain_scenarios(name, "test", 2, 3, 31) for name in COMPOUND_DOMAINS}
        for name in COMPOUND_DOMAINS:
            self.assertEqual(rows[name], domain_scenarios(name, "test", 2, 3, 31))
        groups = [{row["group_id"] for row in rows[name]} for name in COMPOUND_DOMAINS]
        self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))

    def test_compound_suites_combine_registered_shifts(self):
        weak = domain_scenarios("long_weak", "test", 2, 10, 37)
        strong = domain_scenarios("long_strong", "test", 2, 10, 37)
        extreme = domain_scenarios("extreme_scale", "test", 2, 20, 37)
        self.assertTrue(all(row["old_count"] >= 16 and row["current_scale"] <= .5 for row in weak))
        self.assertTrue(all(row["old_count"] >= 16 and row["current_scale"] >= 1.6 for row in strong))
        self.assertTrue(all((row["conflict_count"] / row["old_count"] <= .25 or
                             row["conflict_count"] / row["old_count"] >= .70) and
                            (row["current_scale"] <= .5 or row["current_scale"] >= 1.6)
                            for row in extreme))


if __name__ == "__main__":
    unittest.main()
