import unittest

from tasks.diagnose_v18_locality_constrained_oracle import constrained_selection


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=.1):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "generation": {"normalized_exact": correct}}


def arm(name, joint, locality, nll=.1):
    values = (joint, joint, 1, 1, locality)
    return {"arm_id": name, "queries": [query(kind, value, nll)
                                         for kind, value in zip(KINDS, values)]}


class DiagnoseV18LocalityConstrainedOracleTest(unittest.TestCase):
    def test_budget_changes_optimal_joint_choice(self):
        rows = [{"arms": [arm(f"safe{i}", 0, 1), arm(f"risky{i}", 1, 0)]}
                for i in range(4)]
        failures, total, chosen = constrained_selection(rows, 1)
        self.assertEqual(failures, 1)
        self.assertEqual(total["joint_success"], 1)
        self.assertEqual(total["locality_accuracy"], 3)
        self.assertEqual(sum(name.startswith("risky") for name, _ in chosen), 1)

    def test_ties_prefer_update_retention_then_nll(self):
        safe = arm("safe", 1, 1, .5)
        lower_nll = arm("lower", 1, 1, .1)
        failures, _, chosen = constrained_selection([{"arms": [safe, lower_nll]}], 0)
        self.assertEqual(failures, 0)
        self.assertEqual(chosen[0][0], "lower")


if __name__ == "__main__":
    unittest.main()
