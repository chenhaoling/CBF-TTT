import unittest

from scripts.audit_controlled_interference_v19_outcome import recompute_table
from tasks.evaluate_controlled_interference_v19 import aggregate


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=.1):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "generation": {"normalized_exact": correct}}


def arm(arm_id, joint, locality, nll=.1, kind="fixed"):
    values = (joint, joint, 1, 1, locality)
    return {"arm_id": arm_id, "kind": kind,
            "queries": [query(name, value, nll) for name, value in zip(KINDS, values)]}


def row(index, risky=False):
    fold = index % 5
    arms = [arm("fixed:no_memory", 0, 1), arm("fixed:retain_all", 0, 1),
            arm("fixed:global_reset", 0, 1), arm("fixed:last_1", 0, 1),
            arm("fixed:last_2", 0, 1)]
    arms.append(arm(f"delete_one:s{index}", 1, int(not risky), kind="delete_one"))
    return {"episode_id": f"e{index}", "subject": f"subject{index // 4}",
            "template_group": f"template{index // 16}", "fold": fold,
            "mechanism": "clean_update", "length": 8, "arms": arms}


class EvaluateControlledInterferenceV19Test(unittest.TestCase):
    def test_primary_budget_is_exact_and_registered(self):
        rows = [row(index, risky=index < 20) for index in range(320)]
        result = aggregate(rows)
        primary = result["pareto"]["0.03"]
        self.assertEqual(primary["maximum_locality_failures"], 9)
        self.assertEqual(primary["observed_locality_failures"], 9)
        self.assertEqual(primary["metrics"]["locality_accuracy"], 311 / 320)
        self.assertEqual(primary["metrics"]["joint_success"], 309 / 320)
        self.assertTrue(result["confirmation_gate"]["passed"])

    def test_budget_zero_rejects_all_risky_actions(self):
        rows = [row(index, risky=True) for index in range(320)]
        result = aggregate(rows)
        self.assertEqual(result["pareto"]["0.0"]["observed_locality_failures"], 0)
        self.assertEqual(result["pareto"]["0.0"]["metrics"]["joint_success"], 0)

    def test_independent_auditor_exactly_reconstructs_registered_table(self):
        rows = [row(index, risky=index % 11 == 0) for index in range(320)]
        self.assertEqual(aggregate(rows), recompute_table(rows))


if __name__ == "__main__":
    unittest.main()
