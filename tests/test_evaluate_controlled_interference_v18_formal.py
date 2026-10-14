import unittest

from scripts.audit_controlled_interference_v18_formal_outcome import recompute_table
from tasks.build_controlled_interference_v18 import FIXED_POLICIES
from tasks.evaluate_controlled_interference_v18_formal import aggregate, clustered_interval


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def queries(update, retain, locality=1):
    values = (update, update, retain, retain, locality)
    return [{"kind": kind, "correct": correct, "answer_nll": .1 if correct else 1.0,
             "generation": {"normalized_exact": correct}}
            for kind, correct in zip(KINDS, values)]


def arm(arm_id, update, retain, kind="fixed", **extra):
    return {"arm_id": arm_id, "kind": kind, "memory_session_ids": [],
            "score_source_arm_id": arm_id, "queries": queries(update, retain), **extra}


def formal_rows():
    rows = []
    for index in range(320):
        subject_index, within = divmod(index, 4)
        template_index = subject_index % 20
        fixed = []
        for policy in FIXED_POLICIES:
            if policy == "retain_all":
                success = int(within != 3)
                fixed.append(arm(f"fixed:{policy}", success, 1))
            elif policy in {"global_reset", "last_1", "last_2"}:
                fixed.append(arm(f"fixed:{policy}", 1, 0))
            else:
                fixed.append(arm(f"fixed:{policy}", 0, 0))
        delete = arm(f"delete_key:k{index}", 1, 1, kind="delete_key",
                     deletion_index=0, deleted_key_id=f"k{index}",
                     deleted_session_ids=[f"s{index}"])
        rows.append({"episode_id": f"e{index}", "subject": f"subject{subject_index}",
                     "template_group": f"template{template_index}",
                     "fold": template_index % 5, "length": (8, 16, 32, 64)[within],
                     "mechanism": ("clean_update", "stale_repetition",
                                   "irrelevant_near_duplicate", "useful_repetition")[within],
                     "arms": fixed + [delete]})
    return rows


class ControlledInterferenceFormalScoringTest(unittest.TestCase):
    def test_cluster_interval_is_deterministic(self):
        records = [{"subject": f"s{index // 4}", "gain": int(index % 4 == 0)}
                   for index in range(40)]
        first = clustered_interval(records, "gain", "subject", 7, draws=1000)
        second = clustered_interval(records, "gain", "subject", 7, draws=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], .25)
        self.assertEqual(first["clusters"], 10)

    def test_outer_fold_formal_gate_and_independent_recomputation(self):
        rows = formal_rows()
        result = aggregate(rows)
        self.assertEqual(result, recompute_table(rows))
        self.assertTrue(result["headroom_gate"]["passed"])
        self.assertEqual(result["oracle"]["joint_success"], 1.0)
        self.assertEqual(result["outer_fold_fixed"]["joint_success"], .75)
        self.assertEqual(result["primary_comparison"][
            "oracle_minus_fixed_joint_success_subject_macro"]["mean"], .25)
        self.assertEqual(set(result["fixed_selected_by_outer_fold"].values()),
                         {"retain_all"})
        self.assertTrue(all(value["rows"] == 64 for value in result["by_fold"].values()))


if __name__ == "__main__":
    unittest.main()
