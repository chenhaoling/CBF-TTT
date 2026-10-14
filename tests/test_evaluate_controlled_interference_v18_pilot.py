import unittest

from scripts.audit_controlled_interference_v18_pilot import recompute_table
from tasks.build_controlled_interference_v18 import FIXED_POLICIES
from tasks.evaluate_controlled_interference_v18_pilot import (
    aggregate, arm_metrics, select_oracle, state_arms,
)


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


def rows(retain_all_ceiling=False):
    output = []
    for index in range(40):
        fixed = []
        for policy in FIXED_POLICIES:
            if policy == "retain_all":
                success = 1 if retain_all_ceiling or index % 2 == 0 else 0
                fixed.append(arm(f"fixed:{policy}", success, 1))
            elif policy in {"global_reset", "last_1", "last_2"}:
                fixed.append(arm(f"fixed:{policy}", 1, 0))
            else:
                fixed.append(arm(f"fixed:{policy}", 0, 0))
        delete = arm(f"delete_key:k{index}", 1, 1, kind="delete_key",
                     deletion_index=0, deleted_key_id=f"k{index}",
                     deleted_session_ids=[f"s{index}"])
        output.append({"episode_id": f"e{index}", "mechanism": "stale_repetition",
                       "length": 16, "arms": fixed + [delete]})
    return output


class ControlledInterferenceScoringTest(unittest.TestCase):
    def test_state_arm_order_and_metadata(self):
        state = {"fixed": {policy: [policy] for policy in FIXED_POLICIES},
                 "delete_one": [{"deleted_session_id": "s0",
                                 "memory_session_ids": ["s1"]}],
                 "delete_key": [{"deleted_key_id": "k0",
                                  "deleted_session_ids": ["s0"],
                                  "memory_session_ids": ["s1"]}]}
        values = state_arms(state)
        self.assertEqual([row["arm_id"] for row in values[:5]],
                         [f"fixed:{policy}" for policy in FIXED_POLICIES])
        self.assertEqual(values[-1]["deleted_key_id"], "k0")

    def test_joint_metric_and_exact_oracle(self):
        row = rows()[1]
        oracle, metric = select_oracle(row)
        self.assertEqual(oracle["kind"], "delete_key")
        self.assertEqual(metric["joint_success"], 1)
        self.assertEqual(arm_metrics(oracle)["generation_joint_success"], 1)

    def test_admission_passes_only_with_nontrivial_fixed_and_oracle_gain(self):
        result = aggregate(rows())
        self.assertEqual(result, recompute_table(rows()))
        self.assertTrue(result["corpus_admission_gate"]["passed"])
        self.assertEqual(result["fixed_arms"]["retain_all"]["joint_success"], .5)
        self.assertEqual(result["oracle_minus_fixed_joint_success"], .5)

        ceiling = aggregate(rows(retain_all_ceiling=True))
        self.assertFalse(ceiling["corpus_admission_gate"]["passed"])
        self.assertFalse(ceiling["corpus_admission_gate"]["checks"][
            "retain_all_joint_success_between_20_and_90"])


if __name__ == "__main__":
    unittest.main()
