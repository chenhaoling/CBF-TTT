import unittest

from scripts.audit_mquake_qwen_headroom_v17_formal import recompute_table
from tasks.build_mquake_qwen_headroom_v17_states import FIXED_POLICIES
from tasks.evaluate_mquake_qwen_headroom_v17_formal import aggregate


def queries(primary=(0, 1), nll=(1.0, .1)):
    return [
        {"kind": "primary_old_phrase", "correct": primary[0], "answer_nll": nll[0]},
        {"kind": "primary_current_phrase", "correct": primary[1], "answer_nll": nll[1]},
        {"kind": "hard_retain", "correct": 1, "answer_nll": .1},
        {"kind": "unrelated", "correct": 1, "answer_nll": .1},
    ]


def make_arm(arm_id, primary=(0, 1), nll=(1.0, .1), kind="fixed", **extra):
    return {"arm_id": arm_id, "kind": kind, "memory_session_ids": [],
            "score_source_arm_id": arm_id, "queries": queries(primary, nll), **extra}


def make_rows(delete_primary=(1, 1), delete_nll=(.1, .1), count=20):
    rows = []
    for index in range(count):
        fixed = [make_arm(f"fixed:{policy}") for policy in FIXED_POLICIES]
        deletion = make_arm(
            f"delete_one:old{index}", primary=delete_primary, nll=delete_nll,
            kind="delete_one", deletion_index=0, deleted_session_id=f"old{index}")
        rows.append({"episode_id": f"e{index}", "subject": f"subject{index}",
                     "fold": index % 5, "primary_old_session_id": f"old{index}",
                     "arms": fixed + [deletion]})
    return rows


class FormalHeadroomV17Test(unittest.TestCase):
    def test_supported_headroom_requires_five_points_and_positive_interval(self):
        result = aggregate(make_rows())
        self.assertTrue(result["causal_headroom_supported"])
        self.assertEqual(result["primary_comparison"][
            "oracle_minus_fixed_accuracy_subject_macro"]["mean"], .5)
        self.assertEqual(result["oracle_action_counts"]["delete_primary_old"], 20)
        self.assertEqual(result["next_stage"], "train_readout_utility_policy")

    def test_accuracy_tie_prefers_retain_all_and_stops_controller(self):
        rows = make_rows(delete_primary=(0, 1), delete_nll=(1.0, .1))
        result = aggregate(rows)
        self.assertFalse(result["causal_headroom_supported"])
        self.assertEqual(result["oracle_action_counts"]["retain_all"], 20)
        self.assertEqual(result["primary_comparison"][
            "oracle_minus_fixed_accuracy_subject_macro"]["mean"], 0.0)
        self.assertEqual(result["next_stage"], "stop_controller_training_no_causal_headroom")

    def test_independent_formal_recomputation_matches_scorer(self):
        rows = make_rows()
        self.assertEqual(aggregate(rows), recompute_table(rows))


if __name__ == "__main__":
    unittest.main()
