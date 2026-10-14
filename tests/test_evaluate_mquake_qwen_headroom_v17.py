import unittest

from tasks.evaluate_mquake_qwen_headroom_v17 import (
    FIXED_POLICIES, clustered_interval, select_fixed_by_outer_fold, select_oracle,
    state_arms, summarize_rows,
)


def query(kind, correct, nll):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "candidates": ["a", "b"], "total_tokens_max": 100,
            "resources": {"seconds": .1, "candidate_scores": 2,
                          "gpu_start_allocated_gib": 8.0,
                          "gpu_start_reserved_gib": 8.5,
                          "gpu_peak_allocated_gib": 8.1,
                          "gpu_peak_reserved_gib": 8.6}}


def queries(primary_correct, primary_nll):
    return [
        query("primary_old_phrase", primary_correct[0], primary_nll[0]),
        query("primary_current_phrase", primary_correct[1], primary_nll[1]),
        query("hard_retain", 1, .1), query("unrelated", 1, .1),
    ]


def arm(arm_id, correct=(1, 1), nll=(.1, .1), kind="fixed", **extra):
    return {"arm_id": arm_id, "kind": kind, "memory_session_ids": [],
            "score_source_arm_id": arm_id, "queries": queries(correct, nll), **extra}


class HeadroomV17Test(unittest.TestCase):
    def test_state_arms_preserve_registered_order(self):
        state = {
            "fixed": {policy: [policy] for policy in FIXED_POLICIES},
            "delete_one": [
                {"deleted_session_id": "s0", "memory_session_ids": ["s1", "s2"]},
                {"deleted_session_id": "s1", "memory_session_ids": ["s0", "s2"]},
            ],
        }
        values = state_arms(state)
        self.assertEqual([row["arm_id"] for row in values[:5]],
                         [f"fixed:{policy}" for policy in FIXED_POLICIES])
        self.assertEqual([row["deleted_session_id"] for row in values[5:]], ["s0", "s1"])
        self.assertEqual([row["deletion_index"] for row in values[5:]], [0, 1])

    def test_oracle_uses_accuracy_then_nll_then_retain_then_chronology(self):
        fixed = [arm(f"fixed:{policy}", correct=(0, 1), nll=(1, 1))
                 for policy in FIXED_POLICIES]
        row = {"arms": fixed + [
            arm("delete_one:s0", correct=(1, 1), nll=(.4, .4),
                kind="delete_one", deletion_index=0, deleted_session_id="s0"),
            arm("delete_one:s1", correct=(1, 1), nll=(.2, .2),
                kind="delete_one", deletion_index=1, deleted_session_id="s1"),
        ]}
        chosen, _ = select_oracle(row)
        self.assertEqual(chosen["arm_id"], "delete_one:s1")

        row["arms"][1]["queries"] = queries((1, 1), (.2, .2))  # retain_all ties exactly
        chosen, _ = select_oracle(row)
        self.assertEqual(chosen["arm_id"], "fixed:retain_all")

        row["arms"][1]["queries"] = queries((0, 1), (1, 1))
        row["arms"][-2]["queries"] = queries((1, 1), (.2, .2))
        chosen, _ = select_oracle(row)
        self.assertEqual(chosen["arm_id"], "delete_one:s0")

    def test_outer_fold_fixed_selection_uses_only_other_folds(self):
        rows = []
        for fold in range(5):
            values = []
            for policy in FIXED_POLICIES:
                good = policy == "last_2"
                values.append(arm(f"fixed:{policy}", correct=(int(good), int(good)),
                                  nll=(.1 if good else 1, .1 if good else 1)))
            rows.append({"fold": fold, "arms": values})
        selected, evidence = select_fixed_by_outer_fold(rows)
        self.assertEqual(set(selected.values()), {"last_2"})
        self.assertEqual(evidence["0"]["last_2"]["queries"], 8)

    def test_subject_cluster_interval_is_deterministic(self):
        records = [
            {"subject": "a", "gain": 1.0}, {"subject": "a", "gain": 0.0},
            {"subject": "b", "gain": .5}, {"subject": "b", "gain": .5},
        ]
        first = clustered_interval(records, "gain", draws=100, seed=7)
        second = clustered_interval(records, "gain", draws=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], .5)

    def test_resource_summary_does_not_gate_on_descriptive_headroom(self):
        rows = []
        for index in range(40):
            fixed = [arm(f"fixed:{policy}", correct=(0, 1), nll=(1, 1))
                     for policy in FIXED_POLICIES]
            deletion = arm(f"delete_one:s{index}", correct=(1, 1), nll=(.1, .1),
                           kind="delete_one", deletion_index=0,
                           deleted_session_id=f"s{index}")
            rows.append({"episode_id": f"e{index}", "subject": f"subject{index}",
                         "fold": index % 5, "arms": fixed + [deletion]})
        resources = [
            {"unique_memory_states_scored": 120, "seconds": 10.0},
            {"unique_memory_states_scored": 120, "seconds": 11.0},
        ]
        result = summarize_rows(rows, resources, formal_unique_states=240)
        self.assertTrue(result["pilot_gate"]["passed"])
        self.assertTrue(result["pilot_headroom"]["descriptive_only_not_a_formal_gate"])
        self.assertEqual(result["pilot_headroom"]["oracle_primary_accuracy"], 1.0)
        self.assertEqual(result["pilot_headroom"]["selected_fixed_primary_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
