import unittest

from tasks.train_qwen_natural_forgetting_factor_gate_v2 import ENGINEERED, NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    classification_metrics,
    reconstruct_trace,
)


def decision(step, incoming, action, source, top, before, features=None, kind="update"):
    return {
        "step": step, "incoming_chain_id": incoming, "action": action,
        "desired_source_action": source, "top": top, "before_size": before,
        "after_write_size": before + (0 if action == "scope_top1_a0" else 1),
        "features": features or [float(index) for index in range(len(ENGINEERED))],
        "event_kind": kind,
    }


class SafeDeleteV5Test(unittest.TestCase):
    def test_reconstructs_physical_target_and_deduplicates(self):
        events = [
            decision(0, "a", "retain_all", "retain", None, 0, kind="initial"),
            decision(1, "a", "scope_top1_a0", "local", 0, 1),
            decision(2, "b", "retain_all", "retain", 0, 1, kind="hard_retain"),
        ]
        trace = {"session_id": "fixture", "outer_fold": None,
                 "decisions": {"arm_a": events, "arm_b": events}}
        from tasks.train_qwen_natural_forgetting_safe_delete_v5 import fold_for
        trace["outer_fold"] = fold_for("fixture")
        rows = reconstruct_trace(trace)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["safe_delete"] for row in rows], [0, 1, 0])
        self.assertTrue(all(row["origin_arms"] == ["arm_a", "arm_b"] for row in rows))
        self.assertEqual(rows[1]["memory"], [["a", 0]])
        self.assertEqual(rows[2]["memory"], [["a", 1]])

    def test_local_wrong_top_is_unsafe(self):
        events = [
            decision(0, "a", "retain_all", "retain", None, 0),
            decision(1, "b", "retain_all", "retain", 0, 1),
            decision(2, "a", "retain_all", "local", 1, 2),
        ]
        from tasks.train_qwen_natural_forgetting_safe_delete_v5 import fold_for
        trace = {"session_id": "wrong-top", "outer_fold": fold_for("wrong-top"),
                 "decisions": {"retain": events}}
        rows = reconstruct_trace(trace)
        self.assertEqual(rows[-1]["safe_delete"], 0)

    def test_only_registered_no_schedule_features_enter_model(self):
        self.assertEqual(len(NO_SCHEDULE), 10)
        self.assertFalse({"memory_size", "event_position", "top_recency"} & set(NO_SCHEDULE))
        self.assertFalse({"event_kind", "desired_source_action", "chain_id"} & set(NO_SCHEDULE))

    def test_classification_metrics(self):
        value = classification_metrics([1, 1, 0, 0], [.9, .4, .8, .1], .5, [0, 0, 1, 1])
        self.assertEqual(value["true_positives"], 1)
        self.assertEqual(value["false_positives"], 1)
        self.assertEqual(value["false_negatives"], 1)
        self.assertEqual(value["true_negatives"], 1)
        self.assertEqual(value["precision"], .5)
        self.assertEqual(value["recall"], .5)


if __name__ == "__main__":
    unittest.main()
