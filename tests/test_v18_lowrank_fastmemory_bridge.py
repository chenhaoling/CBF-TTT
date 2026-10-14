import unittest

from scripts.audit_v18_lowrank_fastmemory_bridge import recompute_table
from tasks.build_controlled_interference_v18 import FIXED_POLICIES
from tasks.evaluate_v18_lowrank_fastmemory_bridge import (
    SCALES, aggregate, query_key_id, render_query,
)


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=None):
    if nll is None:
        nll = .1 if correct else 1.0
    return {"kind": kind, "by_scale": {str(scale): {
        "correct": correct, "answer_nll": nll, "scores": {}, "prediction": "x"}
        for scale in SCALES},
        "generation_primary": {"normalized_exact": correct}}


def arm(arm_id, update, retain, kind="fixed", **extra):
    values = (update, update, retain, retain, 1)
    return {"arm_id": arm_id, "kind": kind, "memory_session_ids": [],
            "score_source_arm_id": arm_id,
            "queries": [query(name, value) for name, value in zip(KINDS, values)], **extra}


def control(kind, correct, wrong=0, zero=0):
    return {"kind": kind, "correct": query(kind, correct),
            "wrong": query(kind, wrong), "zero": query(kind, zero)}


def rows(content_correct=1):
    output = []
    for index in range(320):
        subject_index, within = divmod(index, 4); template_index = subject_index % 20
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
        output.append({"episode_id": f"e{index}", "subject": f"s{subject_index}",
                       "template_group": f"t{template_index}", "fold": template_index % 5,
                       "length": (8, 16, 32, 64)[within],
                       "mechanism": ("clean_update", "stale_repetition",
                                     "irrelevant_near_duplicate", "useful_repetition")[within],
                       "arms": fixed + [delete],
                       "content_controls": [control(kind, content_correct) for kind in KINDS[:4]]})
    return output


class LowrankFastMemoryBridgeTest(unittest.TestCase):
    def test_query_key_mapping_and_fresh_prompt(self):
        episode = {"keys": ["k0", "k1", "k2"]}
        self.assertEqual(query_key_id(episode, "update_old_surface"), "k0")
        self.assertEqual(query_key_id(episode, "retain_2"), "k2")
        self.assertIsNone(query_key_id(episode, "locality"))
        prompt = render_query("question")
        self.assertIn("question", prompt)
        self.assertNotIn("Session notes", prompt)

    def test_bridge_gate_and_independent_recomputation(self):
        values = rows()
        result = aggregate(values)
        self.assertEqual(result, recompute_table(values))
        self.assertTrue(result["bridge_gate"]["passed"])
        self.assertEqual(result["scales"]["8.0"]["joint_gain_micro"], .25)
        self.assertEqual(result["content_binding"]["correct"]["accuracy"], 1.0)
        self.assertEqual(result["content_binding"]["wrong"]["accuracy"], 0.0)

    def test_content_binding_failure_blocks_bridge(self):
        result = aggregate(rows(content_correct=0))
        self.assertFalse(result["bridge_gate"]["passed"])
        self.assertFalse(result["bridge_gate"]["checks"][
            "correct_component_accuracy_at_least_90"])


if __name__ == "__main__":
    unittest.main()
