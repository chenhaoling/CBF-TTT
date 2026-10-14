import unittest

from tasks.build_v18_causal_deletion_examples import (
    FEATURE_NAMES, component_features, episode_examples, utility_target,
)
from tasks.evaluate_v18_lowrank_fastmemory_bridge import SCALES


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll):
    return {"kind": kind, "by_scale": {str(scale): {
        "correct": correct, "answer_nll": nll, "scores": {}, "prediction": "x"}
        for scale in SCALES}, "generation_primary": {"normalized_exact": correct}}


def arm(arm_id, joint, nll, kind="fixed", **extra):
    correct = (joint, joint, 1, 1, 1)
    return {"arm_id": arm_id, "kind": kind,
            "queries": [query(name, value, nll) for name, value in zip(KINDS, correct)],
            **extra}


def fixture():
    sessions = [
        {"session_id": "s0", "key_id": "a", "value": "old"},
        {"session_id": "s1", "key_id": "b", "value": "keep"},
        {"session_id": "s2", "key_id": "a", "value": "old"},
        {"session_id": "s3", "key_id": "a", "value": "new"},
    ]
    episode = {"episode_id": "e", "subject": "subject", "template_group": "template",
               "fold": 0, "sessions": sessions, "current_session_id": "s3"}
    arms = [arm("fixed:retain_all", 0, 1.0)]
    for session in sessions[:-1]:
        # Single deletions do not solve the stale-repetition interaction.
        arms.append(arm(f"delete_one:{session['session_id']}", 0, .9,
                        kind="delete_one", deleted_session_id=session["session_id"]))
    arms.extend((arm("delete_key:a", 1, .1, kind="delete_key", deleted_key_id="a"),
                 arm("delete_key:b", 0, 1.2, kind="delete_key", deleted_key_id="b")))
    bridge = {"arms": arms}
    physical = {"key_similarities": {"a||a": 1.0, "a||b": .2, "b||b": 1.0}}
    return episode, bridge, physical


class BuildV18CausalDeletionExamplesTest(unittest.TestCase):
    def test_three_point_utility(self):
        retain = {"joint_success": 0}; better = {"joint_success": 1}
        same = {"joint_success": 0}; worse_retain = {"joint_success": 1}
        self.assertEqual(utility_target(retain, better), 1.0)
        self.assertEqual(utility_target(retain, same), .5)
        self.assertEqual(utility_target(worse_retain, same), 0.0)

    def test_write_time_features_exclude_query_and_absolute_length(self):
        episode, _, physical = fixture()
        values = component_features(episode, physical["key_similarities"], 0)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertNotIn("query", " ".join(FEATURE_NAMES))
        self.assertNotIn("length", FEATURE_NAMES)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[3], 1.0)

    def test_group_counterfactual_credits_all_stale_components(self):
        episode, bridge, physical = fixture()
        rows = episode_examples(episode, bridge, physical)
        by_id = {row["component_session_id"]: row for row in rows}
        self.assertEqual(by_id["s0"]["single_target"], .5)
        self.assertEqual(by_id["s0"]["group_target"], 1.0)
        self.assertEqual(by_id["s0"]["target"], 1.0)
        self.assertEqual(by_id["s2"]["target"], 1.0)
        self.assertEqual(by_id["s1"]["target"], .5)
        self.assertEqual(by_id["s1"]["conservation_target"], 1)
        self.assertTrue(all(row["current_session_id"] == "s3" for row in rows))


if __name__ == "__main__":
    unittest.main()
