import unittest

from tasks.build_v19_nested_policy_examples import (
    FEATURE_NAMES, common_prefix, component_features, episode_examples,
    trigram_jaccard, utility_target,
)
from scripts.audit_v19_nested_policy_examples import recompute_episode


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=.2):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "generation": {"normalized_exact": correct}}


def arm(arm_id, joint, locality=1, kind="fixed", **extra):
    values = (joint, joint, 1, 1, locality)
    return {"arm_id": arm_id, "kind": kind, **extra,
            "queries": [query(name, value, .1 if joint else .3)
                        for name, value in zip(KINDS, values)]}


def fixture():
    sessions = [
        {"session_id": "h0", "session_index": 0, "key_id": "QAAA",
         "value": "OLD", "role": "initial"},
        {"session_id": "h1", "session_index": 1, "key_id": "QBBB",
         "value": "KEEP", "role": "initial"},
        {"session_id": "h2", "session_index": 2, "key_id": "QBBBX",
         "value": "KEEP2", "role": "near_duplicate"},
        {"session_id": "current", "session_index": 3, "key_id": "QAAA",
         "value": "NEW", "role": "current_update"},
    ]
    episode = {"episode_id": "e", "subject": "s", "template_group": "t",
               "fold": 0, "current_session_id": "current", "sessions": sessions}
    arms = [arm("fixed:retain_all", 0, policy="retain_all")]
    for session in sessions[:-1]:
        helpful = session["session_id"] == "h0"
        arms.append(arm(f"delete_one:{session['session_id']}", int(helpful),
                        kind="delete_one", deleted_session_id=session["session_id"]))
    for key in ("QAAA", "QBBB", "QBBBX"):
        arms.append(arm(f"delete_key:{key}", int(key == "QAAA"), kind="delete_key",
                        deleted_key_id=key))
    outcome = {"episode_id": "e", "arms": arms}
    return episode, outcome


class BuildV19NestedPolicyExamplesTest(unittest.TestCase):
    def test_string_features_distinguish_exact_near_and_unrelated(self):
        self.assertEqual(trigram_jaccard("QAAA", "QAAA"), 1)
        self.assertGreater(trigram_jaccard("QBBB", "QBBBX"),
                           trigram_jaccard("QBBB", "QAAA"))
        self.assertGreater(common_prefix("QBBB", "QBBBX"), .75)

    def test_feature_vector_is_finite_and_write_time_only(self):
        episode, _ = fixture()
        features = component_features(episode, 0)
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(features[FEATURE_NAMES.index("newest_exact_key")], 1)
        self.assertEqual(features[FEATURE_NAMES.index("newest_value_conflict")], 1)

    def test_labels_reward_stale_deletion_and_conserve_unrelated(self):
        episode, outcome = fixture()
        rows = episode_examples(episode, outcome)
        index = {row["component_session_id"]: row for row in rows}
        self.assertEqual(index["h0"]["target"], 1)
        self.assertEqual(index["h0"]["exact_conflict"], 1)
        self.assertEqual(index["h1"]["target"], .5)
        self.assertEqual(index["h1"]["conservation_target"], 1)
        self.assertNotIn("current", index)

    def test_independent_auditor_reconstructs_examples(self):
        episode, outcome = fixture()
        self.assertEqual(episode_examples(episode, outcome),
                         recompute_episode(episode, outcome))

    def test_utility_target_uses_joint_success_only(self):
        retain = {"joint_success": 0, "joint_answer_nll": .1}
        self.assertEqual(utility_target(retain,
                                       {"joint_success": 0, "joint_answer_nll": 9}), .5)


if __name__ == "__main__":
    unittest.main()
