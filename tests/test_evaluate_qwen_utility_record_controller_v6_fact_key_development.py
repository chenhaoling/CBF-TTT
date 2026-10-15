import unittest

from tasks.evaluate_qwen_utility_record_controller_v6_fact_key_development import (
    fact_key_matches,
    fact_key_projected_actions,
    safe_fact_key_oracle_actions,
)
from tests.test_train_qwen_utility_record_controller_v5_residual import fixture


def v6_fixture():
    label, feature = fixture()
    label["event_index"] = 2
    label["candidate_record_ids"] = ["r0", "r1"]

    def record(identity, relation, key):
        return {
            "record_id": identity,
            "audit": {
                "semantic_key": key,
                "relation_id": relation,
                "subject_norm": "shared subject",
            },
        }

    episode = {
        "episode_id": "e",
        "events": [
            {"record": record("r0", "P1", "fact-p1")},
            {"record": record("r1", "P2", "fact-p2")},
            {"record": record("incoming", "P1", "fact-p1")},
        ],
    }
    return label, feature, episode


class TestEvaluateUtilityRecordControllerV6FactKeyDevelopment(unittest.TestCase):
    def test_fact_key_requires_subject_and_relation(self):
        label, _, episode = v6_fixture()
        matches = fact_key_matches([label], [episode])
        self.assertTrue(matches[("e", 0, 0)]["same_fact_key"])
        self.assertFalse(matches[("e", 0, 1)]["same_fact_key"])

    def test_projection_rejects_cross_relation_deletion(self):
        label, feature, episode = v6_fixture()
        matches = fact_key_matches([label], [episode])
        projected = fact_key_projected_actions(
            [label], [feature], {("e", 0): "grid_1_0"}, matches
        )
        self.assertEqual(projected[("e", 0)], "grid_1_1")
        kept = fact_key_projected_actions(
            [label], [feature], {("e", 0): "grid_0_1"}, matches
        )
        self.assertEqual(kept[("e", 0)], "grid_0_1")

    def test_fact_key_oracle_keeps_helpful_same_fact_deletion(self):
        label, feature, episode = v6_fixture()
        matches = fact_key_matches([label], [episode])
        actions = safe_fact_key_oracle_actions([label], [feature], matches)
        self.assertEqual(actions[("e", 0)], "grid_0_1")


if __name__ == "__main__":
    unittest.main()
