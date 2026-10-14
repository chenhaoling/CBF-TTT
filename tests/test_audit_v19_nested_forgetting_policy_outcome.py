import unittest

from scripts.audit_v19_nested_forgetting_policy_outcome import (
    expected_shard_state_counts,
)
from tasks.train_v19_nested_forgetting_policy import SEEDS


def state(identifier, memories):
    return {"episode_id": identifier, "per_seed": {
        str(seed): {"retained_session_ids": list(memory), "deleted_session_ids": []}
        for seed, memory in zip(SEEDS, memories)}}


class AuditV19NestedPolicyOutcomeTest(unittest.TestCase):
    def test_shard_counts_follow_episode_order_not_state_file_order(self):
        episodes = [{"episode_id": value} for value in ("e0", "e1", "e2", "e3")]
        states = [
            state("e3", [("a",), ("a",), ("b",)]),
            state("e0", [("a",), ("a",), ("a",)]),
            state("e2", [("a",), ("b",), ("c",)]),
            state("e1", [("a",), ("b",), ("b",)]),
        ]
        index = {row["episode_id"]: row for row in states}
        unique, reused = expected_shard_state_counts(episodes, index)
        self.assertEqual(unique, {0: 4, 1: 4})
        self.assertEqual(reused, {0: 2, 1: 2})
        reverse_index = {row["episode_id"]: row for row in reversed(states)}
        self.assertEqual(expected_shard_state_counts(episodes, reverse_index),
                         (unique, reused))


if __name__ == "__main__":
    unittest.main()
