import unittest

from tasks.train_long_session_sparse_gate_v1 import (
    aggregate,
    matched_random_schedules,
    replay_episode,
)


class FakeEvidence:
    def prepare_episode(self, episode): return None

    def top(self, memory, event, step, length, episode_scores):
        target = event.get("target_session_id")
        index = next((i for i, row in enumerate(memory) if row["session_id"] == target), 0)
        return index, [0.] * 23, {"score": 1.}


class FixedController:
    threshold = .5
    def probability(self, feature): return 1.


def episode():
    return {"episode_id": "e", "primary_subject_hash": "s", "length": 4,
            "sessions": [
                {"session_id": "old", "units": ["a"], "desired": "retain",
                 "target_session_id": None, "event_kind": "primary_old"},
                {"session_id": "x", "units": ["b"], "desired": "retain",
                 "target_session_id": None, "event_kind": "hard_retain"},
                {"session_id": "new", "units": ["c"], "desired": "local",
                 "target_session_id": "old", "event_kind": "primary_update"},
                {"session_id": "y", "units": ["d"], "desired": "retain",
                 "target_session_id": None, "event_kind": "distractor"}]}


class SparseGateTest(unittest.TestCase):
    def test_matched_random_schedule_has_exact_count_and_is_deterministic(self):
        episodes = [episode(), {**episode(), "episode_id": "e2"}]
        first = matched_random_schedules(episodes, 3, 17)
        second = matched_random_schedules(episodes, 3, 17)
        self.assertEqual(first, second)
        self.assertEqual(sum(map(sum, first.values())), 3)
        self.assertFalse(any(row[0] for row in first.values()))

    def test_oracle_replay_keeps_current_and_deletes_stale(self):
        row = replay_episode(episode(), FakeEvidence(), "oracle")
        self.assertEqual(row["pair_state_correct"], 1)
        self.assertEqual(row["current_retained"], 1)
        self.assertEqual(row["stale_deleted"], 1)

    def test_always_delete_is_unsafe(self):
        row = replay_episode(episode(), FakeEvidence(), "controller", FixedController())
        self.assertLess(row["non_target_retention"], 1.)
        metric = aggregate([row])
        self.assertEqual(metric["by_event_kind"]["primary_update"]["events"], 1)


if __name__ == "__main__": unittest.main()
