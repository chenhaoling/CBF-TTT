import math
import unittest

from tasks.train_mquake_jackknife_consensus_v15 import (
    PROTOCOL,
    QUORUM,
    empirical_threshold,
    gate_seed,
    replay_consensus,
)


class FakeEvidence:
    @staticmethod
    def prepare_episode(_episode):
        return {}


class FakeResolver:
    @staticmethod
    def resolve(_memory, _event, step, _length, _episode_scores):
        return 0, step, None


class FakeEnsemble:
    @staticmethod
    def score(step):
        return {1: 0.0, 2: 10.0}[step]


class JackknifeConsensusTest(unittest.TestCase):
    def test_protocol_and_seed_are_stable(self):
        self.assertEqual(PROTOCOL, "mquake_jackknife_consensus_v15")
        self.assertEqual(gate_seed(2, 4, 1), 44401)

    def test_empirical_threshold_obeys_upper_tail_budget(self):
        values = list(range(1000))
        threshold = empirical_threshold(values, .001)
        self.assertLessEqual(sum(value >= threshold for value in values), 1)
        self.assertTrue(math.isfinite(threshold))

    def test_empirical_threshold_above_max_when_budget_rounds_to_zero(self):
        threshold = empirical_threshold([1.0, 2.0], .0005)
        self.assertGreater(threshold, 2.0)

    def test_three_of_four_consensus_uses_only_prior_scores(self):
        episode = {
            "episode_id": "e", "primary_subject_hash": "s", "length": 3,
            "sessions": [
                {"session_id": "old", "event_kind": "primary_old",
                 "desired": "retain", "target_session_id": None},
                {"session_id": "d", "event_kind": "distractor",
                 "desired": "retain", "target_session_id": None},
                {"session_id": "new", "event_kind": "primary_update",
                 "desired": "local", "target_session_id": "old"},
            ],
        }
        voters = [(FakeEnsemble(), 5.0) for _ in range(4)]
        record = replay_consensus(
            episode, FakeEvidence(), FakeResolver(), voters
        )
        self.assertEqual(record["pair_state_correct"], 1)
        self.assertEqual(record["decisions"][0]["votes"], [])
        self.assertEqual(record["decisions"][1]["history_sizes"], [0] * 4)
        self.assertEqual(record["decisions"][1]["vote_count"], 0)
        self.assertEqual(record["decisions"][2]["history_sizes"], [1] * 4)
        self.assertEqual(record["decisions"][2]["vote_count"], 4)
        self.assertEqual(record["decisions"][2]["deleted"], "old")
        self.assertEqual(record["decisions"][2]["quorum"], QUORUM)


if __name__ == "__main__":
    unittest.main()
