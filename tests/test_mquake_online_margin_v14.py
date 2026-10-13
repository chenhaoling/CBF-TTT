import unittest

from scripts.audit_mquake_online_margin_v14 import selected_threshold
from tasks.train_mquake_online_margin_v14 import (
    grouped_indices,
    replay_online,
    relative_margin,
    select_threshold_from_rows,
)


def threshold_row(threshold, pair, lcb, feasible, macro=None):
    return {"threshold": threshold,
            "metric": {"pair_state_accuracy": pair, "stale_deletion": pair},
            "cluster": {"pair_macro": pair if macro is None else macro,
                        "safety_lcb95": lcb, "safety_macro": lcb},
            "feasible": feasible}


class OnlineMarginV14Test(unittest.TestCase):
    def test_relative_margin_is_additive_shift_invariant_and_causal(self):
        self.assertIsNone(relative_margin(3., []))
        self.assertEqual(relative_margin(4., [1., 2., 3.]), 2.)
        self.assertEqual(relative_margin(14., [11., 12., 13.]), 2.)

    def test_grouping_requires_one_positive_and_preserves_steps(self):
        rows = [
            {"episode_id": "b", "subject": "s", "step": 2, "label": 1.},
            {"episode_id": "b", "subject": "s", "step": 1, "label": 0.},
            {"episode_id": "a", "subject": "t", "step": 1, "label": 1.},
        ]
        groups = grouped_indices(rows)
        self.assertEqual([row["episode_id"] for row in groups], ["a", "b"])
        self.assertEqual(groups[1]["indices"], [1, 0])
        self.assertEqual(groups[1]["target"], 1)

    def test_registered_selection_prioritizes_subject_macro_pair(self):
        rows = [threshold_row(1., .95, .96, False, .95),
                threshold_row(2., .91, .98, True, .88),
                threshold_row(3., .90, .99, True, .92)]
        chosen, feasible = select_threshold_from_rows(rows)
        self.assertTrue(feasible)
        self.assertEqual(chosen["threshold"], 3.)
        grid = {str(row["threshold"]): {key: row[key]
                for key in ("metric", "cluster", "feasible")} for row in rows}
        self.assertEqual(selected_threshold(grid), (3., True))

    def test_online_replay_uses_only_prior_scores_and_resets_history(self):
        class Evidence:
            @staticmethod
            def prepare_episode(episode):
                return None

        class Resolver:
            @staticmethod
            def resolve(memory, incoming, step, length, scores):
                del memory, step, length, scores
                return 0, incoming["test_score"], {}

        class Ensemble:
            @staticmethod
            def score(feature):
                return feature

        sessions = [
            {"session_id": "old", "event_kind": "primary_old", "desired": "retain",
             "target_session_id": None, "test_score": 0.},
            {"session_id": "d1", "event_kind": "distractor", "desired": "retain",
             "target_session_id": None, "test_score": 1.},
            {"session_id": "new", "event_kind": "primary_update", "desired": "local",
             "target_session_id": "old", "test_score": 3.},
            {"session_id": "d2", "event_kind": "distractor", "desired": "retain",
             "target_session_id": None, "test_score": 1.},
        ]
        episode = {"episode_id": "e", "primary_subject_hash": "s", "length": 4,
                   "sessions": sessions}
        result = replay_online(episode, Evidence(), Resolver(), Ensemble(), .5)
        self.assertEqual([row["history_size"] for row in result["decisions"]],
                         [0, 0, 1, 2])
        self.assertIsNone(result["decisions"][1]["deleted"])
        self.assertEqual(result["decisions"][2]["deleted"], "old")
        self.assertEqual(result["pair_state_correct"], 1)

    def test_torch_listwise_fit_ranks_simple_positive(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        from tasks.train_mquake_online_margin_v14 import (
            ListwiseMember,
            fit_listwise_gate,
        )
        rows = []
        for subject in ("a", "b"):
            for episode in range(2):
                identifier = f"{subject}-{episode}"
                rows.extend([
                    {"episode_id": identifier, "subject": subject, "step": 1,
                     "label": 0., "feature": torch.tensor([0., 0.])},
                    {"episode_id": identifier, "subject": subject, "step": 2,
                     "label": 1., "feature": torch.tensor([2., 1.])},
                    {"episode_id": identifier, "subject": subject, "step": 3,
                     "label": 0., "feature": torch.tensor([0., 1.])},
                ])
        artifact = fit_listwise_gate(rows, 7, epoch=20)
        member = ListwiseMember(artifact)
        self.assertGreater(member.score(torch.tensor([2., 1.])),
                           member.score(torch.tensor([0., 0.])))


if __name__ == "__main__":
    unittest.main()
