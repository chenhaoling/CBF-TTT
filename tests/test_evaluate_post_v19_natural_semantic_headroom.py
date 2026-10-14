import unittest

from tasks.evaluate_post_v19_natural_semantic_headroom import aggregate
from scripts.audit_post_v19_natural_semantic_outcome import expected_state_counts


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def queries(correct):
    return [{"kind": kind, "correct": int(correct), "answer_nll": 1.0,
             "generation": {"normalized_exact": int(correct)}} for kind in KINDS]


def arm(identifier, memory, correct, source=None):
    return {"arm_id": identifier, "memory_session_ids": memory,
            "score_source_arm_id": source or identifier, "queries": queries(correct)}


def row(index):
    fold = index % 5
    retain = [f"m{index}", f"c{index}"]
    semantic = [f"c{index}"]
    arms = [
        arm("fixed:no_memory", [], False),
        arm("fixed:retain_all", retain, False),
        arm("fixed:global_reset", semantic, True),
        arm("fixed:last_1", semantic, True, "fixed:global_reset"),
        arm("fixed:last_2", retain, False, "fixed:retain_all"),
        arm("special:exact_conflict", retain, False, "fixed:retain_all"),
        arm("special:semantic_oracle", semantic, True, "fixed:global_reset"),
    ]
    return {"episode_id": f"e{index}", "subject": f"s{index}",
            "relation_id": f"r{index % 10}", "source": "counterfact",
            "fold": fold, "mechanism": "single_conflict", "length": 8,
            "arms": arms}


class NaturalSemanticEvaluationTest(unittest.TestCase):
    def test_pilot_aggregation_passes_known_positive_fixture(self):
        result = aggregate([row(index) for index in range(80)], "pilot")
        self.assertTrue(result["pilot_gate"]["passed"])
        self.assertEqual(result["arm_metrics"]["semantic_oracle"]["joint_success"], 1.0)
        self.assertEqual(result["arm_metrics"]["exact_surface"]["joint_success"], 0.0)

    def test_expected_counts_follow_episode_order(self):
        episodes = [{"episode_id": "e0"}, {"episode_id": "e1"}]
        states = [
            {"episode_id": "e1", "fixed": {"no_memory": [], "retain_all": ["a"],
             "global_reset": ["b"], "last_1": ["b"], "last_2": ["a"]},
             "special": {"exact_conflict": ["a"], "semantic_oracle": ["b"]},
             "exact_removed": [], "semantic_removed": ["a"]},
            {"episode_id": "e0", "fixed": {"no_memory": [], "retain_all": ["x"],
             "global_reset": ["y"], "last_1": ["y"], "last_2": ["x"]},
             "special": {"exact_conflict": ["x"], "semantic_oracle": ["y"]},
             "exact_removed": [], "semantic_removed": ["x"]},
        ]
        unique, reused = expected_state_counts(episodes, states)
        self.assertEqual(unique, {0: 3, 1: 3})
        self.assertEqual(reused, {0: 4, 1: 4})


if __name__ == "__main__":
    unittest.main()
