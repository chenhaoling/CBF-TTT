import itertools
import unittest

from tasks.evaluate_qwen_utility_record_controller_v2_templama_development import (
    majority_actions,
    metric,
    random_matched_actions,
)


def action(name, values, correct=4, unrelated=2):
    roles = {
        role: {"queries": 1, "correct": int(index < correct),
               "total_answer_nll": 1.0}
        for index, role in enumerate((
            "current_exact", "current_rephrase", "retained_1",
            "retained_2", "anchor",
        ))
    }
    return {
        "spec": {"name": name, "kind": "grid", "alphas": values},
        "queries": 5,
        "correct": correct,
        "total_answer_nll": 5.0,
        "unrelated_queries": 3,
        "unrelated_correct": unrelated,
        "mean_memory_mass": sum(values),
        "selected_null": 0,
        "per_query_role": roles,
    }


def row():
    actions = {}
    for values in itertools.product((0, 1), repeat=3):
        name = "grid_" + "_".join(map(str, values))
        actions[name] = action(name, list(values))
    actions["global_clear"] = {
        **action("global_clear", [], correct=1, unrelated=1),
        "spec": {"name": "global_clear", "kind": "global_clear", "alphas": []},
    }
    return {
        "episode_id": "e0", "source_group": "g0", "boundary_index": 0,
        "candidate_count": 3, "retain_action": "grid_1_1_1",
        "safe_oracle_action": "grid_0_1_1", "mechanism": "semantic_overwrite",
        "target_relation": "P1", "actions": actions,
    }


class TestEvaluateUtilityRecordControllerV2TempLamaDevelopment(unittest.TestCase):
    def test_random_control_matches_deletion_count(self):
        value = row(); identity = ("e0", 0)
        reference = {identity: "grid_0_1_0"}
        for draw in range(20):
            selected = random_matched_actions([value], reference, draw)[identity]
            self.assertEqual(
                value["actions"][selected]["spec"]["alphas"].count(0), 2
            )

    def test_majority_votes_per_record(self):
        value = row(); identity = ("e0", 0)
        actions = {
            1: {identity: "grid_0_1_1"},
            2: {identity: "grid_0_0_1"},
            3: {identity: "grid_1_0_1"},
        }
        self.assertEqual(majority_actions([value], actions)[identity], "grid_0_0_1")

    def test_metric_reports_relation_mechanism_and_roles(self):
        value = row(); identity = ("e0", 0)
        features = {identity: {
            "targets": {"delete_targets": [1, 0, 0]},
        }}
        result = metric([value], {identity: "grid_0_1_1"}, features)
        self.assertEqual(result["accuracy"], .8)
        self.assertEqual(result["deletion_precision"], 1.0)
        self.assertEqual(result["per_relation_accuracy"], {"P1": .8})
        self.assertEqual(
            set(result["per_query_role"]),
            {"current_exact", "current_rephrase", "retained_1", "retained_2", "anchor"},
        )


if __name__ == "__main__":
    unittest.main()
