import json
from pathlib import Path
import unittest

from tasks.evaluate_v18_physical_conflict_rules import (
    COSINE_THRESHOLDS, RANDOM_SEEDS, aggregate, exact_conflict_ids, matched_random_ids,
    most_recent_same_key_ids, physical_states, select_physical_rule,
)
from tasks.evaluate_v18_lowrank_fastmemory_bridge import SCALES


ROOT = Path(__file__).parents[1]
FORMAL = (ROOT / "experiments/cbf_ttt/qwen3_4b_final_1b_20260927"
          / "memory_interference_v18_formal_data/result/formal.jsonl")


def toy_episode():
    sessions = [
        {"session_id": "s0", "key_id": "a", "value": "old"},
        {"session_id": "s1", "key_id": "b", "value": "keep"},
        {"session_id": "s2", "key_id": "b", "value": "keep"},
        {"session_id": "s3", "key_id": "a", "value": "new"},
    ]
    return {"episode_id": "e0", "sessions": sessions, "current_session_id": "s3"}


def metric(joint, update=1.0, retain=1.0, locality=1.0, nll=.1):
    return {"joint_success": joint, "update_accuracy": update,
            "retained_key_accuracy": retain, "locality_accuracy": locality,
            "joint_answer_nll": nll, "generation_joint_success": joint}


def scored_query(kind, correct):
    return {"kind": kind, "by_scale": {str(scale): {
        "correct": correct, "answer_nll": .1 if correct else 1.0,
        "scores": {}, "prediction": "x"} for scale in SCALES},
        "generation_primary": {"normalized_exact": correct}}


def scored_arm(name, correct):
    kinds = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
             "locality")
    return {"arm_name": name, "queries": [scored_query(kind, correct) for kind in kinds]}


class V18PhysicalRulesTest(unittest.TestCase):
    def test_exact_conflict_preserves_agreeing_repetition(self):
        episode = toy_episode()
        self.assertEqual(exact_conflict_ids(episode), ["s1", "s2", "s3"])
        self.assertEqual(most_recent_same_key_ids(episode), ["s2", "s3"])

    def test_matched_random_is_stable_count_matched_and_keeps_current(self):
        episode = toy_episode(); reference = ["s1", "s2", "s3"]
        first = matched_random_ids(episode, reference, RANDOM_SEEDS[0], "exact")
        second = matched_random_ids(episode, reference, RANDOM_SEEDS[0], "exact")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(reference))
        self.assertIn("s3", first)

    def test_all_states_keep_current_and_have_registered_controls(self):
        episode = toy_episode()
        similarities = {("a", "a"): 1.0, ("b", "b"): 1.0,
                        ("a", "b"): .5}
        states = physical_states(episode, similarities)
        expected = 3 + len(COSINE_THRESHOLDS) + (2 + len(COSINE_THRESHOLDS)) * len(RANDOM_SEEDS)
        self.assertEqual(len(states), expected)
        self.assertTrue(all("s3" in values for values in states.values()))

    def test_outer_training_selects_threshold_and_family(self):
        rows = []
        for _ in range(4):
            table = {"exact_conflict": metric(.75), "most_recent_same_key": metric(.5)}
            for threshold in COSINE_THRESHOLDS:
                table[f"qwen_cosine_conflict:{threshold}"] = metric(
                    1.0 if threshold == .95 else .6)
            rows.append({"metrics": table})
        selected = select_physical_rule(rows)
        self.assertEqual(selected["family"], "qwen_cosine_conflict")
        self.assertEqual(selected["cosine_threshold"], .95)

    def test_aggregate_uses_outer_fold_rule_and_matched_random(self):
        rows = []
        for index in range(10):
            arms = [scored_arm("exact_conflict", 1),
                    scored_arm("most_recent_same_key", 0)]
            arms.extend(scored_arm(f"qwen_cosine_conflict:{threshold}", 0)
                        for threshold in COSINE_THRESHOLDS)
            arms.extend(scored_arm(f"matched_random:exact_conflict:{seed}", 0)
                        for seed in RANDOM_SEEDS)
            rows.append({"episode_id": f"e{index}", "subject": f"s{index // 2}",
                         "template_group": f"t{index % 5}", "fold": index % 5,
                         "length": 8, "mechanism": "clean_update", "arms": arms,
                         "bridge_reference": {"fixed": metric(0, 0, 1),
                                              "oracle": metric(1)}})
        result = aggregate(rows)
        self.assertEqual(result["strongest_physical"]["joint_success"], 1.0)
        self.assertEqual(result["matched_random"]["joint_success"], 0.0)
        self.assertEqual(result["joint_gain_micro"], 1.0)
        self.assertEqual(result["fixed_to_oracle_gap_closed"], 1.0)
        self.assertTrue(all(value["family"] == "exact_conflict"
                            for value in result["outer_fold_selection"].values()))

    @unittest.skipUnless(FORMAL.exists(), "archived V18 formal corpus unavailable")
    def test_frozen_formal_exact_rule_targets_only_conflicting_key(self):
        episodes = [json.loads(line) for line in FORMAL.read_text().splitlines()]
        for episode in episodes:
            retained = set(exact_conflict_ids(episode))
            removed = [row for row in episode["sessions"] if row["session_id"] not in retained]
            self.assertTrue(removed)
            self.assertTrue(all(row["key_id"] == episode["target_key_id"] for row in removed))
            self.assertTrue(all(row["value"] != episode["primary_current_value"]
                                for row in removed))
            self.assertIn(episode["current_session_id"], retained)


if __name__ == "__main__":
    unittest.main()
