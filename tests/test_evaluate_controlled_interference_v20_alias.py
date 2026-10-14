import copy
import unittest

from scripts.audit_controlled_interference_v20_alias_outcome import recompute
from tasks.build_controlled_interference_v20_alias import build_episode, build_states
from tasks.evaluate_controlled_interference_v20_alias import aggregate, state_arms


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=.1):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "generation": {"normalized_exact": correct}}


def arm(arm_id, joint, locality=1, kind="fixed", policy=None):
    values = (joint, joint, 1, 1, locality)
    result = {"arm_id": arm_id, "kind": kind,
              "queries": [query(name, value) for name, value in zip(KINDS, values)]}
    if policy is not None:
        result["policy"] = policy
    return result


def row(index):
    shared = arm("fixed:retain_all", 0, policy="retain_all")
    arms = [arm("fixed:no_memory", 0, policy="no_memory"), shared,
            arm("fixed:global_reset", 0, policy="global_reset"),
            arm("fixed:last_1", 0, policy="last_1"),
            arm("fixed:last_2", 0, policy="last_2")]
    exact = arm("special:exact_conflict", 0, kind="semantic_rule",
                policy="exact_conflict")
    exact["queries"] = copy.deepcopy(shared["queries"])
    arms.extend((exact, arm("special:semantic_oracle", 1, kind="semantic_rule",
                            policy="semantic_oracle")))
    return {"episode_id": f"e{index}", "subject": f"subject{index // 4}",
            "template_group": f"template{index // 16}", "fold": (index // 16) % 5,
            "mechanism": "clean_update", "length": 8, "arms": arms}


class EvaluateControlledInterferenceV20AliasTest(unittest.TestCase):
    def test_state_arms_exact_is_noop_and_semantic_removes_stale_slot(self):
        episode = build_episode(1)
        state = build_states(episode)
        arms = {value["arm_id"]: value for value in state_arms(state)}
        self.assertEqual(arms["special:exact_conflict"]["memory_session_ids"],
                         arms["fixed:retain_all"]["memory_session_ids"])
        self.assertEqual(arms["special:exact_conflict"]["removed_session_ids"], [])
        self.assertGreater(len(arms["special:semantic_oracle"]["removed_session_ids"]), 0)
        self.assertIn(episode["current_session_id"],
                      arms["special:semantic_oracle"]["memory_session_ids"])

    def test_registered_semantic_gate_passes_clear_positive_fixture(self):
        result = aggregate([row(index) for index in range(320)])
        self.assertEqual(result["semantic_oracle_vs_exact_conflict"][
            "joint_gain_micro"], 1)
        self.assertTrue(result["semantic_headroom_gate"]["passed"])
        self.assertEqual(result["next_stage"],
                         "train_nested_oof_semantic_forgetting_controller")

    def test_independent_auditor_reconstructs_registered_table(self):
        rows = [row(index) for index in range(320)]
        self.assertEqual(aggregate(rows), recompute(rows))


if __name__ == "__main__":
    unittest.main()
