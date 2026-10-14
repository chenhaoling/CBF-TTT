import unittest

from tasks.evaluate_v19_nested_forgetting_policy import aggregate, exact_arm
from scripts.audit_v19_nested_forgetting_policy_outcome import recompute
from tasks.train_v19_nested_forgetting_policy import SEEDS


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct, nll=.1):
    return {"kind": kind, "correct": correct, "answer_nll": nll,
            "generation": {"normalized_exact": correct}}


def arm(arm_id, joint, locality=1, kind="fixed", **extra):
    values = (joint, joint, 1, 1, locality)
    return {"arm_id": arm_id, "kind": kind, **extra,
            "queries": [query(name, value) for name, value in zip(KINDS, values)]}


def fixture():
    episodes, policies, raw = [], [], []
    selections = []
    for index in range(320):
        fold = (index // 16) % 5
        episode_id = f"e{index}"
        episodes.append({
            "episode_id": episode_id, "subject": f"s{index // 4}",
            "template_group": f"t{index // 16}", "fold": fold,
            "sessions": [{"session_id": "old", "key_id": "K", "value": "OLD"},
                         {"session_id": "current", "key_id": "K", "value": "NEW"}]})
        fixed = arm("fixed:retain_all", 0, policy="retain_all")
        exact = arm("delete_key:K", 0, kind="delete_key", deleted_key_id="K")
        oracle = arm("delete_one:old", 1, kind="delete_one", deleted_session_id="old")
        raw.append({"episode_id": episode_id, "arms": [fixed, exact, oracle]})
        policies.append({
            "episode_id": episode_id, "subject": f"s{index // 4}",
            "template_group": f"t{index // 16}", "fold": fold,
            "mechanism": "clean_update", "length": 8,
            "arms": [arm(f"policy:{seed}", 1, kind="nested_oof_policy", seed=seed)
                     for seed in SEEDS]})
        selections.append({"episode_id": episode_id, "arm_id": "delete_one:old"})
    outcome = {"fixed_selected_by_outer_fold": {str(fold): "retain_all"
                                                 for fold in range(5)},
               "selections": {"0.03": selections}}
    policy = {"selection": {str(fold): {"constraints_satisfied": True}
                            for fold in range(5)}}
    return episodes, policies, raw, outcome, policy


class EvaluateV19NestedForgettingPolicyTest(unittest.TestCase):
    def test_exact_rule_is_reconstructed_from_write_time_key(self):
        episodes, _, raw, _, _ = fixture()
        self.assertEqual(exact_arm(episodes[0], raw[0])["arm_id"], "delete_key:K")

    def test_clear_positive_fixture_passes_full_policy_gate(self):
        args = fixture()
        result = aggregate(*args)
        self.assertEqual(result["arm_metrics"]["policy_three_seed_mean"][
            "joint_success"], 1)
        self.assertEqual(result["policy_vs_fixed"]["fraction_oracle_gap_closed"], 1)
        self.assertTrue(result["controlled_policy_gate"]["passed"])
        self.assertEqual(result["next_stage"],
                         "build_source_disjoint_semantic_policy_confirmation")
        self.assertEqual(result, recompute(*args))


if __name__ == "__main__":
    unittest.main()
