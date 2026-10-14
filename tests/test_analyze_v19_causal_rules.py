import unittest

from scripts.audit_v19_causal_rules import recompute
from tasks.analyze_v19_causal_rules import aggregate, choose_rule


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct):
    return {"kind": kind, "correct": correct, "answer_nll": .1 if correct else 1.,
            "generation": {"normalized_exact": correct}}


def arm(arm_id, joint, locality=1, kind="fixed", **extra):
    values = (joint, joint, 1, 1, locality)
    return {"arm_id": arm_id, "kind": kind,
            "queries": [query(name, value) for name, value in zip(KINDS, values)], **extra}


def fixtures():
    episodes, rows = [], []
    for index in range(320):
        fold = index % 5
        sessions = [{"session_id": f"e{index}-old", "key_id": "target", "value": "old"},
                    {"session_id": f"e{index}-keep", "key_id": "keep", "value": "kept"},
                    {"session_id": f"e{index}-new", "key_id": "target", "value": "new"}]
        episodes.append({"episode_id": f"e{index}", "subject": f"s{index // 4}",
                         "template_group": f"t{index // 16}", "fold": fold,
                         "length": 8, "mechanism": "clean_update", "sessions": sessions})
        fixed = [arm(f"fixed:{name}", 0) for name in (
            "no_memory", "retain_all", "global_reset", "last_1", "last_2")]
        deletes = [arm(f"delete_one:{sessions[0]['session_id']}", 1, kind="delete_one",
                       deleted_session_id=sessions[0]["session_id"]),
                   arm(f"delete_one:{sessions[1]['session_id']}", 0, kind="delete_one",
                       deleted_session_id=sessions[1]["session_id"]),
                   arm("delete_key:target", 1, kind="delete_key", deleted_key_id="target"),
                   arm("delete_key:keep", 0, kind="delete_key", deleted_key_id="keep")]
        rows.append({"episode_id": f"e{index}", "subject": f"s{index // 4}",
                     "template_group": f"t{index // 16}", "fold": fold,
                     "length": 8, "mechanism": "clean_update", "arms": fixed + deletes})
    outcome = {"fixed_selected_by_outer_fold": {str(fold): "retain_all" for fold in range(5)},
               "pareto": {"0.03": {"metrics": {"joint_success": 1.}}},
               "confirmation_gate": {"passed": True}}
    return episodes, rows, outcome


class AnalyzeV19CausalRulesTest(unittest.TestCase):
    def test_rules_use_current_session_content(self):
        episodes, rows, _ = fixtures()
        self.assertEqual(choose_rule(episodes[0], rows[0], "delete_newest_exact")["arm_id"],
                         "delete_one:e0-old")
        self.assertEqual(choose_rule(episodes[0], rows[0], "delete_key_exact")["arm_id"],
                         "delete_key:target")

    def test_explicit_key_gate_and_independent_recomputation(self):
        episodes, rows, outcome = fixtures()
        table = aggregate(episodes, rows, outcome)
        self.assertTrue(table["explicit_key_explanation_gate"]["passed"])
        self.assertEqual(table["delete_key_exact_vs_fixed"]["fraction_oracle_gap_closed"], 1.)
        self.assertEqual(table, recompute(episodes, rows, outcome))
        self.assertEqual(table["next_stage"], "build_v20_semantic_alias_confirmation")


if __name__ == "__main__":
    unittest.main()
