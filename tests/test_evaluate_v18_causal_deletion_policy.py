import json
from pathlib import Path
import tempfile
import unittest

from tasks.evaluate_v18_causal_deletion_policy import aggregate
from tasks.evaluate_v18_lowrank_fastmemory_bridge import SCALES
from tasks.train_v18_causal_deletion_policy import SEEDS


KINDS = ("update_old_surface", "update_current_surface", "retain_1", "retain_2",
         "locality")


def query(kind, correct):
    return {"kind": kind, "by_scale": {str(scale): {
        "correct": correct, "answer_nll": .1 if correct else 1.0,
        "scores": {}, "prediction": "x"} for scale in SCALES},
        "generation_primary": {"normalized_exact": correct}}


def arm(correct, **extra):
    return {"queries": [query(kind, correct) for kind in KINDS], **extra}


def metric(correct):
    return {"joint_success": correct, "update_accuracy": correct,
            "retained_key_accuracy": 1.0, "locality_accuracy": 1.0,
            "joint_answer_nll": .1 if correct else 1.0,
            "generation_joint_success": correct}


class EvaluateV18CausalDeletionPolicyTest(unittest.TestCase):
    def test_controlled_gate_and_exact_rule_comparison(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); raw = [] ; selections = [] ; rows = []
            for index in range(10):
                episode_id = f"e{index}"
                rows.append({"episode_id": episode_id, "subject": f"s{index}",
                             "template_group": f"t{index}", "fold": index % 5,
                             "length": 8, "mechanism": "clean_update",
                             "arms": [arm(1, seed=seed) for seed in SEEDS]})
                raw.append({"episode_id": episode_id,
                            "arms": [arm(1, arm_name="exact_conflict")]})
                selections.append({"episode_id": episode_id, "fixed": metric(0),
                                   "physical": metric(1), "oracle": metric(1)})
            (root / "shard0.jsonl").write_text("".join(
                json.dumps(row) + "\n" for row in raw[::2]))
            (root / "shard1.jsonl").write_text("".join(
                json.dumps(row) + "\n" for row in raw[1::2]))
            (root / "summary.json").write_text(json.dumps({
                "selection_records": selections}))
            result = aggregate(rows, root)
            self.assertTrue(result["controlled_policy_gate"]["passed"])
            self.assertEqual(result["joint_gain_over_fixed"], 1.0)
            self.assertEqual(result["joint_gain_over_physical"], 0.0)
            self.assertEqual(result["joint_gain_over_exact_rule"], 0.0)
            self.assertFalse(result["learned_policy_claim_supported"])
            self.assertFalse(result["semantic_alias_gate"]["evaluated"])


if __name__ == "__main__":
    unittest.main()
