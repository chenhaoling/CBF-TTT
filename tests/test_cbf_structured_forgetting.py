import unittest

from tasks.build_cbf_event_curriculum import generate_group
from tasks.train_cbf_structured_forgetting import (
    ACTIONS, apply_action, build_examples, gate, lookup, parse_block, visible_features,
)


class StructuredMemoryTests(unittest.TestCase):
    def test_last_actions_remove_requested_recency(self):
        state = [{"batch": "a", "facts": {"x": "old"}},
                 {"batch": "b", "facts": {"x": "new"}}]
        assignment = {"batch": "c", "facts": {"y": "value"}}
        self.assertEqual(lookup(apply_action(state, "last1", assignment), "x"), "old")
        self.assertEqual(lookup(apply_action(state, "last2", assignment), "x"), "new")
        self.assertEqual(lookup(apply_action(state, "global", assignment), "x"), "unknown")

    def test_visible_parser_generalizes_across_templates(self):
        for split in ("train", "dev", "test"):
            inp, _ = generate_group(7, split, 0)[0]
            parsed = parse_block(inp["blocks"][0]["text"], [])
            self.assertTrue(parsed["batch"].startswith("batch_"))
            self.assertGreaterEqual(len(parsed["facts"]), 5)

    def test_parser_keeps_old_reference_when_policy_state_diverged(self):
        pairs = generate_group(9, "dev", 0)
        inp, target = next(pair for pair in pairs if pair[1]["family"] == "revoke_last1")
        boundary = inp["decision_boundaries"][0]
        block = next(row for row in inp["blocks"] if row["index"] == boundary)
        parsed = parse_block(block["text"], [])
        self.assertEqual(len(parsed["old_refs"]), 1)
        self.assertNotEqual(parsed["old_refs"][0], parsed["batch"])

    def test_counterfactual_labels_cover_registered_actions(self):
        pairs = generate_group(11, "train", 0)
        examples, meta = build_examples(pairs, True)
        self.assertEqual(set(meta["label_counts"]), set(ACTIONS))
        self.assertEqual(sum(meta["label_counts"].values()), len(examples))
        self.assertEqual({row["action"] for row in examples}, set(ACTIONS))

    def test_history_ablation_only_changes_features(self):
        pairs = generate_group(13, "train", 0)
        full, _ = build_examples(pairs, True)
        blind, _ = build_examples(pairs, False)
        self.assertEqual([row["label"] for row in full], [row["label"] for row in blind])
        self.assertTrue(any(a["features"] != b["features"] for a, b in zip(full, blind)))

    def test_gate_requires_fixed_policy_gain(self):
        fixed = {f"fixed_{action}": {"query_accuracy": .7, "groups": {"a": .7, "b": .7}}
                 for action in ACTIONS}
        summaries = {**fixed, "learned": {"query_accuracy": 1., "action_accuracy": 1.,
                                            "groups": {"a": 1., "b": 1.}}}
        self.assertTrue(gate(summaries)["passed"])
        summaries["learned"]["query_accuracy"] = .8
        self.assertFalse(gate(summaries)["passed"])


if __name__ == "__main__":
    unittest.main()
