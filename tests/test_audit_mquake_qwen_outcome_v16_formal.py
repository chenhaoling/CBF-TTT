"""Focused tests for the independent V16 formal outcome auditor."""

import unittest

from scripts.audit_mquake_qwen_outcome_v16_formal import (
    comparison_rows, fixed_selection, normalize, reconstruct_memory, render, specs,
)
from tests.test_evaluate_mquake_qwen_outcome_v16_formal import row


class FormalOutcomeAuditTest(unittest.TestCase):
    def test_prompt_and_specs_are_independently_defined(self):
        sample = {"old": {"statement": "old?", "value": "A"},
                  "current": {"statement": "new?", "value": "B"},
                  "hard_retain": {"statement": "hard?", "value": "C"},
                  "unrelated": {"statement": "other?", "value": "D"}}
        self.assertEqual([value[0] for value in specs(sample)],
                         ["primary_old_phrase", "primary_current_phrase", "hard_retain", "unrelated"])
        memory = [{"units": ["a"]}, {"units": ["b"]}]
        prompt = render(memory, {"a": "one", "b": "two"}, "query")
        self.assertLess(prompt.index("one"), prompt.index("two"))
        self.assertTrue(prompt.endswith("Answer:\n"))

    def test_causal_reconstruction_and_normalization(self):
        episode = {"sessions": [{"session_id": "a"}, {"session_id": "b"},
                                {"session_id": "future"}]}
        self.assertEqual([value["session_id"] for value in
            reconstruct_memory(episode, {"query_step": 1}, ["b", "a"])], ["b", "a"])
        with self.assertRaises(ValueError):
            reconstruct_memory(episode, {"query_step": 1}, ["future"])
        self.assertEqual(normalize(" NEW   York "), "new york")

    def test_crossfit_comparison_uses_selected_policy(self):
        rows = [row(index) for index in range(20)]
        selected, _ = fixed_selection(rows)
        values = comparison_rows(rows, selected)
        self.assertEqual(len(values), 40)
        self.assertTrue(all(value["fixed_policy"] == "no_memory" for value in values))
        self.assertTrue(all(value["accuracy_gain"] == 1 for value in values))


if __name__ == "__main__":
    unittest.main()
