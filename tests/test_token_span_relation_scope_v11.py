"""Pure tests for V11 span labels, relation views, and gate."""

import unittest

from tasks.train_token_span_relation_scope_v11 import (
    character_spans, development_gate, masked_relation_view,
    normalized_span_tokens, select_contiguous_span, token_labels,
)


class TokenSpanRelationScopeV11Tests(unittest.TestCase):
    def test_case_insensitive_character_and_token_labels(self):
        text = "Noise. Ada Lovelace worked in London."
        spans = character_spans(text, "ada lovelace")
        self.assertEqual(spans, [(7, 19)])
        offsets = [(0, 5), (7, 10), (11, 19), (20, 26), (27, 29), (30, 36)]
        self.assertEqual(token_labels(offsets, spans), [0, 1, 1, 0, 0, 0])

    def test_contiguous_span_and_relation_mask(self):
        logits = [-3.0, 2.0, 3.0, -2.0, -3.0]
        span = select_contiguous_span(logits, .5)
        self.assertEqual(span, (1, 3))
        text = "Noise Ada Lovelace worked London"
        offsets = [(0, 5), (6, 9), (10, 18), (19, 25), (26, 32)]
        self.assertEqual(normalized_span_tokens(text, offsets, span), ("ada", "lovelace"))
        self.assertEqual(masked_relation_view(text, offsets, span),
                         "noise <ENTITY> worked london")

    def test_single_token_fallback(self):
        self.assertEqual(select_contiguous_span([-4.0, -1.0, -2.0], .7), (1, 2))

    def test_gate(self):
        result = {"top1_accuracy": .97, "mrr": .985}
        gain = {"top1_gain": .02, "bootstrap_95ci": [.005, .03]}
        self.assertTrue(development_gate(result, gain, .978)["passed"])
        gain["bootstrap_95ci"] = [-.001, .03]
        self.assertFalse(development_gate(result, gain, .978)["passed"])


if __name__ == "__main__":
    unittest.main()
