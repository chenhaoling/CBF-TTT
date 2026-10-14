"""Tests for causal V16 pilot memory and prompt construction."""

import unittest

from tasks.evaluate_mquake_qwen_outcome_v16 import (
    aggregate, encode_candidates, encode_pair, policy_memory, query_specs, render_prompt,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())


class QwenOutcomePilotTest(unittest.TestCase):
    def setUp(self):
        self.episode = {"sessions": [
            {"session_id": "d0", "units": ["d"]},
            {"session_id": "old", "units": ["o"]},
            {"session_id": "hard", "units": ["h"]},
            {"session_id": "current", "units": ["c"]},
            {"session_id": "future", "units": ["f"]},
        ]}
        self.sample = {"query_step": 3, "primary_old_session_id": "old",
                       "primary_current_session_id": "current",
                       "old": {"statement": "old?", "value": "A"},
                       "current": {"statement": "current?", "value": "B"},
                       "hard_retain": {"statement": "hard?", "value": "C"},
                       "unrelated": {"statement": "other?", "value": "D"}}

    def test_oracle_is_causal_and_deletes_only_old(self):
        memory = policy_memory(self.episode, self.sample, "oracle")
        self.assertEqual([row["session_id"] for row in memory], ["d0", "hard", "current"])

    def test_no_memory_is_empty(self):
        self.assertEqual(policy_memory(self.episode, self.sample, "no_memory"), [])

    def test_prompt_preserves_chronology_and_excludes_future(self):
        memory = policy_memory(self.episode, self.sample, "oracle")
        prompt = render_prompt(memory, {key: f"text-{key}" for key in "dohcf"}, "question")
        self.assertLess(prompt.index("text-d"), prompt.index("text-h"))
        self.assertLess(prompt.index("text-h"), prompt.index("text-c"))
        self.assertNotIn("text-f", prompt)

    def test_query_matrix(self):
        rows = query_specs(self.sample)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["candidates"], ["A", "B"])
        self.assertEqual(rows[0]["answer"], "B")
        self.assertEqual(rows[2]["answer"], "C")

    def test_answer_boundary(self):
        prompt, answer = encode_pair(Tokenizer(), "Answer:\n", "Alice")
        self.assertEqual(bytes(prompt).decode(), "Answer:\n")
        self.assertEqual(bytes(answer).decode(), "Alice")

    def test_candidates_share_exact_prompt_prefix(self):
        prompt, answers = encode_candidates(Tokenizer(), "Answer:\n", ["Alice", "Bob"])
        self.assertEqual(bytes(prompt).decode(), "Answer:\n")
        self.assertEqual(bytes(answers["Alice"]).decode(), "Alice")
        self.assertEqual(bytes(answers["Bob"]).decode(), "Bob")
        with self.assertRaises(ValueError):
            encode_candidates(Tokenizer(), "Answer:\n", ["Alice", "Alice"])

    def test_aggregate_reports_per_query_resources(self):
        def query(kind, correct):
            return {"kind": kind, "correct": correct, "answer_nll": 1.0,
                    "prompt_tokens": 10, "total_tokens_max": 12,
                    "candidates": ["A", "B"], "resources": {
                        "seconds": .25, "candidate_scores": 2,
                        "gpu_start_allocated_gib": 8.0,
                        "gpu_start_reserved_gib": 8.5,
                        "gpu_peak_allocated_gib": 9.0,
                        "gpu_peak_reserved_gib": 9.5}}
        rows = []
        for _ in range(40):
            rows.append({"policies": {
                "no_memory": {"queries": [query(kind, 0) for kind in
                                             ("primary_old_phrase", "primary_current_phrase",
                                              "hard_retain", "unrelated")]},
                "oracle": {"queries": [query(kind, 1) for kind in
                                          ("primary_old_phrase", "primary_current_phrase",
                                           "hard_retain", "unrelated")]},
            }})
        result = aggregate(rows)
        self.assertTrue(result["pilot_gate"]["passed"])
        self.assertEqual(result["policies"]["oracle"]["resources"]
                         ["max_gpu_peak_reserved_gib"], 9.5)
        self.assertEqual(result["policies"]["oracle"]["resources"]
                         ["p95_seconds_per_query"], .25)


if __name__ == "__main__":
    unittest.main()
