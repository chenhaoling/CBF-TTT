import unittest

from tasks.build_qwen_natural_value_bridge_v1 import (
    episodes_from_facts,
    normalize,
)
from tasks.evaluate_qwen_natural_value_bridge_v1 import (
    ARMS,
    aggregate,
    render_query,
)


def facts(count=8):
    return [
        {
            "subject": f"subject {index}",
            "prompt": f"What is value {index}?",
            "rephrase_prompt": f"Return value {index}.",
            "original": f"old answer {index}",
            "updated": f"new answer {index}",
            "source_row_sha256": f"{index:064x}",
        }
        for index in range(count)
    ]


class NaturalValueBridgeTest(unittest.TestCase):
    def test_model_free_candidates_are_deterministic_and_unique(self):
        left = episodes_from_facts(facts(), "calibration")
        right = episodes_from_facts(list(reversed(facts())), "calibration")
        self.assertEqual(left, right)
        self.assertEqual(len(left), 8)
        for row in left:
            self.assertEqual(len(row["candidates"]), 6)
            self.assertEqual(len({normalize(value) for value in row["candidates"]}), 6)
            self.assertIn(row["correct"], row["candidates"])
            self.assertIn(row["wrong"], row["candidates"])
            self.assertFalse(row["candidate_construction_used_model_scores"])

    def test_normalization_is_case_and_whitespace_stable(self):
        self.assertEqual(normalize("  New   York "), normalize("new york"))
        self.assertNotEqual(normalize("new york"), normalize("new jersey"))

    def test_rendered_prompt_has_answer_boundary(self):
        prompt = render_query("Who made the item?")
        self.assertIn("Query: Who made the item?", prompt)
        self.assertTrue(prompt.endswith("Answer:\n"))

    def test_frozen_gate_aggregation_passes_clean_content_separation(self):
        rows = []
        for episode in ("a", "b"):
            for query_type in ("exact", "rephrase"):
                arm_rows = {}
                for arm in ARMS:
                    prediction = "new" if arm == "correct" else "old"
                    arm_rows[arm] = {
                        "scores": {"new": .1 if arm == "correct" else 1.0,
                                   "old": 1.0 if arm == "correct" else .1,
                                   "d1": 2.0, "d2": 2.0, "d3": 2.0, "d4": 2.0},
                        "prediction": prediction,
                        "candidate_correct": int(prediction == "new"),
                        "injected_wrong": int(prediction == "old"),
                        "answer_nll": .1 if arm == "correct" else 1.0,
                    }
                arm_rows["correct"]["generation"] = {
                    "token_exact": 1, "ids": [1], "tokens": 1,
                    "conditioned_length": 1,
                }
                rows.append({
                    "episode_id": episode,
                    "query_type": query_type,
                    "scales": {"4.0": {"arms": arm_rows}},
                })
        result = aggregate(rows, 4.0)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["metrics"]["correct"]["candidate_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["wrong"]["injected_wrong_accuracy"], 1.0)
        self.assertGreater(result["comparisons"]["zero"]["bootstrap_95ci"][0], 0)


if __name__ == "__main__":
    unittest.main()
