import unittest

from tasks.analyze_qwen_natural_causal_memory_v1 import reader_diagnosis


class NaturalMemoryDiagnosisTest(unittest.TestCase):
    def test_reconstructs_correct_top_behind_defer_gate(self):
        memory = [{"chain_id": "c1", "version": 0, "target": "answer"}]
        states = [{
            "session_id": "s1",
            "snapshots": [{"phase": "initial", "states": {"oracle": memory}}],
        }]
        rows = []
        for surface, defer, probability in (
                ("exact", 0, .9), ("rephrase", 1, .2)):
            rows.append({
                "session_id": "s1", "group_id": "g1", "phase": "initial",
                "surface": surface, "chain_kind": "primary", "chain_id": "c1",
                "active_version": 0, "final": True,
                "arms": {
                    "oracle_auto": {
                        "scope": {"top": 0, "retrieve_probability": probability},
                        "defer": defer, "selection_correct": 1 - defer,
                        "candidate_correct": 1 - defer,
                    },
                    "oracle_oracle": {
                        "candidate_correct": 1,
                        "generation": {"token_exact": 1},
                    },
                },
            })
        result = reader_diagnosis(rows, states)
        self.assertEqual(result["top_accuracy"], 1.0)
        self.assertEqual(result["selection_accuracy"], .5)
        self.assertEqual(result["categories"], {
            "defer_correct_top": 1, "retrieve_correct_top": 1,
        })
        self.assertTrue(result["force_top_counterfactual"][
            "exact_if_covered_fraction_is_one"])
        self.assertEqual(result["force_top_counterfactual"][
            "candidate_accuracy_on_covered"], 1.0)


if __name__ == "__main__":
    unittest.main()
