import unittest

from tasks.build_pair_localized_relation_memory_s4 import (
    centered_indices,
    deterministic_windows,
    dialogue_candidates,
    mask_dialogue_turns,
)


class PairLocalizedRelationMemoryS4BuilderTests(unittest.TestCase):
    def test_mask_and_centered_candidate(self):
        turns = ["Alice entered.", "Bob met Alice.", "They left."]
        masked = mask_dialogue_turns(turns, "Alice", "Bob")
        self.assertEqual(masked[1], "[ARG2] met [ARG1].")
        candidates = dialogue_candidates(masked, [0, 1], [1], "m")
        self.assertEqual([row["center_turn"] for row in candidates], [0, 1, 2])
        self.assertEqual(candidates[1]["turn_indices"], [0, 1, 2])
        self.assertTrue(candidates[1]["features"]["arg1_present"])
        self.assertTrue(candidates[1]["features"]["arg2_present"])

    def test_centered_indices_clip(self):
        self.assertEqual(centered_indices(0, 3), [0, 1])
        self.assertEqual(centered_indices(2, 3), [1, 2])

    def test_deterministic_missing_argument_falls_back(self):
        windows = deterministic_windows(["a", "b"], [], [1])
        self.assertTrue(windows["0"]["full_dialogue_fallback"])
        self.assertEqual(windows["0"]["turn_indices"], [0, 1])


if __name__ == "__main__":
    unittest.main()
