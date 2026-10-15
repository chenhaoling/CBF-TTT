import unittest

from scripts.inspect_pair_localization_s4_source import (
    nearest_entity_window,
    normalized,
    phrase_turns,
)


class PairLocalizationS4SourceInspectionTests(unittest.TestCase):
    def test_phrase_turns_uses_boundaries_and_normalization(self):
        dialogue = ["Ross called Rachel.", "Crossing the road", "RACHEL replied"]
        self.assertEqual(phrase_turns(dialogue, " rachel "), [0, 2])
        self.assertEqual(phrase_turns(dialogue, "Ross"), [0])

    def test_nearest_window_is_deterministic(self):
        self.assertEqual(nearest_entity_window([0, 6], [4, 8], 10, 0), {4, 5, 6})
        self.assertEqual(nearest_entity_window([0, 6], [4, 8], 10, 1), {3, 4, 5, 6, 7})

    def test_missing_argument_has_no_window(self):
        self.assertEqual(nearest_entity_window([], [1], 3, 1), set())
        self.assertEqual(normalized(" A   B "), "a b")


if __name__ == "__main__":
    unittest.main()
