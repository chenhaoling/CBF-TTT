"""Pure policy tests for V4 cross-view consensus."""

import unittest

from tasks.evaluate_qwen_natural_forgetting_consensus_v4 import (
    candidate_action,
    parse_thresholds,
)


class ConsensusV4Tests(unittest.TestCase):
    def test_threshold_name_parser(self):
        self.assertEqual(parse_thresholds("n_b50_r050"), (.5, .05))
        self.assertEqual(parse_thresholds("x_n_b80_r100"), (.8, .1))

    def test_cross_view_intersection(self):
        engineered = {1: (.9, .05), 2: (.9, .05), 3: (.9, .05)}
        no_schedule = {1: (.8, .05), 2: (.8, .05), 3: (.8, .05)}
        self.assertTrue(candidate_action("x_n_b70_r100", engineered, no_schedule))
        unsafe = {1: (.9, .2), 2: (.9, .2), 3: (.9, .2)}
        self.assertFalse(candidate_action("x_n_b70_r100", unsafe, no_schedule))

    def test_vote_rules(self):
        engineered = {1: (0., 1.), 2: (0., 1.), 3: (0., 1.)}
        votes = {1: (.8, .1), 2: (.7, .1), 3: (.4, .1)}
        self.assertTrue(candidate_action("n_majority", engineered, votes))
        self.assertFalse(candidate_action("n_unanimity", engineered, votes))


if __name__ == "__main__":
    unittest.main()
