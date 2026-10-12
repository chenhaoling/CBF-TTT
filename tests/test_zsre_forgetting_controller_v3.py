"""Tests for subject-excluded calibration pair supervision."""

import unittest

from tasks.train_zsre_forgetting_controller_v3 import auxiliary_examples, subject_hash


class AugmentedGateTests(unittest.TestCase):
    def test_each_fact_yields_one_positive_and_two_negatives(self):
        sessions = [{"facts": [{"subject": "Ada", "prompt": "p", "rephrase_prompt": "r",
                                "locality_prompts": ["l1", "l2"]}]}]
        rows = auxiliary_examples(sessions)
        self.assertEqual([row["label"] for row in rows], [1.0, 0.0, 0.0])
        self.assertTrue(all(row["subject"] == "Ada" for row in rows))

    def test_subject_hash_is_order_invariant(self):
        self.assertEqual(subject_hash(["a", "b"]), subject_hash(["b", "a"]))


if __name__ == "__main__": unittest.main()
