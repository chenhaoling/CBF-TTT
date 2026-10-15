import unittest

from scripts.audit_structured_fact_comparator_s1_training import CHECKPOINTS, TARGETS


class StructuredFactComparatorS1TrainingAuditTests(unittest.TestCase):
    def test_registered_checkpoints_cover_pilot_epochs(self):
        self.assertEqual(CHECKPOINTS, {"checkpoint_epoch_0.25": 188,
                                      "checkpoint_epoch_0.50": 375,
                                      "checkpoint_epoch_1.00": 750,
                                      "checkpoint_epoch_2.00": 1500})

    def test_only_attention_projection_targets_are_allowed(self):
        self.assertEqual(TARGETS, {"q_proj", "k_proj", "v_proj", "o_proj"})


if __name__ == "__main__":
    unittest.main()
