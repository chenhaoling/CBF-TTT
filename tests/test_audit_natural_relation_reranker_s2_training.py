import unittest

from scripts.audit_natural_relation_reranker_s2_training import CHECKPOINTS, TARGETS


class NaturalRelationRerankerS2TrainingAuditTests(unittest.TestCase):
    def test_registered_checkpoints_cover_one_epoch(self):
        self.assertEqual(CHECKPOINTS, {"checkpoint_epoch_0.25": 75,
                                      "checkpoint_epoch_0.50": 150,
                                      "checkpoint_epoch_1.00": 300})

    def test_only_attention_projections_are_allowed(self):
        self.assertEqual(TARGETS, {"q_proj", "k_proj", "v_proj", "o_proj"})


if __name__ == "__main__":
    unittest.main()
