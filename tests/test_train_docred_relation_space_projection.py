import unittest

from tasks.train_docred_relation_space_projection import (
    description_to_statement_loss, selective_operating_point,
    supervised_contrastive_loss,
)


class DocredRelationProjectionTest(unittest.TestCase):
    def test_losses_prefer_separated_relations(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed locally")
        targets = torch.tensor([0, 0, 1, 1])
        separated = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        mixed = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        self.assertLess(float(supervised_contrastive_loss(separated, targets)),
                        float(supervised_contrastive_loss(mixed, targets)))
        logits_good = torch.tensor([[4., 0.], [4., 0.], [0., 4.], [0., 4.]])
        logits_bad = logits_good.flip(1)
        self.assertLess(float(description_to_statement_loss(logits_good, targets, 2)),
                        float(description_to_statement_loss(logits_bad, targets, 2)))

    def test_selective_gate_accepts_clean_high_score_tail(self):
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        scores = [.9, .8, .7, .6, .4, .3, .2, .1]
        relations = ["A", "A", "B", "B", "A", "A", "B", "B"]
        result = selective_operating_point(labels, scores, relations)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["selected"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
