import importlib.util
import unittest

from tasks.train_retrieval_evidence_forgetting_s3_projection import (
    relation_batches,
    roc_auc,
    selection_key,
)


class RetrievalEvidenceForgettingS3ProjectionTests(unittest.TestCase):
    def test_auc_handles_ties(self):
        self.assertEqual(roc_auc([True, False], [0.5, 0.5]), 0.5)
        self.assertEqual(roc_auc([True, False], [1.0, 0.0]), 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is exercised remotely")
    def test_relation_batches_contain_four_per_relation(self):
        import torch

        relations = [f"r{relation}" for relation in range(20) for _ in range(4)]
        batches = relation_batches(relations, torch.Generator().manual_seed(1), steps=2)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(len(batch), 64)
            counts = {}
            for index in batch:
                counts[relations[index]] = counts.get(relations[index], 0) + 1
            self.assertEqual(set(counts.values()), {4})

    def test_selection_prefers_retrieval_then_auc_then_size(self):
        base = {"retrieval": {"recall_at_16": .8}, "direct_pair_roc_auc": .7,
                "config": {"dimension": 512, "epoch": 25,
                           "pubmed_pretraining": True, "temperature": .1, "margin": .2}}
        better = {"retrieval": {"recall_at_16": .81}, "direct_pair_roc_auc": .6,
                  "config": {"dimension": 512, "epoch": 25,
                             "pubmed_pretraining": True, "temperature": .1, "margin": .2}}
        smaller = {"retrieval": {"recall_at_16": .8}, "direct_pair_roc_auc": .7,
                   "config": {"dimension": 256, "epoch": 25,
                              "pubmed_pretraining": True, "temperature": .1, "margin": .2}}
        self.assertGreater(selection_key(better), selection_key(base))
        self.assertGreater(selection_key(smaller), selection_key(base))

    def test_selection_prefers_no_pubmed_only_after_primary_ties(self):
        common = {"retrieval": {"recall_at_16": .8}, "direct_pair_roc_auc": .7}
        without = {**common, "config": {"dimension": 256, "epoch": 10,
                   "pubmed_pretraining": False, "temperature": .05, "margin": .1}}
        with_pretraining = {**common, "config": {"dimension": 256, "epoch": 10,
                            "pubmed_pretraining": True, "temperature": .05, "margin": .1}}
        self.assertGreater(selection_key(without), selection_key(with_pretraining))


if __name__ == "__main__":
    unittest.main()
