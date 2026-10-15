import unittest

from scripts.audit_natural_relation_reranker_s2_development import release


class NaturalRelationRerankerS2DevelopmentAuditTests(unittest.TestCase):
    def test_release_rejects_missing_retrieval(self):
        selected = {"delete": {"recall": .8, "precision": .995, "fp": 1},
                    "same_relation": {"recall": .9, "specificity": .96},
                    "relation_macro_balanced_accuracy": .9,
                    "same_entity_other_relation_specificity": .99}
        baseline = {"delete": {"recall": .2, "precision": .995}}
        values = {"zero_shot": baseline, "trigram": baseline,
                  "frozen_qwen_cosine": baseline}
        self.assertFalse(release(selected, values, {"recall_at_16": .94})["passed"])


if __name__ == "__main__": unittest.main()
