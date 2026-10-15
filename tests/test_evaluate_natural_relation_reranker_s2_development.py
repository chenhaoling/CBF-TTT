import unittest

from tasks.evaluate_natural_relation_reranker_s2_development import gate


def metric(recall=.8, precision=.995, specificity=.96, macro=.86, fp=2, hard=.99):
    return {"delete": {"recall": recall, "precision": precision, "fp": fp},
            "same_relation": {"recall": .85, "specificity": specificity},
            "relation_macro_balanced_accuracy": macro,
            "same_entity_other_relation_specificity": hard}


class NaturalRelationRerankerS2DevelopmentTests(unittest.TestCase):
    def test_gate_passes_only_all_registered_checks(self):
        selected = metric()
        baselines = {"zero_shot": metric(.2), "trigram": metric(.1),
                     "frozen_qwen_cosine": metric(.3)}
        retrieval = {"recall_at_16": .96}
        self.assertTrue(gate(selected, baselines, retrieval)["passed"])
        selected["delete"]["fp"] = 5
        self.assertFalse(gate(selected, baselines, retrieval)["passed"])

    def test_unsafe_baseline_does_not_set_gain_target(self):
        selected = metric(.7)
        unsafe = metric(.9, precision=.8)
        safe = metric(.2)
        value = gate(selected, {"zero_shot": unsafe, "trigram": safe,
                                "frozen_qwen_cosine": safe}, {"recall_at_16": .96})
        self.assertEqual(value["best_safe_nonadapted_delete_recall"], .2)


if __name__ == "__main__":
    unittest.main()
