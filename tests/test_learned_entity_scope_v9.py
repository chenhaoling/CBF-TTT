"""Pure tests for V9 entity supervision and gate."""
import unittest
from tasks.train_learned_entity_scope_v9 import development_gate,entity_examples

class LearnedEntityScopeV9Tests(unittest.TestCase):
    def test_examples_pair_same_relation_different_subject(self):
        rows=[]
        for case,subject in enumerate(("Ada","Grace")):
            rows.append({"case_id":case,"requested_rewrite":{"subject":subject,"relation_id":"P1",
                "prompt":"{} was born in"},"paraphrase_prompts":[f"Birthplace of {subject}",f"Where was {subject} born"]})
        examples=entity_examples(rows)
        self.assertEqual(sum(x["label"] for x in examples),4)
        self.assertEqual(len(examples),8)
        self.assertTrue(all(x["left_subject"]!=x["right_subject"] for x in examples if not x["label"]))
    def test_gate(self):
        r={"top1_accuracy":.97,"mrr":.985,"entity_pair_auc":.98}
        v8={"top1_gain":.02,"bootstrap_95ci":[.005,.04]};v4={"top1_gain":.14,"bootstrap_95ci":[.11,.17]}
        self.assertTrue(development_gate(r,v8,v4)["passed"])

if __name__=="__main__":unittest.main()

