import unittest

from tasks.train_natural_relation_reranker_s2 import (
    canonical_answer, checkpoint_steps, prompt,
)


ROW = {"left_text": "[SUBJECT] founded [VALUE] in 1990",
       "right_text": "[SUBJECT] established [VALUE] during 1991",
       "same_relation": 1, "left_relation_audit": "hidden"}


class NaturalRelationRerankerS2TrainTests(unittest.TestCase):
    def test_answer_is_binary_canonical_json(self):
        self.assertEqual(canonical_answer(True), '{"same_relation":true}')
        self.assertEqual(canonical_answer(False), '{"same_relation":false}')

    def test_prompt_uses_only_masked_text(self):
        value = prompt(ROW)
        self.assertIn(ROW["left_text"], value)
        self.assertIn(ROW["right_text"], value)
        self.assertNotIn("hidden", value)

    def test_registered_checkpoint_steps(self):
        self.assertEqual(checkpoint_steps(300), {75: .25, 150: .5, 300: 1.})


if __name__ == "__main__":
    unittest.main()
