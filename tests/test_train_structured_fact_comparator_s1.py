import unittest

from tasks.train_structured_fact_comparator_s1 import (
    canonical_answer, checkpoint_steps, cosine_schedule, prompt,
)


ROW = {"left_text": "[SUBJECT] A [/SUBJECT] likes [VALUE] B [/VALUE]",
       "right_text": "[SUBJECT] A [/SUBJECT] likes [VALUE] C [/VALUE]",
       "same_entity": 1, "same_relation": 1, "same_value": 0, "operation": "update"}


class StructuredFactComparatorS1TrainTests(unittest.TestCase):
    def test_canonical_answer_has_fixed_order_and_no_spaces(self):
        self.assertEqual(canonical_answer(ROW),
                         '{"same_entity":true,"same_relation":true,'
                         '"same_value":false,"operation":"update"}')

    def test_prompt_contains_only_visible_text_and_schema(self):
        value = prompt(ROW)
        self.assertIn(ROW["left_text"], value)
        self.assertIn(ROW["right_text"], value)
        self.assertNotIn("left_relation_audit", value)

    def test_registered_checkpoint_steps(self):
        self.assertEqual(checkpoint_steps(750), {188: .25, 375: .5, 750: 1., 1500: 2.})

    def test_schedule_warms_then_decays(self):
        self.assertLess(cosine_schedule(0, 100, 10), cosine_schedule(9, 100, 10))
        self.assertGreater(cosine_schedule(10, 100, 10), cosine_schedule(99, 100, 10))


if __name__ == "__main__":
    unittest.main()
