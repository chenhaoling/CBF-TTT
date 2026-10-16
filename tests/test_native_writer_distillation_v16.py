"""Tensor tests for contextual-to-native writer distillation."""

import unittest

import torch

from tasks.train_native_writer_distillation_v16 import distillation_terms


class NativeWriterDistillationV16Test(unittest.TestCase):
    def test_correct_teacher_states_have_lower_loss_than_swapped_states(self):
        zero = torch.zeros(4)
        left_target = torch.tensor([2.0, 0.0, 0.0, 0.0])
        right_target = torch.tensor([0.0, 2.0, 0.0, 0.0])
        good_mse, good_contrast, _ = distillation_terms(
            left_target, right_target, zero, left_target, right_target, 2.0)
        bad_mse, bad_contrast, _ = distillation_terms(
            right_target, left_target, zero, left_target, right_target, 2.0)
        self.assertLess(good_mse, bad_mse)
        self.assertLess(good_contrast, bad_contrast)

    def test_scale_must_be_positive(self):
        value = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "positive"):
            distillation_terms(value, value, value, value, value, 0.0)


if __name__ == "__main__":
    unittest.main()
