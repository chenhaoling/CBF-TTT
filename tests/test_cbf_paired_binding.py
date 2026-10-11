import unittest

from tasks.train_cbf_paired_binding import paired_objective, paired_schedule


def rows():
    return [
        {"context_id": context, "split": "train", "id": f"{context}.{index}"}
        for context in ("a", "b") for index in range(4)
    ]


class PairedScheduleTests(unittest.TestCase):
    def test_each_context_covers_all_assignments_once(self):
        schedule = paired_schedule(rows(), 2, 11)
        self.assertEqual(len(schedule), 4)
        for context in ("a", "b"):
            shifts = [shift for item, left, right in schedule if item == context
                      for shift in (left, right)]
            self.assertEqual(sorted(shifts), [0, 1, 2, 3])
            self.assertTrue(all({left, right} in ({0, 2}, {1, 3})
                                for item, left, right in schedule if item == context))

    def test_round_count_is_fixed(self):
        with self.assertRaises(ValueError):
            paired_schedule(rows(), 1, 11)


class PairedObjectiveTests(unittest.TestCase):
    def test_swap_gradient_prefers_own_memory(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch unavailable")
        candidates = ["a", "b", "c", "d"]
        left_rows, right_rows, left_items, right_items = [], [], [], []
        for index in range(4):
            left_scores = torch.zeros(4, requires_grad=True)
            right_scores = torch.zeros(4, requires_grad=True)
            left_target, right_target = index, (index + 2) % 4
            left_rows.append({"answer": candidates[left_target], "query_ids": [index]})
            right_rows.append({"answer": candidates[right_target], "query_ids": [index]})
            left_items.append((F.cross_entropy((-left_scores).unsqueeze(0),
                                                torch.tensor([left_target])), left_scores, 0))
            right_items.append((F.cross_entropy((-right_scores).unsqueeze(0),
                                                 torch.tensor([right_target])), right_scores, 0))
        loss, retrieval, swap, _ = paired_objective(
            left_items, right_items, left_rows, right_rows, candidates, "paired_swap")
        self.assertAlmostEqual(float(retrieval.detach()), 1.386294, places=5)
        self.assertGreater(float(swap.detach()), 0)
        loss.backward()
        # NLL gradients must reduce the own target under its memory and increase it under the other memory.
        self.assertGreater(float(left_items[0][1].grad[0]), 0)
        self.assertLess(float(right_items[0][1].grad[0]), 0)

    def test_listwise_arm_has_zero_swap_contribution(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch unavailable")
        candidates = ["a", "b", "c", "d"]
        left_rows, right_rows, left_items, right_items = [], [], [], []
        for index in range(4):
            left = torch.zeros(4, requires_grad=True)
            right = torch.zeros(4, requires_grad=True)
            other = (index + 2) % 4
            left_rows.append({"answer": candidates[index], "query_ids": [index]})
            right_rows.append({"answer": candidates[other], "query_ids": [index]})
            left_items.append((F.cross_entropy((-left).unsqueeze(0), torch.tensor([index])), left, 0))
            right_items.append((F.cross_entropy((-right).unsqueeze(0), torch.tensor([other])), right, 0))
        loss, retrieval, _, _ = paired_objective(
            left_items, right_items, left_rows, right_rows, candidates, "paired_listwise")
        self.assertAlmostEqual(float(loss.detach()), float(retrieval.detach()), places=7)


if __name__ == "__main__":
    unittest.main()
