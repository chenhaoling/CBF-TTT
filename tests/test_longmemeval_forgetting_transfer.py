import unittest

from tasks.evaluate_longmemeval_forgetting_transfer import (
    oracle_rollout, random_matched_rollout, render_session, score_memory,
)


class FakeTable:
    def __init__(self, rows): self.rows = rows
    def __getitem__(self, item):
        if isinstance(item, list): return FakeMatrix([self.rows[index] for index in item])
        return self.rows[item]


class FakeMatrix:
    def __init__(self, rows): self.rows = rows
    @property
    def T(self): return self
    def __matmul__(self, other):
        left = self.rows; right = other.rows
        return FakeScores([[sum(a*b for a,b in zip(x,y)) for y in right] for x in left])


class FakeScores:
    def __init__(self, rows): self.rows = rows
    def max(self): return max(max(row) for row in self.rows)


class TransferTest(unittest.TestCase):
    def test_render_and_oracle_pair(self):
        text = render_session([{"role": "assistant", "content": ""},
                               {"role": "user", "content": "hello"}], "2024-01-01")
        self.assertEqual(text, "Session date: 2024-01-01\nuser: hello")
        ids = ["a", "old", "b", "new", "c"]
        self.assertEqual(oracle_rollout(ids, ["old", "new"]), [0, 2, 3, 4])

    def test_random_matched_replays_delete_schedule(self):
        decisions = [{"deleted": False}, {"deleted": True}, {"deleted": False}, {"deleted": True}]
        memory = random_matched_rollout("q", 4, decisions)
        self.assertEqual(len(memory), 2)
        self.assertIn(3, memory)

    def test_state_metrics(self):
        ids = ["old", "x", "new"]
        table = FakeTable([[1., 0.], [0., 1.], [1., 1.], [1., 1.]])
        result = score_memory([1, 2], ids, ["old", "new"], [3], [[0], [1], [2]],
                              {str(seed): table for seed in (5101, 5102, 5103)})
        self.assertEqual(result["pair_state_correct"], 1.)
        self.assertEqual(result["latest_retained"], 1.)
        self.assertEqual(result["stale_deleted"], 1.)


if __name__ == "__main__": unittest.main()
