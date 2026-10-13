import unittest

from tasks.train_episode_barrier_forgetting_v6 import BCE_WEIGHT, MARGIN, PROTOCOL


class EpisodeBarrierV6Test(unittest.TestCase):
    def test_registered_objective_constants(self):
        self.assertEqual(PROTOCOL, "episode_barrier_forgetting_v6")
        self.assertEqual(MARGIN, .5)
        self.assertEqual(BCE_WEIGHT, .25)


if __name__ == "__main__": unittest.main()
