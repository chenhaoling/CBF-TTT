import unittest

from tasks.train_long_session_candidate_v2 import bootstrap, release_gate


class CandidateV2Test(unittest.TestCase):
    def test_subject_cluster_bootstrap(self):
        left = [{"episode_id": f"e{i}", "subject": f"s{i // 2}", "correct": 1}
                for i in range(8)]
        right = [{**row, "correct": 0} for row in left]
        result = bootstrap(left, right, 3, draws=100)
        self.assertEqual(result["gain"], 1.)
        self.assertEqual(result["bootstrap_95ci"], [1., 1.])
        self.assertEqual(result["subject_clusters"], 4)

    def test_release_gate_requires_every_check(self):
        metric = {"top1": .96, "mrr": .98,
                  "by_length": {str(length): {"top1": .95}
                                for length in (16, 32, 48, 64)}}
        per_seed = {str(seed): {"top1": .94} for seed in (1, 2, 3)}
        comparison = {"bootstrap_95ci": [.21, .3]}
        self.assertTrue(release_gate(metric, per_seed, comparison)["passed"])
        metric["by_length"]["64"]["top1"] = .92
        self.assertFalse(release_gate(metric, per_seed, comparison)["passed"])


if __name__ == "__main__":
    unittest.main()
