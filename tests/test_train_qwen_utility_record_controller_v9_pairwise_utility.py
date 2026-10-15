import unittest

from tasks.train_qwen_utility_record_controller_v9_pairwise_utility import (
    grouped_batches,
    pairwise_accuracy,
    v9_gate,
)


class TestV9PairwiseUtility(unittest.TestCase):
    def test_grouped_batches_keep_boundary_complete(self):
        rows = [
            {"training_domain": "d", "episode_id": "e",
             "boundary_index": boundary}
            for boundary in range(100) for _ in range(3)
        ]
        batches = grouped_batches(rows, 7)
        locations = {}
        for batch_index, batch in enumerate(batches):
            for index in batch:
                boundary = rows[index]["boundary_index"]
                locations.setdefault(boundary, set()).add(batch_index)
        self.assertTrue(all(len(value) == 1 for value in locations.values()))

    def test_pairwise_accuracy_is_boundary_local(self):
        examples = [
            {"episode_id": "e", "boundary_index": 0, "slot": 0,
             "benefit_target": 1},
            {"episode_id": "e", "boundary_index": 0, "slot": 1,
             "benefit_target": 0},
        ]
        predictions = {
            ("e", 0, 0): {"benefit": .9},
            ("e", 0, 1): {"benefit": .1},
        }
        self.assertEqual(pairwise_accuracy(examples, predictions)["accuracy"], 1.0)

    def test_gate_requires_effect_and_v8_gain(self):
        mechanisms = {name: 0.0 for name in (
            "different_subject_same_relation", "explicit_reset_revoke",
            "mixed_lifecycle", "same_subject_different_relation",
            "semantic_duplicate", "semantic_overwrite", "stale_repetition",
        )}
        development = {name: {"oracle_headroom": .02, "physics": {
            "per_mechanism_accuracy": mechanisms,
        }} for name in ("templama", "lama")}
        metric_tt = {"gain_over_lifecycle_physics": .04,
                     "oracle_gap_closure": 1.0, "added_unrelated_errors": 0,
                     "per_mechanism_gain": mechanisms}
        metric_lama = {"gain_over_lifecycle_physics": .016,
                       "oracle_gap_closure": .8, "added_unrelated_errors": 0,
                       "per_mechanism_gain": mechanisms}
        seeds = [{"selected_metric": {"domains": {
            "templama": metric_tt, "lama": metric_lama,
        }}} for _ in range(3)]
        gate = v9_gate(seeds, development, {"mean_gain": {"lama": .012}})
        self.assertTrue(gate["passed_pending_independent_audit"])


if __name__ == "__main__":
    unittest.main()
