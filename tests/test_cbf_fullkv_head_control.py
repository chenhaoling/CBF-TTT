import unittest

from tasks.train_cbf_fullkv_head_control import aggregate, feature_key, gate, required_items


class FullKVHeadControlTests(unittest.TestCase):
    def test_feature_key_is_assignment_specific(self):
        self.assertNotEqual(feature_key("c", 0, "q"), feature_key("c", 1, "q"))

    def test_required_features_cover_train_four_and_dev_two_shifts(self):
        def episode(split, group):
            return {shift: {"rows": [{"id": f"q{i}", "split": split, "group_id": group}
                                      for i in range(4)]} for shift in range(4)}
        rows = [{"context_id": "train", "split": "train"}, {"context_id": "dev", "split": "dev"}]
        items = required_items(rows, {"train": episode("train", "a"), "dev": episode("dev", "b")})
        self.assertEqual(len(items), 24)
        self.assertEqual({shift for context, shift, _ in items if context == "dev"}, {0, 2})

    def test_gate_requires_generalized_fullkv_readout(self):
        summary = {"dev": {"sets": {
            "context4": {"top1": .8},
            "global16": {"top1": .6, "top1_group_bootstrap_95ci": [.3, .8]},
        }}}
        self.assertTrue(gate(summary, .9)["passed"])
        summary["dev"]["sets"]["global16"]["top1_group_bootstrap_95ci"][0] = .05
        self.assertFalse(gate(summary, .9)["passed"])

    def test_aggregate_uses_all_rows(self):
        metric = {"top1": 1, "mrr": 1., "rank": 1, "margin": .2,
                  "true_digit_nll": .1, "true_answer_nll": .1, "prediction": "x"}
        records = []
        for split in ("train_probe", "dev"):
            for group in ("a", "b"):
                records.append({"evaluation_split": split, "group_id": group,
                                "sets": {"context4": metric, "global16": metric}})
        result = aggregate(records)
        self.assertEqual(result["dev"]["rows"], 2)
        self.assertEqual(result["dev"]["sets"]["global16"]["top1"], 1)


if __name__ == "__main__":
    unittest.main()
