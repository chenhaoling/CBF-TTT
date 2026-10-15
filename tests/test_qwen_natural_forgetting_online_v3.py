"""Pure accounting tests for on-policy natural forgetting V3."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tasks.evaluate_qwen_natural_forgetting_online_v3 import (
    ENGINEERED,
    decide,
    paired_accounting,
    quantile,
)


class FakeFactor:
    configs = {
        "engineered": {"benefit_threshold": .8, "risk_threshold": .1},
        "no_schedule": {"benefit_threshold": .5, "risk_threshold": .2},
    }

    def __init__(self, probabilities):
        self.values = probabilities

    def probabilities(self, features, feature_set="engineered"):
        return self.values[feature_set]


class OnlineV3Tests(unittest.TestCase):
    def test_quantile_uses_registered_conservative_index(self):
        self.assertEqual(quantile([4, 1, 3, 2], .25), 1)
        self.assertEqual(quantile([4, 1, 3, 2], 1.), 4)

    @patch("tasks.evaluate_qwen_natural_forgetting_online_v3.engineered_features")
    def test_factor_ensemble_deletes_exact_current_top(self, feature_mock):
        feature_mock.return_value = ([0.] * len(ENGINEERED), 1, .9)
        factor = FakeFactor({
            "engineered": {9401: (.9, .05), 9402: (.85, .08), 9403: (.88, .02)},
            "no_schedule": {9401: (.9, .05), 9402: (.9, .05), 9403: (.9, .05)},
        })
        memory = [{"chain_id": "a", "prompt": "A"},
                  {"chain_id": "b", "prompt": "B"}]
        event = {"kind": "primary_update", "action": "local", "chain_id": "b"}
        updated, record = decide(
            "factor_ensemble", memory, event, {"prompt": "new"}, 3, 8,
            object(), object(), factor)
        self.assertEqual([row["chain_id"] for row in updated], ["a"])
        self.assertEqual(record["deleted"][0]["chain_id"], "b")
        self.assertEqual(record["action"], "scope_top1_a0")
        self.assertEqual(record["before_size"], 2)

    @patch("tasks.evaluate_qwen_natural_forgetting_online_v3.engineered_features")
    def test_risk_and_benefit_gates_are_distinct(self, feature_mock):
        feature_mock.return_value = ([0.] * len(ENGINEERED), 0, .9)
        factor = FakeFactor({
            "engineered": {9401: (.9, .4), 9402: (.9, .4), 9403: (.9, .4)},
            "no_schedule": {9401: (.9, .4), 9402: (.9, .4), 9403: (.9, .4)},
        })
        memory = [{"chain_id": "a", "prompt": "A"}]
        event = {"kind": "primary_update", "action": "local", "chain_id": "a"}
        guarded = decide("factor_ensemble", memory, event, {"prompt": "new"},
                         1, 2, object(), object(), factor)[1]
        benefit = decide("benefit_only_ensemble", memory, event, {"prompt": "new"},
                         1, 2, object(), object(), factor)[1]
        self.assertEqual(guarded["action"], "retain_all")
        self.assertEqual(benefit["action"], "scope_top1_a0")

    def test_paired_accounting_separates_collateral_errors(self):
        rows = [
            {"chain_id": "a", "recent_chain_id": "a",
             "arms": {"retain_all": {"correct": 1}, "policy": {"correct": 0}}},
            {"chain_id": "b", "recent_chain_id": "a",
             "arms": {"retain_all": {"correct": 1}, "policy": {"correct": 0}}},
            {"chain_id": "c", "recent_chain_id": "a",
             "arms": {"retain_all": {"correct": 0}, "policy": {"correct": 1}}},
        ]
        self.assertEqual(paired_accounting(rows, "policy"), {
            "added_errors": 2, "recovered_errors": 1, "collateral_added_errors": 1,
        })


if __name__ == "__main__":
    unittest.main()
