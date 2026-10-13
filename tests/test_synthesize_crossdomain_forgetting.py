import copy
import unittest

from tasks.synthesize_crossdomain_forgetting import synthesize


def metric(events, correct):
    accuracy = correct / events
    return {
        "mean_accuracy": .6,
        "mean_action_accuracy": (events - 1 + events - 2 + correct) / (3 * events),
        "by_event_kind": {
            "primary_initial": {"accuracy": (events - 1) / events},
            "hard_retain": {"accuracy": (events - 2) / events},
            "primary_update": {"accuracy": accuracy},
        },
        "per_seed": {"1": {"sessions": 2, "actions": 3 * events,
                             "by_event_kind": {
                                 "primary_initial": {"events": events},
                                 "hard_retain": {"events": events},
                                 "primary_update": {"events": events}}}},
    }


def comparison():
    return {"query_accuracy_gain": .01, "query_bootstrap_95ci": [.001, .02],
            "action_accuracy_gain": .1, "action_bootstrap_95ci": [.05, .15]}


class SynthesisTest(unittest.TestCase):
    def setUp(self):
        self.cf = {"protocol": "causal_onpolicy_scope_v13",
                   "data_status": "observed_development_only",
                   "terminal_status": "completed_causal_onpolicy_scope_v13",
                   "arms": {"onpolicy_mlp": metric(100, 90)},
                   "onpolicy_vs_fixed_retain": comparison()}
        self.zsre = {"protocol": "zsre_counterfactual_forgetting_final_test_v1",
                     "test_loaded": True, "test_scored": True,
                     "terminal_status": "completed_one_time_heldout_test",
                     "test": {"controller": metric(100, 90), "comparison": comparison()}}

    def test_pooled_counts_and_status_are_explicit(self):
        result = synthesize(self.cf, self.zsre)
        self.assertTrue(result["replicated"])
        self.assertEqual(result["descriptive_pooled_counts"]["actions"], 600)
        self.assertEqual(result["descriptive_pooled_counts"]["by_event_kind"]["primary_update"]["correct"], 180)
        self.assertEqual(result["domains"]["counterfact"]["data_status"],
                         "grouped_oof_observed_development")
        self.assertFalse(result["claim_boundary"]["native_dense_fast_weight_proven"])

    def test_rejects_incomplete_heldout_test(self):
        bad = copy.deepcopy(self.zsre); bad["test_scored"] = False
        with self.assertRaises(ValueError):
            synthesize(self.cf, bad)

    def test_rejects_noninteger_reported_accuracy(self):
        bad = copy.deepcopy(self.cf)
        bad["arms"]["onpolicy_mlp"]["by_event_kind"]["primary_update"]["accuracy"] = .901
        with self.assertRaises(ValueError):
            synthesize(bad, self.zsre)


if __name__ == "__main__":
    unittest.main()
