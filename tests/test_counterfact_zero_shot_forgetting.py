"""Construction and gate tests for CounterFact zero-shot transfer."""

import unittest

from tasks.build_counterfact_zero_shot_forgetting import make_chains, pack, select_pairs
from tasks.evaluate_counterfact_zero_shot_forgetting import transfer_gate


def row(case, subject, relation, old, new, paraphrase):
    return {"case_id": case, "requested_rewrite": {
        "prompt": "{} has " + relation, "relation_id": relation,
        "target_true": {"str": old}, "target_new": {"str": new}, "subject": subject},
        "paraphrase_prompts": [paraphrase]}


class CounterFactConstructionTests(unittest.TestCase):
    def test_selection_requires_same_subject_distinct_relations(self):
        rows = [row(1, "A", "r1", "old", "new", "A relation one"),
                row(2, "A", "r2", "aux", "unused", "A relation two"),
                row(3, "B", "r1", "x", "y", "B relation")]
        selected = select_pairs(rows)
        self.assertEqual(len(selected), 1)
        primary, auxiliary = make_chains(*selected[0])
        self.assertEqual(primary["subject"], auxiliary["subject"])
        self.assertNotEqual(primary["versions"][0]["prompt"], auxiliary["versions"][0]["prompt"])
        session = pack([(primary, auxiliary)])[0]
        self.assertEqual([event["kind"] for event in session["events"]],
                         ["primary_initial", "hard_retain", "primary_update"])

    def test_transfer_gate_requires_relative_and_absolute_checks(self):
        controller = {"mean_action_accuracy": .91, "by_event_kind": {
            "hard_retain": {"accuracy": .92}, "primary_update": {"accuracy": .86}}}
        comparison = {"query_bootstrap_95ci": [.01, .02],
                      "action_bootstrap_95ci": [.01, .03]}
        self.assertTrue(transfer_gate(controller, comparison)["passed"])
        controller["by_event_kind"]["primary_update"]["accuracy"] = .84
        self.assertFalse(transfer_gate(controller, comparison)["passed"])


if __name__ == "__main__":
    unittest.main()
