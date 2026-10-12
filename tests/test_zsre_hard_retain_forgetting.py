"""Tests for same-subject hard-retain ZsRE construction."""

import unittest

from tasks.build_zsre_hard_retain_forgetting import auxiliary, pack
from tasks.evaluate_zsre_hard_retain_forgetting import active_queries


def primary():
    return {"id": "main", "split": "calibration", "subject": "subject", "versions": [
        {"version": 0, "prompt": "main q", "rephrase_prompt": "main r", "target": "new",
         "source_row_sha256": "x"},
        {"version": 1, "prompt": "main q2", "rephrase_prompt": "main r2", "target": "new2",
         "source_row_sha256": "y"}]}


def row(equal=True):
    return {"locality": {"Relation_Specificity": [
        {"prompt": "local q", "ground_truth": ["answer"]},
        {"prompt": "local r", "ground_truth": ["answer" if equal else "different"]}]}}


class HardRetainTests(unittest.TestCase):
    def test_auxiliary_uses_two_aliases_and_shared_value(self):
        aux = auxiliary(primary(), row())
        self.assertEqual(aux["versions"][0]["prompt"], "local q")
        self.assertEqual(aux["versions"][0]["rephrase_prompt"], "local r")
        self.assertEqual(aux["versions"][0]["target"], "answer")

    def test_disagreeing_aliases_are_rejected(self):
        with self.assertRaises(ValueError): auxiliary(primary(), row(False))

    def test_event_order_contains_hard_retain_before_local(self):
        main = primary(); aux = {"id": "aux", "versions": [{}]}
        events = pack([(main, aux)], "calibration")[0]["events"]
        self.assertEqual([(e["chain_id"], e["action"]) for e in events],
                         [("main", "retain"), ("aux", "retain"), ("main", "local")])
        self.assertEqual([e["kind"] for e in events],
                         ["primary_initial", "hard_retain", "primary_update"])

    def test_primary_and_auxiliary_share_bootstrap_group(self):
        main = primary(); main["kind"] = "primary"
        aux = {"id": "aux", "kind": "auxiliary", "parent_chain_id": "main",
               "versions": [{"prompt": "aux q", "rephrase_prompt": "aux r"}]}
        session = {"chains": [main, aux]}
        rows = active_queries(session, {"main": 0, "aux": 0})
        self.assertEqual({row["group_id"] for row in rows}, {"main"})
        self.assertEqual({row["chain_kind"] for row in rows}, {"primary", "auxiliary"})


if __name__ == "__main__": unittest.main()
