"""Tests for natural ZsRE forgetting policy replay."""

import unittest

from tasks.evaluate_zsre_natural_forgetting import POLICIES, active_queries, apply_update


class MockScorer:
    def __init__(self, top=0, retrieve=True, cosine=True):
        self.result = {"top": top, "retrieve": retrieve, "cosine_retrieve": cosine}
    def score(self, query, keys, seed): return self.result


def version(name="q", target="v"):
    return {"prompt": name, "rephrase_prompt": name + " r", "target": target,
            "locality_prompts": [name + " l1", name + " l2"]}


class ReplayTests(unittest.TestCase):
    def test_registered_policies(self):
        self.assertEqual(POLICIES, ("fixed_retain", "fixed_global", "fixed_top",
                                   "cosine_local", "learned_local", "oracle_local"))

    def test_learned_local_requires_correct_chain_deletion(self):
        memory = [{"chain_id": "a", "version": 0, "prompt": "a", "target": "x"},
                  {"chain_id": "b", "version": 0, "prompt": "b", "target": "y"}]
        event = {"chain_id": "b", "version": 1, "action": "local"}
        _, good = apply_update("learned_local", memory, event, version("b2"), MockScorer(top=1), 1)
        _, bad = apply_update("learned_local", memory, event, version("b2"), MockScorer(top=0), 1)
        self.assertEqual((good["correct"], bad["correct"]), (1, 0))

    def test_historical_queries_target_current_version(self):
        chain = {"id": "a", "versions": [version("old"), version("new")]}
        rows = active_queries({"chains": [chain]}, {"a": 1})
        historical = [row for row in rows if row["type"] == "historical"]
        self.assertEqual(len(historical), 2)
        self.assertTrue(all(row["active_version"] == 1 for row in historical))


if __name__ == "__main__": unittest.main()
