"""Construction tests for natural ZsRE sequential forgetting chains."""

import unittest

from tasks.build_zsre_natural_forgetting import build_chains, pack


def row(subject, prompt, old, new, suffix):
    return {"subject": subject, "prompt": prompt, "rephrase_prompt": prompt + " rephrased",
            "ground_truth": [old], "target_new": new, "cond": suffix,
            "locality": {"Relation_Specificity": [
                {"prompt": prompt + " local a", "ground_truth": ["a"]},
                {"prompt": prompt + " local b", "ground_truth": ["b"]}]},
            "portability": {}}


class ConstructionTests(unittest.TestCase):
    def test_distinct_targets_form_ordered_chain(self):
        rows = [row("s", "q1", "old", "new a", "1"),
                row("s", "q2", "old", "new b", "2"),
                row("s", "q3", "old", "new a", "3")]
        from tasks.build_zsre_natural_forgetting import source_hash
        owner = {"s": source_hash(rows[1])}
        chains = build_chains(rows, owner, "calibration")
        self.assertEqual(len(chains), 1); self.assertEqual(len(chains[0]["versions"]), 2)
        self.assertEqual(chains[0]["versions"][0]["prompt"], "q2")

    def test_single_target_group_is_excluded(self):
        rows = [row("s", "q1", "old", "same", "1"), row("s", "q2", "old", "same", "2")]
        self.assertEqual(build_chains(rows, {"s": "none"}, "dev"), [])

    def test_round_robin_pack_has_retain_then_local(self):
        chains = [{"id": "a", "subject": "a", "versions": [{}, {}, {}]},
                  {"id": "b", "subject": "b", "versions": [{}, {}]}]
        events = pack(chains, "calibration")[0]["events"]
        self.assertEqual([(row["chain_id"], row["version"], row["action"]) for row in events],
                         [("a", 0, "retain"), ("b", 0, "retain"),
                          ("a", 1, "local"), ("b", 1, "local"), ("a", 2, "local")])


if __name__ == "__main__": unittest.main()
