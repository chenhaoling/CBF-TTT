"""Tests for full ZsRE counterfactual forgetting sequence construction."""

import unittest

from tasks.build_zsre_counterfactual_forgetting import chains, pack


def fact():
    return {"subject": "Ada", "prompt": "Who is Ada's mother?",
            "rephrase_prompt": "Which person gave birth to Ada?", "original": "Ann",
            "updated": "Beth", "source_row_sha256": "hash"}


def source():
    from tasks.build_zsre_counterfactual_forgetting import source_hash
    row = {"locality": {"Relation_Specificity": [
        {"prompt": "Ada occupation", "ground_truth": ["scientist"]},
        {"prompt": "What job does Ada have?", "ground_truth": ["scientist"]}]}}
    row["_computed_hash"] = source_hash(row)
    return row


class CounterfactualForgettingTests(unittest.TestCase):
    def test_sequence_uses_original_then_counterfactual(self):
        row = source(); item = fact(); item["source_row_sha256"] = row.pop("_computed_hash")
        primary, auxiliary = chains(item, row, "calibration")
        self.assertEqual([version["target"] for version in primary["versions"]], ["Ann", "Beth"])
        self.assertEqual(primary["versions"][1]["prompt"], item["rephrase_prompt"])
        self.assertEqual(auxiliary["versions"][0]["target"], "scientist")

    def test_pack_orders_two_retains_before_local_update(self):
        row = source(); item = fact(); item["source_row_sha256"] = row.pop("_computed_hash")
        primary, auxiliary = chains(item, row, "calibration")
        events = pack([(primary, auxiliary)], "calibration")[0]["events"]
        self.assertEqual([(event["action"], event["kind"]) for event in events], [
            ("retain", "primary_initial"), ("retain", "hard_retain"),
            ("local", "primary_update")])


if __name__ == "__main__": unittest.main()
