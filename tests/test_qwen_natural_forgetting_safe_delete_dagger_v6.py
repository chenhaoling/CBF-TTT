import json
from pathlib import Path
import tempfile
import unittest

from tasks.train_qwen_natural_forgetting_factor_gate_v2 import NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import (
    core_row,
    merged_corpus,
    oof_metrics,
)


def row(identity="x", safe=1, feature=0.0):
    return {
        "protocol": "fixture", "state_id": identity, "session_id": "s",
        "outer_fold": 0, "event_step": 0, "memory": [], "top": None,
        "incoming_chain_id": "c", "incoming_version": 0,
        "event_kind": "initial", "desired_source_action": "retain",
        "features": [feature] * len(NO_SCHEDULE), "safe_delete": safe,
        "origin_arms": ["base"],
    }


class SafeDeleteDaggerV6Test(unittest.TestCase):
    def test_core_row_excludes_behavior_outputs(self):
        value = {**row(), "v5_probability": .9, "v5_action": "retain_all"}
        self.assertNotIn("v5_probability", core_row(value))
        self.assertNotIn("v5_action", core_row(value))
        self.assertEqual(len(core_row(value)["features"]), 10)

    def test_merge_deduplicates_identical_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v5 = root / "v5" / "result"; v5.mkdir(parents=True)
            dagger = root / "dagger"; dagger.mkdir()
            (v5 / "safe_delete_corpus.jsonl").write_text(json.dumps(row()) + "\n")
            duplicate = {**row(), "origin_arms": ["dagger"],
                         "v5_probability": .8, "v5_action": "retain_all"}
            (dagger / "dagger_rows_shard0.jsonl").write_text(json.dumps(duplicate) + "\n")
            (dagger / "dagger_rows_shard1.jsonl").write_text("")
            merged = merged_corpus(root / "v5", dagger)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["origin_arms"], ["base", "dagger"])

    def test_merge_rejects_feature_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v5 = root / "v5" / "result"; v5.mkdir(parents=True)
            dagger = root / "dagger"; dagger.mkdir()
            (v5 / "safe_delete_corpus.jsonl").write_text(json.dumps(row()) + "\n")
            (dagger / "dagger_rows_shard0.jsonl").write_text(
                json.dumps({**row(feature=1.0), "origin_arms": ["dagger"]}) + "\n")
            (dagger / "dagger_rows_shard1.jsonl").write_text("")
            with self.assertRaises(ValueError):
                merged_corpus(root / "v5", dagger)

    def test_oof_metrics_respects_fold_specific_actions(self):
        rows = [row("a", 1), {**row("b", 0), "outer_fold": 1}]
        oof = [{"action": "scope_top1_a0"}, {"action": "retain_all"}]
        metrics = oof_metrics(rows, oof)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 0)


if __name__ == "__main__":
    unittest.main()
