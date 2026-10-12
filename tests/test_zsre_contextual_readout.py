"""Tests for subject-disjoint ZsRE readout construction and gates."""

import unittest

from tasks.build_zsre_memory_benchmark import select_rows, split_rows
from tasks.evaluate_zsre_contextual_readout import QUERY_TYPES, gate, select_config, SEEDS, UNKNOWN_GRID


def source_row(index, subject=None):
    subject = subject or f"subject_{index}"
    return {"subject": subject, "prompt": f"question {index}",
            "rephrase_prompt": f"rephrase {index}", "ground_truth": [f"old {index}"],
            "target_new": f"new {index}", "locality": {"Relation_Specificity": [
                {"prompt": f"local a {index}", "ground_truth": ["a"]},
                {"prompt": f"local b {index}", "ground_truth": ["b"]}]}}


class ZsREConstructionTests(unittest.TestCase):
    def test_subject_deduplication_and_split(self):
        rows = [source_row(index) for index in range(1068)]
        rows.append(source_row(9999, "subject_0"))
        selected = select_rows(rows); split = split_rows(selected)
        self.assertEqual(len(selected), 1068)
        self.assertEqual({name: len(part) for name, part in split.items()},
                         {"calibration": 267, "dev": 267, "test": 534})
        sets = {name: {row["subject"] for row in part} for name, part in split.items()}
        self.assertFalse(sets["calibration"] & sets["dev"])
        self.assertFalse(sets["dev"] & sets["test"])

    def test_config_selection_is_train_only_and_deterministic(self):
        grid = {}
        for regularizer, score in ((1e-4, .91), (1e-3, .905), (1e-2, .901)):
            label = f"context_last_r{regularizer:g}"; grid[label] = {}
            for threshold in UNKNOWN_GRID:
                value = score if threshold == .5 else score - .1
                grid[label][str(threshold)] = {str(seed): value for seed in SEEDS}
        selected = select_config(grid)
        self.assertEqual(selected["regularizer"], 1e-2)
        self.assertEqual(selected["unknown_raw"], .5)

    def test_gate_requires_exact_rephrase_locality_and_seed(self):
        by_type = {kind: {"accuracy": .95, "nll": .1} for kind in QUERY_TYPES}
        summary = {"by_type": by_type, "min_seed_accuracy": .9}
        self.assertTrue(gate(summary)["passed"])
        summary["by_type"]["updated_rephrase"]["accuracy"] = .69
        self.assertFalse(gate(summary)["passed"])


if __name__ == "__main__": unittest.main()
