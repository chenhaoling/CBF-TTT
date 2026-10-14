import json
from pathlib import Path
import tempfile
import unittest

from tasks.build_qwen_causal_ledger_residual import (
    CONFIRMATION_SEED,
    DEVELOPMENT_SEED,
    FAMILIES,
    PILOT_SEED,
    RESOURCE_V2_SEED,
    code_value,
    generate_group,
    load_pilot,
    materialize_pilot,
    null_value,
    parse_visible_block,
    validate,
)
from tasks.build_qwen_causal_ledger_residual_v2 import (
    load as load_v2,
    materialize as materialize_v2,
)
from tasks.build_qwen_causal_ledger_development_v2 import (
    GROUPS_PER_FAMILY as DEVELOPMENT_GROUPS_PER_FAMILY,
    load as load_development_v2,
    materialize as materialize_development_v2,
)
from tasks.pilot_qwen_causal_ledger_residual import (
    ACTIONS,
    apply_grid_action,
    lexicographic_label,
    semantic_apply,
    visible_facts,
)
from tasks.diagnose_qwen_causal_ledger_context_scale import (
    SCALES as CONTEXT_SCALES,
    select_scale,
)


class CausalLedgerResidualCorpusTest(unittest.TestCase):
    def test_all_phases_and_families_are_causal_and_parseable(self):
        pairs = []
        for phase, seed in (
            ("pilot", PILOT_SEED),
            ("development", DEVELOPMENT_SEED),
            ("confirmation", CONFIRMATION_SEED),
        ):
            for family in FAMILIES:
                group = generate_group(seed, phase, family, 2)
                pairs.extend(group)
                for inp, target in group:
                    for block, semantic in zip(inp["blocks"], target["semantic_events"]):
                        parsed = parse_visible_block(block["text"], phase)
                        self.assertEqual(parsed["record"], semantic[-1]["record"])
                        self.assertEqual(parsed["facts"], semantic[-1]["facts"])
        audit = validate(pairs)
        self.assertTrue(audit["causal_answers_verified"])
        self.assertTrue(audit["phase_symbols_disjoint"])

    def test_candidates_have_truth_null_and_six_unique_codes(self):
        for family in FAMILIES:
            for inp, target in generate_group(PILOT_SEED, "pilot", family, 1):
                answers = {row["id"]: row for row in target["answers"]}
                for query in inp["queries"]:
                    self.assertEqual(len(query["candidates"]), 6)
                    self.assertEqual(len(set(query["candidates"])), 6)
                    self.assertIn(answers[query["id"]]["answer"], query["candidates"])
                    self.assertIn(null_value("pilot"), query["candidates"])

    def test_registered_pilot_materialization_and_hash_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            first = materialize_pilot(root)
            manifest, pairs = load_pilot(root)
            self.assertEqual(first, manifest)
            self.assertEqual(len(pairs), 8 * len(FAMILIES) * 2)
            self.assertFalse(manifest["development_generated"])
            with (root / "pilot.inputs.jsonl").open("a") as handle:
                handle.write(json.dumps({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_pilot(root)

    def test_registered_pilot_size_and_seed_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "eight groups"):
                materialize_pilot(Path(directory) / "bad", groups_per_family=7)
            with self.assertRaisesRegex(ValueError, "frozen seed"):
                materialize_pilot(Path(directory) / "bad_seed", seed=PILOT_SEED + 1)

    def test_semantic_ledger_restores_older_active_assignment(self):
        state = []
        state = semantic_apply(state, {
            "batch": "r0", "facts": {"k": "old"}, "old_refs": [], "reset": False,
        }, 1)
        state = semantic_apply(state, {
            "batch": "r1", "facts": {"k": "new"}, "old_refs": [], "reset": False,
        }, 2)
        self.assertEqual(visible_facts(state)["k"]["value"], "new")
        state = semantic_apply(state, {
            "batch": "r2", "facts": {"x": "v"}, "old_refs": ["r1"], "reset": False,
        }, 3)
        self.assertEqual(visible_facts(state)["k"]["value"], "old")

    def test_grid_action_changes_only_selected_records_and_keeps_current(self):
        state = [
            {"batch": "r0", "facts": {"a": "x"}, "weight": 1.0, "block_index": 1},
            {"batch": "r1", "facts": {"b": "y"}, "weight": 1.0, "block_index": 2},
        ]
        changed = apply_grid_action(state, {
            "batch": "r2", "facts": {"c": "z"}, "old_refs": [], "reset": False,
        }, 3, (0, 1), (0, (0.0, 0.5, 1.0)))
        self.assertEqual([row["weight"] for row in changed], [0.0, 0.5, 1.0])
        self.assertEqual([row["weight"] for row in state], [1.0, 1.0])

    def test_lexicographic_label_prefers_accuracy_before_nll(self):
        actions = ACTIONS[:3]
        outcomes = []
        for accuracy, nll, removed in ((2 / 3, 0.1, 0), (1.0, 3.0, 1), (1.0, 2.0, 2)):
            outcomes.append({
                "8.0": {"accuracy": accuracy, "nll": nll, "removed_mass": removed}
            })
        index, action = lexicographic_label(outcomes, actions)
        self.assertEqual(index, 2)
        self.assertEqual(action, actions[2])

    def test_context_scale_selection_uses_smallest_passing_scale(self):
        table = {
            str(scale): {
                "free_exact": 0.90 if scale >= 32 else 0.80,
                "teacher_sequence_exact": 0.97 if scale >= 24 else 0.90,
            }
            for scale in CONTEXT_SCALES
        }
        self.assertEqual(select_scale(table), 32.0)
        for row in table.values():
            row["free_exact"] = 0.80
        self.assertIsNone(select_scale(table))

    def test_v2_resource_namespace_is_fresh_and_hash_locked(self):
        self.assertTrue(code_value("resource_v2", 0).startswith("Y "))
        self.assertTrue(code_value("pilot", 0).startswith("Z "))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v2"
            manifest = materialize_v2(root)
            loaded, pairs = load_v2(root)
            self.assertEqual(manifest, loaded)
            self.assertEqual(manifest["seed"], RESOURCE_V2_SEED)
            self.assertEqual(len(pairs), 8 * len(FAMILIES) * 2)
            self.assertTrue(all(
                row["answer"].startswith("Y ")
                for _, target in pairs for row in target["answers"]
            ))

    def test_development_v2_size_seed_and_sealing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "development"
            manifest = materialize_development_v2(root)
            loaded, pairs = load_development_v2(root)
            self.assertEqual(manifest, loaded)
            self.assertEqual(manifest["groups_per_family"], DEVELOPMENT_GROUPS_PER_FAMILY)
            self.assertEqual(len(pairs), DEVELOPMENT_GROUPS_PER_FAMILY * len(FAMILIES) * 2)
            self.assertFalse(manifest["confirmation_generated"])
            self.assertFalse(manifest["ood_generated"])


if __name__ == "__main__":
    unittest.main()
