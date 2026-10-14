import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:  # The repository's lightweight local test host.
    torch = None

from tasks.build_qwen_dynamic_record_forgetting import (
    DATA_SEED,
    FAMILIES,
    generate_group,
    load_corpus,
    materialize_pilot,
    parse_visible_block,
    replay,
    validate,
)
from tasks.pilot_qwen_dynamic_record_forgetting import (
    ACTIONS,
    apply_dynamic_action,
    candidate_indices,
    lexicographic_label,
    valid_actions,
    summarize,
)


class FakeEmbeddings:
    seed = 2301
    dim = 3

    def get(self, kind, text):
        del kind
        value = sum(text.encode()) % 3
        output = torch.zeros(3)
        output[value] = 1.0
        return output


class DynamicRecordCorpusTest(unittest.TestCase):
    def test_every_family_is_causal_and_parseable(self):
        pairs = []
        for family in FAMILIES:
            pairs.extend(generate_group(DATA_SEED, "pilot", family, 0))
        audit = validate(pairs)
        self.assertTrue(audit["causal_answers_verified"])
        self.assertTrue(audit["counterfactual_twins_verified"])
        for inp, target in pairs:
            events = []
            for block, semantic in zip(inp["blocks"], target["semantic_events"]):
                parsed = parse_visible_block(block["text"], "pilot")
                self.assertEqual(parsed["record"], semantic[-1]["record"])
                events.extend(semantic)
            self.assertIsInstance(replay(events), dict)

    def test_registered_pilot_materialization_and_hash_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            first = materialize_pilot(root)
            manifest, pairs = load_corpus(root)
            self.assertEqual(first, manifest)
            self.assertEqual(len(pairs), 8 * len(FAMILIES) * 2)
            self.assertEqual(manifest["audit"]["groups"], 8 * len(FAMILIES))
            with (root / "pilot.inputs.jsonl").open("a") as handle:
                handle.write(json.dumps({"tampered": True}) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_corpus(root)

    def test_pilot_size_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly eight"):
                materialize_pilot(Path(directory) / "bad", groups_per_family=7)

    @unittest.skipIf(torch is None, "torch is exercised in the remote project environment")
    def test_dynamic_candidates_and_physical_masks(self):
        state = [
            {"batch": "r0", "facts": {"unit_a": "v0"}, "delta": torch.eye(3),
             "weight": 1.0, "block_index": 1},
            {"batch": "r1", "facts": {"unit_b": "v1"}, "delta": torch.eye(3),
             "weight": 1.0, "block_index": 2},
            {"batch": "r2", "facts": {"unit_a": "v2"}, "delta": torch.eye(3),
             "weight": 1.0, "block_index": 3},
        ]
        assignment = {"batch": "r3", "facts": {"unit_a": "v3"},
                      "old_refs": ["r0"], "reset": False}
        slots = candidate_indices(state, assignment, FakeEmbeddings())
        self.assertEqual(slots[:2], (0, 2))
        actions = valid_actions(assignment, slots)
        self.assertTrue(all(action[0] == 0 for action in actions))
        changed = apply_dynamic_action(
            state, assignment, torch.eye(3), 4, slots, (0, (0.0, 0.5, 1.0))
        )
        self.assertEqual(changed[0]["weight"], 0.0)
        self.assertEqual(changed[2]["weight"], 0.5)
        self.assertEqual(changed[-1]["weight"], 1.0)

    def test_lexicographic_label_prioritizes_accuracy_then_nll(self):
        actions = ACTIONS[:3]
        outcomes = {
            actions[0]: {"accuracy": 2 / 3, "nll": 0.1, "removed_mass": 0.0},
            actions[1]: {"accuracy": 1.0, "nll": 3.0, "removed_mass": 1.0},
            actions[2]: {"accuracy": 1.0, "nll": 2.0, "removed_mass": 2.0},
        }
        self.assertEqual(lexicographic_label(outcomes, actions), actions[2])

    def test_resource_rule_scales_from_total_groups(self):
        rows = []
        for seed in (2301, 2302, 2303):
            for group in range(56):
                rows.append({
                    "group_id": str(group), "projection_seed": seed,
                    "family": FAMILIES[group % len(FAMILIES)],
                    "label_seconds": 1.0, "seconds_per_action": 1 / 27,
                    "peak_allocated_gib": 8.0, "peak_reserved_gib": 9.0,
                    "max_prompt_tokens": 128, "max_answer_tokens": 5,
                    "active_records": 12, "valid_action_count": 27,
                    "label": [0, [1.0, 1.0, 1.0]],
                })
        summary = summarize(
            rows,
            wall_seconds=56.0,
            extraction_peak_allocated_gib=8.0,
            extraction_peak_reserved_gib=9.0,
        )
        self.assertEqual(summary["estimated_two_gpu_seconds"]["512"], 512.0)
        self.assertEqual(summary["selected_formal_train_groups"], 512)
        self.assertEqual(summary["overall_peak_allocated_gib"], 8.0)


if __name__ == "__main__":
    unittest.main()
